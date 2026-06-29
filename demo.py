import numpy as np
import argparse
import os
import glob
import cv2
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import Sam3Processor, Sam3Model
from PIL import Image

from amr.models import MultiAMR
from amr.configs import get_config
from amr.utils import recursive_to
from amr.utils.renderer import Renderer, cam_full_to_crop, cam_crop_s_to_t
from amr.datasets.utils import gen_trans_from_patch_cv, convert_cvimg_to_tensor

LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)
MAPPING = [0, 1, -1, 7, 10, 13, 16, 4, 5, 8, 11, 14, -1, -1, 6, 9, 12, 15, 3, -1, -1, -1, -1, -1, 2, -1]
BBOX_EXPAND_FACTOR = 1.2


def ensure_egl_rendering():
    os.environ["PYOPENGL_PLATFORM"] = "egl"


def load_image_paths(input_path: str):
    """Load image paths from a single image file or a folder of images."""
    if os.path.isfile(input_path):
        return [input_path]
    elif os.path.isdir(input_path):
        exts = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        paths = []
        for ext in exts:
            paths.extend(glob.glob(os.path.join(input_path, ext)))
        paths.sort()
        return paths
    else:
        raise ValueError(f"Input path {input_path} is not a valid file or directory.")


def load_sam3_model():
    """Load SAM3 model from Hugging Face."""
    processor = Sam3Processor.from_pretrained("data/sam3")
    model = Sam3Model.from_pretrained("data/sam3")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    return processor, model, device


def segment_animals_with_sam3(processor, model, device, image_rgb, text_prompt="animals", score_threshold=0.5):
    """
    Use SAM3 to segment animals in the image with text prompt.
    Returns masks, bounding boxes, and confidence scores.
    Only returns detections above score_threshold.
    """
    # Convert numpy array to PIL Image
    pil_image = Image.fromarray(image_rgb)

    # Prepare inputs with text prompt
    inputs = processor(images=pil_image, text=text_prompt, return_tensors="pt").to(device)

    # Generate masks
    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process results with correct parameters
    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=0.5,
        mask_threshold=0.5,
        target_sizes=inputs.get("original_sizes").tolist()
    )[0]

    # Extract masks, bboxes and scores from results
    valid_masks = []
    bboxes = []
    scores = []

    result_masks = results.get('masks', [])
    result_scores = results.get('scores', [])
    result_boxes = results.get('boxes', [])

    for i, mask in enumerate(result_masks):
        mask_np = mask.cpu().numpy() if torch.is_tensor(mask) else mask
        if mask_np.sum() == 0:
            continue

        # Get score (use returned score if available, else default 1.0)
        score = float(result_scores[i].item()) if i < len(result_scores) and torch.is_tensor(result_scores[i]) else 1.0
        if score < score_threshold:
            continue

        valid_masks.append(mask_np)
        scores.append(score)

        # Use returned bbox if available, otherwise compute from mask
        if i < len(result_boxes):
            box = result_boxes[i]
            if torch.is_tensor(box):
                box = box.cpu().numpy()
            bboxes.append(np.array(box, dtype=np.float32))
        else:
            ys, xs = np.where(mask_np > 0)
            x1, y1 = xs.min(), ys.min()
            x2, y2 = xs.max(), ys.max()
            bboxes.append(np.array([x1, y1, x2, y2], dtype=np.float32))

    return valid_masks, bboxes, np.array(scores, dtype=np.float32)


def xyxy_to_coco(xyxy: np.ndarray) -> np.ndarray:
    """Convert bbox from [x1, y1, x2, y2] to COCO format [x, y, w, h]."""
    bbox_coco = xyxy.astype(np.float32).copy()
    bbox_coco[2] = bbox_coco[2] - bbox_coco[0]
    bbox_coco[3] = bbox_coco[3] - bbox_coco[1]
    return bbox_coco


def infer_vitpose(vitpose_model, image_rgb: np.ndarray, bboxes: list) -> dict:
    """
    Run ViTPose inference on the image with detected bboxes.

    Args:
        vitpose_model: ViTPose model instance
        image_rgb: RGB image array [H, W, 3]
        bboxes: List of bboxes in [x1, y1, x2, y2] format

    Returns:
        Dictionary with keypoints for each detected animal
    """
    if len(bboxes) == 0:
        return {}

    keypoints_list = []
    for bbox_xyxy in bboxes:
        if bbox_xyxy[2] <= bbox_xyxy[0] or bbox_xyxy[3] <= bbox_xyxy[1]:
            keypoints_list.append(np.zeros((0, 3), dtype=np.float32))
            continue

        bbox_coco = xyxy_to_coco(bbox_xyxy)
        pose_results = vitpose_model(image_rgb, bbox_coco)

        image_pose_result, _ = pose_results
        if len(image_pose_result) == 0:
            keypoints_list.append(np.zeros((0, 3), dtype=np.float32))
            continue

        # Take the first (best) detection
        pose_result = image_pose_result[0]
        kpts = pose_result["keypoints"]  # [K, 3] with (x, y, score)
        if isinstance(kpts, torch.Tensor):
            kpts = kpts.detach().cpu().numpy()
        kpts = kpts[MAPPING, :]
        kpts[np.where(np.array(MAPPING) == -1)[0], :] = np.array([0, 0, 0])
        keypoints_list.append(kpts.astype(np.float32))

    # Debug: visualize keypoints on image
    # debug_img = image_rgb.copy()
    # for animal_idx, kpts in enumerate(keypoints_list):
    #     for idx, (x, y, score) in enumerate(kpts):
    #         if score > 0.3:  # Only draw confident keypoints
    #             cv2.circle(debug_img, (int(x), int(y)), 5, (255, 0, 0), -1)
    #             # cv2.putText(debug_img, f"{animal_idx}_{idx}", (int(x)+5, int(y)-5),
    #             #            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    # cv2.imwrite(f'debug_kpts_{len(keypoints_list)}.jpg', cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR))

    return {"keypoints": keypoints_list}


def preprocess_image(image_rgb: np.ndarray, cfg, merged_mask=None, vitpose_keypoints=None):
    """
    Preprocess a full RGB image for multi-animal model inference.
    The full image is resized (with aspect-ratio-preserving padding) into the
    model's square input size, and the corresponding batch dict is constructed.

    Args:
        image_rgb: RGB image array [H, W, 3]
        cfg: Config object
        merged_mask: Optional merged segmentation mask [H, W], same size as image_rgb
        vitpose_keypoints: Optional dict with keypoints from ViTPose
    """
    h, w = image_rgb.shape[:2]
    img_size = cfg.MODEL.IMAGE_SIZE
    mean = 255.0 * np.array(cfg.MODEL.IMAGE_MEAN)
    std = 255.0 * np.array(cfg.MODEL.IMAGE_STD)
    focal_length = cfg.EXTRA.FOCAL_LENGTH

    center_x, center_y = w / 2.0, h / 2.0
    bbox_size = float(max(w, h))

    # Prepare mask: if not provided, use all-white mask
    if merged_mask is None:
        merged_mask = np.ones((h, w), dtype=np.uint8) * 255
    else:
        # Convert to uint8 [0, 255] range for consistency with training
        merged_mask = (merged_mask * 255).astype(np.uint8)

    # Concatenate image and mask as RGBA (same as training pipeline)
    img_rgba = np.concatenate([image_rgb, merged_mask[:, :, None]], axis=2)

    # Affine transform that maps the original image into the model input patch
    trans = gen_trans_from_patch_cv(
        center_x, center_y, bbox_size, bbox_size,
        img_size, img_size, 1.0, 0
    )

    # Apply affine transform to RGBA image (same as training)
    img_patch_rgba = cv2.warpAffine(
        img_rgba, trans, (img_size, img_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    # Split into RGB and mask channels
    img_patch_rgb = img_patch_rgba[:, :, :3]
    mask_patch = img_patch_rgba[:, :, 3]

    # Convert image to tensor and normalize
    img_patch = convert_cvimg_to_tensor(img_patch_rgb)  # [C, H, W]
    for c in range(3):
        img_patch[c] = (img_patch[c] - mean[c]) / std[c]

    # Process mask: normalize to [0, 1] and check validity
    mask = (mask_patch / 255.0).clip(0, 1).astype(np.float32)
    if (mask < 0.5).all():
        mask = np.ones_like(mask)
    mask_score = 1.0 if mask.sum() > 0 else 0.0

    cam_int = np.array([
        [focal_length, 0.0, w / 2.0],
        [0.0, focal_length, h / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)

    # Prepare targets dict with keypoints if available
    targets_dict = {}
    if vitpose_keypoints is not None and 'keypoints' in vitpose_keypoints:
        kpts_list = vitpose_keypoints['keypoints']
        # Transform keypoints to crop space and stack for all animals
        transformed_kpts = []
        for kpts in kpts_list:
            if kpts.shape[0] > 0:
                # Apply affine transform to keypoints
                kpts_xy = kpts[:, :2]  # [K, 2]
                ones = np.ones((kpts_xy.shape[0], 1))
                kpts_homo = np.concatenate([kpts_xy, ones], axis=1)  # [K, 3]
                kpts_transformed = (trans @ kpts_homo.T).T  # [K, 2] pixel coords
                kpts_transformed = kpts_transformed / img_size - 0.5  # normalize to [-0.5, 0.5]
                # Combine with scores
                kpts_with_score = np.concatenate([kpts_transformed, kpts[:, 2:3]], axis=1)
                transformed_kpts.append(kpts_with_score)
            else:
                # If no keypoints detected, create zero array with expected shape
                transformed_kpts.append(np.zeros((20, 3), dtype=np.float32))  # Assuming 20 keypoints

        # Stack all keypoints: [num_animals, num_keypoints, 3]
        if len(transformed_kpts) > 0:
            keypoints_2d = np.stack(transformed_kpts, axis=0)
            targets_dict['keypoints_2d'] = torch.from_numpy(keypoints_2d).float()

            # Debug: visualize transformed keypoints on crop image
            # debug_img = img_patch_rgb.copy()
            # colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
            # for animal_idx, kpts in enumerate(transformed_kpts):
            #     color = colors[animal_idx % len(colors)]
            #     for kpt_idx, (x, y, score) in enumerate(kpts):
            #         if score > 0.3:  # Only draw confident keypoints
            #             cv2.circle(debug_img, (int(x), int(y)), 3, color, -1)
            #             # cv2.putText(debug_img, f"{animal_idx}_{kpt_idx}", (int(x)+5, int(y)-5),
            #             #            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            # cv2.imwrite('debug_transformed_kpts.jpg', cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR))

    batch = {
        'img': torch.from_numpy(img_patch).unsqueeze(0).float(),
        'mask': torch.from_numpy(mask).unsqueeze(0).float(),
        'mask_score': torch.tensor([mask_score]).float(),
        'bbox_center': torch.tensor([[center_x, center_y]]).float(),
        'bbox_scale': torch.tensor([[bbox_size, bbox_size]]).float(),
        'orig_bbox_scale': torch.tensor([[bbox_size, bbox_size]]).float(),
        'bbox_expand_factor': torch.tensor([1.0]).float(),
        'ori_img_size': torch.tensor([[w, h]]).float(),
        'img_size': torch.tensor([[img_size, img_size]]).float(),
        'input_size': torch.tensor([[img_size, img_size]]).float(),
        'affine_trans': torch.from_numpy(trans).unsqueeze(0).float(),
        'affine_trans_worot': torch.from_numpy(trans).unsqueeze(0).float(),
        'cam_int': torch.from_numpy(cam_int).unsqueeze(0).float(),
        'num_animals': torch.tensor([cfg.MODEL.NUM_ANIMALS]).int(),
        'targets': [targets_dict],
    }

    return batch


def preprocess_crop(
    image_rgb: np.ndarray,
    bbox_xyxy: np.ndarray,
    cfg,
    mask: np.ndarray = None,
    keypoints: np.ndarray = None,
):
    """Preprocess a single-animal crop from the full image."""
    h, w = image_rgb.shape[:2]
    img_size = cfg.MODEL.IMAGE_SIZE
    mean = 255.0 * np.array(cfg.MODEL.IMAGE_MEAN)
    std = 255.0 * np.array(cfg.MODEL.IMAGE_STD)
    focal_length = cfg.EXTRA.FOCAL_LENGTH

    x1, y1, x2, y2 = bbox_xyxy
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    bbox_w = x2 - x1
    bbox_h = y2 - y1
    bbox_size = float(max(bbox_w, bbox_h)) * BBOX_EXPAND_FACTOR

    if mask is not None:
        mask_full = (mask * 255).astype(np.uint8)
    else:
        mask_full = np.ones((h, w), dtype=np.uint8) * 255

    img_rgba = np.concatenate([image_rgb, mask_full[:, :, None]], axis=2)
    trans = gen_trans_from_patch_cv(
        cx, cy, bbox_size, bbox_size,
        img_size, img_size, 1.0, 0,
    )

    img_patch_rgba = cv2.warpAffine(
        img_rgba, trans, (img_size, img_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    img_patch_rgb = img_patch_rgba[:, :, :3]
    mask_patch = img_patch_rgba[:, :, 3]

    img_patch = convert_cvimg_to_tensor(img_patch_rgb)
    for c in range(3):
        img_patch[c] = (img_patch[c] - mean[c]) / std[c]

    mask_norm = (mask_patch / 255.0).clip(0, 1).astype(np.float32)
    if (mask_norm < 0.5).all():
        mask_norm = np.ones_like(mask_norm)
    mask_score = 1.0 if mask is not None and mask_norm.sum() > 0 else 0.0

    cam_int = np.array([
        [focal_length, 0.0, w / 2.0],
        [0.0, focal_length, h / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)

    num_animals = cfg.MODEL.NUM_ANIMALS
    targets_dict = {}
    if keypoints is not None and keypoints.shape[0] > 0:
        kpts_xy = keypoints[:, :2]
        ones = np.ones((kpts_xy.shape[0], 1))
        kpts_homo = np.concatenate([kpts_xy, ones], axis=1)
        kpts_transformed = (trans @ kpts_homo.T).T
        kpts_transformed = kpts_transformed / img_size - 0.5
        kpts_with_score = np.concatenate([kpts_transformed, keypoints[:, 2:3]], axis=1)

        dummy = np.zeros_like(kpts_with_score)
        kpts_all = [kpts_with_score] + [dummy] * (num_animals - 1)
        targets_dict['keypoints_2d'] = torch.from_numpy(np.stack(kpts_all, axis=0)).float()

    batch = {
        'img': torch.from_numpy(img_patch).unsqueeze(0).float(),
        'mask': torch.from_numpy(mask_norm).unsqueeze(0).float(),
        'mask_score': torch.tensor([mask_score]).float(),
        'bbox_center': torch.tensor([[cx, cy]]).float(),
        'bbox_scale': torch.tensor([[bbox_size, bbox_size]]).float(),
        'orig_bbox_scale': torch.tensor([[bbox_size, bbox_size]]).float(),
        'bbox_expand_factor': torch.tensor([1.0]).float(),
        'ori_img_size': torch.tensor([[w, h]]).float(),
        'img_size': torch.tensor([[img_size, img_size]]).float(),
        'input_size': torch.tensor([[img_size, img_size]]).float(),
        'affine_trans': torch.from_numpy(trans).unsqueeze(0).float(),
        'affine_trans_worot': torch.from_numpy(trans).unsqueeze(0).float(),
        'cam_int': torch.from_numpy(cam_int).unsqueeze(0).float(),
        'num_animals': torch.tensor([num_animals]).int(),
        'targets': [targets_dict],
    }

    return batch


def select_predictions(pred_output):
    """
    Select which animal predictions to render based on confidence scores.

    Returns:
        vertices_to_render: Tensor [K, V, 3]
        cam_t_to_render:    Tensor [K, 3]
        num_rendered:       int
        source:             str  ("confidence" | "all")
    """
    pred_vertices = pred_output['pred_vertices'].detach()   # [1, N, V, 3]
    pred_cam_t    = pred_output['pred_cam_t'].detach()      # [1, N, 3]
    N = pred_vertices.shape[1]

    # --- Confidence-based filtering ---
    if 'pred_keep_mask' in pred_output:
        keep_mask = pred_output['pred_keep_mask'][0]   # [N] bool
        if keep_mask.any():
            return (
                pred_vertices[0][keep_mask],
                pred_cam_t[0][keep_mask],
                int(keep_mask.sum().item()),
                "confidence",
            )

    # --- Return all predictions ---
    return (
        pred_vertices[0],
        pred_cam_t[0],
        N,
        "all",
    )


def select_best_prediction(pred_output):
    """Select the single best prediction for a per-crop animal."""
    pred_confs = pred_output['pred_confs'][0]
    best_idx = pred_confs[:, 0].argmax().item()
    return {
        'vertices': pred_output['pred_vertices'][0, best_idx].detach().cpu().numpy(),
        'cam_t': pred_output['pred_cam_t'][0, best_idx].detach().cpu().numpy(),
        'cam': pred_output['pred_cam'][0, best_idx].detach().cpu().numpy(),
        'conf': pred_confs[best_idx, 0].item(),
    }


def render_on_full_frame(
    renderer, image_rgb, vertices_to_render, cam_t_to_render, focal_length
):
    """Render reconstructed meshes onto the original full-resolution image."""
    ensure_egl_rendering()
    h, w = image_rgb.shape[:2]
    n = vertices_to_render.shape[0]
    vertices_list = [vertices_to_render[i].cpu().numpy() for i in range(n)]
    cam_t_list    = [cam_t_to_render[i].cpu().numpy()    for i in range(n)]

    cam_view = renderer.render_rgba_multiple(
        vertices_list, cam_t=cam_t_list,
        mesh_base_color=LIGHT_BLUE,
        scene_bg_color=(1, 1, 1),
        render_res=[w, h],
        focal_length=float(focal_length),
    )

    input_img  = image_rgb.astype(np.float32) / 255.0
    alpha      = cam_view[:, :, 3:]
    output_img = input_img * (1 - alpha) + cam_view[:, :, :3] * alpha
    return output_img


def render_on_crop(renderer, batch, vertices_to_render, cam_t_to_render, device, cfg):
    """Render reconstructed meshes onto the model-input crop view."""
    ensure_egl_rendering()
    n = vertices_to_render.shape[0]
    focal_length = batch['cam_int'][:, 0, 0]   # [1]

    cam_crop_wpersp = cam_full_to_crop(
        cam_t_to_render,
        batch['bbox_center'].expand(n, -1),
        batch['bbox_scale'][:, 0].expand(n),
        batch['ori_img_size'].expand(n, -1),
        focal_length=focal_length.expand(n),
    )
    cam_crop_t = cam_crop_s_to_t(
        cam_crop_wpersp,
        cfg.MODEL.IMAGE_SIZE,
        focal_length=focal_length.expand(n),
    )

    vertices_list = [vertices_to_render[i].cpu().numpy() for i in range(n)]
    cam_t_list    = [cam_crop_t[i].cpu().numpy()         for i in range(n)]

    # Un-normalise the crop image for overlay
    img_tensor = batch['img'].clone()
    img_tensor = img_tensor * torch.tensor(
        [0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    img_tensor = img_tensor + torch.tensor(
        [0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    crop_img = np.clip(img_tensor[0].permute(1, 2, 0).cpu().numpy(), 0, 1)

    cam_view = renderer.render_rgba_multiple(
        vertices_list, cam_t=cam_t_list,
        mesh_base_color=LIGHT_BLUE,
        scene_bg_color=(1, 1, 1),
        render_res=[cfg.MODEL.IMAGE_SIZE, cfg.MODEL.IMAGE_SIZE],
        focal_length=float(focal_length[0].item()),
    )

    alpha      = cam_view[:, :, 3:]
    output_img = crop_img * (1 - alpha) + cam_view[:, :, :3] * alpha
    return output_img


def render_per_crop_full_frame(renderer, image_rgb, all_verts, all_cam_t, cfg):
    """Render all per-crop predictions onto the original full-resolution image."""
    ensure_egl_rendering()
    h, w = image_rgb.shape[:2]
    focal_length = float(cfg.EXTRA.FOCAL_LENGTH)
    cam_view = renderer.render_rgba_multiple(
        all_verts,
        cam_t=all_cam_t,
        mesh_base_color=LIGHT_BLUE,
        scene_bg_color=(1, 1, 1),
        render_res=[w, h],
        focal_length=focal_length,
    )
    input_img = image_rgb.astype(np.float32) / 255.0
    alpha = cam_view[:, :, 3:]
    return input_img * (1 - alpha) + cam_view[:, :, :3] * alpha


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mode = "per_crop" if getattr(args, "per_crop", False) else getattr(args, "mode", "per_crop")
    score_threshold = getattr(args, "score_threshold", 0.5)

    # Load SAM3 model
    if args.use_sam3:
        print("Loading SAM3 model...")
        sam3_processor, sam3_model, sam3_device = load_sam3_model()
    else:
        sam3_processor, sam3_model, sam3_device = None, None, None

    # Load ViTPose model
    if args.use_vitpose:
        from amr.models.pose_models import ViTPose

        print("Loading ViTPose model...")
        vitpose_model = ViTPose(
            cfg_path="third-party/ViTPose/configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/apt36k/ViTPose_huge_apt36k_256x192.py",
            device=device,
            return_pose_image=True
        )
        ensure_egl_rendering()
    else:
        vitpose_model = None

    # Load config — auto-detect from checkpoint directory if not provided
    if args.config is not None:
        cfg = get_config(args.config, update_cachedir=True)
    else:
        cfg_path = str(Path(args.checkpoint).parent.parent / '.hydra/config.yaml')
        cfg = get_config(cfg_path, update_cachedir=True)

    # Remove pretrained backbone weights key (not needed at inference)
    if 'PRETRAINED_WEIGHTS' in cfg.MODEL.BACKBONE:
        cfg.defrost()
        cfg.MODEL.BACKBONE.pop('PRETRAINED_WEIGHTS')
        cfg.freeze()

    model = MultiAMR.load_from_checkpoint(
        args.checkpoint, cfg=cfg, strict=False, map_location='cpu'
    )
    model = model.to(device)
    model.eval()
    model.use_mask = not args.no_mask

    renderer = Renderer(cfg, faces=model.smal.faces.to(device))

    image_paths = load_image_paths(args.input_path)
    print(f"Found {len(image_paths)} image(s).")

    os.makedirs(args.out_folder, exist_ok=True)

    for img_path in tqdm(image_paths, desc="Processing images"):
        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            print(f"  WARNING: Failed to read {img_path}, skipping.")
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        if mode == "per_crop":
            masks, bboxes = [], []
            if args.use_sam3 and sam3_model is not None:
                masks, bboxes, scores = segment_animals_with_sam3(
                    sam3_processor, sam3_model, sam3_device, image_rgb,
                    text_prompt="animals",
                    score_threshold=score_threshold,
                )

            if len(bboxes) == 0:
                continue

            keypoints_list = [None] * len(bboxes)
            if args.use_vitpose and vitpose_model is not None:
                vitpose_results = infer_vitpose(vitpose_model, image_rgb, bboxes)
                if 'keypoints' in vitpose_results:
                    keypoints_list = vitpose_results['keypoints']

            stem = os.path.splitext(os.path.basename(img_path))[0]
            all_verts, all_cam_t = [], []
            for animal_idx, bbox in enumerate(bboxes):
                animal_mask = masks[animal_idx] if animal_idx < len(masks) else None
                animal_kpts = keypoints_list[animal_idx] if animal_idx < len(keypoints_list) else None

                batch = preprocess_crop(
                    image_rgb, bbox, cfg,
                    mask=animal_mask,
                    keypoints=animal_kpts,
                )
                batch = recursive_to(batch, device)

                with torch.no_grad():
                    output = model.forward_pose_branch(batch)

                pred = select_best_prediction(output['smal'])
                all_verts.append(pred['vertices'])
                all_cam_t.append(pred['cam_t'])

            if len(all_verts) > 0:
                output_img = render_per_crop_full_frame(
                    renderer, image_rgb, all_verts, all_cam_t, cfg,
                )
                out_path = os.path.join(args.out_folder, stem + '_render.jpg')
                output_img_uint8 = (np.clip(output_img, 0, 1) * 255).astype(np.uint8)
                cv2.imwrite(out_path, cv2.cvtColor(output_img_uint8, cv2.COLOR_RGB2BGR))

            continue

        # ---- Use SAM3 for segmentation if enabled ----
        merged_mask = None
        bboxes = []
        if args.use_sam3 and sam3_model is not None:
            masks, bboxes, scores = segment_animals_with_sam3(
                sam3_processor, sam3_model, sam3_device, image_rgb,
                text_prompt="animals",
                score_threshold=score_threshold,
            )
            if len(masks) == 0:
                print(f"  {os.path.basename(img_path)}: no animals found by SAM3, "
                      "falling back to confidence filtering.")
            else:
                # Merge all masks into one
                h, w = image_rgb.shape[:2]
                merged_mask = np.zeros((h, w), dtype=np.float32)
                for mask in masks:
                    merged_mask = np.maximum(merged_mask, mask.astype(np.float32))

        # ---- Run ViTPose if enabled ----
        vitpose_keypoints = None
        if args.use_vitpose and vitpose_model is not None and len(bboxes) > 0:
            vitpose_keypoints = infer_vitpose(vitpose_model, image_rgb, bboxes)
            print(f"  {os.path.basename(img_path)}: detected {len(vitpose_keypoints.get('keypoints', []))} keypoint sets.")

        # ---- Run multi-animal reconstruction model ----
        batch = preprocess_image(image_rgb, cfg, merged_mask=merged_mask, vitpose_keypoints=vitpose_keypoints)
        batch = recursive_to(batch, device)

        with torch.no_grad():
            output = model.forward_pose_branch(batch)

        pred_output = output['smal']

        # ---- Select which predictions to render ----
        try:
            vertices_to_render, cam_t_to_render, num_rendered, source = \
                select_predictions(pred_output)
        except Exception as e:
            print(f"  {os.path.basename(img_path)}: prediction selection failed ({e}), skipping.")
            continue

        if num_rendered == 0:
            print(f"  {os.path.basename(img_path)}: no valid predictions to render, skipping.")
            continue

        print(f"  {os.path.basename(img_path)}: rendering {num_rendered} animal(s) "
              f"[source: {source}].")

        # ---- Render ----
        try:
            if args.full_frame:
                focal_length = float(batch['cam_int'][0, 0, 0].item())
                output_img = render_on_full_frame(
                    renderer, image_rgb, vertices_to_render, cam_t_to_render, focal_length
                )
            else:
                output_img = render_on_crop(
                    renderer, batch, vertices_to_render, cam_t_to_render, device, cfg
                )
        except Exception as e:
            print(f"  {os.path.basename(img_path)}: rendering failed ({e}), skipping.")
            continue

        # ---- Save result ----
        stem = os.path.splitext(os.path.basename(img_path))[0]
        out_name = stem + '_render.jpg'
        out_path = os.path.join(args.out_folder, out_name)
        output_img_uint8 = (np.clip(output_img, 0, 1) * 255).astype(np.uint8)
        cv2.imwrite(out_path, cv2.cvtColor(output_img_uint8, cv2.COLOR_RGB2BGR))

    print(f"Results saved to {args.out_folder}")


def build_parser(
    default_out_folder="demo_out",
    default_mode="per_crop",
    default_score_threshold=0.5,
    description=None,
):
    parser = argparse.ArgumentParser(
        description=description or "Multi-animal mesh reconstruction from images."
    )
    parser.add_argument("--input_path", type=str, required=True,
                        help="Path to a single image or a folder of images")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to model config file (auto-detected from checkpoint if not provided)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--out_folder", type=str, default=default_out_folder,
                        help="Output directory for rendered images")
    parser.add_argument("--use_sam3", action="store_true",
                        help="Use SAM3 for animal segmentation with text prompt 'animals'")
    parser.add_argument("--use_vitpose", action="store_true",
                        help="Use ViTPose for keypoint detection on SAM3-detected animals")
    parser.add_argument("--full_frame", action="store_true",
                        help="In full_image mode, render on the full-resolution original image "
                             "(per_crop mode always saves one full-resolution overlay)")
    parser.add_argument("--no_mask", action="store_true",
                        help="Disable mask prompt during inference")
    parser.add_argument("--mode", choices=["full_image", "per_crop"], default=default_mode,
                        help="Inference pipeline: per_crop runs each SAM3 animal crop independently "
                             "and saves one full-frame result; full_image uses the original whole-image behavior")
    parser.add_argument("--per_crop", action="store_true",
                        help="Shortcut for --mode per_crop")
    parser.add_argument("--score_threshold", type=float, default=default_score_threshold,
                        help="SAM3 confidence threshold to filter detections")
    return parser


def parse_args(
    default_out_folder="demo_out",
    default_mode="per_crop",
    default_score_threshold=0.5,
    description=None,
):
    parser = build_parser(
        default_out_folder=default_out_folder,
        default_mode=default_mode,
        default_score_threshold=default_score_threshold,
        description=description,
    )
    args = parser.parse_args()
    if args.per_crop:
        args.mode = "per_crop"
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
