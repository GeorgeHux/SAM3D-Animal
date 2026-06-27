from typing import Any, Dict, List, Optional
import random

import torch
import wandb
import torch.nn.functional as F
from torch import nn, Tensor
from scipy.optimize import linear_sum_assignment
import pytorch_lightning as pl
from torchvision.utils import make_grid, draw_bounding_boxes
from pytorch3d.transforms import axis_angle_to_matrix

from .amr import AMR
from .backbones.vit import get_abs_pos
from .components.detr_utils import inverse_sigmoid
from .heads import build_smal_head
from .components.matcher import Matcher
from ..utils.renderer import cam_full_to_crop, cam_crop_s_to_t
from ..utils.mesh_renderer import SilhouetteRenderer


class MultiAMR(AMR):
    """
    Multi-Animal-Mesh-Recovery that operates on full images and supports multiple animals per image.
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        # Interpolate the positional embeddings to the target image size
        image_size = cfg.MODEL.get("IMAGE_SIZE", 512)
        target_h, target_w = image_size if isinstance(image_size, (list, tuple)) else (int(image_size), int(image_size))
        patch_embed = self.backbone.patch_embed
        kernel_h, kernel_w = patch_embed.proj.kernel_size
        stride_h, stride_w = patch_embed.proj.stride
        pad_h, pad_w = patch_embed.proj.padding
        Hp = (target_h + 2 * pad_h - kernel_h) // stride_h + 1
        Wp = (target_w + 2 * pad_w - kernel_w) // stride_w + 1
        ori_h, ori_w = patch_embed.patch_shape
        with torch.no_grad():
            resized_pos_embed = get_abs_pos(self.backbone.pos_embed, Hp, Wp, ori_h, ori_w, has_cls_token=True)
        self.backbone.pos_embed = nn.Parameter(resized_pos_embed)
        self.backbone.pos_embed.requires_grad = True

        self.num_animals = cfg.MODEL.NUM_ANIMALS
        self.grad_accum_steps = int(cfg.GENERAL.get("GRAD_ACCUM_STEPS", 1))
        self.conf_thresh = cfg.MODEL.get("CONF_THRESH", 0.5)
        self.conf_focal_alpha = 0.25
        self.conf_focal_gamma = 2.0
        self.box_token_embed = nn.Embedding(self.num_animals, self.cfg.MODEL.DECODER.DIM)
        self.box_refpoint_embed = nn.Embedding(self.num_animals, 4)
        self.bbox_head = build_smal_head(cfg, head_type="bbox")
        self.box_refpoint_embed.weight.data[:, :2].uniform_(0, 1)
        self.box_refpoint_embed.weight.data[:, :2] = inverse_sigmoid(
            self.box_refpoint_embed.weight.data[:, :2]
        )

        # Re-initialize the SMAL token embedding layer to account for multiple animals
        self.init_pose = nn.Embedding(self.num_animals, self.smal_head.npose)
        self.init_camera = nn.Embedding(self.num_animals, 3)
        self.keypoint_embedding_idxs = list(range(26))
        self.keypoint_embedding = torch.nn.Embedding(
            self.num_animals * len(self.keypoint_embedding_idxs), self.cfg.MODEL.DECODER.DIM
        )
        self.keypoint3d_embedding_idxs = list(range(26))
        self.keypoint3d_embedding = torch.nn.Embedding(
            self.num_animals * len(self.keypoint3d_embedding_idxs), self.cfg.MODEL.DECODER.DIM
        )
        num_keypoints = len(self.keypoint_embedding_idxs)
        keypoint_indices = (
            torch.arange(self.num_animals, dtype=torch.long)[:, None] * num_keypoints
            + torch.arange(num_keypoints, dtype=torch.long)[None, :]
        )
        self.register_buffer("keypoint_token_indices", keypoint_indices, persistent=False)
        num_keypoints3d = len(self.keypoint3d_embedding_idxs)
        keypoint3d_indices = (
            torch.arange(self.num_animals, dtype=torch.long)[:, None] * num_keypoints3d
            + torch.arange(num_keypoints3d, dtype=torch.long)[None, :]
        )
        self.register_buffer("keypoint3d_token_indices", keypoint3d_indices, persistent=False)

        if cfg.MODEL.DECODER.CONDITION_TYPE == "none":
            self.init_to_token_smal = torch.nn.Linear(self.smal_head.npose + self.camera_head.ncam, self.cfg.MODEL.DECODER.DIM)

        self.matcher = Matcher(cfg)

        # (#3) Fused confidence head: concatenated pose + box tokens → confidence
        dim = self.cfg.MODEL.DECODER.DIM
        self.conf_head = nn.Linear(2 * dim, 1)
        prior_prob = 0.01
        self.conf_head.bias.data.fill_(
            -torch.log(torch.tensor((1 - prior_prob) / prior_prob)).item()
        )

        # (#2) Denoising (DN) training
        self.dn_num_groups = cfg.MODEL.get("DN_NUM_GROUPS", 5)
        self.dn_box_noise_scale = cfg.MODEL.get("DN_BOX_NOISE_SCALE", 0.4)
        if self.dn_num_groups > 0:
            self.dn_embed = nn.Embedding(1, dim)
            self.dn_pos_proj = nn.Linear(4, dim)

        self.use_gt_prompt = True
        self.use_mask = True
        self.num_prompt_keypoints = None  # None = use all; int = subsample for ablation
        # Prompt dropout settings:
        #   prompt_drop_rate      – probability of dropping the *entire* prompt (prompt-free training)
        #   point_drop_rate_max   – upper bound for per-point dropout rate (sampled uniformly each step)
        self.prompt_drop_rate = cfg.MODEL.get("PROMPT_DROP_RATE", 0.2)
        self.point_drop_rate_max = cfg.MODEL.get("POINT_DROP_RATE_MAX", 0.7)

        # Initialize silhouette renderer for mask loss
        focal_default = float(getattr(cfg.EXTRA, 'FOCAL_LENGTH', 5000))
        self.silhouette_renderer = SilhouetteRenderer(
            size=cfg.MODEL.IMAGE_SIZE,
            focal=focal_default,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

        self._freeze_unused_trainable_params()

    def _freeze_unused_trainable_params(self) -> None:
        # MultiAMR uses the fused confidence head below, so the legacy
        # confidence branch inside bbox_head never contributes to the loss.
        if hasattr(self.bbox_head, "conf_head"):
            for param in self.bbox_head.conf_head.parameters():
                param.requires_grad = False

        # The current prompt builder only emits joint ids and -2 (invalid).
        # It never emits -1, so this embedding would otherwise stay unused.
        if hasattr(self, "prompt_encoder") and hasattr(
            self.prompt_encoder, "not_a_point_embed"
        ):
            self.prompt_encoder.not_a_point_embed.weight.requires_grad = False

    # def on_after_backward(self):
    #     for name, param in self.named_parameters():
    #         if param.grad is None:
    #             print(name)

    def _focal_loss_probs(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        inputs = inputs.float().clamp(1e-6, 1 - 1e-6)
        targets = targets.float()
        ce_loss = -(targets * inputs.log() + (1 - targets) * (1 - inputs).log())
        p_t = inputs * targets + (1 - inputs) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.conf_focal_gamma)
        alpha_t = (
            self.conf_focal_alpha * targets
            + (1 - self.conf_focal_alpha) * (1 - targets)
        )
        loss = alpha_t * loss
        if valid_mask is None:
            return loss.mean()
        valid_mask = valid_mask.to(loss.dtype)
        return (loss * valid_mask).sum() / valid_mask.sum().clamp(min=1.0)

    def _get_conf_valid_mask(
        self, batch: Dict, conf_targets: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        valid_mask = conf_targets > 0.5
        targets = batch.get("targets", [])
        if not targets:
            return valid_mask
        detect_all = []
        for target in targets:
            flag = True   # flag = False
            # for key in ("detect_all_animals", "detect_all_people", "detect_all"):
            #     if key in target:
            #         flag = bool(target[key])
            #         break
            detect_all.append(flag)
        return valid_mask | torch.tensor(detect_all, device=device)[:, None]

    def _attach_confidence_selection(self, pose_output: Dict) -> Dict:
        pred_confs = pose_output.get("pred_confs")
        if pred_confs is None or self.conf_thresh is None:
            return pose_output
        keep_mask = pred_confs[..., 0] > float(self.conf_thresh)
        pose_output["pred_keep_mask"] = keep_mask
        pose_output["pred_keep_indices"] = [
            torch.where(mask)[0] for mask in keep_mask
        ]
        for key, value in list(pose_output.items()):
            if torch.is_tensor(value) and value.ndim >= 2 and value.shape[:2] == keep_mask.shape:
                pose_output[f"selected_{key}"] = [
                    value[b][keep_mask[b]] for b in range(value.shape[0])
                ]
        return pose_output

    def _prepare_dn(self, batch: Dict, batch_size: int, device, dtype):
        """Build denoising queries from GT boxes."""
        gt_boxes = [b["bbox"] for b in batch["targets"]]
        img_size = batch.get("img_size")
        with torch.no_grad():
            gt_norm = self.matcher._normalize_gt_boxes(gt_boxes, img_size)
        num_gts = [len(b) for b in gt_norm]
        max_gt = max(num_gts) if num_gts else 0
        if max_gt == 0:
            return None

        single_pad = 2 * max_gt
        dn_pad = single_pad * self.dn_num_groups

        dn_box_refs = torch.zeros(batch_size, dn_pad, 4, device=device, dtype=dtype)
        dn_box_targets = torch.zeros_like(dn_box_refs)
        dn_conf_targets = torch.zeros(batch_size, dn_pad, device=device, dtype=dtype)
        dn_valid = torch.zeros(batch_size, dn_pad, device=device, dtype=torch.bool)

        for b in range(batch_size):
            n = num_gts[b]
            padded = torch.zeros(max_gt, 4, device=device, dtype=dtype)
            padded[:n] = gt_norm[b]
            for g in range(self.dn_num_groups):
                base = g * single_pad
                pos_n = torch.randn(max_gt, 4, device=device, dtype=dtype) * self.dn_box_noise_scale
                dn_box_refs[b, base:base + max_gt] = (padded + pos_n).clamp(1e-4, 1 - 1e-4)
                dn_box_targets[b, base:base + max_gt] = padded
                dn_conf_targets[b, base:base + n] = 1.0
                dn_valid[b, base:base + n] = True
                nb = base + max_gt
                neg_n = torch.randn(max_gt, 4, device=device, dtype=dtype) * self.dn_box_noise_scale * 2
                dn_box_refs[b, nb:nb + max_gt] = (padded + neg_n).clamp(1e-4, 1 - 1e-4)
                dn_box_targets[b, nb:nb + max_gt] = padded
                dn_valid[b, nb:nb + n] = True

        # Learnable projections (need gradients)
        dn_augment = self.dn_pos_proj(dn_box_refs)
        dn_tokens = self.dn_embed.weight[0:1].unsqueeze(0).expand(batch_size, dn_pad, -1).clone()
        dn_box_refs_inv = inverse_sigmoid(dn_box_refs)

        return {
            "dn_tokens": dn_tokens,
            "dn_augment": dn_augment,
            "dn_box_refs": dn_box_refs_inv,
            "dn_box_targets": dn_box_targets,
            "dn_conf_targets": dn_conf_targets,
            "dn_valid": dn_valid,
            "dn_pad_size": dn_pad,
            "single_pad": single_pad,
        }

    def _build_dn_attn_mask(
        self, dn_pad: int, normal_count: int, single_pad: int, device
    ):
        total = dn_pad + normal_count
        # In this decoder path, boolean masks use True=can attend.
        mask = torch.ones(total, total, dtype=torch.bool, device=device)
        # Standard DETR DN: matching queries cannot see DN queries.
        mask[dn_pad:, :dn_pad] = False
        # Each DN group can only see itself among DN queries, but can still
        # attend to all normal queries.
        for g in range(self.dn_num_groups):
            s = g * single_pad
            e = s + single_pad
            mask[s:e, :s] = False
            mask[s:e, e:dn_pad] = False
        return mask.unsqueeze(0)

    def forward_step(self, batch: Dict, train: bool = False) -> Dict:
        output = self.forward_pose_branch(batch)
        # Save DN tensors before matching — the matcher reorders tensors by
        # shape (B, num_animals, …) which collides when dn_pad == num_animals.
        dn_saved = {}
        for k in list(output["smal"].keys()):
            if k.startswith("dn_"):
                dn_saved[k] = output["smal"].pop(k)
        output["smal"] = self.matcher(batch, output["smal"])
        output["smal"].update(dn_saved)
        return output

    def compute_loss(self, batch: Dict, output: Dict, train: bool = True) -> torch.Tensor:
        """
        Compute losses given the input batch and the regression output
        Args:
            batch (Dict): Dictionary containing batch data
            output (Dict): Dictionary containing the regression output
            train (bool): Flag indicating whether it is training or validation mode
        Returns:
            torch.Tensor : Total loss for current batch
        """
        # Decoder output for the pose branch
        pose_output = output["smal"]

        # ------------------------- Predictions -------------------------
        # Pose parameters are stored as 6D reps (global + body). Convert back to rotmats.
        pred_pose_raw = pose_output["pred_pose_raw"]  # (B, (1 + J) * 6)
        pred_global_orient = pose_output["pred_global_orient"]          # (B,1,3,3)
        pred_body_pose = pose_output["pred_pose"]              # (B,J,3,3)
        pred_betas = pose_output["pred_betas"]                     # (B, ~145) shape comps
        # Full-image coords are kept for logging; use cropped coords for loss to match GT preprocessing
        pred_keypoints_2d_full = pose_output["pred_keypoints_2d"]
        pred_keypoints_2d_cropped = pose_output["pred_keypoints_2d_cropped"]
        pred_keypoints_3d = pose_output["pred_keypoints_3d"]   # camera coords
        pred_vertices = pose_output.get("pred_vertices", None)
        pred_cam_t = pose_output.get("pred_cam_t", None)
        pred_cam = pose_output.get("pred_cam", None)

        pred_smal_params = {
            "global_orient": pred_global_orient,
            "pose": pred_body_pose,
            "betas": pred_betas,
        }

        # Expose commonly-used predictions for downstream logging/visualization
        output.update(
            {
                "pred_smal_params": pred_smal_params,
                "pred_keypoints_2d": pred_keypoints_2d_full,
                "pred_keypoints_2d_cropped": pred_keypoints_2d_cropped,
                "pred_keypoints_3d": pred_keypoints_3d,
                "pred_vertices": pred_vertices,
                "pred_cam_t": pred_cam_t,
                "pred_cam": pred_cam,
            }
        )

        # --------------------------- Targets ---------------------------
        gt_keypoints_2d = torch.cat([b["keypoints_2d"] for b in batch["targets"]])
        gt_keypoints_3d = torch.cat([b["keypoints_3d"] for b in batch["targets"]])
        gt_smal_params = {
            "global_orient": torch.cat([b["smal_params"]["global_orient"] for b in batch["targets"]]),
            "pose": torch.cat([b["smal_params"]["pose"] for b in batch["targets"]]),
            "betas": torch.cat([b["smal_params"]["betas"] for b in batch["targets"]]),
        }
        has_smal_params = {
            "global_orient": torch.cat([b["has_smal_params"]["global_orient"] for b in batch["targets"]]),
            "pose": torch.cat([b["has_smal_params"]["pose"] for b in batch["targets"]]),
            "betas": torch.cat([b["has_smal_params"]["betas"] for b in batch["targets"]]),
        }
        gt_global_orient = axis_angle_to_matrix(gt_smal_params["global_orient"].view(-1, 1, 3))          # (N, 1, 3, 3)
        gt_pose = axis_angle_to_matrix(gt_smal_params["pose"].view(-1, 34, 3))              # (B, J, 3, 3)
        gt_betas = gt_smal_params["betas"]                     # (B, 145) shape comps

        # ---------------------------- Losses ---------------------------
        loss_keypoints_2d = self.keypoint_2d_loss(pred_keypoints_2d_cropped, gt_keypoints_2d)
        loss_keypoints_3d = self.keypoint_3d_loss(
            pred_keypoints_3d, gt_keypoints_3d, pelvis_id=0
        )

        loss_smal_params = {}
        loss_smal_params["global_orient"] = self.smal_parameter_loss(
            pred_global_orient, gt_global_orient, has_smal_params["global_orient"]
        )
        loss_smal_params["pose"] = self.smal_parameter_loss(
            pred_body_pose, gt_pose, has_smal_params["pose"]
        )
        loss_smal_params["betas"] = self.smal_parameter_loss(
            pred_betas, gt_betas, has_smal_params["betas"]
        )

        loss = (
            self.cfg.LOSS_WEIGHTS["KEYPOINTS_3D"] * loss_keypoints_3d
            + self.cfg.LOSS_WEIGHTS["KEYPOINTS_2D"] * loss_keypoints_2d
            + loss_smal_params["global_orient"] * self.cfg.LOSS_WEIGHTS.get("GLOBAL_ORIENT", 1.0)
            + loss_smal_params["pose"] * self.cfg.LOSS_WEIGHTS.get("POSE", 1.0)
            + loss_smal_params["betas"] * self.cfg.LOSS_WEIGHTS.get("BETAS", 1.0)
        )

        losses = dict(
            loss=loss.detach(),
            loss_keypoints_2d=loss_keypoints_2d.detach(),
            loss_keypoints_3d=loss_keypoints_3d.detach(),
            loss_global_orient=loss_smal_params["global_orient"].detach(),
            loss_pose=loss_smal_params["pose"].detach(),
            loss_betas=loss_smal_params["betas"].detach(),
        )

        # Add bbox and GIOU losses for multiple animals mesh recovery
        pred_boxes = pose_output.get("pred_boxes")
        gt_boxes = [b["bbox"] for b in batch["targets"]]
        img_size = batch.get("img_size")
        gt_boxes_norm = self.matcher._normalize_gt_boxes(gt_boxes, img_size)
        gt_boxes_norm = torch.cat(gt_boxes_norm)

        loss_bbox = F.l1_loss(pred_boxes, gt_boxes_norm, reduction="none").sum()

        pred_xyxy = self.matcher._box_cxcywh_to_xyxy(pred_boxes)
        gt_xyxy = self.matcher._box_cxcywh_to_xyxy(gt_boxes_norm)
        giou = self.matcher._generalized_box_iou(pred_xyxy, gt_xyxy)
        loss_giou = (1 - torch.diag(giou)).sum()
        loss_conf = torch.tensor(0.0, device=loss.device)
        match_info = pose_output.get("_match_info")
        if match_info is not None and match_info["full_pred_confs"] is not None:
            full_pred_confs = match_info["full_pred_confs"]   # (B, Q, 1)
            match_indices = match_info["match_indices"]
            conf_targets = torch.zeros(
                full_pred_confs.shape[:2], device=loss.device, dtype=full_pred_confs.dtype
            )
            for b, (pred_idx, _) in enumerate(match_indices):
                if len(pred_idx) > 0:
                    conf_targets[b, pred_idx] = 1.0
            conf_valid_mask = self._get_conf_valid_mask(batch, conf_targets, loss.device)
            loss_conf = self._focal_loss_probs(
                full_pred_confs[..., 0], conf_targets, conf_valid_mask
            )
        loss = (
            loss
            + self.cfg.LOSS_WEIGHTS.get("BBOX", 1.0) * loss_bbox
            + self.cfg.LOSS_WEIGHTS.get("GIOU", 1.0) * loss_giou
            + self.cfg.LOSS_WEIGHTS.get("CONF", 1.0) * loss_conf
        )
        # (#1) Auxiliary confidence losses on intermediate decoder layers
        loss_aux_conf = torch.tensor(0.0, device=loss.device)
        aux_conf_w = self.cfg.LOSS_WEIGHTS.get("AUX_CONF", 1.0)
        for aux_out in output.get("smal_aux", []):
            aux_m = self.matcher(batch, aux_out)
            aux_mi = aux_m.get("_match_info")
            if aux_mi is not None and aux_mi["full_pred_confs"] is not None:
                afc = aux_mi["full_pred_confs"]
                act = torch.zeros(afc.shape[:2], device=loss.device, dtype=afc.dtype)
                for b, (pi, _) in enumerate(aux_mi["match_indices"]):
                    if len(pi) > 0:
                        act[b, pi] = 1.0
                avm = self._get_conf_valid_mask(batch, act, loss.device)
                loss_aux_conf = loss_aux_conf + self._focal_loss_probs(afc[..., 0], act, avm)
        loss = loss + aux_conf_w * loss_aux_conf

        # (#2) Denoising losses
        loss_dn_conf = torch.tensor(0.0, device=loss.device)
        loss_dn_box = torch.tensor(0.0, device=loss.device)
        dn_meta = output.get("dn_meta")
        if dn_meta is not None:
            dn_confs = pose_output.get("dn_pred_confs")
            dn_boxes = pose_output.get("dn_pred_boxes")
            valid = dn_meta["dn_valid"]                       # (B, dn_pad)
            if dn_confs is not None and valid.any():
                dn_confs_2d = dn_confs.squeeze(-1)            # (B, dn_pad)
                loss_dn_conf = self._focal_loss_probs(
                    dn_confs_2d[valid], dn_meta["dn_conf_targets"][valid]
                )
                pos = valid & (dn_meta["dn_conf_targets"] > 0.5)
                if pos.any():
                    loss_dn_box = F.l1_loss(
                        dn_boxes[pos], dn_meta["dn_box_targets"][pos],
                    )
        loss = loss + (
            self.cfg.LOSS_WEIGHTS.get("DN_CONF", 1.0) * loss_dn_conf
            + self.cfg.LOSS_WEIGHTS.get("DN_BBOX", 1.0) * loss_dn_box
        )

        # Mask/Silhouette loss
        loss_mask = torch.tensor(0.0, device=loss.device)
        if "mask" in batch and pred_vertices is not None and pred_cam_t is not None:
            loss_mask = self._compute_mask_loss(
                batch, pred_vertices, pred_cam_t
            )
            loss = loss + self.cfg.LOSS_WEIGHTS.get("MASK", 1.0) * loss_mask

        losses.update(
            {
                "loss": loss.detach(),
                "loss_bbox": loss_bbox.detach(),
                "loss_giou": loss_giou.detach(),
                "loss_conf": loss_conf.detach(),
                "loss_aux_conf": loss_aux_conf.detach(),
                "loss_dn_conf": loss_dn_conf.detach(),
                "loss_dn_box": loss_dn_box.detach(),
                "loss_mask": loss_mask.detach(),
            }
        )
        output["losses"] = losses
        return loss

    def _render_silhouettes(
        self,
        batch: Dict,
        pred_vertices: torch.Tensor,
        pred_cam_t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Render per-image silhouettes for all animals by converting full-image cam_t to
        crop space, applying it to vertices, and merging per-animal silhouettes via union.

        Args:
            batch: Batch dict with 'num_animals', 'cam_int', 'bbox_center', 'bbox_scale',
                   'ori_img_size'.
            pred_vertices: [N_total, V, 3], N_total = sum(num_animals across batch).
            pred_cam_t:    [N_total, 3], full-image-space camera translation.

        Returns:
            pred_silhouettes: [B, H, W] float32 in [0, 1].
        """
        device = pred_vertices.device
        batch_size = batch['img'].shape[0]
        num_animals_per_image = batch['num_animals']  # [B]
        render_size = self.silhouette_renderer.size

        faces = self.smal.faces
        if not isinstance(faces, torch.Tensor):
            faces = torch.from_numpy(faces)
        faces = faces.to(device=device, dtype=torch.int64)

        # Split predictions per image
        pred_vertices_list = torch.split(pred_vertices, num_animals_per_image.tolist(), dim=0)
        pred_cam_t_list = torch.split(pred_cam_t, num_animals_per_image.tolist(), dim=0)

        # Convert full-image cam_t → crop-space 3D translation for each image
        focal_length = batch["cam_int"][:, 0, 0]
        bbox_center = batch['bbox_center']
        bbox_scale = batch["bbox_scale"][:, 0]
        ori_img_size = batch["ori_img_size"]
        cam_crop_ts = []
        for b in range(batch_size):
            cam_crop_wpersp = cam_full_to_crop(
                pred_cam_t_list[b],
                bbox_center[[b]],
                bbox_scale[[b]],
                ori_img_size[[b]],
                focal_length=focal_length[[b]],
            )
            cam_crop_t = cam_crop_s_to_t(
                cam_crop_wpersp,
                self.cfg.MODEL.IMAGE_SIZE,
                focal_length=focal_length[[b]],
            )
            cam_crop_ts.append(cam_crop_t)

        # Render per-image merged silhouettes
        pred_silhouettes_list = []
        for b in range(batch_size):
            n_animals = num_animals_per_image[b].item()
            if n_animals == 0:
                pred_silhouettes_list.append(
                    torch.zeros(render_size, render_size, device=device, dtype=torch.float32)
                )
                continue

            verts_b = pred_vertices_list[b]      # [n_animals, V, 3]
            cam_t_b = cam_crop_ts[b]             # [n_animals, 3] crop-space
            vertices_cam = verts_b + cam_t_b.unsqueeze(1)  # [n_animals, V, 3]

            animal_silhouettes = []
            for i in range(n_animals):
                sil_i = self.silhouette_renderer(
                    vertices_cam[i:i+1], faces.unsqueeze(0)
                )  # [1, H, W]
                animal_silhouettes.append(sil_i[0])  # [H, W]

            merged = torch.stack(animal_silhouettes, dim=0).max(dim=0)[0]  # [H, W]
            pred_silhouettes_list.append(torch.clamp(merged, 0.0, 1.0))

        return torch.stack(pred_silhouettes_list, dim=0)  # [B, H, W]

    def _compute_mask_loss(
        self,
        batch: Dict,
        pred_vertices: torch.Tensor,
        pred_cam_t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute mask/silhouette L1 loss for multiple animals per image.
        """
        render_size = self.silhouette_renderer.size

        # Prepare GT masks
        gt_masks = batch['mask']
        if gt_masks.ndim == 4:
            gt_masks = gt_masks.squeeze(1)  # [B, H, W]
        if gt_masks.shape[-2:] != (render_size, render_size):
            gt_masks = F.interpolate(
                gt_masks.unsqueeze(1).float(),
                size=(render_size, render_size),
                mode='bilinear',
                align_corners=False,
            ).squeeze(1)
        gt_masks = gt_masks.float()
        if gt_masks.max() > 1.0:
            gt_masks = gt_masks / 255.0
        gt_masks = torch.clamp(gt_masks, 0.0, 1.0)

        pred_silhouettes = self._render_silhouettes(batch, pred_vertices, pred_cam_t)
        return F.l1_loss(pred_silhouettes, gt_masks)

    @pl.utilities.rank_zero.rank_zero_only
    def log_visualizations_to_wandb(
        self, batch: Dict, output: Dict, step_count: int, train: bool = True
    ) -> None:
        """
        Log results to W&B, including bbox visualizations for multi-animal cases.
        """
        losses = output["losses"]
        # Keep metrics before narrowing to SMAL outputs so validation metrics are logged
        metrics = output.get("metric")
        output = output["smal"] if "smal" in output else output
        mode = "train" if train else "val"

        images = batch["img"]

        images = images.view(-1, *images.shape[-3:])
        # Un-normalize images for visualization
        images = images * torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
        images = images + torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)

        def _to_float32_numpy(tensor: torch.Tensor) -> Any:
            if isinstance(tensor, list) or isinstance(tensor, tuple):
                return [_to_float32_numpy(t) for t in tensor]
            else:
                return tensor.to(torch.float32).cpu().numpy()

        def _xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
            x, y, w, h = boxes.unbind(-1)
            return torch.stack([x, y, x + w, y + h], dim=-1)

        def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
            cx, cy, w, h = boxes.unbind(-1)
            return torch.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=-1)

        def _denormalize_boxes(boxes: List[torch.Tensor], img_size: torch.Tensor) -> torch.Tensor:
            denorm_boxes = []
            for b in range(img_size.shape[0]):
                img_h = img_size[b, 0]
                img_w = img_size[b, 1]
                denorm_boxes.append(boxes[b].clamp(0, 1) * torch.tensor([img_w, img_h, img_w, img_h], device=img_size.device))
            return denorm_boxes

        def _draw_boxes_on_images(imgs: torch.Tensor, boxes_per_image: List[torch.Tensor], color: str) -> torch.Tensor:
            drawn = []
            for img, boxes in zip(imgs, boxes_per_image):
                img_uint8 = (img.clamp(0, 1) * 255).to(torch.uint8)
                h, w = img_uint8.shape[1], img_uint8.shape[2]
                boxes = boxes.to(img_uint8.device).clone()
                boxes[:, 0::2] = boxes[:, 0::2].clamp(0, w - 1)
                boxes[:, 1::2] = boxes[:, 1::2].clamp(0, h - 1)
                drawn_img = draw_bounding_boxes(img_uint8, boxes, colors=color, width=2)
                drawn.append(drawn_img.to(torch.float32) / 255.0)
            return torch.stack(drawn, dim=0)

        # Create a dictionary to hold all logging data
        log_data = {}

        # Add losses to the log dictionary
        for loss_name, val in losses.items():
            log_data[f"{mode}/{loss_name}"] = val.detach().item()

        # If in validation mode, add metrics to the log dictionary
        if not train and metrics is not None:
            for metric_name, val in metrics.items():
                log_data[f"{mode}/{metric_name}"] = val

        num_images_to_log = min(images.shape[0], self.cfg.EXTRA.NUM_LOG_IMAGES)

        # --- Draw GT bbox on input images (full image) ---
        gt_boxes = torch.cat([b["bbox"] for b in batch["targets"]])
        img_size = batch.get("img_size")
        gt_boxes_xyxy = _xywh_to_xyxy(gt_boxes)
        images_for_gt = images
        gt_boxes_per_img = torch.split(gt_boxes_xyxy, batch['num_animals'].tolist(), dim=0)
        images_with_gt = _draw_boxes_on_images(images_for_gt, gt_boxes_per_img, color="green")

        # --- Prepare predictions and draw pred bbox on render inputs ---
        pred_boxes = output['pred_boxes'].detach()
        images_pred = images
        pred_boxes_xyxy = _cxcywh_to_xyxy(pred_boxes)
        pred_boxes_per_img = torch.split(pred_boxes_xyxy, batch['num_animals'].tolist(), dim=0)
        pred_boxes_xyxy = _denormalize_boxes(pred_boxes_per_img, img_size)
        images_pred = _draw_boxes_on_images(images_pred, pred_boxes_xyxy, color="red")

        pred_vertices = torch.split(output["pred_vertices"].detach(), batch['num_animals'].tolist(), dim=0)
        pred_cam_t = torch.split(output["pred_cam_t"].detach(), batch['num_animals'].tolist(), dim=0)
        pred_keypoints_2d = torch.split(output["pred_keypoints_2d_cropped"].detach(), batch['num_animals'].tolist(), dim=0)
        gt_keypoints_2d = [b["keypoints_2d"] for b in batch["targets"]]
        # Create the visualization grid
        focal_length = batch["cam_int"][:, 0, 0]
        bbox_center = batch['bbox_center']
        bbox_scale = batch["bbox_scale"][:, 0]
        ori_img_size = batch["ori_img_size"]
        cam_crop_ts = []
        for n in range(len(pred_cam_t)):
            cam_crop_wpersp = cam_full_to_crop(
                pred_cam_t[n],
                bbox_center[[n]],
                bbox_scale[[n]],
                ori_img_size[[n]],
                focal_length=focal_length[[n]],
            )
            cam_crop_t = cam_crop_s_to_t(
                cam_crop_wpersp,
                self.cfg.MODEL.IMAGE_SIZE,
                focal_length=focal_length[[n]],
            )
            cam_crop_ts.append(cam_crop_t)

        # Render predicted silhouettes and prepare GT masks for visualization
        with torch.no_grad():
            pred_silhouettes = self._render_silhouettes(
                batch,
                output["pred_vertices"].detach(),
                output["pred_cam_t"].detach(),
            )  # [B, H, W]
        gt_masks_vis = batch.get("mask")
        if gt_masks_vis is not None:
            if gt_masks_vis.ndim == 4:
                gt_masks_vis = gt_masks_vis.squeeze(1)
            render_size = self.silhouette_renderer.size
            if gt_masks_vis.shape[-2:] != (render_size, render_size):
                gt_masks_vis = F.interpolate(
                    gt_masks_vis.unsqueeze(1).float(),
                    size=(render_size, render_size),
                    mode='bilinear',
                    align_corners=False,
                ).squeeze(1)
            gt_masks_vis = gt_masks_vis.float()
            if gt_masks_vis.max() > 1.0:
                gt_masks_vis = gt_masks_vis / 255.0
            gt_masks_vis = torch.clamp(gt_masks_vis, 0.0, 1.0)

        num_images_to_log_pred = min(images_pred.shape[0], self.cfg.EXTRA.NUM_LOG_IMAGES)
        pred_masks_np = _to_float32_numpy(pred_silhouettes[:num_images_to_log_pred].cpu()) if pred_silhouettes is not None else None
        gt_masks_np = _to_float32_numpy(gt_masks_vis[:num_images_to_log_pred].cpu()) if gt_masks_vis is not None else None
        predictions_grid = self._get_mesh_renderer().visualize_wandb_for_multiple_animals(
            _to_float32_numpy(pred_vertices[:num_images_to_log_pred]),
            _to_float32_numpy(cam_crop_ts[:num_images_to_log_pred]),
            _to_float32_numpy(images[:num_images_to_log_pred]),
            focal_length[0].item(),
            _to_float32_numpy(pred_keypoints_2d[:num_images_to_log_pred]),
            _to_float32_numpy(gt_keypoints_2d[:num_images_to_log_pred]),
            _to_float32_numpy(images_with_gt),
            _to_float32_numpy(images_pred),
            pred_masks=pred_masks_np,
            gt_masks=gt_masks_np,
        )
        predictions_grid = [img.float().cpu() for img in predictions_grid]
        nrow = 7 + (1 if pred_masks_np is not None else 0) + (1 if gt_masks_np is not None else 0)
        predictions_grid = make_grid(predictions_grid, nrow=nrow, padding=2)
        predictions_grid_for_wandb = (predictions_grid * 255).to(torch.uint8)
        log_data[f"{mode}/predictions"] = wandb.Image(predictions_grid_for_wandb)

        # Log the entire dictionary to W&B
        self._log_wandb_payload(log_data, step_count, mode)

    def forward_decoder(
        self,
        image_embeddings: torch.Tensor,
        init_estimate: Optional[torch.Tensor] = None,
        keypoints: Optional[torch.Tensor] = None,
        prev_estimate: Optional[torch.Tensor] = None,
        condition_info: Optional[torch.Tensor] = None,
        batch=None,
    ):
        """
        Args:
            image_embeddings: image features from the backbone, shape (B, C, H, W)
            init_estimate: initial estimate to be refined on, shape (B, 1, C)
            keypoints: optional prompt input, shape (B, N, 3),
                3 for coordinates (x,y) + label.
                (x, y) should be normalized to range [0, 1].
                label==-1 indicates incorrect points,
                label==-2 indicates invalid points
            prev_estimate: optional prompt input, shape (B, 1, C),
                previous estimate for pose refinement.
            condition_info: optional condition information that is concatenated with
                the input tokens, shape (B, c)
        """
        batch_size = image_embeddings.shape[0]

        # Initial estimation for residual prediction.
        if init_estimate is None:
            # [B, num_animals, npose]
            init_pose = self.init_pose.weight.expand(batch_size, -1, -1)
            if hasattr(self, "init_camera"):
                # [B, num_animals, 3]
                init_camera = self.init_camera.weight.expand(batch_size, -1, -1)

            init_estimate = (
                init_pose
                if not hasattr(self, "init_camera")
                else torch.cat([init_pose, init_camera], dim=-1)
            )  # B x num_animals x (npose + 3)

        if condition_info is not None:
            condition_info = condition_info.view(batch_size, 1, -1).expand(
                batch_size, self.num_animals, -1
            )
            init_input = torch.cat(
                [condition_info, init_estimate], dim=-1
            )  # B x num_animals x (npose + 3 + c)
        else:
            init_input = init_estimate

        # [B, num_animals, 1024]
        token_embeddings = self.init_to_token_smal(init_input)

        num_pose_token = token_embeddings.shape[1]
        assert num_pose_token == self.num_animals

        image_augment, token_augment, token_mask = None, None, None
        box_tokens = self.box_token_embed.weight[None, :, :].repeat(batch_size, 1, 1)
        box_token_start_idx = None
        kps_emb_start_idx = None
        kps3d_emb_start_idx = None
        if hasattr(self, "prompt_encoder") and keypoints is not None:
            if prev_estimate is None:
                # Use initial embedding if no previous embedding
                prev_estimate = init_estimate
            # Previous estimate w/o the CLIFF condition.
            # [B, num_animals, 1024]
            prev_embeddings = self.prev_to_token_smal(prev_estimate)

            # ViT backbone assumes a different aspect ratio as input size
            image_augment = self.prompt_encoder.get_dense_pe((32, 32))
            image_embeddings = self.ray_cond_emb(image_embeddings, batch["ray_cond"])

            # To start, keypoints is all [0, 0, -2]. The points get sent into self.pe_layer._pe_encoding,
            # the labels determine the embedding weight (special one for -2, -1, then each of joint.)
            prompt_embeddings, prompt_mask = self.prompt_encoder(
                keypoints=keypoints
            )  # B x 1 x 1280
            prompt_embeddings = self.prompt_to_token(
                prompt_embeddings
            )  # Linear layered: B x 1 x 1024

            # Concatenate pose tokens and prompt embeddings as decoder input
            token_embeddings = torch.cat(
                [
                    token_embeddings,
                    prev_embeddings,
                    prompt_embeddings,
                ],
                dim=1,
            )

            token_augment = torch.zeros_like(token_embeddings)
            token_augment[:, num_pose_token : num_pose_token + num_pose_token] = prev_embeddings
            token_augment[:, (num_pose_token * 2) :] = prompt_embeddings
            token_mask = None

            box_token_start_idx = token_embeddings.shape[1]
            token_embeddings = torch.cat([token_embeddings, box_tokens], dim=1)
            token_augment = torch.cat(
                [token_augment, torch.zeros_like(box_tokens)], dim=1
            )

            # Put in a token for each keypoint per animal
            kps_emb_start_idx = token_embeddings.shape[1]
            kps_tokens = self.keypoint_embedding(self.keypoint_token_indices)
            kps_tokens = kps_tokens[None, :, :, :].expand(batch_size, -1, -1, -1)
            kps_tokens = kps_tokens.reshape(batch_size, -1, kps_tokens.shape[-1])
            token_embeddings = torch.cat([token_embeddings, kps_tokens], dim=1)  
            # No positional embeddings
            token_augment = torch.cat(
                [
                    token_augment,
                    torch.zeros_like(kps_tokens),
                ],
                dim=1,
            )
            if self.cfg.MODEL.DECODER.get("DO_KEYPOINT3D_TOKENS", False):
                # Put in a token for each keypoint per animal
                kps3d_emb_start_idx = token_embeddings.shape[1]
                kps3d_tokens = self.keypoint3d_embedding(self.keypoint3d_token_indices)
                kps3d_tokens = kps3d_tokens[None, :, :, :].expand(batch_size, -1, -1, -1)
                kps3d_tokens = kps3d_tokens.reshape(
                    batch_size, -1, kps3d_tokens.shape[-1]
                )
                token_embeddings = torch.cat([token_embeddings, kps3d_tokens], dim=1)
                # No positional embeddings
                token_augment = torch.cat(
                    [
                        token_augment,
                        torch.zeros_like(kps3d_tokens),
                    ],
                    dim=1,
                )
        else:
            box_token_start_idx = token_embeddings.shape[1]
            token_embeddings = torch.cat([token_embeddings, box_tokens], dim=1)

        # --- DN: prepend denoising tokens ---
        dn_meta = None
        dn_pad = 0
        if self.training and self.dn_num_groups > 0:
            dn_meta = self._prepare_dn(batch, batch_size, token_embeddings.device, token_embeddings.dtype)
            if dn_meta is not None:
                dn_pad = dn_meta["dn_pad_size"]
                old_len = token_embeddings.shape[1]
                token_embeddings = torch.cat([dn_meta["dn_tokens"], token_embeddings], dim=1)
                if token_augment is not None:
                    token_augment = torch.cat([dn_meta["dn_augment"], token_augment], dim=1)
                else:
                    token_augment = torch.cat([
                        dn_meta["dn_augment"],
                        torch.zeros(batch_size, old_len, self.cfg.MODEL.DECODER.DIM,
                                    device=token_embeddings.device, dtype=token_embeddings.dtype),
                    ], dim=1)
                box_token_start_idx += dn_pad
                if kps_emb_start_idx is not None:
                    kps_emb_start_idx += dn_pad
                if kps3d_emb_start_idx is not None:
                    kps3d_emb_start_idx += dn_pad

        self_attn_mask = None
        if dn_pad > 0:
            normal_count = token_embeddings.shape[1] - dn_pad
            self_attn_mask = self._build_dn_attn_mask(
                dn_pad, normal_count, dn_meta["single_pad"], token_embeddings.device
            )

        # We're doing intermediate model predictions
        def token_to_pose_output_fn(tokens, prev_pose_output, layer_idx):
            # Get the pose tokens (offset by dn_pad)
            pose_tokens = tokens[:, dn_pad:dn_pad + self.num_animals]
            box_token_end_idx = box_token_start_idx + self.num_animals
            box_tokens_out = tokens[:, box_token_start_idx:box_token_end_idx]
            box_refpoints = self.box_refpoint_embed.weight[None, :, :].expand(
                batch_size, -1, -1
            )
            # Predict boxes (ignore bbox_head's conf; use fused conf_head)
            pred_boxes, _ = self.bbox_head(
                box_tokens_out, init_estimate=box_refpoints
            )
            # (#3) Fused confidence from concatenated pose + box tokens
            pred_confs = self.conf_head(
                torch.cat([pose_tokens, box_tokens_out], dim=-1)
            ).sigmoid()

            pose_tokens_flat = pose_tokens.reshape(batch_size * self.num_animals, -1)
            prev_pose = init_pose.reshape(batch_size * self.num_animals, -1)
            prev_camera = init_camera.reshape(batch_size * self.num_animals, -1)

            # Get pose outputs
            pred_smal_params, pred_pose_6d = self.smal_head(
                pose_tokens_flat, prev_pose
            )
            smal_output = self.smal(**pred_smal_params)
            pose_output = {
                'pred_pose_raw': pred_pose_6d.view(batch_size, self.num_animals, -1),
                'pred_global_orient': pred_smal_params['global_orient'].view(
                    batch_size, self.num_animals, *pred_smal_params['global_orient'].shape[1:]
                ),
                'pred_pose': pred_smal_params['pose'].view(
                    batch_size, self.num_animals, *pred_smal_params['pose'].shape[1:]
                ),
                'pred_betas': pred_smal_params['betas'].view(
                    batch_size, self.num_animals, -1
                ),
                'pred_keypoints_3d': smal_output.joints.view(
                    batch_size, self.num_animals, *smal_output.joints.shape[1:]
                ),
                'pred_vertices': smal_output.vertices.view(
                    batch_size, self.num_animals, *smal_output.vertices.shape[1:]
                ),
                'pred_boxes': pred_boxes,
                'pred_confs': pred_confs,
            }
            # Get Camera Translation
            if hasattr(self, "camera_head"):
                pred_cam = self.camera_head(pose_tokens_flat, prev_camera)
                pose_output["pred_cam"] = pred_cam.view(batch_size, self.num_animals, -1)
            # Run camera projection
            if hasattr(self, "camera_head"):
                flat_pose_output = {
                    "pred_keypoints_3d": smal_output.joints,
                    "pred_vertices": smal_output.vertices,
                    "pred_cam": pred_cam,
                }
                repeat_fields = [
                    "bbox_center",
                    "bbox_scale",
                    "ori_img_size",
                    "cam_int",
                    "affine_trans_worot",
                    "img_size",
                ]
                flat_batch = dict(batch)
                for field in repeat_fields:
                    if field in flat_batch and flat_batch[field].shape[0] == batch_size:
                        flat_batch[field] = flat_batch[field].repeat_interleave(
                            self.num_animals, dim=0
                        )
                flat_pose_output = self.camera_project(flat_pose_output, flat_batch)  # TODO: check if this is correct by using the GT params
                pose_output.update(
                    {
                        "pred_keypoints_2d": flat_pose_output["pred_keypoints_2d"].view(
                            batch_size, self.num_animals, *flat_pose_output["pred_keypoints_2d"].shape[1:]
                        ),
                        "pred_cam_t": flat_pose_output["pred_cam_t"].view(
                            batch_size, self.num_animals, -1
                        ),
                        "pred_keypoints_2d_depth": flat_pose_output[
                            "pred_keypoints_2d_depth"
                        ].view(batch_size, self.num_animals, -1),
                    }
                )

            # Get 2D KPS in crop
            if "pred_keypoints_2d" in pose_output:
                flat_batch = dict(batch)
                if flat_batch["affine_trans_worot"].shape[0] == batch_size:
                    flat_batch["affine_trans_worot"] = flat_batch[
                        "affine_trans_worot"
                    ].repeat_interleave(self.num_animals, dim=0)
                    flat_batch["img_size"] = flat_batch["img_size"].repeat_interleave(
                        self.num_animals, dim=0
                    )
                pred_keypoints_2d_flat = pose_output["pred_keypoints_2d"].reshape(
                    batch_size * self.num_animals, -1, 2
                )
                pred_keypoints_2d_cropped = self._full_to_crop(
                    flat_batch, pred_keypoints_2d_flat
                )
                pose_output["pred_keypoints_2d_cropped"] = (
                    pred_keypoints_2d_cropped.view(
                        batch_size, self.num_animals, -1, 2
                    )
                )

            # (#2) DN outputs: box + fused confidence for denoising queries
            if dn_meta is not None and dn_pad > 0:
                dn_tok = tokens[:, :dn_pad]
                dn_pred_boxes, _ = self.bbox_head(dn_tok, init_estimate=dn_meta["dn_box_refs"])
                dn_pred_confs = self.conf_head(
                    torch.cat([dn_tok, dn_meta["dn_augment"]], dim=-1)
                ).sigmoid()
                pose_output["dn_pred_boxes"] = dn_pred_boxes
                pose_output["dn_pred_confs"] = dn_pred_confs

            return pose_output

        kp_token_update_fn = self.keypoint_token_update_fn

        # Now for 3D
        kp3d_token_update_fn = self.keypoint3d_token_update_fn

        # Combine the 2D and 3D functions
        def keypoint_token_update_fn_comb(*args):
            if kp_token_update_fn is not None:
                args = kp_token_update_fn(kps_emb_start_idx, image_embeddings, *args)
            if kp3d_token_update_fn is not None:
                args = kp3d_token_update_fn(kps3d_emb_start_idx, *args)
            return args

        pose_token, pose_output = self.decoder(
            token_embeddings,
            image_embeddings,
            token_augment,
            image_augment,
            token_mask,
            token_to_pose_output_fn=token_to_pose_output_fn,
            keypoint_token_update_fn=keypoint_token_update_fn_comb,
            self_attn_mask=self_attn_mask,
        )

        return pose_token, pose_output, dn_meta

    def keypoint_token_update_fn(
        self,
        kps_emb_start_idx,
        image_embeddings,
        token_embeddings,
        token_augment,
        pose_output,
        layer_idx,
    ):
        # It's already after the last layer, we're done.
        if layer_idx == len(self.decoder.layers) - 1:
            return token_embeddings, token_augment, pose_output, layer_idx

        # Clone
        token_embeddings = token_embeddings.clone()
        token_augment = token_augment.clone()

        num_keypoints_per_animal = len(self.keypoint_embedding_idxs)
        num_keypoints = self.num_animals * num_keypoints_per_animal

        # Get current 2D KPS predictions
        pred_keypoints_2d_cropped = pose_output[
            "pred_keypoints_2d_cropped"
        ].clone()  # B x A x K x 2
        pred_keypoints_2d_depth = pose_output["pred_keypoints_2d_depth"].clone()

        if pred_keypoints_2d_cropped.ndim == 3:
            pred_keypoints_2d_cropped = pred_keypoints_2d_cropped[:, None, ...]
            pred_keypoints_2d_depth = pred_keypoints_2d_depth[:, None, ...]

        pred_keypoints_2d_cropped = pred_keypoints_2d_cropped[
            :, :, self.keypoint_embedding_idxs
        ]
        pred_keypoints_2d_depth = pred_keypoints_2d_depth[
            :, :, self.keypoint_embedding_idxs
        ]

        # Get 2D KPS to be 0 ~ 1
        pred_keypoints_2d_cropped_01 = pred_keypoints_2d_cropped + 0.5

        # Get a mask of those that are 1) beyond image boundaries or 2) behind the camera
        invalid_mask = (
            (pred_keypoints_2d_cropped_01[:, :, :, 0] < 0)
            | (pred_keypoints_2d_cropped_01[:, :, :, 0] > 1)
            | (pred_keypoints_2d_cropped_01[:, :, :, 1] < 0)
            | (pred_keypoints_2d_cropped_01[:, :, :, 1] > 1)
            | (pred_keypoints_2d_depth[:, :, :] < 1e-5)
        )

        # Flatten animals for embedding
        pred_keypoints_2d_cropped_flat = pred_keypoints_2d_cropped.reshape(
            -1, num_keypoints, 2
        )
        invalid_mask_flat = invalid_mask.reshape(-1, num_keypoints)

        # Run them through the prompt encoder's pos emb function
        posemb = self.keypoint_posemb_linear(pred_keypoints_2d_cropped_flat)
        posemb = posemb * (~invalid_mask_flat[:, :, None])
        posemb = posemb.view(token_embeddings.shape[0], num_keypoints, -1)
        token_augment[
            :,
            kps_emb_start_idx : kps_emb_start_idx + num_keypoints,
            :,
        ] = posemb

        # Also maybe update token_embeddings with the grid sampled 2D feature.
        # Remember that pred_keypoints_2d_cropped are -0.5 ~ 0.5. We want -1 ~ 1
        pred_keypoints_2d_cropped_sample_points = pred_keypoints_2d_cropped * 2
        pred_keypoints_2d_cropped_sample_points = (
            pred_keypoints_2d_cropped_sample_points.reshape(
                token_embeddings.shape[0], num_keypoints, 2
            )
        )
        if self.cfg.MODEL.BACKBONE.TYPE in [
            "vit_amr",
            "vit",
            "vit_b",
            "vit_l",
            "vit_512_384",
        ]:
            pred_keypoints_2d_cropped_sample_points[:, :, 0] = (
                pred_keypoints_2d_cropped_sample_points[:, :, 0]
            )

        pred_keypoints_2d_cropped_feats = (
            torch.nn.functional.grid_sample(
                image_embeddings,
                pred_keypoints_2d_cropped_sample_points[:, :, None, :],  # -1 ~ 1, xy
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            .squeeze(3)
            .permute(0, 2, 1)
        )  # B x (A*K) x C
        invalid_mask_flat = invalid_mask.reshape(
            token_embeddings.shape[0], num_keypoints
        )
        pred_keypoints_2d_cropped_feats = pred_keypoints_2d_cropped_feats * (
            ~invalid_mask_flat[:, :, None]
        )
        token_embeddings[
            :,
            kps_emb_start_idx : kps_emb_start_idx + num_keypoints,
            :,
        ] += self.keypoint_feat_linear(pred_keypoints_2d_cropped_feats)

        return token_embeddings, token_augment, pose_output, layer_idx

    def keypoint3d_token_update_fn(
        self,
        kps3d_emb_start_idx,
        token_embeddings,
        token_augment,
        pose_output,
        layer_idx,
    ):
        # It's already after the last layer, we're done.
        if layer_idx == len(self.decoder.layers) - 1:
            return token_embeddings, token_augment, pose_output, layer_idx

        num_keypoints3d_per_animal = len(self.keypoint3d_embedding_idxs)
        num_keypoints3d = self.num_animals * num_keypoints3d_per_animal

        # Get current 3D kps predictions
        pred_keypoints_3d = pose_output["pred_keypoints_3d"].clone()
        if pred_keypoints_3d.ndim == 3:
            pred_keypoints_3d = pred_keypoints_3d[:, None, ...]

        # Now, pelvis normalize
        pred_keypoints_3d = (
            pred_keypoints_3d
            - (
                pred_keypoints_3d[:, :, [self.pelvis_idx[0]], :]
                + pred_keypoints_3d[:, :, [self.pelvis_idx[1]], :]
            )
            / 2
        )

        # Get the kps we care about, _after_ pelvis norm (just in case idxs shift)
        pred_keypoints_3d = pred_keypoints_3d[:, :, self.keypoint3d_embedding_idxs]

        pred_keypoints_3d_flat = pred_keypoints_3d.reshape(-1, num_keypoints3d, 3)
        posemb = self.keypoint3d_posemb_linear(pred_keypoints_3d_flat)
        posemb = posemb.view(token_embeddings.shape[0], num_keypoints3d, -1)

        token_augment[
            :,
            kps3d_emb_start_idx : kps3d_emb_start_idx + num_keypoints3d,
            :,
        ] = posemb

        return token_embeddings, token_augment, pose_output, layer_idx

    def _get_inference_keypoints_prompt(
        self, batch: Dict, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        keypoints_prompt = torch.zeros((batch_size, 1, 3), device=device, dtype=dtype)
        keypoints_prompt[:, :, -1] = -2

        if "keypoints_prompt" in batch and batch["keypoints_prompt"] is not None:
            return batch["keypoints_prompt"].to(device=device, dtype=dtype)

        # Eval / demo: only use GT prompt when explicitly enabled
        if not self.training and not self.use_gt_prompt:
            return keypoints_prompt

        # Training: drop entire prompt with probability prompt_drop_rate
        if self.training and random.random() < self.prompt_drop_rate:
            return keypoints_prompt

        num_animals = torch.max(batch['num_animals']).item()
        try:
            gt_keypoints_2d = []
            for b in batch['targets']:
                dummy = torch.zeros((num_animals - b['keypoints_2d'].shape[0], b['keypoints_2d'].shape[1], b['keypoints_2d'].shape[2]), device=b['keypoints_2d'].device)
                dummy[..., 2] = -2
                gt_keypoints_2d.append(
                    torch.cat(
                        (
                        b['keypoints_2d'],
                        dummy
                        ),
                        dim=0)
                    )
            gt_keypoints_2d = torch.stack(gt_keypoints_2d, dim=0)
        except Exception:
            return keypoints_prompt

        if gt_keypoints_2d.numel() == 0:
            return keypoints_prompt

        gt_xy = torch.clamp(gt_keypoints_2d[..., :2] + 0.5, min=0.0, max=1.0)
        vis = gt_keypoints_2d[..., 2] > 0.5
        in_img = (
            (gt_keypoints_2d[..., :2] <= 0.5) & (gt_keypoints_2d[..., :2] >= -0.5)
        ).all(dim=-1)
        valid = vis & in_img

        num_kps = gt_keypoints_2d.shape[-2]
        if hasattr(self, "prompt_keypoints"):
            label_map = torch.tensor(
                [self.prompt_keypoints.get(i, -2) for i in range(num_kps)],
                device=device,
                dtype=dtype,
            )
        else:
            label_map = torch.arange(num_kps, device=device, dtype=dtype)
        labels = label_map.view(1, 1, num_kps).expand_as(gt_keypoints_2d[..., 2]).clone()
        labels[~valid] = -2

        keypoints_prompt = torch.cat([gt_xy, labels.unsqueeze(-1)], dim=-1)
        # shape: [B, num_animals, num_kps, 3]

        # Per-point random dropout during training:
        # Randomly mask out individual keypoints so the model learns to work with
        # partial observations (simulating occlusion, detection failure, etc.).
        if self.training and self.point_drop_rate_max > 0:
            drop_rate = random.uniform(0.0, self.point_drop_rate_max)
            # [B, num_animals, num_kps] — True means drop this point
            point_drop = torch.rand(
                keypoints_prompt.shape[0],
                num_animals,
                num_kps,
                device=device,
            ) < drop_rate
            # Only drop points that are currently valid (label != -2)
            already_invalid = keypoints_prompt[..., -1] == -2
            point_drop = point_drop & ~already_invalid
            keypoints_prompt[point_drop] = torch.tensor([0.0, 0.0, -2.0], device=device, dtype=dtype)

        # Ablation: subsample a fixed number of prompt keypoints per animal during eval.
        if not self.training and self.num_prompt_keypoints is not None:
            n_keep = self.num_prompt_keypoints
            B_kp = keypoints_prompt.shape[0]
            for b_idx in range(B_kp):
                for a_idx in range(num_animals):
                    valid_mask = keypoints_prompt[b_idx, a_idx, :, -1] != -2
                    valid_indices = valid_mask.nonzero(as_tuple=False).view(-1)
                    if valid_indices.numel() > n_keep:
                        perm = torch.randperm(valid_indices.numel(), device=device)[:valid_indices.numel() - n_keep]
                        drop_indices = valid_indices[perm]
                        keypoints_prompt[b_idx, a_idx, drop_indices] = torch.tensor([0.0, 0.0, -2.0], device=device, dtype=dtype)

        return keypoints_prompt.view(batch_size, -1, 3).to(device=device, dtype=dtype)

    def forward_pose_branch(self, batch: Dict) -> Dict:
        """
        Run a forward pass for the crop-image (pose) branch.
        Args:
            batch (Dict): Dictionary containing batch data
        Returns:
            Dict: Dictionary containing the regression output
        """
        # Optionally get ray conditioining
        ray_cond = self.get_ray_condition(batch)  # This is B x 2 x H x W
        x = batch['img']

        batch_size = x.shape[0]
        batch["ray_cond"] = ray_cond.clone()
        ray_cond = None

        image_embeddings = self.backbone(
            x.type(self.backbone_dtype), extra_embed=ray_cond
        )  # (B, C, H, W)

        if isinstance(image_embeddings, tuple):
            image_embeddings = image_embeddings[-1]
        image_embeddings = image_embeddings.type(x.dtype)

        # Mask condition if available
        if self.cfg.MODEL.PROMPT_ENCODER.get("MASK_EMBED_TYPE", None) is not None:
            # v1: non-iterative mask conditioning
            if self.cfg.MODEL.PROMPT_ENCODER.get("MASK_PROMPT", "v1") == "v1":
                mask_embeddings = self._get_mask_prompt(batch, image_embeddings)
                image_embeddings = image_embeddings + mask_embeddings
            else:
                raise NotImplementedError

        # Prepare input for promptable decoder
        condition_info = self._get_decoder_condition(batch)

        keypoints_prompt = self._get_inference_keypoints_prompt(
            batch=batch, batch_size=batch_size, device=x.device, dtype=x.dtype
        )

        # Forward promptable decoder to get updated pose tokens and regression output
        tokens_output, all_pose_outputs, dn_meta = self.forward_decoder(
            image_embeddings,
            init_estimate=None,
            keypoints=keypoints_prompt,
            prev_estimate=None,
            condition_info=condition_info,
            batch=batch,
        )
        pose_output = all_pose_outputs[-1]
        pose_output = self._attach_confidence_selection(pose_output)

        output = {
            "smal": pose_output,
            "smal_aux": all_pose_outputs[:-1],  # (#1) intermediate outputs for aux losses
            "dn_meta": dn_meta,                 # (#2) denoising metadata
            "condition_info": condition_info,
            "image_embeddings": image_embeddings,
        }

        return output

    def _get_mask_prompt(self, batch: Dict, image_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Get mask prompt
        Args:
            batch (Dict): Dictionary containing batch data
            image_embeddings (torch.Tensor): Image embeddings
        Returns:
            torch.Tensor: Mask embeddings
        """
        x_mask = batch["mask"] if len(batch["mask"].shape) == 4 else batch["mask"].unsqueeze(1)
        mask_embeddings, no_mask_embeddings = self.prompt_encoder.get_mask_embeddings(
            x_mask, image_embeddings.shape[0], image_embeddings.shape[2:]
        )
        mask_score = batch["mask_score"].view(-1, 1, 1, 1)
        # 50% dropout during training: randomly discard valid masks so the model
        # learns to work without mask prompts (mask_score == 0 → no_mask_embeddings).
        if self.training and random.random() < 0.5:
            mask_score = torch.zeros_like(mask_score)
        if not self.training and not self.use_mask:
            mask_score = torch.zeros_like(mask_score)
        mask_embeddings = torch.where(
            mask_score > 0,
            mask_score * mask_embeddings.to(image_embeddings),
            no_mask_embeddings.to(image_embeddings),
        )
        return mask_embeddings

    def on_train_start(self) -> None:
        super().on_train_start()
        if self.trainer is None:
            return
        if self.grad_accum_steps > 1 and self.trainer.accumulate_grad_batches == 1:
            self.trainer.accumulate_grad_batches = self.grad_accum_steps
            self.print(
                f"Using gradient accumulation: {self.trainer.accumulate_grad_batches} steps"
            )

    def training_step(self, batch: Dict, batch_idx: int = 0) -> Dict:
        """
        Run a full training step
        Args:
            batch (Dict): Dictionary containing {'img', 'mask', 'keypoints_2d', 'keypoints_3d', 'orig_keypoints_2d',
                                                'box_center', 'box_size', 'img_size', 'smal_params',
                                                'smal_params_is_axis_angle', '_trans', 'imgname', 'focal_length'}
        Returns:
            Dict: Dictionary containing regression output.
        """
        batch = batch['img']

        batch_size = batch['img'].shape[0]
        output = self.forward_step(batch, train=True)
        if self.cfg.get('UPDATE_GT_SPIN', False):
            self.update_batch_gt_spin(batch, output)
        loss = self.compute_loss(batch, output, train=True)

        # Error if Nan
        if torch.isnan(loss):
            raise ValueError('Loss is NaN')

        if self.global_step > 0 and self.global_step % self.cfg.GENERAL.LOG_STEPS == 0:
            self.log_visualizations_to_wandb(batch, output, self.global_step, train=True)

        self.log('train/loss', output['losses']['loss'], on_step=True, on_epoch=True, prog_bar=True, logger=False,
                 batch_size=batch_size, sync_dist=True)

        return loss

    def compute_metric(self, batch: Dict, output: Dict):
        output = output['smal'] if 'smal' in output else output
        batch['keypoints_3d'] = torch.cat([b['keypoints_3d'] for b in batch['targets']], dim=0)
        batch['keypoints_2d'] = torch.cat([b['keypoints_2d'] for b in batch['targets']], dim=0)
        batch['smal_params'] = {
            "global_orient": torch.cat([b['smal_params']['global_orient'] for b in batch['targets']], dim=0),
            "pose": torch.cat([b['smal_params']['pose'] for b in batch['targets']], dim=0),
            "betas": torch.cat([b['smal_params']['betas'] for b in batch['targets']], dim=0),
        }
        batch['has_smal_params'] = {
            "global_orient": torch.cat([b['has_smal_params']['global_orient'] for b in batch['targets']], dim=0),
            "pose": torch.cat([b['has_smal_params']['pose'] for b in batch['targets']], dim=0),
            "betas": torch.cat([b['has_smal_params']['betas'] for b in batch['targets']], dim=0),
        }
        with torch.no_grad():
            pa_mpjpe, pa_mpvpe = self.evaluator.eval_3d(output, batch)
            # pck, auc = self.evaluator.eval_2d(output, batch)
        # return dict(PCK=pck[1], AUC=auc, PA_MPJPE=pa_mpjpe, PA_MPVPE=pa_mpvpe)
        return dict(PA_MPJPE=pa_mpjpe, PA_MPVPE=pa_mpvpe)