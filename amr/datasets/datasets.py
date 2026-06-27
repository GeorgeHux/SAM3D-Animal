import copy
import os
import numpy as np
import torch
from yacs.config import CfgNode
import cv2
import pyrootutils
from torch.utils.data import ConcatDataset
from typing import List
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

import json
import hydra
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple
from amr.datasets.utils import get_example, expand_to_aspect_ratio, trans_point2d


def collate_fn_multi_animal(batch):
    (img_patch, 
    mask_patch,
    mask_score,
    box_center,
    box_scale,
    orig_bbox_scale,
    bbox_expand_factor,
    ori_img_size,
    img_size,
    input_size,
    affine_trans,
    affine_trans_worot,
    cam_int,
    num_animals,
    targets) = zip(*batch)
    batch = {"img": np.stack(img_patch, axis=0),
             "mask": np.stack(mask_patch, axis=0),
             "mask_score": np.stack(mask_score, axis=0),
             'bbox_center': np.stack(box_center, axis=0),
             'bbox_scale': np.stack(box_scale, axis=0),
             'orig_bbox_scale': np.stack(orig_bbox_scale, axis=0),
             'bbox_expand_factor': np.stack(bbox_expand_factor, axis=0),
             'ori_img_size': np.stack(ori_img_size, axis=0),
             'img_size': np.stack(img_size, axis=0),
             'input_size': np.stack(input_size, axis=0),
             'affine_trans': np.stack(affine_trans, axis=0),
             'affine_trans_worot': np.stack(affine_trans_worot, axis=0),
             'cam_int': np.stack(cam_int, axis=0),
             'num_animals': np.stack(num_animals, axis=0),
             "targets": targets}
    return batch


class TrainDataset(Dataset):
    def __init__(self, cfg: CfgNode, is_train: bool, root_image: str, json_file: str):
        super().__init__()
        self.root_image = root_image
        self.focal_length = cfg.EXTRA.get("FOCAL_LENGTH", 5000)

        json_file = json_file
        with open(json_file, 'r') as f:
            self.data = json.load(f)

        self.is_train = is_train
        self.IMG_SIZE = cfg.MODEL.IMAGE_SIZE
        self.MEAN = 255. * np.array(cfg.MODEL.IMAGE_MEAN)
        self.STD = 255. * np.array(cfg.MODEL.IMAGE_STD)
        self.use_skimage_antialias = cfg.DATASETS.get('USE_SKIMAGE_ANTIALIAS', False)
        self.border_mode = {
            'constant': cv2.BORDER_CONSTANT,
            'replicate': cv2.BORDER_REPLICATE,
        }[cfg.DATASETS.get('BORDER_MODE', 'constant')]

        self.augm_config = cfg.DATASETS.CONFIG

    def __len__(self):
        return len(self.data['data'])

    def __getitem__(self, item):
        data = self.data['data'][item]
        key = data['img_path']
        image = np.array(Image.open(os.path.join(self.root_image, key)).convert("RGB"))
        h, w = image.shape[:2]
        mask = np.array(Image.open(os.path.join(self.root_image, data['mask_path'])).convert('L'))
        keypoint_2d = np.array(data['keypoint_2d'], dtype=np.float32)
        if keypoint_2d.shape[1] == 2:
            keypoint_2d = np.concatenate([keypoint_2d, np.ones((len(keypoint_2d), 1), dtype=np.float32)], axis=-1)
        if 'keypoint_3d' in data:
            keypoint_3d = np.concatenate(
                (data['keypoint_3d'], np.ones((len(data['keypoint_3d']), 1))), axis=-1).astype(np.float32)
        else:
            keypoint_3d = np.zeros((len(keypoint_2d), 4), dtype=np.float32)
        bbox = data['bbox']  # [x, y, w, h]
        center = np.array([(bbox[0] * 2 + bbox[2]) // 2, (bbox[1] * 2 + bbox[3]) // 2])
        pose = np.array(data['pose'], dtype=np.float32) if 'pose' in data else np.zeros(105, dtype=np.float32)  # [105, ]
        betas = np.array(data['shape'], dtype=np.float32) if 'shape' in data else np.zeros(145, dtype=np.float32)  # [145, ]
        translation = np.array(data['trans'], dtype=np.float32) if 'trans' in data else np.zeros(3, dtype=np.float32)  # [3, ]
        # Mark SMAL parameters as available when any value is non-zero (arrays of zeros indicate absence).
        has_pose = np.array(float(np.any(pose)), dtype=np.float32)
        has_betas = np.array(float(np.any(betas)), dtype=np.float32)
        has_translation = np.array(float(np.any(translation)), dtype=np.float32)
        ori_keypoint_2d = keypoint_2d.copy()
        center_x, center_y = center[0], center[1]
        bbox_size = max([bbox[2], bbox[3]])

        smal_params = {'global_orient': pose[:3],
                       'pose': pose[3:],
                       'betas': betas,
                       'transl': translation,
                       }
        has_smal_params = {'global_orient': has_pose,
                           'pose': has_pose,
                           'betas': has_betas,
                           'transl': has_translation,
                           }

        augm_config = copy.deepcopy(self.augm_config)
        img_rgba = np.concatenate([image, mask[:, :, None]], axis=2)
        img_patch_rgba, keypoints_2d, keypoints_3d, smal_params, has_smal_params, img_size, trans_worot, trans, img_border_mask, box_center, scale = get_example(
            img_rgba,
            center_x, center_y,
            bbox_size, bbox_size,
            keypoint_2d, keypoint_3d,
            smal_params, has_smal_params,
            self.IMG_SIZE, self.IMG_SIZE,
            self.MEAN, self.STD, self.is_train, augm_config,
            is_bgr=False, return_trans=True,
            use_skimage_antialias=self.use_skimage_antialias,
            border_mode=self.border_mode
        )
        # Keep numerical outputs in float32 to avoid dtype mismatches with AMP
        keypoints_2d = keypoints_2d.astype(np.float32)
        trans_worot = trans_worot.astype(np.float32)
        trans = trans.astype(np.float32)
        img_patch = (img_patch_rgba[:3, :, :])
        mask_patch = (img_patch_rgba[3, :, :] / 255.0).clip(0, 1)
        if (mask_patch < 0.5).all():
            mask_patch = np.ones_like(mask_patch)

        item = {'img': img_patch,
                'bbox': np.array(bbox, dtype=np.float32),
                'bbox_format': 'xywh',
                'mask': mask_patch,
                'mask_score': np.array(1., dtype=np.float32),
                'bbox_center': box_center,
                'bbox_scale': np.array([bbox_size, bbox_size], dtype=np.float32) * scale,
                'orig_bbox_scale': np.array([bbox_size, bbox_size], dtype=np.float32),
                'bbox_expand_factor': np.array(scale, dtype=np.float32),
                'ori_img_size': np.array([w, h], dtype=np.float32),
                'img_size': np.array([self.IMG_SIZE, self.IMG_SIZE], dtype=np.float32),
                'input_size': np.array([self.IMG_SIZE, self.IMG_SIZE], dtype=np.float32),
                'affine_trans': trans,
                'affine_trans_worot': trans_worot,
                'cam_int': np.array([[self.focal_length, 0., w / 2.], [0., self.focal_length, h / 2.], [0., 0., 1.]], dtype=np.float32),
                'keypoints_2d': keypoints_2d,
                'orig_keypoints_2d': ori_keypoint_2d,
                'keypoints_3d': keypoints_3d,
                'smal_params': smal_params,
                'has_smal_params': has_smal_params,
                }
        return item
    

class EvaluationDataset(Dataset):
    def __init__(self, root_image: str,  json_file: str, augm_config,
                 focal_length: int=5000, image_size: int=256, 
                 mean: List[float]=[0.485, 0.456, 0.406], std: List[float]=[0.229, 0.224, 0.225]):
        super().__init__()
        self.root_image = root_image
        self.focal_length = focal_length

        with open(json_file, 'r') as f:
            self.data = json.load(f)

        self.is_train = False
        self.IMG_SIZE = image_size
        self.MEAN = 255. * np.array(mean)
        self.STD = 255. * np.array(std)
        self.use_skimage_antialias = False
        self.border_mode = cv2.BORDER_CONSTANT
        self.augm_config = augm_config

    def __len__(self):
        return len(self.data['data'])

    def __getitem__(self, item):
        data = self.data['data'][item]
        key = data['img_path']
        image = np.array(Image.open(os.path.join(self.root_image, key)).convert("RGB"))
        h, w = image.shape[:2]
        mask = np.array(Image.open(os.path.join(self.root_image, data['mask_path'])).convert('L'))
        keypoint_2d = np.array(data['keypoint_2d'], dtype=np.float32)
        if keypoint_2d.shape[1] == 2:
            keypoint_2d = np.concatenate([keypoint_2d, np.ones((len(keypoint_2d), 1), dtype=np.float32)], axis=-1)
        if 'keypoint_3d' in data:
            keypoint_3d = np.concatenate(
                (data['keypoint_3d'], np.ones((len(data['keypoint_3d']), 1))), axis=-1).astype(np.float32)
        else:
            keypoint_3d = np.zeros((len(keypoint_2d), 4), dtype=np.float32)
        bbox = data['bbox']  # [x, y, w, h]
        center = np.array([(bbox[0] * 2 + bbox[2]) // 2, (bbox[1] * 2 + bbox[3]) // 2])
        pose = np.array(data['pose'], dtype=np.float32) if 'pose' in data else np.zeros(105, dtype=np.float32)  # [105, ]
        betas = np.array(data['shape'], dtype=np.float32) if 'shape' in data else np.zeros(145, dtype=np.float32)  # [145, ]
        translation = np.array(data['trans'], dtype=np.float32) if 'trans' in data else np.zeros(3, dtype=np.float32)  # [3, ]
        has_pose = np.array(float(np.any(pose)), dtype=np.float32)
        has_betas = np.array(float(np.any(betas)), dtype=np.float32)
        has_translation = np.array(float(np.any(translation)), dtype=np.float32)
        ori_keypoint_2d = keypoint_2d.copy()
        center_x, center_y = center[0], center[1]
        bbox_size = max([bbox[2], bbox[3]])

        smal_params = {'global_orient': pose[:3],
                       'pose': pose[3:],
                       'betas': betas,
                       'transl': translation,
                       }
        has_smal_params = {'global_orient': has_pose,
                           'pose': has_pose,
                           'betas': has_betas,
                           'transl': has_translation,
                           }

        augm_config = copy.deepcopy(self.augm_config)
        img_rgba = np.concatenate([image, mask[:, :, None]], axis=2)
        img_patch_rgba, keypoints_2d, keypoints_3d, smal_params, has_smal_params, img_size, trans_worot, trans, img_border_mask, box_center, scale = get_example(
            img_rgba,
            center_x, center_y,
            bbox_size, bbox_size,
            keypoint_2d, keypoint_3d,
            smal_params, has_smal_params,
            self.IMG_SIZE, self.IMG_SIZE,
            self.MEAN, self.STD, self.is_train, augm_config,
            is_bgr=False, return_trans=True,
            use_skimage_antialias=self.use_skimage_antialias,
            border_mode=self.border_mode
        )
        keypoints_2d = keypoints_2d.astype(np.float32)
        trans_worot = trans_worot.astype(np.float32)
        trans = trans.astype(np.float32)
        img_patch = (img_patch_rgba[:3, :, :])
        mask_patch = (img_patch_rgba[3, :, :] / 255.0).clip(0, 1)
        if (mask_patch < 0.5).all():
            mask_patch = np.ones_like(mask_patch)

        item = {'img': img_patch,
                'bbox': np.array(bbox, dtype=np.float32),
                'bbox_format': 'xywh',
                'mask': mask_patch,
                'mask_score': np.array(1., dtype=np.float32),
                'bbox_center': box_center,
                'bbox_scale': np.array([bbox_size, bbox_size], dtype=np.float32) * scale,
                'orig_bbox_scale': np.array([bbox_size, bbox_size], dtype=np.float32),
                'bbox_expand_factor': np.array(scale, dtype=np.float32),
                'ori_img_size': np.array([w, h], dtype=np.float32),
                'img_size': np.array([self.IMG_SIZE, self.IMG_SIZE], dtype=np.float32),
                'input_size': np.array([self.IMG_SIZE, self.IMG_SIZE], dtype=np.float32),
                'affine_trans': trans,
                'affine_trans_worot': trans_worot,
                'cam_int': np.array([[self.focal_length, 0., w / 2.], [0., self.focal_length, h / 2.], [0., 0., 1.]], dtype=np.float32),
                'keypoints_2d': keypoints_2d,
                'orig_keypoints_2d': ori_keypoint_2d,
                'keypoints_3d': keypoints_3d,
                'smal_params': smal_params,
                'has_smal_params': has_smal_params,
                }
        return item


class OptionAnimalDataset(Dataset):
    def __init__(self, cfg: CfgNode):
        datasets = []
        weights = []

        dataset_configs = cfg.DATASETS
        for dataset_name in dataset_configs:
            if dataset_name != "CONFIG":
                weight = dataset_configs[dataset_name].get("WEIGHT", 0.0)
                if weight <= 0.0:
                    continue
                datasets.append(TrainDataset(cfg, 
                                            is_train=True,
                                            root_image=dataset_configs[dataset_name].ROOT_IMAGE,
                                            json_file=dataset_configs[dataset_name].JSON_FILE.TRAIN))
                weights.extend([weight] * len(datasets[-1]))

        # Concatenate all enabled datasets
        if datasets:
            self.dataset = ConcatDataset(datasets)
            self.weights = torch.tensor(weights, dtype=torch.float32)
            # Track number of source datasets for sampling strategy decisions
            self.num_datasets = len(datasets)
        else:
            raise ValueError("No datasets enabled in the configuration.")
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]


class MultiAnimalTrainDataset(Dataset):
    def __init__(self, cfg: CfgNode, is_train: bool, root_image: str, json_file: str):
        super().__init__()
        self.root_image = root_image
        self.focal_length = cfg.EXTRA.get("FOCAL_LENGTH", 5000)

        json_file = json_file
        with open(json_file, 'r') as f:
            self.data = json.load(f)

        self.is_train = is_train
        self.IMG_SIZE = cfg.MODEL.IMAGE_SIZE
        self.MEAN = 255. * np.array(cfg.MODEL.IMAGE_MEAN)
        self.STD = 255. * np.array(cfg.MODEL.IMAGE_STD)
        self.use_skimage_antialias = cfg.DATASETS.get('USE_SKIMAGE_ANTIALIAS', False)
        self.border_mode = {
            'constant': cv2.BORDER_CONSTANT,
            'replicate': cv2.BORDER_REPLICATE,
        }[cfg.DATASETS.get('BORDER_MODE', 'constant')]

        self.augm_config = cfg.DATASETS.CONFIG

    def __len__(self):
        return len(self.data['data'])

    def __getitem__(self, item):
        data = self.data['data'][item]
        num_animals = data['num_animals']
        key = data['img_path']
        image = np.array(Image.open(os.path.join(self.root_image, key)).convert("RGB"))
        h, w = image.shape[:2]
        mask = np.array(Image.open(os.path.join(self.root_image, data['mask_path'])).convert('L')) if data['mask_path'] else np.zeros((h, w), dtype=np.uint8)
        keypoint_2d = np.array(data['keypoint_2d'], dtype=np.float32)
        if keypoint_2d.shape[-1] == 2:
            keypoint_2d = np.concatenate([keypoint_2d, np.ones((keypoint_2d.shape[0], keypoint_2d.shape[1], 1), dtype=np.float32)], axis=-1)
        if 'keypoint_3d' in data:
            keypoint_3d = np.array(data['keypoint_3d'], dtype=np.float32)
            if keypoint_3d.shape[-1] == 3:
                keypoint_3d = np.concatenate(
                    (keypoint_3d, np.ones((keypoint_3d.shape[0], keypoint_3d.shape[1], 1))), axis=-1).astype(np.float32)
        else:
            keypoint_3d = np.zeros((keypoint_2d.shape[0], keypoint_2d.shape[1], 4), dtype=np.float32)
        bbox = np.array(data['bbox'], dtype=np.float32)  # [N, 4] in xywh
        center = np.array([w / 2.0, h / 2.0], dtype=np.float32)
        pose = np.array(data['pose'], dtype=np.float32) if 'pose' in data else np.zeros((num_animals, 105), dtype=np.float32)  # [N, 105]
        betas = np.array(data['shape'], dtype=np.float32) if 'shape' in data else np.zeros((num_animals, 145), dtype=np.float32)  # [N,145]
        translation = np.array(data['trans'], dtype=np.float32) if 'trans' in data else np.zeros((num_animals, 3), dtype=np.float32)  # [N,3]
        # Mark SMAL parameters as available when any value is non-zero (arrays of zeros indicate absence).
        has_pose = np.any(pose, axis=1).astype(np.float32).reshape(-1, 1)
        has_betas = np.any(betas, axis=1).astype(np.float32).reshape(-1, 1)
        has_translation = np.any(translation, axis=1).astype(np.float32).reshape(-1, 1)
        ori_keypoint_2d = keypoint_2d.copy()
        center_x, center_y = center[0], center[1]
        bbox_size = max(w, h)
        bbox_width = bbox_size
        bbox_height = bbox_size

        smal_params = {'global_orient': pose[:, :3],
                       'pose': pose[:, 3:],
                       'betas': betas,
                       'transl': translation,
                       }
        has_smal_params = {'global_orient': has_pose,
                           'pose': has_pose,
                           'betas': has_betas,
                           'transl': has_translation,
                           }

        augm_config = copy.deepcopy(self.augm_config)
        img_rgba = np.concatenate([image, mask[:, :, None]], axis=2)
        img_patch_rgba, keypoints_2d, keypoints_3d, smal_params, has_smal_params, img_size, trans_worot, trans, img_border_mask, box_center, scale = get_example(
            img_rgba,
            center_x, center_y,
            bbox_width, bbox_height,
            keypoint_2d, keypoint_3d,
            smal_params, has_smal_params,
            self.IMG_SIZE, self.IMG_SIZE,
            self.MEAN, self.STD, self.is_train, augm_config,
            is_bgr=False, return_trans=True,
            use_skimage_antialias=self.use_skimage_antialias,
            border_mode=self.border_mode
        )
        # Transform bbox annotations using the same affine transform.
        bbox_xyxy = np.stack(
            [
                bbox[:, 0],
                bbox[:, 1],
                bbox[:, 0] + bbox[:, 2],
                bbox[:, 1] + bbox[:, 3],
            ],
            axis=-1,
        )
        transformed_bbox = []
        for i in range(bbox_xyxy.shape[0]):
            x1, y1, x2, y2 = bbox_xyxy[i]
            corners = np.array(
                [[x1, y1], [x2, y1], [x1, y2], [x2, y2]], dtype=np.float32
            )
            corners = np.stack([trans_point2d(c, trans) for c in corners], axis=0)
            x_min, y_min = corners.min(axis=0)
            x_max, y_max = corners.max(axis=0)
            transformed_bbox.append([x_min, y_min, x_max - x_min, y_max - y_min])
        bbox = np.array(transformed_bbox, dtype=np.float32)
        # Keep numerical outputs in float32 to avoid dtype mismatches with AMP
        keypoints_2d = keypoints_2d.astype(np.float32)
        trans_worot = trans_worot.astype(np.float32)
        trans = trans.astype(np.float32)
        img_patch = (img_patch_rgba[:3, :, :])
        mask_patch = (img_patch_rgba[3, :, :] / 255.0).clip(0, 1)
        if (mask_patch < 0.5).all():
            mask_patch = np.ones_like(mask_patch)

        # Draw Bbox and 2D keypoints on the image and save it as temp.png
        # vis_img = img_patch.transpose(1, 2, 0)
        # if self.MEAN is not None and self.STD is not None:
        #     vis_img = vis_img * self.STD + self.MEAN
        # vis_img = np.clip(vis_img, 0, 255).astype(np.uint8)
        # vis_bgr = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
        # for i in range(bbox.shape[0]):
        #     x, y, bw, bh = bbox[i]
        #     x1, y1 = int(round(x)), int(round(y))
        #     x2, y2 = int(round(x + bw)), int(round(y + bh))
        #     cv2.rectangle(vis_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        #     kpts = keypoints_2d[i]
        #     for j in range(kpts.shape[0]):
        #         if kpts[j, -1] > 0:
        #             px = int(round((kpts[j, 0] + 0.5) * self.IMG_SIZE))
        #             py = int(round((kpts[j, 1] + 0.5) * self.IMG_SIZE))
        #             cv2.circle(vis_bgr, (px, py), 2, (0, 0, 255), -1)
        # vis_img = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        # cv2.imwrite("temp.png", (vis_img[:, :, ::-1]).astype(np.uint8))
        # exit()
        mask_score = np.array(1., dtype=np.float32) if data['mask_path'] else np.array(0., dtype=np.float32)
        orig_bbox_scale = np.stack([bbox_width, bbox_height], axis=-1).astype(np.float32)
        bbox_expand_factor = np.array(scale, dtype=np.float32)
        box_scale = (np.stack([bbox_width, bbox_height], axis=-1) * bbox_expand_factor).astype(np.float32)
        ori_img_size = np.array([w, h], dtype=np.float32)
        img_size = np.array([self.IMG_SIZE, self.IMG_SIZE], dtype=np.float32)
        input_size = np.array([self.IMG_SIZE, self.IMG_SIZE], dtype=np.float32)
        affine_trans = trans
        affine_trans_worot = trans_worot
        cam_int = np.array([[self.focal_length, 0., w / 2.], [0., self.focal_length, h / 2.], [0., 0., 1.]], dtype=np.float32)
        orig_bbox = np.array(data['bbox'], dtype=np.float32)  # [N, 4] xywh in original image pixels
        item = {'bbox': bbox,
                'bbox_format': 'xywh',
                'img_path': key,
                'orig_bbox': orig_bbox,
                'keypoints_2d': keypoints_2d,
                'orig_keypoints_2d': ori_keypoint_2d,
                'keypoints_3d': keypoints_3d,
                'smal_params': smal_params,
                'has_smal_params': has_smal_params,
                }
        return (img_patch,
                mask_patch,
                mask_score,
                box_center,
                box_scale,
                orig_bbox_scale,
                bbox_expand_factor,
                ori_img_size,
                img_size,
                input_size,
                affine_trans,
                affine_trans_worot,
                cam_int,
                np.array(num_animals, dtype=np.int32),
                item
                )


class OptionMultiAnimalDataset(Dataset):
    def __init__(self, cfg: CfgNode):
        datasets = []
        weights = []

        dataset_configs = cfg.DATASETS
        for dataset_name in dataset_configs:
            weight = dataset_configs[dataset_name].get("WEIGHT", 0.0)
            if weight <= 0.0:
                continue
            if dataset_name != "CONFIG":
                datasets.append(
                    MultiAnimalTrainDataset(
                        cfg,
                        is_train=True,
                        root_image=dataset_configs[dataset_name].ROOT_IMAGE,
                        json_file=dataset_configs[dataset_name].JSON_FILE.TRAIN,
                    )
                )
                weights.extend([weight] * len(datasets[-1]))

        if datasets:
            self.dataset = ConcatDataset(datasets)
            self.weights = torch.tensor(weights, dtype=torch.float32)
            self.num_datasets = len(datasets)
        else:
            raise ValueError("No datasets enabled in the configuration.")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


def test_multi_animal_dataset():
    from amr.configs import get_config
    cfg = get_config("amr/configs_hydra/experiment/multi_animal_det.yaml")
    dataset = MultiAnimalTrainDataset(cfg, is_train=True, root_image=cfg.DATASETS.ANIMAL3D.ROOT_IMAGE, json_file=cfg.DATASETS.ANIMAL3D.JSON_FILE.TRAIN)
    for i in range(len(dataset)):
        (img_patch, 
        mask_patch,
        mask_score,
        box_center,
        box_scale,
        orig_bbox_scale,
        bbox_expand_factor,
        ori_img_size,
        img_size,
        input_size,
        affine_trans,
        affine_trans_worot,
        cam_int,
        num_animals,
        item) = dataset[i]
        print(item.keys())
        break


if __name__ == "__main__":
    test_multi_animal_dataset()