import torch
import pickle
import numpy as np
import pytorch_lightning as pl
import torch.nn.functional as F
from functools import partial
from torchvision.utils import make_grid
from typing import Any, Dict, Optional
from pytorch3d.transforms import matrix_to_axis_angle, rotation_6d_to_matrix, axis_angle_to_matrix, matrix_to_rotation_6d
from ..utils.evaluate_metric import Evaluator
from yacs.config import CfgNode
from ..utils import MeshRenderer
from ..utils.renderer import cam_full_to_crop, cam_crop_s_to_t
from ..utils.geometry import aa_to_rotmat, perspective_projection
from ..utils.pylogger import get_pylogger
from ..utils.fp16_utils import convert_module_to_f16, convert_to_fp16_safe
from .backbones import create_backbone
from .losses import Keypoint3DLoss, Keypoint2DLoss, ParameterLoss
from .heads import build_smal_head
from .smal_warapper import SMAL
from .components.transformer import FFN
from .decoders import build_decoder, build_keypoint_sampler, PromptEncoder
from .components.camera_embed import CameraEncoder
import wandb

log = get_pylogger(__name__)
PROMPT_KEYPOINTS = {"animals": {i: i for i in range(26)}}
KEY_BODY = list(range(26))


class AMR(pl.LightningModule):
    def __init__(self, cfg: CfgNode):
        """
        Setup Model
        Args:
            cfg (CfgNode): Config file as a yacs CfgNode
        """
        super().__init__()
        self.pelvis_idx = [10, 11]
        # Save hyperparameters
        self.save_hyperparameters(logger=False)

        self.cfg = cfg
        self._wandb_metrics_defined = False
        self._wandb_vis_step = {"train": 0, "val": 0}
        # Create backbone feature extractor
        self.backbone = create_backbone(cfg)
        if cfg.MODEL.BACKBONE.get('PRETRAINED_WEIGHTS', None):
            log.info(f'Loading backbone weights from {cfg.MODEL.BACKBONE.PRETRAINED_WEIGHTS}')
            state_dict = torch.load(cfg.MODEL.BACKBONE.PRETRAINED_WEIGHTS, map_location='cpu', weights_only=True)['state_dict']
            state_dict = {k.replace('backbone.', ''): v for k, v in state_dict.items()}
            missing, unexpected = self.backbone.load_state_dict(state_dict, strict=False)
            log.info(f'Missing keys: {missing}')
            log.info(f'Unexpected keys: {unexpected}')
        # Manually convert the torso of the model to fp16.
        if self.cfg.TRAIN.USE_FP16:
            self.convert_to_fp16()
            if self.cfg.TRAIN.get("FP16_TYPE", "float16") == "float16":
                self.backbone_dtype = torch.float16
            else:
                self.backbone_dtype = torch.bfloat16
        else:
            self.backbone_dtype = torch.float32

        # Create SMAL head
        self.smal_head = build_smal_head(cfg, head_type="sam3d")
        # Initialize pose token with zero-pose
        self.init_pose = torch.nn.Embedding(1, self.smal_head.npose)

        # Create camera head
        self.camera_head = build_smal_head(cfg, head_type="perspective")
        # Initialize camera token with zero-pose
        self.init_camera = torch.nn.Embedding(1, self.camera_head.ncam)
        torch.nn.init.zeros_(self.init_camera.weight)

        # Support conditioned information for decoder
        cond_dim = 3
        init_dim = self.smal_head.npose + self.camera_head.ncam + cond_dim
        self.init_to_token_smal = torch.nn.Linear(init_dim, self.cfg.MODEL.DECODER.DIM)
        self.prev_to_token_smal = torch.nn.Linear(
            init_dim - cond_dim, self.cfg.MODEL.DECODER.DIM
        )

        # Create prompt encoder
        self.max_num_clicks = 0
        if self.cfg.MODEL.PROMPT_ENCODER.ENABLE:
            self.max_num_clicks = self.cfg.MODEL.PROMPT_ENCODER.MAX_NUM_CLICKS
            self.prompt_keypoints = PROMPT_KEYPOINTS[
                self.cfg.MODEL.PROMPT_ENCODER.PROMPT_KEYPOINTS
            ]

            self.prompt_encoder = PromptEncoder(
                embed_dim=self.backbone.embed_dims,  # need to match backbone dims for PE
                num_body_joints=len(set(self.prompt_keypoints.values())),
                frozen=self.cfg.MODEL.PROMPT_ENCODER.get("frozen", False),
                mask_embed_type=self.cfg.MODEL.PROMPT_ENCODER.get(
                    "MASK_EMBED_TYPE", None
                ),
            )
            self.prompt_to_token = torch.nn.Linear(
                self.backbone.embed_dims, self.cfg.MODEL.DECODER.DIM
            )

            self.keypoint_prompt_sampler = build_keypoint_sampler(
                self.cfg.MODEL.PROMPT_ENCODER.get("KEYPOINT_SAMPLER", {}),
                prompt_keypoints=self.prompt_keypoints,
                keybody_idx=(KEY_BODY),
            )
            # To keep track of prompting history
            self.prompt_hist = np.zeros(
                (len(set(self.prompt_keypoints.values())) + 2, self.max_num_clicks),
                dtype=np.float32,
            )

            if self.cfg.MODEL.DECODER.FROZEN:
                for param in self.prompt_to_token.parameters():
                    param.requires_grad = False

        # Create promptable decoder
        self.decoder = build_decoder(
            self.cfg.MODEL.DECODER, context_dim=self.backbone.embed_dims
        )

        # Create Camera Encoder
        self.ray_cond_emb = CameraEncoder(
            self.backbone.embed_dim,
            self.backbone.patch_size,
        )

        self.keypoint_embedding_idxs = list(range(26))
        self.keypoint_embedding = torch.nn.Embedding(
            len(self.keypoint_embedding_idxs), self.cfg.MODEL.DECODER.DIM
        )
        self.keypoint_posemb_linear = FFN(
            embed_dims=2,
            feedforward_channels=self.cfg.MODEL.DECODER.DIM,
            output_dims=self.cfg.MODEL.DECODER.DIM,
            num_fcs=2,
            add_identity=False,
        )
        self.keypoint_feat_linear = torch.nn.Linear(
            self.backbone.embed_dims, self.cfg.MODEL.DECODER.DIM
        )

        # Do all KPS
        self.keypoint3d_embedding_idxs = list(range(26))
        self.keypoint3d_embedding = torch.nn.Embedding(
            len(self.keypoint3d_embedding_idxs), self.cfg.MODEL.DECODER.DIM
        )
        self.keypoint3d_posemb_linear = FFN(
            embed_dims=3,
            feedforward_channels=self.cfg.MODEL.DECODER.DIM,
            output_dims=self.cfg.MODEL.DECODER.DIM,
            num_fcs=2,
            add_identity=False,
        )

        # Define loss functions
        self.keypoint_3d_loss = Keypoint3DLoss(loss_type='l1')
        self.keypoint_2d_loss = Keypoint2DLoss(loss_type='l1')
        self.smal_parameter_loss = ParameterLoss()

        with open(cfg.SMAL.MODEL_PATH, 'rb') as f:
            smal_cfg = pickle.load(f, encoding='latin1')
        self.smal = SMAL(**smal_cfg)

        # Buffer that shows whetheer we need to initialize ActNorm layers
        self.register_buffer('initialized', torch.tensor(False))
        # Lazily initialize renderer to keep the module picklable for ddp_spawn.
        # pyrender.OffscreenRenderer contains ctypes pointers that cannot be pickled.
        self.mesh_renderer = None
        self.evaluator = Evaluator(smal_model=self.smal)

    def _define_wandb_step_metric(self) -> None:
        if self._wandb_metrics_defined:
            return
        if self.logger is None or not hasattr(self.logger, "experiment"):
            return
        define_metric = getattr(self.logger.experiment, "define_metric", None)
        if callable(define_metric):
            define_metric("trainer/global_step")
            define_metric("train/*", step_metric="trainer/global_step")
            define_metric("val/*", step_metric="trainer/global_step")
            define_metric("train/vis_step")
            define_metric("val/vis_step")
            # Keep image panels on a dedicated axis to avoid sparse-media warnings in DDP.
            define_metric("train/predictions", step_metric="train/vis_step")
            define_metric("val/predictions", step_metric="val/vis_step")
        self._wandb_metrics_defined = True

    def _log_wandb_payload(self, log_data: Dict[str, Any], step_count: int, mode: str) -> None:
        if self.logger is None or not hasattr(self.logger, "experiment"):
            return
        self._define_wandb_step_metric()
        log_data["trainer/global_step"] = int(step_count)
        pred_key = f"{mode}/predictions"
        if pred_key in log_data:
            self._wandb_vis_step[mode] += 1
            log_data[f"{mode}/vis_step"] = self._wandb_vis_step[mode]
        self.logger.experiment.log(log_data)

    def convert_to_fp16(self) -> torch.dtype:
        """
        Convert the torso of the model to float16.
        """
        fp16_type = (
            torch.float16
            if self.cfg.TRAIN.get("FP16_TYPE", "float16") == "float16"
            else torch.bfloat16
        )
        if hasattr(self, "backbone"):
            log.info("Converting backbone to fp16")
            self._set_fp16(self.backbone, fp16_type)
        if hasattr(self, "full_encoder"):
            log.info("Converting full_encoder to fp16")
            self._set_fp16(self.full_encoder, fp16_type)
        return fp16_type

    def _get_mesh_renderer(self) -> MeshRenderer:
        """
        Build renderer on first use to avoid pickling issues with spawn strategies.
        """
        if self.mesh_renderer is None:
            self.mesh_renderer = MeshRenderer(self.cfg, faces=self.smal.faces.cpu().numpy())
        return self.mesh_renderer

    def _set_fp16(self, module, fp16_type):
        if hasattr(module, "pos_embed"):
            module.apply(partial(convert_module_to_f16, dtype=fp16_type))
            module.pos_embed.data = module.pos_embed.data.to(fp16_type)
        elif hasattr(module.encoder, "rope_embed"):
            # DINOv3
            module.encoder.apply(partial(convert_to_fp16_safe, dtype=fp16_type))
            module.encoder.rope_embed = module.encoder.rope_embed.to(fp16_type)
        else:
            # DINOv2
            module.encoder.pos_embed.data = module.encoder.pos_embed.data.to(fp16_type)

    def get_parameters(self):
        # Collect every trainable parameter from the full module hierarchy
        return list(self.parameters())

    def configure_optimizers(self):
        """
        Setup model Optimizers
        Returns:
            torch.optim.Optimizer: Model and discriminator optimizers
        """
        param_groups = [{'params': filter(lambda p: p.requires_grad, self.get_parameters()), 'lr': self.cfg.TRAIN.LR * self.cfg.GENERAL.get("GRAD_ACCUM_STEPS", 1)}]
        optimizer = torch.optim.AdamW(params=param_groups, weight_decay=self.cfg.TRAIN.WEIGHT_DECAY)
        warmup_steps = max(1, int(0.05 * self.cfg.GENERAL.TOTAL_STEPS))
        cosine_steps = max(1, self.cfg.GENERAL.TOTAL_STEPS - warmup_steps)

        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-3,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cosine_steps, eta_min=1.25e-6
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1}}

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
        if self.cfg.MODEL.BACKBONE.TYPE in [
            "vit_amr",
            "vit",
        ]:
            # ViT backbone assumes a different aspect ratio as input size
            mask_embeddings = mask_embeddings[:, :, :, 2:-2]
        elif self.cfg.MODEL.BACKBONE.TYPE in [
            "vit_512_384",
        ]:
            # for x2 resolution
            mask_embeddings = mask_embeddings[:, :, :, 4:-4]

        mask_score = batch["mask_score"].view(-1, 1, 1, 1)
        mask_embeddings = torch.where(
            mask_score > 0,
            mask_score * mask_embeddings.to(image_embeddings),
            no_mask_embeddings.to(image_embeddings),
        )
        return mask_embeddings

    def _get_decoder_condition(self, batch: Dict) -> Optional[torch.Tensor]:
        """
        Get decoder condition
        Args:
            batch (Dict): Dictionary containing batch data
        Returns:
            Optional[torch.Tensor]: Decoder condition
        """
        if self.cfg.MODEL.DECODER.CONDITION_TYPE == "cliff":
            # CLIFF-style condition info (cx/f, cy/f, b/f)
            cx, cy = torch.chunk(batch["bbox_center"], chunks=2, dim=-1)
            img_w, img_h = torch.chunk(batch["ori_img_size"], chunks=2, dim=-1)
            b = batch["bbox_scale"][:, [0]]
            focal_length = batch["cam_int"][:, 0, 0]
            if not self.cfg.MODEL.DECODER.get("USE_INTRIN_CENTER", False):
                condition_info = torch.cat(
                    [cx - img_w / 2.0, cy - img_h / 2.0, b], dim=-1
                )
            else:
                full_img_cxy = batch["cam_int"][:, [0, 1], [2, 2]]
                condition_info = torch.cat([cx - full_img_cxy[:, [0]], cy - full_img_cxy[:, [1]], b], dim=-1)
            condition_info[:, :2] = condition_info[:, :2] / focal_length.unsqueeze(
                -1
            )  # [-1, 1]
            condition_info[:, 2] = condition_info[:, 2] / focal_length  # [-1, 1]
        elif self.cfg.MODEL.DECODER.CONDITION_TYPE == "none":
            return None
        else:
            raise NotImplementedError

        return condition_info.type(batch["img"].dtype)

    def camera_project(self, pose_output: Dict, batch: Dict) -> Dict:
        """
        Project 3D keypoints to 2D using the camera parameters.
        Args:
            pose_output (Dict): Dictionary containing the pose output.
            batch (Dict): Dictionary containing the batch data.
        Returns:
            Dict: Dictionary containing the projected 2D keypoints.
        """
        if hasattr(self, "camera_head"):
            camera_head = self.camera_head
            pred_cam = pose_output["pred_cam"]
        else:
            assert False

        cam_out = camera_head.perspective_projection(
            pose_output["pred_keypoints_3d"],
            pred_cam,
            batch["bbox_center"],
            batch["bbox_scale"][:, 0],
            batch["ori_img_size"],
            batch["cam_int"],
            use_intrin_center=self.cfg.MODEL.DECODER.get("USE_INTRIN_CENTER", False),
        )
        if pose_output.get("pred_vertices", None) is not None:
            cam_out_vertices = camera_head.perspective_projection(
                pose_output["pred_vertices"],
                pred_cam,
                batch["bbox_center"],
                batch["bbox_scale"][:, 0],
                batch["ori_img_size"],
                batch["cam_int"],
                use_intrin_center=self.cfg.MODEL.DECODER.get(
                    "USE_INTRIN_CENTER", False
                ),
            )
            pose_output["pred_keypoints_2d_verts"] = cam_out_vertices[
                "pred_keypoints_2d"
            ]

        pose_output.update(cam_out)

        return pose_output

    def _full_to_crop(
        self,
        batch: Dict,
        pred_keypoints_2d: torch.Tensor,
    ) -> torch.Tensor:
        """Convert full-image keypoints coordinates to crop and normalize to [-0.5. 0.5]"""
        pred_keypoints_2d_cropped = torch.cat(
            [pred_keypoints_2d, torch.ones_like(pred_keypoints_2d[:, :, [-1]])], dim=-1
        )
        affine_trans = batch["affine_trans_worot"].to(pred_keypoints_2d_cropped)
        img_size = batch["img_size"].unsqueeze(1)
        pred_keypoints_2d_cropped = pred_keypoints_2d_cropped @ affine_trans.mT
        pred_keypoints_2d_cropped = pred_keypoints_2d_cropped[..., :2] / img_size - 0.5

        return pred_keypoints_2d_cropped

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
            init_pose = self.init_pose.weight.expand(batch_size, -1).unsqueeze(dim=1)
            if hasattr(self, "init_camera"):
                init_camera = self.init_camera.weight.expand(batch_size, -1).unsqueeze(
                    dim=1
                )

            init_estimate = (
                init_pose
                if not hasattr(self, "init_camera")
                else torch.cat([init_pose, init_camera], dim=-1)
            )  # This is basically pose & camera translation at the end. B x 1 x (npose + 3)

        if condition_info is not None:
            init_input = torch.cat(
                [condition_info.view(batch_size, 1, -1), init_estimate], dim=-1
            )  # B x 1 x (npose + 3 + c) (this is with the CLIFF condition)
        else:
            init_input = init_estimate
        token_embeddings = self.init_to_token_smal(init_input).view(
            batch_size, 1, -1
        )  # B x 1 x 1024 (linear layered)

        num_pose_token = token_embeddings.shape[1]
        assert num_pose_token == 1

        image_augment, token_augment, token_mask = None, None, None
        if hasattr(self, "prompt_encoder") and keypoints is not None:
            if prev_estimate is None:
                # Use initial embedding if no previous embedding
                prev_estimate = init_estimate
            # Previous estimate w/o the CLIFF condition.
            prev_embeddings = self.prev_to_token_smal(prev_estimate).view(
                batch_size, 1, -1
            )  # npose + 3 + c -> B x 1 x 1024; linear layer-ed

            if self.cfg.MODEL.BACKBONE.TYPE in [
                "vit_amr",
                "vit",
                "vit_b",
                "vit_l",
            ]:
                # ViT backbone assumes a different aspect ratio as input size
                image_augment = self.prompt_encoder.get_dense_pe((16, 16))[
                    :, :, :, 2:-2
                ]
            elif self.cfg.MODEL.BACKBONE.TYPE in [
                "vit_512_384",
            ]:
                # ViT backbone assumes a different aspect ratio as input size
                image_augment = self.prompt_encoder.get_dense_pe((32, 32))[
                    :, :, :, 4:-4
                ]
            else:
                image_augment = self.prompt_encoder.get_dense_pe(
                    image_embeddings.shape[-2:]
                )  # (1, C, H, W)

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
            token_augment[:, [num_pose_token]] = prev_embeddings
            token_augment[:, (num_pose_token + 1) :] = prompt_embeddings
            token_mask = None

            # Put in a token for each keypoint
            kps_emb_start_idx = token_embeddings.shape[1]
            token_embeddings = torch.cat(
                [
                    token_embeddings,
                    self.keypoint_embedding.weight[None, :, :].repeat(batch_size, 1, 1),
                ],
                dim=1,
            )  # B x 3 + 26 x 1024
            # No positional embeddings
            token_augment = torch.cat(
                [
                    token_augment,
                    torch.zeros_like(token_embeddings[:, token_augment.shape[1] :, :]),
                ],
                dim=1,
            )  # B x 3 + 26 x 1024
            if self.cfg.MODEL.DECODER.get("DO_KEYPOINT3D_TOKENS", False):
                # Put in a token for each keypoint
                kps3d_emb_start_idx = token_embeddings.shape[1]
                token_embeddings = torch.cat(
                    [
                        token_embeddings,
                        self.keypoint3d_embedding.weight[None, :, :].repeat(
                            batch_size, 1, 1
                        ),
                    ],
                    dim=1,
                )  # B x 3 + 26 + 26 x 1024
                # No positional embeddings
                token_augment = torch.cat(
                    [
                        token_augment,
                        torch.zeros_like(
                            token_embeddings[:, token_augment.shape[1] :, :]
                        ),
                    ],
                    dim=1,
                )  # B x 3 + 26 + 26 x 1024

        # We're doing intermediate model predictions
        def token_to_pose_output_fn(tokens, prev_pose_output, layer_idx):
            # Get the pose token
            pose_token = tokens[:, 0]

            prev_pose = init_pose.view(batch_size, -1)
            prev_camera = init_camera.view(batch_size, -1)

            # Get pose outputs
            pred_smal_params, pred_pose_6d = self.smal_head(pose_token, prev_pose)
            smal_output = self.smal(**pred_smal_params)
            pose_output = {
                'pred_pose_raw': pred_pose_6d,
                'pred_global_orient': pred_smal_params['global_orient'],
                'pred_pose': pred_smal_params['pose'],
                'pred_betas': pred_smal_params['betas'],
                'pred_keypoints_3d': smal_output.joints,
                'pred_vertices': smal_output.vertices,
            }
            # Get Camera Translation
            if hasattr(self, "camera_head"):
                pred_cam = self.camera_head(pose_token, prev_camera)
                pose_output["pred_cam"] = pred_cam
            # Run camera projection
            pose_output = self.camera_project(pose_output, batch)

            # Get 2D KPS in crop
            pose_output["pred_keypoints_2d_cropped"] = self._full_to_crop(
                batch, pose_output["pred_keypoints_2d"]
            )

            return pose_output

        kp_token_update_fn = self.keypoint_token_update_fn

        # Now for 3D
        kp3d_token_update_fn = self.keypoint3d_token_update_fn

        # Combine the 2D and 3D functionse
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
        )

        return pose_token, pose_output

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

        num_keypoints = self.keypoint_embedding.weight.shape[0]

        # Get current 2D KPS predictions
        pred_keypoints_2d_cropped = pose_output[
            "pred_keypoints_2d_cropped"
        ].clone()  # These are -0.5 ~ 0.5
        pred_keypoints_2d_depth = pose_output["pred_keypoints_2d_depth"].clone()
        if pred_keypoints_2d_cropped.ndim == 4:
            pred_keypoints_2d_cropped = pred_keypoints_2d_cropped[:, 0]
            pred_keypoints_2d_depth = pred_keypoints_2d_depth[:, 0]

        pred_keypoints_2d_cropped = pred_keypoints_2d_cropped[
            :, self.keypoint_embedding_idxs
        ]
        pred_keypoints_2d_depth = pred_keypoints_2d_depth[
            :, self.keypoint_embedding_idxs
        ]

        # Get 2D KPS to be 0 ~ 1
        pred_keypoints_2d_cropped_01 = pred_keypoints_2d_cropped + 0.5

        # Get a mask of those that are 1) beyond image boundaries or 2) behind the camera
        invalid_mask = (
            (pred_keypoints_2d_cropped_01[:, :, 0] < 0)
            | (pred_keypoints_2d_cropped_01[:, :, 0] > 1)
            | (pred_keypoints_2d_cropped_01[:, :, 1] < 0)
            | (pred_keypoints_2d_cropped_01[:, :, 1] > 1)
            | (pred_keypoints_2d_depth[:, :] < 1e-5)
        )

        # Run them through the prompt encoder's pos emb function
        token_augment[:, kps_emb_start_idx : kps_emb_start_idx + num_keypoints, :] = (
            self.keypoint_posemb_linear(pred_keypoints_2d_cropped)
            * (~invalid_mask[:, :, None])
        )

        # Also maybe update token_embeddings with the grid sampled 2D feature.
        # Remember that pred_keypoints_2d_cropped are -0.5 ~ 0.5. We want -1 ~ 1
        # Sample points...
        ## Get sampling points
        pred_keypoints_2d_cropped_sample_points = pred_keypoints_2d_cropped * 2
        if self.cfg.MODEL.BACKBONE.TYPE in [
            "vit_amr",
            "vit",
            "vit_b",
            "vit_l",
            "vit_512_384",
        ]:
            # Need to go from 256 x 256 coords to 256 x 192 (HW) because image_embeddings is 16x12
            # Aka, for x, what was normally -1 ~ 1 for 256 should be -16/12 ~ 16/12 (since to sample at original 256, need to overflow)
            pred_keypoints_2d_cropped_sample_points[:, :, 0] = (
                pred_keypoints_2d_cropped_sample_points[:, :, 0] / 12 * 16
            )

        # Version 2 is projecting & bilinear sampling
        pred_keypoints_2d_cropped_feats = (
            F.grid_sample(
                image_embeddings,
                pred_keypoints_2d_cropped_sample_points[:, :, None, :],  # -1 ~ 1, xy
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            .squeeze(3)
            .permute(0, 2, 1)
        )  # B x kps x C
        # Zero out invalid locations...
        pred_keypoints_2d_cropped_feats = pred_keypoints_2d_cropped_feats * (
            ~invalid_mask[:, :, None]
        )
        # This is ADDING
        token_embeddings = token_embeddings.clone()
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

        num_keypoints3d = self.keypoint3d_embedding.weight.shape[0]

        # Get current 3D kps predictions
        pred_keypoints_3d = pose_output["pred_keypoints_3d"].clone()

        # Now, pelvis normalize
        pred_keypoints_3d = (
            pred_keypoints_3d
            - (
                pred_keypoints_3d[:, [self.pelvis_idx[0]], :]
                + pred_keypoints_3d[:, [self.pelvis_idx[1]], :]
            )
            / 2
        )

        # Get the kps we care about, _after_ pelvis norm (just in case idxs shift)
        pred_keypoints_3d = pred_keypoints_3d[:, self.keypoint3d_embedding_idxs]

        # Run through embedding MLP & put in
        token_augment = token_augment.clone()
        token_augment[
            :,
            kps3d_emb_start_idx : kps3d_emb_start_idx + num_keypoints3d,
            :,
        ] = self.keypoint3d_posemb_linear(pred_keypoints_3d)

        return token_embeddings, token_augment, pose_output, layer_idx

    def get_ray_condition(self, batch):
        B, _, H, W = batch["img"].shape
        meshgrid_xy = (
            torch.stack(
                torch.meshgrid(torch.arange(H), torch.arange(W), indexing="xy"), dim=2
            )[None, :, :, :]
            .repeat(B, 1, 1, 1)
            .to(batch["img"].device)
        )  # B x H x W x 2
        meshgrid_xy = (
            meshgrid_xy / batch["affine_trans"][:, None, None, [0, 1], [0, 1]]
        )
        meshgrid_xy = (
            meshgrid_xy
            - batch["affine_trans"][:, None, None, [0, 1], [2, 2]]
            / batch["affine_trans"][:, None, None, [0, 1], [0, 1]]
        )

        # Subtract out center & normalize to be rays
        meshgrid_xy = (
            meshgrid_xy - batch["cam_int"][:, None, None, [0, 1], [2, 2]]
        )
        meshgrid_xy = (
            meshgrid_xy / batch["cam_int"][:, None, None, [0, 1], [0, 1]]
        )

        return meshgrid_xy.permute(0, 3, 1, 2).to(
            batch["img"].dtype
        )  # This is B x 2 x H x W

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
        if self.cfg.MODEL.BACKBONE.TYPE in [
            "vit_amr",
            "vit",
            "vit_b",
            "vit_l",
        ]:
            x = batch['img'][:, :, :, 32:-32]
            ray_cond = ray_cond[:, :, :, 32:-32]
        elif self.cfg.MODEL.BACKBONE.TYPE in [
            "vit_512_384",
        ]:
            x = batch['img'][:, :, :, 64:-64]
            ray_cond = ray_cond[:, :, :, 64:-64]

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

        # Initial estimate with a dummy prompt
        keypoints_prompt = torch.zeros((batch_size, 1, 3)).to(x.device)
        keypoints_prompt[:, :, -1] = -2

        # Forward promptable decoder to get updated pose tokens and regression output
        pose_output = None
        tokens_output, pose_output = self.forward_decoder(
            image_embeddings,
            init_estimate=None,
            keypoints=keypoints_prompt,
            prev_estimate=None,
            condition_info=condition_info,
            batch=batch,
        )
        pose_output = pose_output[-1]

        output = {
            # "pose_token": pose_token,
            "smal": pose_output,  # smal prediction output
            "condition_info": condition_info,
            "image_embeddings": image_embeddings,
        }

        return output

    def forward_step(self, batch: Dict, train: bool = False) -> Dict:
        """
        Run a forward step of the network
        Args:
            batch (Dict): Dictionary containing batch data
            train (bool): Flag indicating whether it is training or validation mode
        Returns:
            Dict: Dictionary containing the regression output
        """
        # Crop-image (pose) branch
        output = self.forward_pose_branch(batch)
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
        batch_size = pred_pose_raw.shape[0]
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
        gt_keypoints_2d = batch["keypoints_2d"].view(batch_size, -1, 3)
        gt_keypoints_3d = batch["keypoints_3d"].view(batch_size, -1, 4)
        gt_smal_params = batch["smal_params"]
        has_smal_params = batch["has_smal_params"]
        gt_global_orient = axis_angle_to_matrix(gt_smal_params["global_orient"].view(batch_size, -1, 3))          # (B,1,3,3)
        gt_pose = axis_angle_to_matrix(gt_smal_params["pose"].view(batch_size, -1, 3))              # (B,J,3,3)
        gt_betas = gt_smal_params["betas"]                     # (B, ~145) shape comps

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

        output["losses"] = losses

        return loss

    def compute_metric(self, batch: Dict, output: Dict):
        output = output['smal'] if 'smal' in output else output
        with torch.no_grad():
            pa_mpjpe, pa_mpvpe = self.evaluator.eval_3d(output, batch)
            pck, auc = self.evaluator.eval_2d(output, batch)
        return dict(PCK=pck[1], AUC=auc, PA_MPJPE=pa_mpjpe, PA_MPVPE=pa_mpvpe)
  
    @pl.utilities.rank_zero.rank_zero_only
    def log_visualizations_to_wandb(self, batch: Dict, output: Dict, step_count: int, train: bool = True) -> None:
        """
        Log results to W&B
        Args:
            batch (Dict): Dictionary containing batch data
            output (Dict): Dictionary containing the regression output
            step_count (int): Global training step count
            train (bool): Flag indicating whether it is training or validation mode
        """
        losses = output['losses']
        # Keep metrics before narrowing to SMAL outputs so validation metrics are logged
        metrics = output.get('metric')
        output = output['smal'] if 'smal' in output else output
        mode = 'train' if train else 'val'
        batch_size = batch['keypoints_2d'].view(-1, *batch['keypoints_2d'].shape[-2:]).shape[0]
        images = batch['img'].view(batch_size, *batch['img'].shape[-3:])
        # Un-normalize images for visualization
        images = images * torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
        images = images + torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)

        pred_vertices = output['pred_vertices'].detach().reshape(batch_size, -1, 3)
        gt_keypoints_2d = batch['keypoints_2d'].view(batch_size, -1, 3)
        pred_cam_t = output['pred_cam_t'].detach().reshape(batch_size, 3)
        # Use crop-normalized keypoints so rendering overlays correctly
        pred_keypoints_2d = output['pred_keypoints_2d_cropped'].detach().reshape(batch_size, -1, 2)

        def _to_float32_numpy(tensor: torch.Tensor) -> Any:
            return tensor.to(torch.float32).cpu().numpy()

        # Create a dictionary to hold all logging data
        log_data = {}

        # Add losses to the log dictionary
        for loss_name, val in losses.items():
            log_data[f"{mode}/{loss_name}"] = val.detach().item()

        # If in validation mode, add metrics to the log dictionary
        if not train and metrics is not None:
            for metric_name, val in metrics.items():
                log_data[f"{mode}/{metric_name}"] = val

        num_images_to_log = min(batch_size, self.cfg.EXTRA.NUM_LOG_IMAGES)

        # Create the visualization grid
        # Renderer expects crop-image camera translation (tx, ty, tz).
        # Convert predicted crop weak-perspective (s, tx, ty) to (tx, ty, tz) in crop coords.
        focal_length = batch["cam_int"][:, 0, 0].to(pred_cam_t)
        cam_crop_wpersp = cam_full_to_crop(
            pred_cam_t,                # full-image (tx, ty, tz)
            batch["bbox_center"],
            batch["bbox_scale"][:, 0],
            batch["ori_img_size"],
            focal_length=focal_length,
        )
        cam_crop_t = cam_crop_s_to_t(
            cam_crop_wpersp,           # (s, tx, ty) in crop coords
            self.cfg.MODEL.IMAGE_SIZE,
            focal_length=focal_length,
        )

        predictions_grid = self._get_mesh_renderer().visualize_tensorboard(
            _to_float32_numpy(pred_vertices[:num_images_to_log]),
            _to_float32_numpy(cam_crop_t[:num_images_to_log]),
            _to_float32_numpy(images[:num_images_to_log]),
            focal_length[0].item(),  # TODO: Assign different focal length for each image
            _to_float32_numpy(pred_keypoints_2d[:num_images_to_log]),
            _to_float32_numpy(gt_keypoints_2d[:num_images_to_log]),
        )
        predictions_grid = [img.float().cpu() for img in predictions_grid]
        predictions_grid = make_grid(predictions_grid, nrow=5, padding=2)
        # Scale to [0, 255] and convert to uint8 to fix wandb warning
        predictions_grid_for_wandb = (predictions_grid * 255).to(torch.uint8)
        # Add the image grid to the log dictionary
        log_data[f'{mode}/predictions'] = wandb.Image(predictions_grid_for_wandb)
        # Log the entire dictionary to W&B
        self._log_wandb_payload(log_data, step_count, mode)

    def forward(self, batch: Dict) -> Dict:
        """
        Run a forward step of the network in val mode
        Args:
            batch (Dict): Dictionary containing batch data
        Returns:
            Dict: Dictionary containing the regression output
        """
        return self.forward_step(batch, train=False)

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
            # self.tensorboard_logging(batch, output, self.global_step, train=True)
            self.log_visualizations_to_wandb(batch, output, self.global_step, train=True)

        self.log('train/loss', output['losses']['loss'], on_step=True, on_epoch=True, prog_bar=True, logger=False,
                 batch_size=batch_size, sync_dist=True)

        return loss

    def validation_step(self, batch: Dict, batch_idx: int, dataloader_idx=0) -> Dict:
        """
        Run a validation step and log to Tensorboard
        Args:
            batch (Dict): Dictionary containing batch data
            batch_idx (int): Unused.
        Returns:
            Dict: Dictionary containing regression output.
        """
        batch_size = batch['img'].shape[0]
        output = self.forward_step(batch, train=False)
        # Compute losses for logging/visualization; keep it inference-only to avoid grad tracking
        with torch.no_grad():
            self.compute_loss(batch, output, train=False)
        metric = self.compute_metric(batch, output)
        output['metric'] = metric

        # Log visualizations for the first batch of each validation epoch
        if batch_idx == 0:
            self.log_visualizations_to_wandb(batch, output, self.global_step, train=False)
        self.log('val/loss', output['losses']['loss'], on_step=True, on_epoch=True, prog_bar=True, logger=False,
            batch_size=batch_size, sync_dist=True)
        return output

    def configure_gradient_clipping(
        self, optimizer, gradient_clip_val, gradient_clip_algorithm, optimizer_idx=None
    ):
        """
        Apply gradient clipping based on config when using automatic optimization.
        Accepts both Lightning signatures (with/without optimizer_idx).
        """
        clip_val = self.cfg.TRAIN.get('GRAD_CLIP_VAL', 0)
        if clip_val and clip_val > 0:
            gn = torch.nn.utils.clip_grad_norm_(self.get_parameters(), clip_val, error_if_nonfinite=True)
            # Log the gradient norm for monitoring
            self.log('train/grad_norm', gn, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
