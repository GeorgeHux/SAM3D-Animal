import torch
import torch.nn as nn
import einops
import roma
from typing import Tuple, Optional
from timm.layers import to_2tuple
from ...utils.geometry import rot6d_to_rotmat, aa_to_rotmat
from ..components.pose_transformer import TransformerDecoder
from ..components.transformer import FFN


def build_smal_head(cfg, head_type: str):
    if head_type == 'transformer_decoder':
        return SMALTransformerDecoderHead(cfg)
    if head_type == "sam3d":
        return SAM3DHead(
            input_dim=cfg.MODEL.DECODER.DIM,
            mlp_depth=cfg.MODEL.SMAL_HEAD.get("MLP_DEPTH", 1),
            mlp_channel_div_factor=cfg.MODEL.SMAL_HEAD.get("MLP_CHANNEL_DIV_FACTOR", 1),
        )
    elif head_type == "perspective":
        return PerspectiveHead(
            input_dim=cfg.MODEL.DECODER.DIM,
            img_size=to_2tuple(cfg.MODEL.IMAGE_SIZE),
            mlp_depth=cfg.MODEL.get("CAMERA_HEAD", dict()).get("MLP_DEPTH", 1),
            mlp_channel_div_factor=cfg.MODEL.get("CAMERA_HEAD", dict()).get(
                "MLP_CHANNEL_DIV_FACTOR", 1
            ),
            default_scale_factor=cfg.MODEL.get("CAMERA_HEAD", dict()).get("DEFAULT_SCALE_FACTOR", 1),
        )
    elif head_type == "bbox":
        return BBoxHead(
            input_dim=cfg.MODEL.DECODER.DIM,
            mlp_depth=cfg.MODEL.get("BBOX_HEAD", dict()).get("MLP_DEPTH", 2),
            mlp_channel_div_factor=cfg.MODEL.get("BBOX_HEAD", dict()).get(
                "MLP_CHANNEL_DIV_FACTOR", 1
            ),
            drop_ratio=cfg.MODEL.get("BBOX_HEAD", dict()).get("DROP_RATIO", 0.0),
        )
    else:
        raise ValueError('Unknown SMAL head type: {}'.format(head_type))


class SMALTransformerDecoderHead(nn.Module):
    """ Cross-attention based SMAL Transformer decoder
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.joint_rep_type = cfg.MODEL.SMAL_HEAD.get('JOINT_REP', '6d')
        self.joint_rep_dim = {'6d': 6, 'aa': 3}[self.joint_rep_type]
        npose = self.joint_rep_dim * (cfg.SMAL.NUM_JOINTS + 1)
        self.npose = npose
        self.input_is_mean_shape = cfg.MODEL.SMAL_HEAD.get('TRANSFORMER_INPUT', 'zero') == 'mean_shape'
        transformer_args = dict(
            num_tokens=1,
            token_dim=(npose + 10 + 3) if self.input_is_mean_shape else 1,
            dim=1024,
        )
        transformer_args = {**transformer_args, **dict(cfg.MODEL.SMAL_HEAD.TRANSFORMER_DECODER)}
        
        self.transformer = TransformerDecoder(
            **transformer_args
        )
        dim = transformer_args['dim']
        self.decpose = nn.Linear(dim, npose)
        self.decshape = nn.Linear(dim, 41)
        self.deccam = nn.Linear(dim, 3)

        if cfg.MODEL.SMAL_HEAD.get('INIT_DECODER_XAVIER', False):
            # True by default in MLP. False by default in Transformer
            nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
            nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
            nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

        init_pose = torch.zeros(size=(1, npose), dtype=torch.float32)
        init_betas = torch.zeros(size=(1, 41), dtype=torch.float32)
        init_cam = torch.tensor([[0.9, 0, 0]], dtype=torch.float32)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_betas', init_betas)
        self.register_buffer('init_cam', init_cam)

    def forward(self, x, **kwargs):
        batch_size = x.shape[0]
        # vit pretrained backbone is channel-first. Change to token-first
        x = einops.rearrange(x, 'b c h w -> b (h w) c')

        init_pose = self.init_pose.expand(batch_size, -1)
        init_betas = self.init_betas.expand(batch_size, -1)
        init_cam = self.init_cam.expand(batch_size, -1)

        pred_pose = init_pose
        pred_betas = init_betas
        pred_cam = init_cam
        pred_pose_list = []
        pred_betas_list = []
        pred_cam_list = []
        for i in range(self.cfg.MODEL.SMAL_HEAD.get('IEF_ITERS', 3)):
            # Input token to transformer is zero token
            if self.input_is_mean_shape:
                token = torch.cat([pred_pose, pred_betas, pred_cam], dim=1)[:, None, :]
            else:
                token = torch.zeros(batch_size, 1, 1).to(x.device)

            # Pass through transformer
            token_out = self.transformer(token, context=x)
            token_out = token_out.squeeze(1)  # (B, C)

            # Readout from token_out
            pred_pose = self.decpose(token_out) + pred_pose
            pred_betas = self.decshape(token_out) + pred_betas
            pred_cam = self.deccam(token_out) + pred_cam
            pred_pose_list.append(pred_pose)
            pred_betas_list.append(pred_betas)
            pred_cam_list.append(pred_cam)

        # Convert self.joint_rep_type -> rotmat
        joint_conversion_fn = {
            '6d': rot6d_to_rotmat,
            'aa': lambda x: aa_to_rotmat(x.view(-1, 3).contiguous())
        }[self.joint_rep_type]

        pred_smal_params_list = {}
        pred_smal_params_list['pose'] = torch.cat(
            [joint_conversion_fn(pbp).view(batch_size, -1, 3, 3)[:, 1:, :, :] for pbp in pred_pose_list], dim=0)
        pred_smal_params_list['betas'] = torch.cat(pred_betas_list, dim=0)
        pred_smal_params_list['cam'] = torch.cat(pred_cam_list, dim=0)
        pred_pose = joint_conversion_fn(pred_pose).view(batch_size, self.cfg.SMAL.NUM_JOINTS + 1, 3, 3)

        pred_smal_params = {'global_orient': pred_pose[:, [0]],
                            'pose': pred_pose[:, 1:],
                            'betas': pred_betas,
                            }
        return pred_smal_params, pred_cam, pred_smal_params_list


def perspective_projection(x, K):
    """
    Computes the perspective projection of a set of points assuming the extrinsinc params have already been applied
    Args:
        - x [bs,N,3]: 3D points
        - K [bs,3,3]: Camera instrincs params
    """
    # Apply perspective distortion
    y = x / x[:, :, -1].unsqueeze(-1)  # (bs, N, 3)

    # Apply camera intrinsics
    y = torch.einsum("bij,bkj->bki", K, y)  # (bs, N, 3)

    return y[:, :, :2]


class PerspectiveHead(nn.Module):
    """
    Predict camera translation (s, tx, ty) and perform full-perspective
    2D reprojection (CLIFF/CameraHMR setup).
    """

    def __init__(
        self,
        input_dim: int,
        img_size: Tuple[int, int],  # model input size (W, H)
        mlp_depth: int = 1,
        drop_ratio: float = 0.0,
        mlp_channel_div_factor: int = 8,
        default_scale_factor: float = 1,
    ):
        super().__init__()

        # Metadata to compute 3D skeleton and 2D reprojection
        self.img_size = to_2tuple(img_size)
        self.ncam = 3  # (s, tx, ty)
        self.default_scale_factor = default_scale_factor

        self.proj = FFN(
            embed_dims=input_dim,
            feedforward_channels=input_dim // mlp_channel_div_factor,
            output_dims=self.ncam,
            num_fcs=mlp_depth,
            ffn_drop=drop_ratio,
            add_identity=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        init_estimate: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            x: pose token with shape [B, C], usually C=DECODER.DIM
            init_estimate: [B, self.ncam]
        """
        pred_cam = self.proj(x)
        if init_estimate is not None:
            pred_cam = pred_cam + init_estimate

        return pred_cam

    def perspective_projection(
        self,
        points_3d: torch.Tensor,
        pred_cam: torch.Tensor,
        bbox_center: torch.Tensor,
        bbox_size: torch.Tensor,
        img_size: torch.Tensor,
        cam_int: torch.Tensor,
        use_intrin_center: bool = False,
    ):
        """
        Args:
            bbox_center / img_size: shape [N, 2], in original image space (w, h)
            bbox_size: shape [N,], in original image space
            cam_int: shape [N, 3, 3]
        """
        batch_size = points_3d.shape[0]
        pred_cam = pred_cam.clone()
        # pred_cam[..., [0, 2]] *= -1  # Camera system difference  # TODO: Check Here

        # Compute camera translation: (scale, x, y) --> (x, y, depth)
        # depth ~= f / s
        # Note that f is in the NDC space (see Zolly section 3.1)
        s, tx, ty = pred_cam[:, 0], pred_cam[:, 1], pred_cam[:, 2]
        bs = bbox_size * s * self.default_scale_factor + 1e-8
        focal_length = cam_int[:, 0, 0]
        tz = 2 * focal_length / bs

        if not use_intrin_center:
            cx = 2 * (bbox_center[:, 0] - (img_size[:, 0] / 2)) / bs
            cy = 2 * (bbox_center[:, 1] - (img_size[:, 1] / 2)) / bs
        else:
            cx = 2 * (bbox_center[:, 0] - (cam_int[:, 0, 2])) / bs
            cy = 2 * (bbox_center[:, 1] - (cam_int[:, 1, 2])) / bs

        pred_cam_t = torch.stack([tx + cx, ty + cy, tz], dim=-1)

        # Compute camera translation
        j3d_cam = points_3d + pred_cam_t.unsqueeze(1)

        # Projection to the image plane.
        # Note that the projection output is in *original* image space now.
        j2d = perspective_projection(j3d_cam, cam_int)

        return {
            "pred_keypoints_2d": j2d.reshape(batch_size, -1, 2),
            "pred_cam_t": pred_cam_t,
            "focal_length": focal_length,
            "pred_keypoints_2d_depth": j3d_cam.reshape(batch_size, -1, 3)[:, :, 2],
        }


class BBoxHead(nn.Module):
    """
    Predict bounding boxes and their confidence from decoder tokens.
    """

    def __init__(
        self,
        input_dim: int,
        mlp_depth: int = 2,
        drop_ratio: float = 0.0,
        mlp_channel_div_factor: int = 1,
    ):
        super().__init__()
        self.nbox = 4
        self.proj = FFN(
            embed_dims=input_dim,
            feedforward_channels=max(1, input_dim // mlp_channel_div_factor),
            output_dims=self.nbox,
            num_fcs=mlp_depth,
            ffn_drop=drop_ratio,
            add_identity=False,
        )
        # Separate confidence head so matching can use a per-query score.
        self.conf_head = nn.Linear(input_dim, 1)
        # Initialize with low prior to stabilize early training.
        prior_prob = 0.01
        bias_value = -torch.log(torch.tensor((1 - prior_prob) / prior_prob))
        self.conf_head.bias.data.fill_(bias_value.item())

    def forward(
        self,
        x: torch.Tensor,
        init_estimate: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            x: token features with shape [..., C]
            init_estimate: optional reference points with shape [..., 4]
        """
        pred_box = self.proj(x)
        if init_estimate is not None:
            pred_box = pred_box + init_estimate
        # Use sigmoid so both box and confidence are in [0, 1].
        pred_box = pred_box.sigmoid()
        pred_conf = self.conf_head(x).sigmoid()
        return pred_box, pred_conf


class SAM3DHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        mlp_depth: int = 1,
        ffn_zero_bias: bool = True,
        mlp_channel_div_factor: int = 8,
    ):
        super().__init__()
        self.num_shape_comps = 145
        self.num_pose = 34 * 6  # 35 joints * 6D dimensions
        self.npose = 6 + self.num_pose + self.num_shape_comps

        self.proj = FFN(
            embed_dims=input_dim,
            feedforward_channels=input_dim // mlp_channel_div_factor,
            output_dims=self.npose,
            num_fcs=mlp_depth,
            ffn_drop=0.0,
            add_identity=False,
        )

        if ffn_zero_bias:
            torch.nn.init.zeros_(self.proj.layers[-2].bias)

    def forward(
        self,
        x: torch.Tensor,
        init_estimate: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            x: pose token with shape [B, C], usually C=DECODER.DIM
            init_estimate: [B, self.npose]
        """
        pred = self.proj(x)
        if init_estimate is not None:
            pred = pred + init_estimate

        # From pred, we want to pull out individual predictions.
        ## First, get globals
        ### Global rotation is first 6.
        count = 6
        global_rot_6d = pred[:, :count]
        global_rot_rotmat = rot6d_to_rotmat(global_rot_6d).view(pred.shape[0], -1, 3, 3)  # B x 1 x 3 x 3

        ## Next, get body pose.
        ### Hold onto raw, continuous version for iterative correction.
        pred_pose_cont_6d = pred[:, count: count + self.num_pose]
        pred_pose_cont_rotmat = rot6d_to_rotmat(pred_pose_cont_6d.view(pred.shape[0], -1, 6)).view(pred.shape[0], -1, 3, 3)
        count += self.num_pose

        ## Get remaining parameters
        pred_shape = pred[:, count: count + self.num_shape_comps]

        pred_smal_params = {'global_orient': global_rot_rotmat,
                            'pose': pred_pose_cont_rotmat,
                            'betas': pred_shape,
                            }
        return pred_smal_params, torch.cat([global_rot_6d, pred_pose_cont_6d], dim=1)
        # Prep outputs
        # output = {
        #     "pred_pose_raw": torch.cat(
        #         [global_rot_6d, pred_pose_cont], dim=1
        #     ),  # Both global rot and continuous pose
        #     "pred_pose_rotmat": None,  # This normally used for mhr pose param rotmat supervision.
        #     "global_rot": global_rot_euler,
        #     "body_pose": pred_pose_euler,  # Unused during training
        #     "shape": pred_shape,
        #     "pred_keypoints_3d": j3d.reshape(batch_size, -1, 3),
        #     "pred_vertices": (
        #         verts.reshape(batch_size, -1, 3) if verts is not None else None
        #     ),
        #     "pred_joint_coords": (
        #         jcoords.reshape(batch_size, -1, 3) if jcoords is not None else None
        #     ),
        #     "joint_global_rots": joint_global_rots,
        #     "mhr_model_params": mhr_model_params,
        # }

        # return output
