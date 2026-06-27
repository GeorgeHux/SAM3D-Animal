import torch
from scipy.optimize import linear_sum_assignment
from typing import Dict, List, Optional
from yacs.config import CfgNode


class Matcher(torch.nn.Module):
    def __init__(self, cfg: CfgNode):
        super().__init__()
        self.cfg = cfg
        self.cost_conf = cfg.MODEL.MATCHER.COST_CONF
        self.cost_bbox = cfg.MODEL.MATCHER.COST_BBOX
        self.cost_giou = cfg.MODEL.MATCHER.COST_GIOU
        self.cost_kpts = cfg.MODEL.MATCHER.COST_KPTS

    @staticmethod
    def _box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
        cx, cy, w, h = boxes.unbind(-1)
        x1 = cx - 0.5 * w
        y1 = cy - 0.5 * h
        x2 = cx + 0.5 * w
        y2 = cy + 0.5 * h
        return torch.stack([x1, y1, x2, y2], dim=-1)

    @staticmethod
    def _generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        # boxes are expected in xyxy format
        area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
        area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[..., 0] * wh[..., 1]
        union = area1[:, None] + area2[None, :] - inter
        iou = inter / union.clamp(min=1e-6)

        lt_c = torch.min(boxes1[:, None, :2], boxes2[:, :2])
        rb_c = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
        wh_c = (rb_c - lt_c).clamp(min=0)
        area_c = wh_c[..., 0] * wh_c[..., 1]
        giou = iou - (area_c - union) / area_c.clamp(min=1e-6)
        return giou

    def _normalize_gt_boxes(self, gt_boxes: List[torch.Tensor], img_size: torch.Tensor) -> List[torch.Tensor]:
        img_h = img_size[:, 0]
        img_w = img_size[:, 1]
        cx = [(gt_box[..., 0] + 0.5 * gt_box[..., 2]) / img_w[b] for b, gt_box in enumerate(gt_boxes)]
        cy = [(gt_box[..., 1] + 0.5 * gt_box[..., 3]) / img_h[b] for b, gt_box in enumerate(gt_boxes)]
        w = [gt_box[..., 2] / img_w[b] for b, gt_box in enumerate(gt_boxes)]
        h = [gt_box[..., 3] / img_h[b] for b, gt_box in enumerate(gt_boxes)]
        boxes = [torch.stack([cx[b], cy[b], w[b], h[b]], dim=-1).clamp(0, 1) for b in range(len(gt_boxes))]
        return boxes

    def _hungarian_match_boxes(
        self,
        pred_boxes: torch.Tensor,
        gt_boxes: List[torch.Tensor],
        pred_confs: Optional[torch.Tensor] = None,
        pred_keypoints_2d: Optional[torch.Tensor] = None,
        gt_keypoints_2d: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        indices = []
        batch_size = pred_boxes.shape[0]
        for b in range(batch_size):
            # L1 distance between predicted and target boxes.
            cost_bbox_matrix = torch.cdist(pred_boxes[b], gt_boxes[b], p=1)
            giou = self._generalized_box_iou(
                self._box_cxcywh_to_xyxy(pred_boxes[b]),
                self._box_cxcywh_to_xyxy(gt_boxes[b]),
            )
            cost_giou_matrix = -giou
            # Confidence cost is per-prediction, expanded to match GT count.
            if pred_confs is not None:
                conf = pred_confs[b].squeeze(-1).clamp(min=1e-8, max=1 - 1e-8)
                alpha = 0.25
                gamma = 2.0
                cost_conf_vec = alpha * ((1 - conf) ** gamma) * (-(conf).log())
                cost_conf_matrix = cost_conf_vec[:, None].expand_as(cost_bbox_matrix)
            else:
                cost_conf_matrix = 0.0
            # Keypoint cost is the mean L1 distance over visible joints.
            if pred_keypoints_2d is not None and gt_keypoints_2d is not None:
                pred_kpts = pred_keypoints_2d[b]
                gt_kpts = gt_keypoints_2d[b]
                gt_xy = gt_kpts[..., :2]
                gt_vis = (gt_kpts[..., 2] > 0.5).to(pred_kpts.dtype)
                diff = (pred_kpts[:, None, :, :] - gt_xy[None, :, :, :]).abs()
                dist = diff.sum(-1)
                vis_cnt = gt_vis.sum(-1).clamp(min=1.0)
                cost_kpts_matrix = (dist * gt_vis[None, :, :]).sum(-1) / vis_cnt[None, :]
            else:
                cost_kpts_matrix = 0.0
            cost_matrix = (
                self.cost_conf * cost_conf_matrix
                + self.cost_bbox * cost_bbox_matrix
                + self.cost_giou * cost_giou_matrix
                + self.cost_kpts * cost_kpts_matrix
            )
            row_ind, col_ind = linear_sum_assignment(cost_matrix.detach().cpu())
            indices.append(
                (
                    torch.as_tensor(row_ind, dtype=torch.long, device=pred_boxes.device),
                    torch.as_tensor(col_ind, dtype=torch.long, device=pred_boxes.device),
                )
            )
        return indices

    def _reorder_pose_output(self, pose_output: Dict, orders: List[torch.Tensor]) -> Dict:
        if not orders:
            return pose_output
        batch_size = pose_output["pred_boxes"].shape[0]
        num_animals = pose_output["pred_boxes"].shape[1]
        for key, value in list(pose_output.items()):
            if not torch.is_tensor(value):
                continue
            if value.ndim < 2 or value.shape[0] != batch_size or value.shape[1] != num_animals:
                continue
            pose_output[key] = torch.cat(
                [value[b].index_select(0, orders[b]) for b in range(batch_size)], dim=0
            )
        return pose_output

    def forward(self, batch: Dict, pose_output: Dict) -> Dict:
        pred_boxes = pose_output.get("pred_boxes", None)
        pred_confs = pose_output.get("pred_confs", None)
        pred_keypoints_2d = pose_output.get("pred_keypoints_2d_cropped", None)
        gt_boxes = [b["bbox"] for b in batch["targets"]]
        gt_keypoints_2d = [b["keypoints_2d"] for b in batch["targets"]]
        if pred_boxes is None or gt_boxes is None:
            return pose_output
        img_size = batch.get("img_size", None)
        gt_boxes_norm = self._normalize_gt_boxes(gt_boxes, img_size)

        indices = self._hungarian_match_boxes(
            pred_boxes,
            gt_boxes_norm,
            pred_confs=pred_confs,
            pred_keypoints_2d=pred_keypoints_2d,
            gt_keypoints_2d=gt_keypoints_2d,
        )
        # Save full (pre-matching) data so the confidence loss can
        # supervise both matched (IoU target) and unmatched (target=0) queries.
        pose_output["_match_info"] = {
            "full_pred_confs": pred_confs,
            "full_pred_boxes": pred_boxes,
            "match_indices": indices,
            "gt_boxes_norm": gt_boxes_norm,
        }

        orders: List[torch.Tensor] = []
        for b in range(pred_boxes.shape[0]):
            pred_idx, gt_idx = indices[b]
            order = pred_idx[torch.argsort(gt_idx)]
            orders.append(order)
        return self._reorder_pose_output(pose_output, orders)
