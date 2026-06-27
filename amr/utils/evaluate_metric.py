import torch
import numpy as np
import open3d as o3d
from typing import Dict, List, Union
from pytorch3d.transforms import axis_angle_to_matrix


def _calc_distances(preds, targets, mask, normalize):
    """Calculate the normalized distances between preds and target.

    Note:
        batch_size: N
        num_keypoints: K
        dimension of keypoints: D (normally, D=2 or D=3)

    Args:
        preds (np.ndarray[N, K, D]): Predicted keypoint location.
        targets (np.ndarray[N, K, D]): Groundtruth keypoint location.
        mask (np.ndarray[N, K]): Visibility of the target. False for invisible
            joints, and True for visible. Invisible joints will be ignored for
            accuracy calculation.
        normalize (np.ndarray[N, D]): Typical value is heatmap_size

    Returns:
        np.ndarray[K, N]: The normalized distances. \
            If target keypoints are missing, the distance is -1.
    """
    N, K, _ = preds.shape
    # set mask=0 when normalize==0
    _mask = mask.copy()
    _mask[np.where((normalize == 0).sum(1))[0], :] = False
    distances = np.full((N, K), -1, dtype=np.float32)
    # handle invalid values
    normalize[np.where(normalize <= 0)] = 1e6
    distances[_mask] = np.linalg.norm(
        ((preds - targets) / normalize[:, None, :])[_mask], axis=-1)
    return distances.T


def _distance_acc(distances, thr=0.5):
    """Return the percentage below the distance threshold, while ignoring
    distances values with -1.

    Note:
        batch_size: N
    Args:
        distances (np.ndarray[N, ]): The normalized distances.
        thr (float): Threshold of the distances.

    Returns:
        float: Percentage of distances below the threshold. \
            If all target keypoints are missing, return -1.
    """
    distance_valid = distances != -1
    num_distance_valid = distance_valid.sum()
    if num_distance_valid > 0:
        return (distances[distance_valid] < thr).sum() / num_distance_valid
    return -1


def keypoint_pck_accuracy(pred, gt, mask, thr, normalize):
    """Calculate the pose accuracy of PCK for each individual keypoint and the
    averaged accuracy across all keypoints for coordinates.

    Note:
        PCK metric measures accuracy of the localization of the body joints.
        The distances between predicted positions and the ground-truth ones
        are typically normalized by the bounding box size.
        The threshold (thr) of the normalized distance is commonly set
        as 0.05, 0.1 or 0.2 etc.

        - batch_size: N
        - num_keypoints: K

    Args:
        pred (np.ndarray[N, K, 2]): Predicted keypoint location.
        gt (np.ndarray[N, K, 2]): Groundtruth keypoint location.
        mask (np.ndarray[N, K]): Visibility of the target. False for invisible
            joints, and True for visible. Invisible joints will be ignored for
            accuracy calculation.
        thr (float): Threshold of PCK calculation.
        normalize (np.ndarray[N, 2]): Normalization factor for H&W.

    Returns:
        tuple: A tuple containing keypoint accuracy.

        - acc (np.ndarray[K]): Accuracy of each keypoint.
        - avg_acc (float): Averaged accuracy across all keypoints.
        - cnt (int): Number of valid keypoints.
    """
    distances = _calc_distances(pred, gt, mask, normalize)

    acc = np.array([_distance_acc(d, thr) for d in distances])
    valid_acc = acc[acc >= 0]
    cnt = len(valid_acc)
    avg_acc = valid_acc.mean() if cnt > 0 else 0
    return avg_acc


def compute_scale_transform(S1: torch.Tensor, S2: torch.Tensor) -> torch.Tensor:
    """
    Computes a scale transform (s) in a batched way that takes
    a set of 3D points S1 (B, N, 3) closest to a set of 3D points S2 (B, N, 3).
    Args:
        S1 (torch.Tensor): First set of points of shape (B, N, 3).
        S2 (torch.Tensor): Second set of points of shape (B, N, 3).
    Returns:
        (torch.Tensor): The first set of points after applying the scale transformation.
    """

    # 1. Remove mean.
    mu1 = S1.mean(dim=1, keepdim=True)
    mu2 = S2.mean(dim=1, keepdim=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = (X1 ** 2).sum(dim=(1, 2), keepdim=True)

    # 3. Compute scale.
    scale = (X2 * X1).sum(dim=(1, 2), keepdim=True) / var1

    # 4. Apply scale transform.
    S1_hat = scale * X1 + mu2

    return S1_hat


def compute_similarity_transform(S1: torch.Tensor, S2: torch.Tensor) -> torch.Tensor:
    """
    Computes a similarity transform (sR, t) in a batched way that takes
    a set of 3D points S1 (B, N, 3) closest to a set of 3D points S2 (B, N, 3),
    where R is a 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    Args:
        S1 (torch.Tensor): First set of points of shape (B, N, 3).
        S2 (torch.Tensor): Second set of points of shape (B, N, 3).
    Returns:
        (torch.Tensor): The first set of points after applying the similarity transformation.
    """

    batch_size = S1.shape[0]
    S1 = S1.permute(0, 2, 1)
    S2 = S2.permute(0, 2, 1)
    # 1. Remove mean.
    mu1 = S1.mean(dim=2, keepdim=True)
    mu2 = S2.mean(dim=2, keepdim=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = (X1 ** 2).sum(dim=(1, 2))

    # 3. The outer product of X1 and X2.
    K = torch.matmul(X1.float(), X2.permute(0, 2, 1))

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are singular vectors of K.
    U, s, V = torch.svd(K.float())
    Vh = V.permute(0, 2, 1)

    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = torch.eye(U.shape[1], device=U.device).unsqueeze(0).repeat(batch_size, 1, 1).float()
    Z[:, -1, -1] *= torch.sign(torch.linalg.det(torch.matmul(U.float(), Vh.float()).float()))

    # Construct R.
    R = torch.matmul(torch.matmul(V, Z), U.permute(0, 2, 1))

    # 5. Recover scale.
    trace = torch.matmul(R, K).diagonal(offset=0, dim1=-1, dim2=-2).sum(dim=-1)
    scale = (trace / var1).unsqueeze(dim=-1).unsqueeze(dim=-1)

    # 6. Recover translation.
    t = mu2 - scale * torch.matmul(R.float(), mu1.float())

    # 7. Error:
    S1_hat = scale * torch.matmul(R.float(), S1.float()).float() + t

    return S1_hat.permute(0, 2, 1)


def pointcloud(points: np.ndarray):
    pcd = o3d.geometry.PointCloud()
    points = o3d.utility.Vector3dVector(points)
    pcd.points = points
    return pcd


class GlobalKeypointAPAccumulator:
    """Accumulate per-instance OKS values across the entire dataset and compute
    AP globally, instead of averaging per-batch AP values.

    Assumes predictions are already matched 1:1 with ground-truth instances
    (e.g. via Hungarian matching in the model's forward pass).
    """

    def __init__(self, image_size: int = 256, oks_sigmas: Union[np.ndarray, None] = None):
        self.image_size = image_size
        self.oks_sigmas = oks_sigmas
        self._all_oks: List[float] = []

    def add_batch(self, output: Dict, batch: Dict):
        """Compute per-instance OKS for one batch of matched pairs and accumulate.

        Args:
            output: model output dict with 'pred_keypoints_2d_cropped' [N, K, 2].
            batch: batch dict with 'keypoints_2d' [N, K, 3] (last dim includes visibility).
        """
        pred_kps = (output['pred_keypoints_2d_cropped'].detach().cpu().float().numpy() + 0.5) * self.image_size
        gt_kps_raw = batch['keypoints_2d'].detach().cpu().float().numpy()
        conf = gt_kps_raw[:, :, 2] > 0.5
        gt_kps = (gt_kps_raw[:, :, :2] + 0.5) * self.image_size

        num_kps = gt_kps.shape[1]
        if self.oks_sigmas is None:
            self.oks_sigmas = np.full(num_kps, 0.05, dtype=np.float32)
        vars_ = (self.oks_sigmas * 2.0) ** 2

        # Per-instance area from visible keypoint bounding box
        x = np.where(conf, gt_kps[:, :, 0], np.inf)
        y = np.where(conf, gt_kps[:, :, 1], np.inf)
        x_max = np.where(conf, gt_kps[:, :, 0], -np.inf).max(axis=1)
        y_max = np.where(conf, gt_kps[:, :, 1], -np.inf).max(axis=1)
        area = (x_max - x.min(axis=1)) * (y_max - y.min(axis=1))
        area = np.clip(area, 1.0, None)

        # OKS per instance
        dist_sq = ((pred_kps - gt_kps) ** 2).sum(axis=-1)
        oks_per_kp = np.exp(-dist_sq / (2.0 * area[:, None] * vars_[None, :] + 1e-6))
        n_vis = np.clip(conf.sum(axis=1), 1, None)
        oks = (oks_per_kp * conf).sum(axis=1) / n_vis

        # Only keep instances with at least one visible keypoint
        valid = conf.sum(axis=1) > 0
        self._all_oks.extend(oks[valid].tolist())

    def summarize(self, oks_thresholds: Union[np.ndarray, None] = None) -> Dict:
        """Compute AP over all accumulated OKS values.

        Returns:
            dict with keys 'mAP', 'AP50', 'AP75'.
        """
        if oks_thresholds is None:
            oks_thresholds = np.arange(0.50, 1.00, 0.05, dtype=np.float32)

        if len(self._all_oks) == 0:
            return {'mAP': 0.0, 'AP50': 0.0, 'AP75': 0.0}

        oks = np.array(self._all_oks)
        ap = (oks[:, None] >= oks_thresholds[None, :]).mean(axis=0)
        return {
            'mAP': float(ap.mean()),
            'AP50': float(ap[np.argmin(np.abs(oks_thresholds - 0.50))]),
            'AP75': float(ap[np.argmin(np.abs(oks_thresholds - 0.75))]),
        }

    def reset(self):
        """Clear accumulated state for reuse across datasets."""
        self._all_oks.clear()


class Evaluator:
    def __init__(self, smal_model, image_size: int=256, pelvis_ind: int = 7):
        self.pelvis_ind = pelvis_ind
        self.smal_model = smal_model
        self.image_size = image_size
    
    # def compute_pck(self, output: Dict, batch: Dict, pck_threshold: Union[List, None]):
    #     pred_keypoints_2d = output['pred_keypoints_2d_cropped'].detach().cpu()
    #     gt_keypoints_2d = batch['keypoints_2d'].detach().cpu()
    #     self.pck_threshold_list = []
        
    #     pred_keypoints_2d = (pred_keypoints_2d + 0.5) * self.image_size  # * batch['bbox_expand_factor'].detach().cpu().numpy().reshape(-1, 1, 1)
    #     conf = gt_keypoints_2d[:, :, -1]
    #     gt_keypoints_2d = (gt_keypoints_2d[:, :, :-1] + 0.5) * self.image_size  # * batch['bbox_expand_factor'].detach().cpu().numpy().reshape(-1, 1, 1)
    #     if pck_threshold is not None:
    #         for i in range(len(pck_threshold)):
    #             self.pck_threshold_list.append(torch.tensor([pck_threshold[i]] * len(pred_keypoints_2d), dtype=torch.float32))

    #     pcks = []
    #     seg_area = torch.sum(batch['mask'].detach().cpu().reshape(batch['mask'].shape[0], -1), dim=-1).unsqueeze(-1)
    #     total_visible = torch.sum(conf, dim=-1)
    #     for th in self.pck_threshold_list:
    #         dist = torch.norm(pred_keypoints_2d - gt_keypoints_2d, dim=-1)

    #         hits = (dist / torch.sqrt(seg_area)) < th.unsqueeze(1)
    #         pck = torch.sum(hits.float() * conf, dim=-1) / total_visible
    #         pcks.append(pck.numpy().tolist())
    #     return torch.mean(torch.tensor(pcks), dim=1)

    def compute_pck(self, output: Dict, batch: Dict, pck_threshold: Union[List, None]):
        pred_keypoints_2d = output['pred_keypoints_2d_cropped'].detach().cpu().float().numpy()
        gt_keypoints_2d = batch['keypoints_2d'].detach().cpu().float().numpy() 
        conf = gt_keypoints_2d[:, :, -1][:, None, :]
        gt_keypoints_2d = gt_keypoints_2d[:, :, :-1]
        self.pck_threshold_list = []
        if pck_threshold is not None:
            for i in range(len(pck_threshold)):
                self.pck_threshold_list.append(np.array([pck_threshold[i]] * len(pred_keypoints_2d), dtype=np.float32))

        batch_size = pred_keypoints_2d.shape[0]
        pred_keypoints_2d = pred_keypoints_2d[:, None, :, :]
        gt_keypoints_2d = gt_keypoints_2d[:, None, :, :]

        pcks = []
        for pck_threshold in self.pck_threshold_list:
            pcks.append([
                keypoint_pck_accuracy(
                    pred_keypoints_2d[i, 0, :, :][None],
                    gt_keypoints_2d[i, 0, :, :][None],
                    conf[i, 0, :][None] > 0.5,
                    thr=pck_threshold[i],
                    normalize=np.ones((1, 2))  # Already in [-0.5,0.5] range. No need to normalize
                )
                for i in range(batch_size)]
            )
        return np.mean(np.array(pcks), axis=1)

    def compute_ap(self, output: Dict, batch: Dict, oks_threshold: Union[List[float], None] = None, oks_sigmas: Union[List[float], None] = None):
        pred_keypoints_2d = (output['pred_keypoints_2d_cropped'].detach().cpu().float().numpy() + 0.5) * self.image_size
        gt_keypoints_2d = batch['keypoints_2d'].detach().cpu().float().numpy()
        conf = gt_keypoints_2d[:, :, -1] > 0.5
        gt_keypoints_2d = (gt_keypoints_2d[:, :, :-1] + 0.5) * self.image_size

        if oks_threshold is None:
            oks_threshold = np.arange(0.50, 1.00, 0.05, dtype=np.float32)
        else:
            oks_threshold = np.asarray(oks_threshold, dtype=np.float32)
        if oks_sigmas is None:
            oks_sigmas = np.full(gt_keypoints_2d.shape[1], 0.05, dtype=np.float32)
        else:
            oks_sigmas = np.asarray(oks_sigmas, dtype=np.float32).reshape(-1)
            if len(oks_sigmas) != gt_keypoints_2d.shape[1]:
                oks_sigmas = np.full(gt_keypoints_2d.shape[1], float(oks_sigmas[0]), dtype=np.float32)

        x = np.where(conf, gt_keypoints_2d[:, :, 0], np.inf)
        y = np.where(conf, gt_keypoints_2d[:, :, 1], np.inf)
        area = (np.where(conf, gt_keypoints_2d[:, :, 0], -np.inf).max(axis=1) - x.min(axis=1)) * \
               (np.where(conf, gt_keypoints_2d[:, :, 1], -np.inf).max(axis=1) - y.min(axis=1))
        # if 'mask' in batch:
        #     mask_area = batch['mask'].detach().cpu().float().reshape(batch['mask'].shape[0], -1).sum(dim=-1).numpy()
        #     area = np.where(np.isfinite(area) & (area > 0), area, mask_area)
        area = np.clip(area, 1.0, None)

        vars = (oks_sigmas * 2.0) ** 2
        dist_sq = ((pred_keypoints_2d - gt_keypoints_2d) ** 2).sum(axis=-1)
        oks = np.exp(-dist_sq / (2.0 * area[:, None] * vars[None, :] + 1e-6))
        oks = (oks * conf).sum(axis=1) / np.clip(conf.sum(axis=1), 1, None)
        oks = oks[conf.sum(axis=1) > 0]
        if oks.size == 0:
            return {'mAP': 0.0, 'AP50': 0.0, 'AP75': 0.0}

        ap = (oks[:, None] >= oks_threshold[None, :]).mean(axis=0)
        return {
            'mAP': float(ap.mean()),
            'AP50': float(ap[np.argmin(np.abs(oks_threshold - 0.50))]),
            'AP75': float(ap[np.argmin(np.abs(oks_threshold - 0.75))]),
        }

    def compute_pa_mpjpe(self, pred_joints, gt_joints):
        S1_hat = compute_similarity_transform(pred_joints, gt_joints)
        pa_mpjpe = torch.sqrt(((S1_hat - gt_joints) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy() * 1000
        return pa_mpjpe.mean()

    def compute_pa_mpvpe(self, gt_vertices: torch.Tensor, pred_vertices: torch.Tensor):
        batch_size = pred_vertices.shape[0]
        S1_hat = compute_similarity_transform(pred_vertices, gt_vertices)
        pa_mpvpe = torch.sqrt(((S1_hat - gt_vertices) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy() * 1000
        return pa_mpvpe.mean()

    def eval_3d(self, output: Dict, batch: Dict):
        """
        Evaluate current batch
        Args:
            output: model output
            batch: model input
        Returns: evaluate metric
        """
        if batch['has_smal_params']["betas"].sum() == 0:
            pa_mpvpe = 0.
        else:
            gt_vertices = self.smal_forward(batch)
            pa_mpvpe = self.compute_pa_mpvpe(gt_vertices, output['pred_vertices'])

        pred_keypoints_3d = output["pred_keypoints_3d"].detach()
        pred_keypoints_3d = pred_keypoints_3d[:, None, :, :]
        batch_size = pred_keypoints_3d.shape[0]
        num_samples = pred_keypoints_3d.shape[1]
        gt_keypoints_3d = batch['keypoints_3d'][:, :, :-1].unsqueeze(1).repeat(1, num_samples, 1, 1)
        
        # Align predictions and ground truth such that the pelvis location is at the origin
        pred_keypoints_3d -= pred_keypoints_3d[:, :, [self.pelvis_ind]]
        gt_keypoints_3d -= gt_keypoints_3d[:, :, [self.pelvis_ind]]
        pa_mpjpe = self.compute_pa_mpjpe(pred_keypoints_3d.reshape(batch_size * num_samples, -1, 3),
                                         gt_keypoints_3d.reshape(batch_size * num_samples, -1, 3))
        return pa_mpjpe, pa_mpvpe
    
    def eval_2d(self, output: Dict, batch: Dict, pck_threshold: List[float]=[0.10, 0.15]):
        pck = self.compute_pck(output, batch, pck_threshold=pck_threshold)
        auc = self.compute_auc(batch, output)
        return pck.tolist(), auc
    
    def compute_auc(self, batch: Dict, output: Dict, threshold_min: int=0.0, threshold_max: int=1.0, steps: int=100):
        thresholds = np.linspace(threshold_min, threshold_max, steps)
        norm_factor = np.trapz(np.ones_like(thresholds), thresholds)
        pck_curve = []
        for th in thresholds:
            # compute_pck returns [adaptive-threshold PCK, fixed-threshold PCK@th].
            # For AUC over thresholds, only integrate the fixed-threshold value.
            pck_at_th = np.asarray(self.compute_pck(output, batch, [th]), dtype=np.float32).reshape(-1)[-1]
            pck_curve.append(float(pck_at_th))
        pck_curve = np.asarray(pck_curve, dtype=np.float32)
        auc = np.trapz(pck_curve, thresholds)
        auc /= norm_factor
        return float(auc)

    def smal_forward(self, batch: Dict):
        batch_size = batch['smal_params']['global_orient'].shape[0]
        smal_params = batch['smal_params']
        smal_params['global_orient'] = axis_angle_to_matrix(smal_params['global_orient'].reshape(batch_size, -1)).unsqueeze(1)
        smal_params['pose'] = axis_angle_to_matrix(smal_params['pose'].reshape(batch_size, -1, 3))
        smal_params = {k: v.cuda() for k, v in smal_params.items()}
        with torch.no_grad():
            smal_output = self.smal_model(**smal_params)
        vertices = smal_output.vertices
        return vertices
