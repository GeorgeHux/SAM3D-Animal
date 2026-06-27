import numpy as np
from tqdm import tqdm
import torch
from amr.utils import recursive_to
from amr.utils.evaluate_metric import Evaluator, GlobalKeypointAPAccumulator
from amr.datasets.datasets import EvaluationDataset, MultiAnimalTrainDataset, collate_fn_multi_animal
import argparse
from torch.utils.data import DataLoader
from amr.models import AMR, MultiAMR
from amr.configs import get_config
torch.multiprocessing.set_sharing_strategy('file_system')


def is_multi_animal_cfg(cfg):
    return int(cfg.MODEL.get("NUM_ANIMALS", 1)) > 1


def _to_tensor_recursive(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    if isinstance(x, np.generic):
        return torch.as_tensor(x)
    if isinstance(x, dict):
        return {k: _to_tensor_recursive(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_to_tensor_recursive(v) for v in x]
    if isinstance(x, tuple):
        return [_to_tensor_recursive(v) for v in x]
    return x


def collate_fn_multi_animal_tensor(batch):
    # Reuse dataset collation layout, then convert all numpy payloads to tensors.
    collated = collate_fn_multi_animal(batch)
    return _to_tensor_recursive(collated)


def flatten_multi_animal_batch_for_metric(batch, pred_num_animals=None):
    flat_batch = dict(batch)
    targets = batch["targets"]
    if pred_num_animals is None:
        pred_num_animals = batch["num_animals"]
    pred_num_animals = pred_num_animals.tolist() if torch.is_tensor(pred_num_animals) else list(pred_num_animals)
    targets = [
        {
            "keypoints_3d": b["keypoints_3d"][:n],
            "keypoints_2d": b["keypoints_2d"][:n],
            "smal_params": {k: v[:n] for k, v in b["smal_params"].items()},
            "has_smal_params": {k: v[:n] for k, v in b["has_smal_params"].items()},
        }
        for b, n in zip(targets, pred_num_animals)
    ]
    flat_batch["keypoints_3d"] = torch.cat([b["keypoints_3d"] for b in targets], dim=0)
    flat_batch["keypoints_2d"] = torch.cat([b["keypoints_2d"] for b in targets], dim=0)
    flat_batch["smal_params"] = {
        "global_orient": torch.cat([b["smal_params"]["global_orient"] for b in targets], dim=0),
        "pose": torch.cat([b["smal_params"]["pose"] for b in targets], dim=0),
        "betas": torch.cat([b["smal_params"]["betas"] for b in targets], dim=0),
    }
    flat_batch["has_smal_params"] = {
        "global_orient": torch.cat([b["has_smal_params"]["global_orient"] for b in targets], dim=0),
        "pose": torch.cat([b["has_smal_params"]["pose"] for b in targets], dim=0),
        "betas": torch.cat([b["has_smal_params"]["betas"] for b in targets], dim=0),
    }
    # Evaluator's 2D metrics normalize by mask area per instance.
    # Multi-animal loader provides one mask per image, so repeat by num_animals.
    if "mask" in batch and "num_animals" in batch:
        repeat_counts = torch.as_tensor(pred_num_animals, device=batch["mask"].device)
        flat_batch["mask"] = batch["mask"].repeat_interleave(repeat_counts, dim=0)
    return flat_batch


def main(args):
    cfg = get_config(args.config)
    default_cfg = get_config(args.default_eval_config)
    model_cls = MultiAMR if is_multi_animal_cfg(cfg) else AMR
    model = model_cls.load_from_checkpoint(args.checkpoint, cfg=cfg, strict=False, map_location=args.device)
    model.eval()
    if hasattr(model, "use_gt_prompt"):
        model.use_gt_prompt = True
    if hasattr(model, "use_mask"):
        model.use_mask = True

    smal_evaluator = Evaluator(smal_model=model.smal, image_size=cfg.MODEL.IMAGE_SIZE)
    cfg_eval_dataset = dict(default_cfg.DATASETS)
    aug_cfg = cfg_eval_dataset.pop("CONFIG", None)  # augmentation config is not used in evaluation

    if args.dataset.upper() == "ALL":
        for key in cfg_eval_dataset.keys():
            print(f"-------- Evaluate {key} dataset ------------")
            eval_one_dataset(cfg_eval_dataset[key], default_cfg, cfg, model,
                             evaluator=smal_evaluator,
                             aug_cfg=aug_cfg,
                             key=key)
            print(f"-------{key} Dataset evaluate finish ------")
    else:
        print(f"-------- Evaluate {args.dataset} dataset ------------")
        eval_one_dataset(cfg_eval_dataset[args.dataset], default_cfg, cfg, model,
                         evaluator=smal_evaluator,
                         aug_cfg=aug_cfg,
                         key=args.dataset)
        print(f"-------{args.dataset} Dataset evaluate finish ------")


def eval_one_dataset(dataset_cfg, default_cfg, cfg, model, evaluator, aug_cfg, key):
    is_multi = is_multi_animal_cfg(cfg)
    dataset_root = dataset_cfg['ROOT_IMAGE']
    if is_multi:
        dataset = MultiAnimalTrainDataset(
            cfg=cfg,
            is_train=False,
            root_image=dataset_root,
            json_file=dataset_cfg['JSON_FILE']['TEST'],
        )
        dataloader = DataLoader(
            dataset,
            batch_size=cfg.TRAIN.BATCH_SIZE,
            num_workers=cfg.GENERAL.NUM_WORKERS,
            collate_fn=collate_fn_multi_animal_tensor,
        )
    else:
        dataset = EvaluationDataset(
            root_image=dataset_root,
            json_file=dataset_cfg['JSON_FILE']['TEST'],
            augm_config=aug_cfg,
            focal_length=cfg.SMAL.get("FOCAL_LENGTH", 5000),
            image_size=cfg.MODEL.IMAGE_SIZE,
        )
        dataloader = DataLoader(dataset, batch_size=cfg.TRAIN.BATCH_SIZE, num_workers=cfg.GENERAL.NUM_WORKERS)

    bar = tqdm(dataloader)
    pa_mpjpe_list, pck_list, auc_list, pa_mpvpe_list = [], [], [], []
    num_instances_list = []
    # Global AP accumulator: collects OKS across all batches, computes AP at the end
    ap_accum = GlobalKeypointAPAccumulator(image_size=cfg.MODEL.IMAGE_SIZE)
    for i, batch in enumerate(bar):
        batch = recursive_to(batch, args.device)

        with torch.no_grad():
            output = model(batch)

        pred_num_animals = torch.clamp(batch["num_animals"], max=cfg.MODEL.NUM_ANIMALS) if is_multi else None
        metric_batch = flatten_multi_animal_batch_for_metric(batch, pred_num_animals) if is_multi else batch
        metric_output = output["smal"] if ("smal" in output and is_multi) else output

        if key in ["Animal3D"]:
            pa_mpjpe, pa_mpvpe = evaluator.eval_3d(metric_output, metric_batch)
        else:
            pa_mpjpe, pa_mpvpe = 0., 0.
        pck, auc = evaluator.eval_2d(metric_output, metric_batch, pck_threshold=default_cfg.METRIC.PCK_THRESHOLD)
        ap_accum.add_batch(metric_output, metric_batch)

        n_instances = metric_batch["keypoints_2d"].shape[0]
        num_instances_list.append(n_instances)
        pa_mpjpe_list.append(pa_mpjpe)
        pa_mpvpe_list.append(pa_mpvpe)
        auc_list.append(auc)
        pck_list.append(pck)

        bar.set_postfix(PA_MPJPE=pa_mpjpe,
                        PA_MPVPE=pa_mpvpe,
                        AUC=auc,
                        pck=pck,)

    weights = np.array(num_instances_list, dtype=np.float64)
    print("---------------- 3D metric -----------------")
    print(f"Avg PA-MPJPE: {np.average(pa_mpjpe_list, weights=weights)}")
    print(f"Avg PA-MPVPE: {np.average(pa_mpvpe_list, weights=weights)}")

    print("--------------- 2D metric ------------------")
    ap_metrics = ap_accum.summarize()
    print(f"mAP: {ap_metrics['mAP']}")
    print(f"AP50: {ap_metrics['AP50']}")
    print(f"AP75: {ap_metrics['AP75']}")
    print(f"AUC: {np.average(auc_list, weights=weights)}")
    pck_list = np.array(pck_list)
    for _, th in enumerate(default_cfg.METRIC.PCK_THRESHOLD):
        print(f"PCK@{th}: {np.average(pck_list[:, _], weights=weights)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to config file", required=True)
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint file", required=True)
    parser.add_argument("--default_eval_config", type=str, default="amr/configs_hydra/experiment/default_val.yaml")
    parser.add_argument("--dataset", type=str, default="ALL")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for evaluation")
    args = parser.parse_args()
    main(args)
