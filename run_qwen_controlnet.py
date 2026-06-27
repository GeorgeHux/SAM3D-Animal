#!/usr/bin/env python3
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:256"
import gc
import re
import json
import argparse
from pathlib import Path
import hashlib
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image
import torch

from diffusers import QwenImageControlNetPipeline, QwenImageControlNetModel


# -------------------------
# utils
# -------------------------
def safe_name(s: str) -> str:
    return "_".join(s.strip().split())


_PAT_IMGDIR = re.compile(r"img_(\d{6})$")


def get_img_idx_from_rgb_path(p: Path) -> int:
    # .../img_000123/img_000123_rgb.png
    m = _PAT_IMGDIR.match(p.parent.name)
    if not m:
        raise ValueError(f"Bad img dir name: {p.parent}")
    return int(m.group(1))


def atomic_write(path: Path, pil_img: Image.Image):
    path.parent.mkdir(parents=True, exist_ok=True)

    # 让 tmp 的“最后后缀”依然是原来的 suffix，比如 .png
    tmp = path.with_name(path.stem + ".tmp" + path.suffix)  # xxx.tmp.png

    pil_img.save(tmp, format=path.suffix.lstrip(".").upper())
    os.replace(tmp, path)



def load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def stable_seed(key: str, base_seed: int) -> int:
    h = hashlib.md5(key.encode("utf-8")).hexdigest()[:8]
    return (int(h, 16) ^ int(base_seed)) & 0xFFFFFFFF


def overlay_rgb(rgb_img: np.ndarray, gen_img: np.ndarray, alpha: float = 0.5) -> Image.Image:
    a = float(alpha)
    rgb = rgb_img.astype(np.float32) / 255.0
    gen = gen_img.astype(np.float32) / 255.0
    out = (1.0 - a) * rgb + a * gen
    out = np.clip(out, 0.0, 1.0)
    return Image.fromarray((out * 255).astype(np.uint8))


def load_species_file(path: Path) -> List[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    out = []
    for ln in lines:
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        out.append(ln)
    if not out:
        raise RuntimeError(f"species_file is empty: {path}")
    return out

def cleanup_cuda(pipe=None):
    # 可选：清 VAE cache，防止越跑越大
    if pipe is not None and hasattr(pipe, "vae"):
        if hasattr(pipe.vae, "_feat_map"):
            pipe.vae._feat_map = None
        if hasattr(pipe.vae, "_enc_feat_map"):
            pipe.vae._enc_feat_map = None

    gc.collect()
    torch.cuda.empty_cache()


# -------------------------
# args
# -------------------------
def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--train_root",
        type=str,
        default="/rds/user/xh365/hpc-work/genzoo/train",
        help="root dir: train/<species>/img_XXXXXX/img_XXXXXX_rgb.png",
    )

    p.add_argument(
        "--animal_name",
        action="append",
        default=[],
        help="species name; can be used multiple times, e.g. --animal_name 'Pomeranian' --animal_name 'Samoyed'",
    )
    p.add_argument(
        "--species_file",
        type=str,
        default=None,
        help="optional text file, one species per line",
    )

    # run selection
    p.add_argument("--start_idx", type=int, default=0, help="min global img idx (per species)")
    p.add_argument("--num_images", type=int, default=-1, help="-1 means all (per species)")
    p.add_argument("--skip_existing", action="store_true", default=False)

    # control
    p.add_argument("--control", type=str, default="depth", choices=["depth", "canny"])
    p.add_argument("--conditioning_scale", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--true_cfg_scale", type=float, default=4.0)

    # prompt
    p.add_argument(
        "--prompt_key",
        type=str,
        default="final_prompt",
        help="key in the json used as prompt",
    )
    p.add_argument(
        "--prompt_fallback",
        type=str,
        default="A photorealistic group of animals.",
        help="used if per-image prompt json is missing",
    )
    p.add_argument(
        "--negative_prompt",
        type=str,
        default="incorrect depth ordering, wrong occlusion, unnatural overlapping, animals merging, body clipping artifacts",
    )

    # output naming (saved under img_xxxxxx/)
    p.add_argument("--out_prefix", type=str, default="qwen_controlnet")
    p.add_argument("--save_overlay", action="store_true", default=True)
    p.add_argument("--no-save_overlay", action="store_false", dest="save_overlay")
    p.add_argument("--overlay_alpha", type=float, default=0.5)

    # perf
    p.add_argument("--batch_size", type=int, default=8, help="batch prompts/images per GPU forward")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--enable_xformers", action="store_true", default=False)
    p.add_argument("--enable_compile", action="store_true", default=False)

    return p.parse_args()


# -------------------------
# main
# -------------------------
def main():
    args = parse_args()

    train_root = Path(args.train_root)
    if not train_root.exists():
        raise RuntimeError(f"train_root not found: {train_root}")

    # species list
    if args.animal_name:
        species_list = [s.strip() for s in args.animal_name if s.strip()]
    elif args.species_file:
        species_list = load_species_file(Path(args.species_file))
    else:
        raise RuntimeError("Provide --animal_name (one or more) or --species_file.")

    # 1) load pipeline once
    base_model = "Qwen/Qwen-Image"
    controlnet_model = "InstantX/Qwen-Image-ControlNet-Union"

    torch.set_float32_matmul_precision("high")

    controlnet = QwenImageControlNetModel.from_pretrained(
        controlnet_model,
        torch_dtype=torch.bfloat16,
    )
    pipe = QwenImageControlNetPipeline.from_pretrained(
        base_model,
        controlnet=controlnet,
        torch_dtype=torch.bfloat16,
    ).to("cuda")
    pipe.enable_attention_slicing("max")   # 降注意力峰值
    pipe.vae.enable_slicing()             # 降 VAE 峰值
    pipe.vae.enable_tiling()              # 进一步降 VAE 峰值（会慢一点但稳）

    # Optional perf knobs
    if args.enable_xformers:
        try:
            pipe.enable_xformers_memory_efficient_attention()
            print("[perf] xformers attention enabled", flush=True)
        except Exception as e:
            print(f"[perf] xformers enable failed: {repr(e)}", flush=True)

    if args.enable_compile:
        # compile has long warm up，but save time for many images
        try:
            pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=False)
            print("[perf] torch.compile enabled (unet)", flush=True)
        except Exception as e:
            print(f"[perf] torch.compile failed: {repr(e)}", flush=True)

    pipe.set_progress_bar_config(disable=False)

    # batching helper
    def chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    for species in species_list:
        sp_dir = train_root / safe_name(species)
        if not sp_dir.exists():
            print(f"[warn] species folder not found, skip: {species} -> {sp_dir}", flush=True)
            continue

        rgb_paths = sorted(sp_dir.glob("img_*/img_*_rgb.png"))
        if not rgb_paths:
            print(f"[warn] no rgb images under: {sp_dir}", flush=True)
            continue

        # filter by global img idx
        picked = []
        for p in rgb_paths:
            try:
                img_idx = get_img_idx_from_rgb_path(p)
            except Exception:
                continue
            if img_idx >= int(args.start_idx):
                picked.append(p)

        # truncate
        if int(args.num_images) >= 0:
            picked = picked[: int(args.num_images)]

        if not picked:
            print(f"[warn] no images selected for {species}", flush=True)
            continue

        print(f"\n=== species={species} | images_to_process={len(picked)} ===", flush=True)

        # Process in GPU batches
        for batch_paths in chunks(picked, max(1, int(args.batch_size))):
            # prepare batch inputs
            prompts: List[str] = []
            neg_prompts: List[str] = []
            control_images: List[Image.Image] = []
            metas: List[Tuple[Path, int, Path, Path, Path]] = []
            # metas: (rgb_path, img_id, img_dir, out_gen, out_overlay)

            # build batch
            for rgb_path in batch_paths:
                img_dir = rgb_path.parent
                img_id = get_img_idx_from_rgb_path(rgb_path)

                # output paths inside img_dir
                out_gen = img_dir / f"{args.out_prefix}_{args.control}_gen.png"
                out_ovl = img_dir / f"{args.out_prefix}_{args.control}_overlay.png"

                if args.skip_existing and out_gen.exists() and (not args.save_overlay or out_ovl.exists()):
                    continue

                # find control image in SAME img_dir
                ctrl_path = img_dir / f"img_{img_id:06d}_{args.control}.png"
                if not ctrl_path.exists():
                    print(f"[skip] missing control image: {ctrl_path}", flush=True)
                    continue

                # prompt from per-image json
                prompt_path = img_dir / f"prompt_{img_dir.name}.json"
                pj = load_json(prompt_path)
                prompt = None
                if isinstance(pj, dict):
                    v = pj.get(args.prompt_key, None)
                    if isinstance(v, str) and v.strip():
                        prompt = v.strip()
                if not prompt:
                    prompt = args.prompt_fallback

                # load control image (PIL)
                control_img = Image.open(ctrl_path).convert("RGB")

                prompts.append(prompt)
                neg_prompts.append(args.negative_prompt)
                control_images.append(control_img)
                metas.append((rgb_path, img_id, img_dir, out_gen, out_ovl))

            if not metas:
                continue

            # generator: per-image deterministic seed
            # diffusers 支持 list[Generator]（多数 pipeline OK），不行就退化成单个 generator
            generators = []
            for (rgb_path, img_id, img_dir, out_gen, out_ovl) in metas:
                s = stable_seed(str(rgb_path), args.seed)
                generators.append(torch.Generator(device="cuda").manual_seed(s))

            # run batch
            try:
                outs = pipe(
                    prompt=prompts,
                    negative_prompt=neg_prompts,
                    control_image=control_images,
                    controlnet_conditioning_scale=float(args.conditioning_scale),
                    width=control_images[0].size[0],
                    height=control_images[0].size[1],
                    num_inference_steps=int(args.steps),
                    true_cfg_scale=float(args.true_cfg_scale),
                    generator=generators,  # if unsupported, exception -> we fallback below
                ).images
            except Exception as e:
                # fallback: one-by-one (more stable)
                print(f"[warn] batched call failed -> fallback to per-image. err={repr(e)}", flush=True)
                outs = []
                for (rgb_path, img_id, img_dir, out_gen, out_ovl), prompt, neg, ctrl, gen in zip(
                    metas, prompts, neg_prompts, control_images, generators
                ):
                    o = pipe(
                        prompt=prompt,
                        negative_prompt=neg,
                        control_image=ctrl,
                        controlnet_conditioning_scale=float(args.conditioning_scale),
                        width=ctrl.size[0],
                        height=ctrl.size[1],
                        num_inference_steps=int(args.steps),
                        true_cfg_scale=float(args.true_cfg_scale),
                        generator=gen,
                    ).images[0]
                    outs.append(o)

            # save outputs
            for (rgb_path, img_id, img_dir, out_gen, out_ovl), out_img in zip(metas, outs):
                # gen
                atomic_write(out_gen, out_img)

                # overlay (optional)
                if args.save_overlay:
                    rgb_np = np.array(Image.open(rgb_path).convert("RGB"))
                    ovl = overlay_rgb(rgb_np, np.array(out_img), alpha=float(args.overlay_alpha))
                    atomic_write(out_ovl, ovl)

                print(f"[ok] {img_dir.name} | {args.control} -> {out_gen.name}", flush=True)
                if hasattr(pipe.vae, "_feat_map"):
                    pipe.vae._feat_map = None
                if hasattr(pipe.vae, "_enc_feat_map"):
                    pipe.vae._enc_feat_map = None
                
                cleanup_cuda(pipe)


    print("\nDone.")


if __name__ == "__main__":
    main()

