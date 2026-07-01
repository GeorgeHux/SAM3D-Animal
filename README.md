# SAM 3D Animal: Promptable Animal 3D Reconstruction from Images in the Wild

[**Arxiv**](https://arxiv.org/abs/2605.07604) | [**Project Page**](http://georgehux.com/SAM3D-Animal-project-page/)

[<p align="center">
  <video width="100%" autoplay muted loop playsinline controls>
    <source src="https://raw.githubusercontent.com/georgehux/SAM3D-Animal/main/teaser/teaser.mp4" type="video/mp4">
  </video>
</p>
](https://github.com/user-attachments/assets/7ce136fe-33e2-4562-a0e4-8f509e6754a9)

## Environment Setup
```bash
conda create -n ENV_NAME python=3.10
# install dependencies
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip install "git+https://github.com/facebookresearch/pytorch3d.git" --no-build-isolation
pip install -e .[all] --no-build-isolation
```

## Checkpoints and Data

Pretrained checkpoints and processed datasets can be downloaded from Google Drive:

[Download checkpoints and data](https://drive.google.com/drive/folders/1JpAodK43prWNk7vr82n8lJg7_bws3pGd?usp=drive_link)

After downloading, place or symlink the files under the repository root. The expected layout is:

```text
data/
  Animal3D/
  APTv2/
  AnimalKingdomTest_cropped/
  AnimalPose/
  AwA2/
  StanfordExtra/
  Herd3D/
  GenZooMultiAnimalv1/
  backbone.pth
  apt36k.pth
  sam3/

logs/
  train/
    runs/
      <run_name>/
        .hydra/config.yaml
        checkpoints/<checkpoint>.ckpt
```

The demo uses `data/sam3/` for SAM3 and `data/apt36k.pth` for ViTPose by default. Training uses `data/backbone.pth` as the default backbone initialization. Training and evaluation configs use relative dataset paths under `data/`.

## Demo

`demo.py` reconstructs animals from a single image or a folder of images. The demo expects the following assets by default:
```text
data/sam3/          # local SAM3 model directory
data/apt36k.pth     # ViTPose checkpoint, used when --use_vitpose is enabled
```
Run the demo:

```bash
python demo.py --input_path data/qualitative --checkpoint /path/to/checkpoint.ckpt --out_folder demo_out --use_sam3 --use_vitpose
```

## Training

Training is configured through Hydra. The main multi-animal training entry point is `main_mamr.py`, and the default multi-animal experiment config is:

```text
amr/configs_hydra/experiment/multi_animal_det.yaml
```

The config uses relative dataset paths under `data/`. Prepare the datasets and annotation files according to the paths in the config, for example:

```text
data/Animal3D/train_multi_animal.json
data/Animal3D/test_multi_animal.json
data/APTv2/train_multi_animal_clean_wmask.json
data/APTv2/test_multi_animal_wmask.json
data/AnimalPose/train_multi_animal_clean_wmask.json
data/AwA2/train_multi_animal_clean_wmask.json
data/StanfordExtra/train_multi_animal_clean_wmask.json
data/Herd3D/train_multi_animal.json
```

The default backbone checkpoint is loaded from:

```text
data/backbone.pth
```

To run the provided two-stage training script:

```bash
bash training_scripts/twostage.sh
```

The script first trains `first_stage`, copies `last.ckpt` into the second-stage run directory, and then trains `second_stage`. Outputs are written under:

```text
logs/train/runs/<exp_name>/
```

For a custom run, launch `main_mamr.py` directly:

```bash
python main_mamr.py \
  exp_name=my_experiment \
  experiment=multi_animal_det \
  trainer=gpu \
  launcher=local \
  WANDB.MODE=offline
```

For multi-GPU DDP training, override the trainer settings:

```bash
python main_mamr.py \
  exp_name=my_experiment \
  experiment=multi_animal_det \
  trainer=ddp \
  launcher=local \
  trainer.devices=4 \
  WANDB.MODE=offline
```

## Eval

`eval.py` evaluates a trained checkpoint on the datasets defined in:

```text
amr/configs_hydra/experiment/default_val.yaml
```

The default evaluation config currently includes `Animal3D`, `APTv2`, and `AnimalKingdom`. Make sure the corresponding files exist under `data/`:

```text
data/Animal3D/test_multi_animal.json
data/APTv2/test_multi_animal_wmask.json
data/AnimalKingdomTest_cropped/test_multi_animal.json
```

Evaluate all configured datasets:

```bash
python eval.py \
  --config /path/to/run/.hydra/config.yaml \
  --checkpoint /path/to/run/checkpoints/epoch-499.ckpt \
  --dataset ALL \
  --device cuda
```

Evaluate a single dataset:

```bash
python eval.py \
  --config /path/to/run/.hydra/config.yaml \
  --checkpoint /path/to/run/checkpoints/checkpoint.ckpt \
  --dataset Animal3D \
  --device cuda
```

## Citation
```bash
@misc{hu2026sam3danimalpromptable,
      title={SAM 3D Animal: Promptable Animal 3D Reconstruction from Images in the Wild},
      author={Xuyi Hu and Jin Lyu and Jiuming Liu and Yebin Liu and Silvia Zuffi and Liang An and Stefan Goetz},
      year={2026},
      eprint={2605.07604},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2605.07604},
}
```
