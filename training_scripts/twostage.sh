stage1="first_stage"
stage2="second_stage"

python main_mamr.py exp_name=$stage1 experiment=multi_animal_det trainer=gpu launcher=local WANDB.MODE=online trainer=ddp trainer.devices=4 \
       DATASETS.AnimalPose.WEIGHT=0.0 DATASETS.APTv2.WEIGHT=0.0 DATASETS.AwA2.WEIGHT=0.0 DATASETS.StanfordExtra.WEIGHT=0.0 \
       GENERAL.TOTAL_STEPS=20000

rm *.log
mkdir -p logs/train/runs/$stage2/checkpoints
cp logs/train/runs/$stage1/checkpoints/last.ckpt logs/train/runs/$stage2/checkpoints/
base_dir="logs/train/runs/$stage1/wandb"
# Acquire the latest run ID
latest_path=$(ls -td ${base_dir}/run-* 2>/dev/null | head -n 1)
# Extract the run ID
run_id="${latest_path##*-}"

python main_mamr.py exp_name=$stage2 experiment=multi_animal_det trainer=gpu launcher=local WANDB.MODE=online trainer=ddp trainer.devices=4 \
       GENERAL.TOTAL_STEPS=100000 WANDB.ID=$run_id
rm *.log

