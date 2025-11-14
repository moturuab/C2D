#!/bin/bash
source "../venv/bin/activate"
groupid=$(date +%F_%T)
python train_clothing1m.py \
  --train_npz /home/moturuab/projects/aip-agoldenb/moturuab/clothing1m.npz \
  --test_npz  /home/moturuab/projects/aip-agoldenb/moturuab/clothing10k_test.npz \
  --model_name resnet50 \
  --pretrained True \
  --epochs 10 \
  --project clothing1m-lilaw-r50 $@

#deactivate
#module load gcc opencv/4.12.0
#pip install opencv-python
#source ../venv/bin/activate

#sbatch --account aip-agoldenb --nodes 1 --gres gpu:l40s:1 --tasks-per-node 1 --mem 128G --cpus-per-task 4 --time 4:00:00 run_killarney.sh --batch_size 256 --lr 2e-3 --epochs=120 --weight_decay 1e-3 --meta_fraction 0.1 --alpha_lr 5e-3 --beta_lr 5e-3 --delta_lr 5e-3 --alpha_wd 1e-4 --beta_wd 1e-4 --delta_wd 1e-4 --use_lilaw False
#sbatch --account aip-agoldenb --nodes 1 --gres gpu:l40s:1 --tasks-per-node 1 --mem 128G --cpus-per-task 4 --time 4:00:00 run_killarney.sh --batch_size 256 --lr 2e-3 --epochs=120 --weight_decay 1e-3 --meta_fraction 0.1 --alpha_lr 5e-3 --beta_lr 5e-3 --delta_lr 5e-3 --alpha_wd 1e-4 --beta_wd 1e-4 --delta_wd 1e-4 --use_lilaw True
#sbatch --account aip-agoldenb --nodes 1 --gres gpu:l40s:1 --tasks-per-node 1 --mem 128G --cpus-per-task 4 --time 4:00:00 run_killarney.sh --batch_size 256 --lr 2e-3 --epochs=120 --weight_decay 1e-3 --meta_fraction 0.1 --alpha_lr 5e-4 --beta_lr 5e-4 --delta_lr 5e-4 --alpha_wd 1e-5 --beta_wd 1e-5 --delta_wd 1e-5 --use_lilaw False
#sbatch --account aip-agoldenb --nodes 1 --gres gpu:l40s:1 --tasks-per-node 1 --mem 128G --cpus-per-task 4 --time 4:00:00 run_killarney.sh --batch_size 256 --lr 2e-3 --epochs=120 --weight_decay 1e-3 --meta_fraction 0.1 --alpha_lr 5e-4 --beta_lr 5e-4 --delta_lr 5e-4 --alpha_wd 1e-5 --beta_wd 1e-5 --delta_wd 1e-5 --use_lilaw True