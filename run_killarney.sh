#!/bin/bash
source "../venv/bin/activate"
groupid=$(date +%F_%T)
python train_clothing1m.py \
  --train_npz /home/moturuab/projects/aip-agoldenb/moturuab/clothing1m.npz \
  --test_npz  /home/moturuab/projects/aip-agoldenb/moturuab/clothing10k_test.npz \
  --use_lilaw True \
  --model_name resnet50 \
  --pretrained True \
  --epochs 10 \
  --batch_size 256 \
  --project clothing1m-lilaw-r50