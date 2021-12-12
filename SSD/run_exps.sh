#!/bin/sh

net=$1      # squeezenet1_1
dataset=$2  # imagenet
loss=$3     # feat


log="log_SSD_${net}_${dataset}_${loss}.txt"
gen_ckpt_dir="${dataset}_${net}_${loss}"
gen_ckpt_name="1"

python -W ignore eval.py  --generator_ckpt_dir=$gen_ckpt_dir --eval_backbone=vgg    --detector_ckpt=./SSD_checkpoints/vgg_ssd300_voc0712/model_final.pth  --gpu_ids=0 --eval_dataset=voc0712 --generator_ckpt_epoch=$gen_ckpt_name --logfile=$log
python -W ignore eval.py  --generator_ckpt_dir=$gen_ckpt_dir --eval_backbone=resnet50    --detector_ckpt=./SSD_checkpoints/resnet50_ssd300_voc0712/model_final.pth  --gpu_ids=0 --eval_dataset=voc0712 --generator_ckpt_epoch=$gen_ckpt_name --logfile=$log
python -W ignore eval.py  --generator_ckpt_dir=$gen_ckpt_dir --eval_backbone=efficient_net_b3  --detector_ckpt=./SSD_checkpoints/efficient_net_b3_ssd300_voc0712/model_final.pth  --gpu_ids=0 --eval_dataset=voc0712 --generator_ckpt_epoch=$gen_ckpt_name --logfile=$log
python -W ignore eval.py  --generator_ckpt_dir=$gen_ckpt_dir --eval_backbone=mobilenet_v3  --detector_ckpt=./SSD_checkpoints/mobilenet_v3_ssd320_voc0712/model_final.pth  --gpu_ids=0 --eval_dataset=voc0712 --generator_ckpt_epoch=$gen_ckpt_name --logfile=$log


