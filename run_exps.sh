#!/bin/sh

# Example command:
# For evaluaating on all models using generator trained against
# squeezenet and imagenet dataset with feature separation loss
# bash exps.sh squeezenet1_1 imagenet feat

net=$1      # squeezenet1_1
dataset=$2  # imagenet
loss=$3     # feat


GEN_CKPT="1"  # loads epoch 1
TRAIN_CONFIG="${dataset}_${net}_${loss}"

EVAL_MODELS="vgg16 resnet152 inception_v3 densenet121 squeezenet1_1 shufflenetv2 mnasnet1_0 mobilenet_v3"

for EVAL_MODEL in $EVAL_MODELS
do
    python eval.py  --train_config=$TRAIN_CONFIG --epoch=$GEN_CKPT --eval_model=$EVAL_MODEL --num_images=-1
done

