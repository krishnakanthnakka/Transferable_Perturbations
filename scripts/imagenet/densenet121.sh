#ImageNet1M
python  train_gen.py --gpu_ids=0 \
        --train_classifier='densenet121' \
        --train_dataset='imagenet'  \
        --loss_type='feat' \
        --max_epochs=1 \
        --act_layer=8 \
        --gen_dropout=0.25 \
        --lr_decay_iters=30 \
        --save_epoch_freq=1 \
        --data_shuffle \
        --seed=42 \

exit 1
