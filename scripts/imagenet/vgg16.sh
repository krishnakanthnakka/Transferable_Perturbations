

python  train_gen.py --gpu_ids=0 \
        --train_classifier='vgg16' \
        --train_dataset='imagenet'  \
        --loss_type='feat' \
        --max_epochs=1 \
        --act_layer=18 \
        --gen_dropout=0.5 \
        --lr_decay_iters=30 \
        --save_epoch_freq=1 \
        --data_shuffle \
        --seed=42




