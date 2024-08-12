#!/bin/bash

train_file='/share/nas165/peng/desktop/ETS_0118/data/LTTC_HS/Unseen_1962/LTTC_Adv_train_0221.csv'
dev_file='/share/nas165/peng/desktop/ETS_0118/data/LTTC_HS/Unseen_1962/LTTC_Adv_dev_0221.csv'
exp_dir="./soft_label_exp/LTTC_0411_SAMAD"

LEARNING_RATE='1e-3' # 1e-3

CUDA_VISIBLE_DEVICES=0 python3 ./model/train.py   --train_file "$train_file" \
                        --dev_file  "$dev_file" \
                        --output_dir "$exp_dir"  \
                        --cuda_id 0 \
                        --train_epochs 8  \
                        --train_batch  8  \
                        --eval_batch   8 \
                        --grad_acc     8 \
                        --learning_rate $LEARNING_RATE \

