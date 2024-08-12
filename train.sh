#!/bin/bash

# ======================= Intermediate ======================= 
form_id='1572' # 1572, 1764, 1766, 1964(Main)
date='0812'
LEARNING_RATE='1e-4' # 1e-3
module_type="SAMAD_MHA" # content_only, content_langUse, SAMAD, SAMAD_wav2vec

train_file="/share/nas165/peng/thesis_project/SAMAD_06/data/LTTC_Intermediate/Unseen_${form_id}/train_${form_id}_0520.csv"
dev_file="/share/nas165/peng/thesis_project/SAMAD_06/data/LTTC_Intermediate/Unseen_${form_id}/dev_${form_id}_0520.csv"

exp_dir="./exp/LTTC-Intermediate/IS-${form_id}/${module_type}_${date}_${LEARNING_RATE}_roundown"


# train: train.py, train_softlabel, train_wav2vec, train_subModel, pretrained Model
CUDA_VISIBLE_DEVICES=0 python3 ./models/train_subModel.py   --train_file "$train_file" \
                        --dev_file  "$dev_file" \
                        --output_dir "$exp_dir"  \
                        --cuda_id 0 \
                        --train_epochs 8  \
                        --train_batch  2  \
                        --eval_batch   2 \
                        --grad_acc     4 \
                        --learning_rate $LEARNING_RATE \


# ======================= High-Intermediate ======================= 
# form_id = '1962' # 1731, 1801, 1862, 1962(Main)
# train_file='/share/nas165/peng/thesis_project/SAMAD_06/data/LTTC_HI/Unseen_${form_id}/train_${form_id}_0520.csv'
# dev_file='/share/nas165/peng/thesis_project/SAMAD_06/data/LTTC_HI/Unseen_${form_id}/dev_${form_id}_0520.csv'

# exp_dir="./exp/LTTC-Intermediate/IS-${form_id}/${module_type}_${date}"
# LEARNING_RATE='1e-3' # 1e-3

# CUDA_VISIBLE_DEVICES=1 python3 ./models/train.py   --train_file "$train_file" \
#                         --dev_file  "$dev_file" \
#                         --output_dir "$exp_dir"  \
#                         --cuda_id 1 \
#                         --train_epochs 8  \
#                         --train_batch  8  \
#                         --eval_batch   8 \
#                         --grad_acc     4 \
#                         --learning_rate $LEARNING_RATE \