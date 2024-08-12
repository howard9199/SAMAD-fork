#!/bin/bash

# ==========================================================
checkpoint_path='LTTC_0411_SAMAD'
date='LTTC_0411_SAMAD'
checkpoint='720'
# ==========================================================
part3='part3'
# known_type='known'
# unknow_type='unknown'
bin='bin'

# ------------------------------Part3---------------------------------------------
model_path="./soft_label_exp/${checkpoint_path}/checkpoint-${checkpoint}" # BEST Model

# ---------------------------------------------------------------------------
known_types=("known" "unknown")
# ("known")
# "unknown") # Define known and unknown types
test_path=("test_1213" "fulltest_1213")
test_path=("LTTC_Adv_test_in_0221" "LTTC_Adv_test_out_0221")
# "test_1213"
# "fulltest_1213")
for index in ${!known_types[@]}; do 
    test_name=${test_path[$index]}
    known_type=${known_types[$index]}

    
    # test_path="/share/nas165/peng/desktop/ETS_0118/data/clean_data/asr_whisper/${test_name}.csv"
    test_path="/share/nas165/peng/desktop/ETS_0118/data/LTTC_HS/Unseen_1962/${test_name}.csv"
    
    output_path="./results/output/${part3}_${known_type}_${date}_ckpt${checkpoint}.csv"
    subscore_path="./results/subscore/${part3}_${known_type}_${date}_ckpt${checkpoint}.csv"
    pic_name="./results/multi_cls/${part3}_${known_type}_${date}_ckpt${checkpoint}.png"
    pic_name_bin="./results/binary_cls/${part3}_${known_type}_${bin}_${date}_ckpt${checkpoint}.png"

    echo "Index:" ${known_types[$index]}
    echo ${output_path}
    echo ${test_path}

    CUDA_VISIBLE_DEVICES=0 python3 ./model/test.py --model_path "$model_path" \
                            --output_path "$output_path" \
                            --test_file  "$test_path" \
                            --pic_name "$pic_name"\
                            --bin_pic_name "$pic_name_bin" \
                            --cuda_id 0 \

done