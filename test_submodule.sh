#!/bin/bash

# ==========================================================
form_id='1964' # 1572, 1764, 1766, 1964(Main)
checkpoint='384'
date='0119'
# ==========================================================
# level='LTTC_Intermediate'
# model_name='SAMAD_MHA_roundown_0610_1e-4' # SAMAD_MHA_0615_1e-4_1.2_roundown, SAMAD_MHA_OLL_0617_1e-4_roundown
# model_path="./exp/LTTC-Intermediate/IS-${form_id}/${model_name}/checkpoint-${checkpoint}"
# 0615_softlabel_roundown
# Single model
level='LTTC_Intermediate'
model_name='SAMAD_QWKloss_0110_1e-5_roundown(epoch32,batch64)' # SAMAD_MHA_content_0620_1e-4_roundown, SAMAD_MHA_content_only_0611_1e-4, SAMAD_MHA_content_delivery_0611_1e-4, SAMAD_MHA_langUse_0611_1e-4
model_path="./exp/LTTC-Intermediate/IS-${form_id}/${model_name}/checkpoint-${checkpoint}"
# ---------------------------------------------------------------------------
known_types=("known" "unknown")
test_path=("test" "fulltest")
#known_types=("known_dev")
#test_path=("dev")

for index in ${!known_types[@]}; 
do 
    test_name=${test_path[$index]}
    known_type=${known_types[$index]}

    #test_path="/share/nas165/peng/thesis_project/(ok)SAMAD_06/data/${level}/Unseen_${form_id}/${test_name}_${form_id}_0520.csv"
    #test_path="/datas/store163/howard/samad/SAMAD/picture-description/cleaned_dev_1764_0520_merged.csv"
    # Create a new folder to store the output results
    folder_name="./results/${level}/${form_id}/${model_name}_${date}/${known_type}/Ckpt_${checkpoint}"
    if [ -d "$folder_name" ]; then
        echo "Folder Existed: $folder_name"
    else
        # Create a folder
        mkdir -p "${folder_name}"
        mkdir "${folder_name}/tsne"
        echo "Create a new folder: $folder_name and $folder_name/tsne "
    fi

    output_path="./${folder_name}/${known_type}_ckpt${checkpoint}.csv"
    subscore_path="./${folder_name}/${known_type}_ckpt${checkpoint}.csv"
    logits_name="./${folder_name}/${known_type}_ckpt${checkpoint}.npy"

    pic_name="./${folder_name}/Multi_${known_type}_ckpt${checkpoint}.png"
    pic_name_bin="./${folder_name}/Binary_${known_type}_ckpt${checkpoint}.png"
    tsne_name="${folder_name}/tsne/tsne_${known_type}_ckpt${checkpoint}.png"
    tsne_name_holistic="${folder_name}/tsne/${known_type}_ckpt${checkpoint}_holistic.png"
    tsne_name_content="${folder_name}/tsne/${known_type}_ckpt${checkpoint}_content.png"
    tsne_name_delivery="${folder_name}/tsne/${known_type}_ckpt${checkpoint}_delivery.png"
    tsne_name_langUse="${folder_name}/tsne/${known_type}_ckpt${checkpoint}_langUSe.png"
    
    
    CUDA_VISIBLE_DEVICES=0 python3 ./models/test.py --model_path "$model_path" \
                                    --output_path "$output_path" \
                                    --dataset_split "$test_name"\
                                    --logit_name "$logits_name" \
                                    --pic_name "$pic_name"\
                                    --bin_pic_name "$pic_name_bin" \
                                    --output_holistic_name "$tsne_name_holistic" \
                                    --output_content_name  "$tsne_name_content" \
                                    --output_delivery_name "$tsne_name_delivery" \
                                    --output_langUse_name  "$tsne_name_langUse" \
                                    --cuda_id 0 \

done

# # ======================High-Intermediate===============================
# form_id='1962' # 1731, 1801, 1862, 1962
# checkpoint='720'
# # ==========================================================
# level='LTTC_HI'
# model_name='test_0520'
# model_path="./exp/LTTC-HI/HI-${form_id}/${model_name}/checkpoint-${checkpoint}"

# # ---------------------------------------------------------------------------
# known_types=("known" "unknown")
# test_path=("test" "fulltest")

# for index in ${!known_types[@]}; 
# do 
#     test_name=${test_path[$index]}
#     known_type=${known_types[$index]}

#     test_path="/share/nas165/peng/thesis_project/SAMAD_06/data/${level}/Unseen_${form_id}/${test_name}_${form_id}_0520.csv"

#     # Create a new folder to store the output results
#     folder_name="./results/${level}/${model_name}/${known_type}"
#     if [ -d "$folder_name" ]; then
#         echo "Folder Existed: $folder_name"
#     else
#         # Create a folder
#         mkdir -p "${folder_name}"
#         mkdir "${folder_name}/tsne"
#         echo "Create a new folder: $folder_name and $folder_name/tsne "
#     fi

#     output_path="./${folder_name}/${known_type}_ckpt${checkpoint}.csv"
#     subscore_path="./${folder_name}/${known_type}_ckpt${checkpoint}.csv"
#     pic_name="./${folder_name}/Multi_${known_type}_ckpt${checkpoint}.png"
#     pic_name_bin="./${folder_name}/Binary_${known_type}_ckpt${checkpoint}.png"
#     tsne_name="${folder_name}/tsne/tsne_${known_type}_ckpt${checkpoint}.png"
#     tsne_name_holistic="${folder_name}/tsne/${known_type}_ckpt${checkpoint}_holistic.png"
#     tsne_name_content="${folder_name}/tsne/${known_type}_ckpt${checkpoint}_content.png"
#     tsne_name_delivery="${folder_name}/tsne/${known_type}_ckpt${checkpoint}_delivery.png"
#     tsne_name_langUse="${folder_name}/tsne/${known_type}_ckpt${checkpoint}_langUSe.png"
    
    
#     CUDA_VISIBLE_DEVICES=0 python3 ./models/test_subModule.py --model_path "$model_path" \
#                                     --output_path "$output_path" \
#                                     --test_file  "$test_path" \
#                                     --pic_name "$pic_name"\
#                                     --bin_pic_name "$pic_name_bin" \
#                                     --output_holistic_name "$tsne_name_holistic" \
#                                     --output_content_name  "$tsne_name_content" \
#                                     --output_delivery_name "$tsne_name_delivery" \
#                                     --output_langUse_name  "$tsne_name_langUse" \
#                                     --cuda_id 0 \

# done
