# train_subModel_by_subscore.py

import pandas as pd
from datasets import Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import spacy
from transformer_model import Transformer
from transformers.modeling_outputs import TokenClassifierOutput
from transformers import BertTokenizer, BertModel
from transformers import Wav2Vec2Processor
import os
import re
import warnings
warnings.filterwarnings('ignore')
from text_preprocessing import *
import json
import librosa
import argparse
import math
from sklearn.metrics import accuracy_score

np.random.seed(42)

def phi(task_label, true_label):
    if task_label == true_label:
        return 0
    elif abs(task_label - true_label) == 1:
        return args.softlabel_para
    else:
        return float("inf")   

def func_softLabels(label):
    exp_terms = torch.exp(-torch.tensor([phi(i, label) for i in range(num_labels)]))
    soft_label = []
    for i in range(num_labels):
        k = torch.exp(-torch.tensor(phi(i, label))) / exp_terms.sum()
        soft_label.append(k.item())

    return soft_label

import librosa
max_audio_length = 16000 * 90  # 假設音訊的最大長度是16000個樣本點
target_sampling_rate = 16000
def speech_file_to_array_fn(path):
    speech_array, sampling_rate = librosa.load(path, sr=target_sampling_rate)
    mono_waveform = librosa.resample(speech_array, orig_sr=sampling_rate, target_sr=target_sampling_rate)

    return mono_waveform

def cal_parameter(model):
    total_params = sum(
        param.numel() for param in model.parameters()
    )
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    print(f'Total parameters: {total_params}')
    print(f'Trainable parameters: {trainable_params}')

def dataset_prepare(file_path):
    df = pd.read_csv(file_path)
    # random score
    # df[args.score] = random_round(df[args.score])
    # baseline
    #print(args.module)
    #print(df[args.module])
    df[args.module] = df[args.module].apply(math.ceil)     # 3.5 -> 3.0
    df = Dataset.from_dict(df)
    return df

def preprocess(examples):

    target_list = [label_to_id(label, label_list) for label in examples[args.module]]
    
    # Softmax 
    # target_list = [func_softLabels(label_to_id(label, label_list)) for label in examples[args.score]]

    vec_list = [json.loads(item) for item in examples['delivery_vec']]
    vec = pad_delivery_sequences(vec_list)

    # ================== Language =================================
    onehot_encoding = [text2BinVec(txt) for txt in examples[args.asr_transcrip]]
    padded_seq = pad_sequence(onehot_encoding) # [batch, sentence_length, 247]
    masked_seq = mask_seq(onehot_encoding)

    # ==================Content=================================
    padded_response_emb = [combineContext(res) for res in examples[args.asr_transcrip]]
    padded_prompt_emb   = [combineContext(prompt) for prompt in examples[args.description]]

    result = {
        "padded_seq": padded_seq,
        "masked_seq": masked_seq,
        "delivery_tra": vec,
    }
    print("**************************************")

    # BERT
    response_tokens = tokenizer(padded_response_emb, 
                                max_length=256,
                                truncation=True,
                                padding='max_length',
                                return_attention_mask=True,
                                return_tensors='pt')

    prompt_tokens = tokenizer(padded_prompt_emb, 
                                max_length=256,
                                truncation=True,
                                padding='max_length',
                                return_attention_mask=True,
                                return_tensors='pt')
    
    # ========================= Wav2vec Delivery =================================
    speech_list = [speech_file_to_array_fn(path) for path in examples[args.path_column]]
    result['input_emb'] = processor(speech_list, 
                                    sampling_rate=target_sampling_rate,
                                    max_length=max_audio_length,
                                    truncation=True,
                                    padding='max_length',
                                    return_tensors="pt"
                                )['input_values']


    result['padded_res_emb'] = response_tokens['input_ids']
    result['attention_mask_res'] = response_tokens['attention_mask']

    result['padded_pro_emb'] = prompt_tokens['input_ids']
    result['attention_mask_pro'] = prompt_tokens['attention_mask']

    result['labels'] = target_list

    return result

def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    references = eval_pred.label_ids
    acc = accuracy_score(references, predictions)
    return {"accuracy": acc}

def compute_loss(model, inputs):
    outputs = model(**inputs)
    labels = inputs.get("labels")
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(outputs, labels)
    return loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda_id', type=int, default=0, help='cuda device')
    parser.add_argument('--train_file', help='訓練資料檔案路徑')
    parser.add_argument('--dev_file', help='驗證資料檔案路徑')
    parser.add_argument('--output_dir', help='模型輸出路徑')
    parser.add_argument('--module', type=str, required=True, choices=['relevance', 'delivery', 'language'], help='要訓練的模組')
    parser.add_argument('--train_epochs', type=int, default=5, help='訓練回合數')
    parser.add_argument('--train_batch', type=int, default=2, help='訓練批次大小')
    parser.add_argument('--eval_batch', type=int, default=2, help='驗證批次大小')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='學習率')
    parser.add_argument('--asr_transcrip', default='whisperX_transcription', help='text_value')
    parser.add_argument('--description',    default='prompt', help='description')
    parser.add_argument('--path_column',    default='wav_path', help='path_value')
    args = parser.parse_args()

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device =  torch.device("cpu")
    print("使用裝置:", device)

    # 載入資料
    train_df = dataset_prepare(args.train_file)
    dev_df = dataset_prepare(args.dev_file)

    processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base')

    # 準備 tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    #label_list2 = train_df.unique(args.score)
    label_list = [1, 2, 3, 4, 5] 
    label_list.sort()
    #label_list2.sort()
    num_labels = len(label_list)

    if args.module == 'relevance':
        from content_module import content_BLSTM
        model = content_BLSTM()
        model.set_state(1)
    elif args.module == 'delivery':
        from delivery_module import Delivery_BLSTM
        model = Delivery_BLSTM()#delivery,relevance,language
        model.set_state(1)
    elif args.module == 'language':
        from langUse_module import Language_BLSTM
        model = Language_BLSTM(device=device)
        model.set_state(1)
    # --------------------------Data Preparation---------------------------
    '''
    train_df = train_df.map(
        preprocess,
        batch_size=100,
        batched=True,
        # remove_columns=['speaker_id', 'score', 'classification', 'description', 'example', 'confidence_score', 'acoustic_score', 'lm_score', 'pitch', 'intensity', 'pause(sec)', 'silence(sec)', 'duration(sec)', 'asr_transcrip', 'wav_path']
        # remove_columns=['speaker_id', 'form_id', 'grade', 'wav_path', 'asr_transcription', 'prompt', 'confidence_score', 'acoustic_score', 'lm_score', 'pitch', 'intensity', 'pause(sec)', 'silence(sec)', 'duration(sec)']
        # WhisperX
        remove_columns=['example', 'wpm', 'total_num_word', 'level_1', 'level_2', 'level_3', 'seq_len', 'key1', 'key2', 'key3', 'key4', 'avg', 'all_avg_score', 'Threshold_Count', 'mean_pitch', 'mean_intensity', 'duration', 'localJitter', 'localShimmer', 'rapJitter', 'long_silence', 'silence', 'long_silence_num', 'silence_num', 'std_energy', 'avg_spectral', 'avg_energy_entropy', 'zero_cross_num', 'v_to_uv_ratio', 'voice_count', 'unvoice_count', 'mean_long_silence', 'mean_silence', 'more3word', 'num_word', 'whisperX_transcription', 'delivery_vec']
        # HI 
        # remove_columns=['mean_pitch', 'mean_intensity', 'duration', 'localJitter', 'localShimmer', 'rapJitter', 'long_silence', 'silence', 'long_silence_num', 'silence_num', 'std_energy', 'avg_spectral', 'avg_energy_entropy', 'zero_cross_num', 'v_to_uv_ratio', 'voice_count', 'unvoice_count', 'more3word', 'num_word', 'whisperX_transcription', 'delivery_vec']
    )
    dev_df = dev_df.map(
        preprocess,
        batch_size=100,
        batched=True,
        # remove_columns=['speaker_id', 'score', 'classification', 'description', 'example', 'confidence_score', 'acoustic_score', 'lm_score', 'pitch', 'intensity', 'pause(sec)', 'silence(sec)', 'duration(sec)', 'asr_transcrip', 'wav_path']
        # remove_columns=['speaker_id', 'form_id', 'grade', 'wav_path', 'asr_transcription', 'prompt', 'confidence_score', 'acoustic_score', 'lm_score', 'pitch', 'intensity', 'pause(sec)', 'silence(sec)', 'duration(sec)']
        # WhisperX
        remove_columns=['example', 'wpm', 'total_num_word', 'level_1', 'level_2', 'level_3', 'seq_len', 'key1', 'key2', 'key3', 'key4', 'avg', 'all_avg_score', 'Threshold_Count', 'mean_pitch', 'mean_intensity', 'duration', 'localJitter', 'localShimmer', 'rapJitter', 'long_silence', 'silence', 'long_silence_num', 'silence_num', 'std_energy', 'avg_spectral', 'avg_energy_entropy', 'zero_cross_num', 'v_to_uv_ratio', 'voice_count', 'unvoice_count', 'mean_long_silence', 'mean_silence', 'more3word', 'num_word', 'whisperX_transcription', 'delivery_vec']
        # HI
        # remove_columns=['mean_pitch', 'mean_intensity', 'duration', 'localJitter', 'localShimmer', 'rapJitter', 'long_silence', 'silence', 'long_silence_num', 'silence_num', 'std_energy', 'avg_spectral', 'avg_energy_entropy', 'zero_cross_num', 'v_to_uv_ratio', 'voice_count', 'unvoice_count', 'more3word', 'num_word', 'whisperX_transcription', 'delivery_vec']

    )
    print('train_df', train_df)
    print("dev_df", dev_df)

    # save the processed data to /datas/store162/howard/samad/dataframe/
    train_df.save_to_disk('/datas/store162/howard/samad/dataframe/train_subscore_df')
    dev_df.save_to_disk('/datas/store162/howard/samad/dataframe/dev_subscore_df')
    '''
    train_df = Dataset.load_from_disk('/datas/store162/howard/samad/dataframe/train_subscore_df')
    dev_df = Dataset.load_from_disk('/datas/store162/howard/samad/dataframe/dev_subscore_df')

    # rename the delivery_emb to input_emb
    #train_df = train_df.rename_column('delivery_emb', 'input_emb')
    #dev_df = dev_df.rename_column('delivery_emb', 'input_emb')
    # load df to gpu
    train_df.set_format(type='torch', columns=['padded_seq', 'masked_seq', 'input_emb', 'padded_res_emb', 'attention_mask_res', 'padded_pro_emb', 'attention_mask_pro', 'labels'])
    dev_df.set_format(type='torch', columns=['padded_seq', 'masked_seq', 'input_emb', 'padded_res_emb', 'attention_mask_res', 'padded_pro_emb', 'attention_mask_pro', 'labels'])

    
    num_labels = len(label_list)
    model.to(device)
    cal_parameter(model)

    from transformers import TrainingArguments, Trainer
    import wandb
    wandb.init()
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.train_batch,
        per_device_eval_batch_size=args.eval_batch,
        num_train_epochs=args.train_epochs,
        learning_rate=args.learning_rate,
        evaluation_strategy="steps",
        save_steps=10,
        eval_steps=10,
        logging_steps=10,
        save_total_limit=5,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        no_cuda=True
    )

    def model_init():
        return model

    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_df,
        eval_dataset=dev_df
    )
    trainer.train()