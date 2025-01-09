import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import sys
from content_module import content_BLSTM
# from langUse_module import Language_BLSTM
from delivery_module import *
from datasets import Dataset, load_dataset
import spacy
from transformer_model import Transformer
from transformers.modeling_outputs import TokenClassifierOutput
from transformers import BertTokenizer, BertModel
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2Model, Wav2Vec2Processor
# from transformers import WhisperProcessor, WhisperModel
import os
import json
import re
import warnings
warnings.filterwarnings('ignore')
from text_preprocessing import *
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import librosa
max_audio_length = 16000 * 90  # 假設音訊的最大長度是16000個樣本點
target_sampling_rate = 16000
holistic_list = []
def speech_file_to_array_fn(path):
    speech_array, sampling_rate = librosa.load(path, sr=target_sampling_rate)
    mono_waveform = librosa.resample(speech_array, orig_sr=sampling_rate, target_sr=target_sampling_rate)

    return mono_waveform
    
def plot_tsne(logits_list, labels, output_name, x_min, x_max, y_min, y_max):
    
    full_logits = np.concatenate(logits_list, axis=0)
    # print(full_logits.shape)
     # 使用 t-SNE 進行降維
    tsne = TSNE(n_components=2, perplexity=20, random_state=0)
    logits_reduced = tsne.fit_transform(full_logits)

    # # 繪製 t-SNE 視覺化結果
    plt.figure(figsize=(7, 7))
    for i, label in enumerate(set(labels)):
        indices = [j for j, l in enumerate(labels) if l == label]
        plt.scatter(logits_reduced[indices, 0], logits_reduced[indices, 1], label=f"Label {label+1}")

    # x_min, x_max = -9, 3
    # y_min, y_max = -10, 0
    # plt.xlim(x_min, x_max)
    # plt.ylim(y_min, y_max)
    plt.legend()
    plt.title("t-SNE visualization of model logits")
    plt.savefig(output_name)


# ------------------------Confusion Matrix------------------------------
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
def plot_confusion_matrix(y_true, y_pred):

    df_record = {'Ground Truth':y_true,  'Model Prediction':y_pred}
    df_record = pd.DataFrame(df_record)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])
    cm_df = pd.DataFrame(cm, index=[1, 2, 3, 4, 5], columns=[1, 2, 3, 4, 5])
    sns.heatmap(cm_df, annot=True, fmt='g', cmap='Blues')

    accuracy = df_record["Ground Truth"].eq(df_record["Model Prediction"]).sum() / len(df_record["Ground Truth"])
    score = f1_score(df_record["Ground Truth"], df_record["Model Prediction"], average="weighted")
    # Calculate micro metrics
    micro_precision = precision_score(df_record["Ground Truth"], df_record["Model Prediction"], average='micro')
    # Calculate macro metrics
    macro_precision = precision_score(df_record["Ground Truth"], df_record["Model Prediction"], average='macro')

    print("\n\nMulit-class:")
    print("Micro Precision:", round(micro_precision, 4))
    print("Macro Precision:", round(macro_precision, 4))
    print(f"F1 score: {round(score, 4)}")
    print('accuracy', round(accuracy*100, 2))
    plt.title("Confusion Matrix (Accuracy: %s%%)" % round(accuracy*100, 2))
    plt.xlabel('Model Predicted')
    plt.ylabel('Ground Truth')
    plt.savefig(args.pic_name)

def plot_binary(y_true, y_pred):
    plt.clf()

    df = {'y_true':y_true,  'y_pred':y_pred}
    df = pd.DataFrame(df)

    df['y_predicted_binary'] = df['y_pred'].apply(lambda x: 1 if x >= 3 else 0)
    df['y_true_binary'] = df['y_true'].apply(lambda x: 1 if x >= 3 else 0)


    # 計算混淆矩陣
    cm = confusion_matrix(df['y_true_binary'], df['y_predicted_binary'], labels=[0, 1])

    # 轉換為DataFrame以更容易地可視化
    cm_df = pd.DataFrame(cm, index=['Failed', 'Pass'], columns=['Failed', 'Pass'])

    # 使用Seaborn創建熱圖
    sns.heatmap(cm_df, annot=True, fmt='g', cmap='Blues')

    accuracy = accuracy_score(df['y_true_binary'], df['y_predicted_binary'])
    score = f1_score(df["y_true_binary"], df["y_predicted_binary"], average="weighted")
    # Calculate micro metrics
    micro_precision = precision_score(df["y_true_binary"], df["y_predicted_binary"], average='micro')
    macro_precision = precision_score(df["y_true_binary"], df["y_predicted_binary"], average='macro')

    print("\n\nBinary:")
    print("Micro Precision:", round(micro_precision, 4))
    print("Macro Precision:", round(macro_precision, 4))
    print('F1 score:', round(score, 4))
    print('Accuracy:', round(accuracy*100, 2))


    plt.title("Confusion Matrix (Accuracy: %s%%)" % round(accuracy*100, 2))
    plt.xlabel('Predicted')
    plt.ylabel('Ground Truth')
    plt.savefig(args.bin_pic_name)

def cal_parameter():
    total_params = sum(
        param.numel() for param in model.parameters()
    )
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    print(f'Total: {total_params}')
    print(f'Trainable: {trainable_params}')

def preprocess(examples):
    target_list = [label_to_id(label, label_list) for label in examples[args.score]]
    # =============== Handcraft Delivery ===================
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
        "delivery_tra": vec
    }
    print("**************************************")

    # ========================= Wav2vec Delivery =================================
    speech_list = [speech_file_to_array_fn(path) for path in examples[args.path_column]]
    result['delivery_emb'] = processor(speech_list, 
                                    sampling_rate=target_sampling_rate,
                                    max_length=max_audio_length,
                                    truncation=True,
                                    padding='max_length',
                                    return_tensors="pt"
                                )['input_values']
    
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

    result['padded_res_emb'] = response_tokens['input_ids']
    result['attention_mask_res'] = response_tokens['attention_mask']

    result['padded_pro_emb'] = prompt_tokens['input_ids']
    result['attention_mask_pro'] = prompt_tokens['attention_mask']

    result['labels'] = target_list

    with torch.no_grad():
        logits = model(        
            padded_res_emb=result['padded_res_emb'].to(device),
            attention_mask_res=result['attention_mask_res'].to(device),
            padded_pro_emb=result['padded_pro_emb'].to(device),
            attention_mask_pro=result['attention_mask_pro'].to(device),
            padded_seq=torch.tensor(result['padded_seq']).to(device),
            masked_seq=torch.tensor(result['masked_seq']).to(device),
            delivery_tra=torch.tensor(result['delivery_tra'], dtype=torch.float32).to(device),
            #input_emb=torch.tensor(result['delivery_emb']).to(device),
            ).logits

    global holistic_list

    holistic_list.append(logits.cpu().detach().numpy())
    #print('holistic_list', holistic_list)

    softmax = nn.Softmax(dim=1)
    holistic_logits = softmax(logits)
    holistic_pred = torch.argmax(holistic_logits, dim=-1).detach().cpu().numpy()

    examples["holistic_predicted"] = holistic_pred
    examples['y_true'] = target_list

    return examples


# np.random.seed(42)
# def random_round(scores):
#     round_up = np.random.rand(*scores.shape) < 0.5 
#     return np.where(round_up, np.ceil(scores), np.floor(scores))

import math
def dataset_prepare(file_path):
    df = pd.read_csv(file_path)
    # df[args.score] = random_round(df[args.score])
    df[args.score] = df[args.score].apply(math.floor)     # 3.5 -> 3.0
    df = Dataset.from_dict(df)
    return df

def dataset_prepare_fromHF(dataset_name, split="fulltest"):
    """
    從 Hugging Face Hub 讀取資料集
    Args:
        dataset_name: Hugging Face Hub 上的資料集名稱
        split: 要讀取的資料split (train/validation/test/fulltest)
    """
    dataset = load_dataset(dataset_name)
    df = dataset[split]
    
    # baseline - 將分數無條件捨去
    df = df.map(lambda x: {"grade": math.floor(x["grade"])})
    
    return df

import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda_id", type=int, default=1, help="cuda device")
    parser.add_argument('--test_file',     help='path')
    parser.add_argument('--model_path',     help='path')
    parser.add_argument('--output_path',     help='path')
    parser.add_argument('--logit_name',     help='path')
    parser.add_argument('--pic_name',     help='path')
    parser.add_argument('--bin_pic_name', help='path')
    parser.add_argument('--output_holistic_name', help='path')
    parser.add_argument('--output_content_name', help='path')
    parser.add_argument('--output_delivery_name', help='path')
    parser.add_argument('--output_langUse_name', help='path')

    parser.add_argument('--asr_transcrip',    default='asr_transcription', help='text_value')
    parser.add_argument('--path_column',    default='wav_path', help='path_value')
    parser.add_argument('--example_column',    default='example', help='text_value')
    parser.add_argument('--description',    default='prompt', help='description')    
    parser.add_argument('--score',   default='grade', help='score_value')
    parser.add_argument('--dataset_name', default="ntnu-smil/Unseen_1964", help='huggingface dataset name')
    # parser.add_argument('--ques_id',        type=int, default=100, help='ques_id')

    args = parser.parse_args()

    # ------------------------Cuda Device------------------------------
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(args.cuda_id, device)

    # --------------------------- Loading Data------------------------------------
    if args.test_file:
        test_df = dataset_prepare(args.test_file)
    else:
        test_df = dataset_prepare_fromHF(args.dataset_name, "fulltest")

    label_list2 = test_df.unique(args.score)
    label_list = [1, 2, 3, 4, 5] 
    label_list.sort()
    label_list2.sort()
    num_labels = len(label_list)
    print('Actually label list', label_list2)
    print(f"A classification problem with {num_labels} classes: {label_list}")
    
    # --------------------------- BERT FeatureExtractor ------------------------------------
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base')
    
    # ======================= Model Definition ===================================
    from multi_subModule import *
    # from multi_model_OLL import *
    # from multi_singleModule import *


    model = MultiFeatureModel(device, num_labels=num_labels)
    state_dict = load_file(args.model_path + '/model.safetensors')
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    result = test_df.map(
        preprocess,
        batch_size=2,
        batched=True,
        # remove_columns=['score', 'classification', 'description', 'example', 'confidence_score', 'acoustic_score', 'lm_score', 'pitch', 'intensity', 'pause(sec)', 'silence(sec)', 'duration(sec)', 'asr_transcrip']
        # remove_columns=['form_id', 'grade', 'asr_transcription', 'prompt', 'confidence_score', 'acoustic_score', 'lm_score', 'pitch', 'intensity', 'pause(sec)', 'silence(sec)', 'duration(sec)']
        remove_columns=['example', 'wpm', 'total_num_word', 'level_1', 'level_2', 'level_3', 'seq_len', 'key1', 'key2', 'key3', 'key4', 'avg', 'all_avg_score', 'Threshold_Count', 'mean_pitch', 'mean_intensity', 'duration', 'localJitter', 'localShimmer', 'rapJitter', 'long_silence', 'silence', 'long_silence_num', 'silence_num', 'std_energy', 'avg_spectral', 'avg_energy_entropy', 'zero_cross_num', 'v_to_uv_ratio', 'voice_count', 'unvoice_count', 'mean_long_silence', 'mean_silence', 'more3word', 'num_word', 'whisperX_transcription', 'delivery_vec']
        # remove_columns=['form_id', 'grade', 'asr_transcription', 'prompt', 'confidence_score', 'acoustic_score', 'lm_score', 'pitch', 'intensity', 'pause(sec)', 'silence(sec)', 'duration(sec)']

    )
    print(result)
    df = {
        'speaker_id': result['speaker_id'],
        'y_true':result['y_true'],
        # 'content_pred': result["content_predicted"],
        # 'delivery_pred': result["delivery_predicted"],
        # 'langUse_pred': result["langUse_predicted"],
        'holistic_pred': result["holistic_predicted"],
    }

    # 假设 logits_datasets 是一个包含多个 NumPy 数组的列表
    logits_datasets = [holistic_list]#, content_list, delivery_list, langUse_list]  # 每个元素是一个数据集的 logits
    tsne_results = []
    # print(len(holistic_list), len(holistic_list[0]))
    global_logits_array = np.vstack(holistic_list)
    np.save(args.logit_name, global_logits_array)

    for logits in logits_datasets:
        temp = np.concatenate(logits, axis=0)
        tsne = TSNE(n_components=2, random_state=42, perplexity=9)
        tsne_result = tsne.fit_transform(temp) # [90, 2]
        tsne_results.append(tsne_result)

    list_x_max = []
    list_x_min = []
    list_y_max = []
    list_y_min = []

    for submodule in tsne_results:
        # submodule: [90, 2]
        list_x_max.append(submodule[:, 0].max())
        list_x_min.append(submodule[:, 0].min())
        list_y_max.append(submodule[:, 1].max())
        list_y_min.append(submodule[:, 1].min())

    # all_x = np.concatenate([result[:, 0] for results in tsne_results])
    # all_y = np.concatenate([result[:, 1] for results in tsne_results])
    print(list_x_max)
    x_min, x_max = min(list_x_min), max(list_x_max)
    y_min, y_max = min(list_y_min), max(list_y_max)

    x_margin = (x_max - x_min) * 0.05
    y_margin = (y_max - y_min) * 0.05

    x_min -= x_margin
    x_max += x_margin
    y_min -= y_margin
    y_max += y_margin

    print(x_min, x_max, y_min, y_max, x_margin, y_margin)

    # speaker_id
    col_order = ['speaker_id', 'y_true', 'holistic_predicted']#, 'content_predicted', 'delivery_predicted', 'langUse_predicted', 'wav_path']
    output_df = pd.DataFrame.from_dict(result)
    output_df = output_df[col_order]
    output_df = output_df.rename_axis('index')
    output_df.to_csv(args.output_path)

    plot_confusion_matrix(y_true=result['y_true'], y_pred=result['holistic_predicted'])
    plot_binary(y_true=result['y_true'], y_pred=result['holistic_predicted'])
    plot_tsne(holistic_list, result["y_true"], args.output_holistic_name, x_min, x_max, y_min, y_max)
    # plot_tsne(holistic_list, result["holistic_predicted"], args.output_holistic_name, x_min, x_max, y_min, y_max)

    # plot_tsne(content_list, result['y_true'], args.output_content_name, x_min, x_max, y_min, y_max)
    # plot_tsne(delivery_list, result['y_true'], args.output_delivery_name, x_min, x_max, y_min, y_max)
    # plot_tsne(langUse_list, result['y_true'], args.output_langUse_name, x_min, x_max, y_min, y_max)

