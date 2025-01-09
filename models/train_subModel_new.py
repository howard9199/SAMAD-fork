
# =======================Preprocessing===============================
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
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2Model, Wav2Vec2Processor
# from transformers import WhisperProcessor, WhisperModel
import os
import re
import warnings
warnings.filterwarnings('ignore')
from text_preprocessing import *
import json
torch.autograd.set_detect_anomaly(True)
# Custom Trainer
from transformers import Trainer
class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(CustomTrainer, self).__init__(*args, **kwargs)

        # Create separate optimizers for each sub-module
        self.optimizer_content = torch.optim.Adam(self.model.content_BLSTM.parameters(), lr=self.args.learning_rate)
        self.optimizer_delivery = torch.optim.Adam(self.model.Delivery_BLSTM.parameters(), lr=self.args.learning_rate)
        self.optimizer_langUse = torch.optim.Adam(self.model.Language_BLSTM.parameters(), lr=self.args.learning_rate)

        # Get parameters not in the sub-modules for holistic optimizer
        sub_module_params = (
            list(self.model.content_BLSTM.parameters()) +
            list(self.model.Delivery_BLSTM.parameters()) +
            list(self.model.Language_BLSTM.parameters())
        )
        holistic_params = self.model.parameters()
        self.optimizer_holistic = torch.optim.Adam(holistic_params, lr=self.args.learning_rate)

    def training_step(self, model, inputs, num_items_in_batch):
        model.train()
        inputs = self._prepare_inputs(inputs)

        # Forward pass
        outputs = model(**inputs)

        # Compute losses
        loss_holistic, loss_content, loss_delivery, loss_langUse = compute_loss_func(
            outputs, inputs['labels'], num_items_in_batch=inputs['labels'].size(0)
        )

        # Backward and optimization for content_BLSTM
        is_not_update = True
        '''
        if torch.isnan(loss_content) == False:
            self.optimizer_content.zero_grad()
            loss_content.backward()
            self.optimizer_content.step()
            is_not_update = False

        # Backward and optimization for Delivery_BLSTM
        if torch.isnan(loss_delivery) == False:
            self.optimizer_delivery.zero_grad()
            loss_delivery.backward()
            self.optimizer_delivery.step()
            is_not_update = False

        # Backward and optimization for Language_BLSTM
        if torch.isnan(loss_langUse) == False:
            self.optimizer_langUse.zero_grad()
            loss_langUse.backward()
            self.optimizer_langUse.step()
            is_not_update = False
        '''
        # Backward and optimization for the holistic part
        if is_not_update:
            if loss_holistic is not None:
                self.optimizer_holistic.zero_grad()
                loss_holistic.backward()
                self.optimizer_holistic.step()
        # Return the holistic loss for logging
        return loss_holistic.detach()
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        model.eval()
        with torch.no_grad():
            inputs = self._prepare_inputs(inputs)
            outputs = model(**inputs)
            # Compute losses
            loss_holistic, loss_content, loss_delivery, loss_langUse = compute_loss_func(
                outputs, inputs['labels'], num_items_in_batch=inputs['labels'].size(0)
            )
            # Return the total loss
            loss = loss_holistic.detach()
            logits = outputs[0]
            labels = inputs.get("labels")
        return (loss, logits, labels)

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

    #target_list = [label_to_id(label, label_list) for label in examples['grade']]
    # 讓 target_list 是 grade, delivery, relevance, language
    target_list = [
        [label_to_id(label, label_list) for label in examples['grade']],
        [label_to_id(label, label_list) for label in examples['delivery']],
        [label_to_id(label, label_list) for label in examples['relevance']],
        [label_to_id(label, label_list) for label in examples['language']]
    ]
    target_list = list(zip(*target_list))


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
    result['delivery_emb'] = processor(speech_list, 
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

from sklearn.metrics import accuracy_score
def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    references = eval_pred.label_ids
    # use only the first element of the tuple
    references = [item[0] for item in references]
    # convert predictions and references to int
    predictions = [int(item) for item in predictions]
    references = [int(item) for item in references]
    acc = accuracy_score(references, predictions)
    return {"accuracy": float(acc)}

def compute_loss_func(outputs, labels, num_items_in_batch):
    # outputs has (holistic, subscore_content, subscore_delivery, subscore_langUse)
    # labels has (label_holistic, label_content, label_delivery, label_langUse)
    '''
    overall_loss = None
    loss_fct = nn.CrossEntropyLoss()
    if labels is not None:
        labels = labels.long().to('cpu')
        loss_holistic = loss_fct(outputs.view(-1, self.num_labels), labels.view(-1))
        overall_loss = loss_holistic
        overall_loss = overall_loss.to('cuda')
    '''
    loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
    # calculate the loss for each sub-module and holistic
    holistic_output = outputs[0].view(-1, num_labels).to('cuda')
    holistic_labels = labels[:, 0].view(-1).long().to('cuda')
    #print('holistic_output', holistic_output)
    #print('holistic_labels', holistic_labels)
    
    content_output = outputs[1].view(-1, num_labels).to('cuda')
    content_labels = labels[:, 1].view(-1).long().to('cuda')
    
    delivery_output = outputs[2].view(-1, num_labels).to('cuda')
    delivery_labels = labels[:, 2].view(-1).long().to('cuda')
    
    langUse_output = outputs[3].view(-1, num_labels).to('cuda')
    langUse_labels = labels[:, 3].view(-1).long().to('cuda')
    

    loss_holistic = loss_fct(holistic_output, holistic_labels)
    loss_content = loss_fct(content_output, content_labels)
    loss_delivery = loss_fct(delivery_output, delivery_labels)
    loss_langUse = loss_fct(langUse_output, langUse_labels)
    #print('holistic_output', holistic_output)
    #print('holistic_labels', holistic_labels)
    #print('content_output', content_output)
    #print('content_labels', content_labels)
    #print('loss_holistic', loss_holistic)
    #print('loss_content', loss_content)
    #print('loss_delivery', loss_delivery)
    #print('loss_langUse', loss_langUse)
    #print(loss_holistic + loss_content + loss_delivery + loss_langUse)
    return loss_holistic, loss_content, loss_delivery, loss_langUse


# soft label
# from sklearn.metrics import accuracy_score
# def compute_metrics(eval_pred):
#     predictions = np.argmax(eval_pred.predictions, axis=1)
#     references = np.argmax(eval_pred.label_ids, axis=1) # softlabel才需這樣算！
#     acc = accuracy_score(references, predictions)
    # return {"accuracy": float(acc)}
    
np.random.seed(42)
def random_round(scores):
    round_up = np.random.rand(*scores.shape) < 0.5 
    return np.where(round_up, np.ceil(scores), np.floor(scores))


import math
def dataset_prepare(file_path):
    df = pd.read_csv(file_path)
    # random score
    # df[args.score] = random_round(df[args.score])
    # baseline
    df[args.score] = df[args.score].apply(math.floor)     # 3.5 -> 3.0
    df = Dataset.from_dict(df)
    return df

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda_id", type=int, default=0, help="cuda device")
    parser.add_argument('--train_file',     help='path')
    parser.add_argument('--dev_file',       help='path')
    parser.add_argument('--output_dir',     help='path')
    # parser.add_argument('--asr_transcrip',    default='asr_transcription', help='text_value')
    parser.add_argument('--asr_transcrip', default='whisperX_transcription', help='text_value')

    parser.add_argument('--description',    default='prompt', help='description')
    parser.add_argument('--score',   default='grade', help='score_value')
    parser.add_argument('--path_column',    default='wav_path', help='path_value')
    parser.add_argument('--train_epochs',   type=int, default=8, help='epochs')
    parser.add_argument('--train_batch',    type=int, default=2, help='epochs')
    parser.add_argument('--eval_batch',     type=int, default=2, help='batch')
    parser.add_argument('--grad_acc',       type=int, default=4, help='grad_acc')
    parser.add_argument('--learning_rate',  type=float, default=1e-3, help='grad_acc')
    parser.add_argument('--softlabel_para', type=float, default=0.8, help='score_value')

    args = parser.parse_args()
    # ------------------------Cuda Device------------------------------
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print(device, str(args.cuda_id))

    # --------------------------- Loading Data------------------------------------
    train_df = dataset_prepare(args.train_file)
    dev_df = dataset_prepare(args.dev_file)

    label_list2 = train_df.unique(args.score)
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
    # --------------------------Data Preparation---------------------------
    '''
    train_df = train_df.map(
        preprocess,
        batch_size=2,
        batched=True,
        # remove_columns=['speaker_id', 'score', 'classification', 'description', 'example', 'confidence_score', 'acoustic_score', 'lm_score', 'pitch', 'intensity', 'pause(sec)', 'silence(sec)', 'duration(sec)', 'asr_transcrip', 'wav_path']
        # remove_columns=['speaker_id', 'form_id', 'grade', 'wav_path', 'asr_transcription', 'prompt', 'confidence_score', 'acoustic_score', 'lm_score', 'pitch', 'intensity', 'pause(sec)', 'silence(sec)', 'duration(sec)']
        # WhisperX
        remove_columns=['example', 'wpm', 'total_num_word', 'level_1', 'level_2', 'level_3', 'seq_len', 'key1', 'key2', 'key3', 'key4', 'avg', 'all_avg_score', 'Threshold_Count', 'mean_pitch', 'mean_intensity', 'duration', 'localJitter', 'localShimmer', 'rapJitter', 'long_silence', 'silence', 'long_silence_num', 'silence_num', 'std_energy', 'avg_spectral', 'avg_energy_entropy', 'zero_cross_num', 'v_to_uv_ratio', 'voice_count', 'unvoice_count', 'mean_long_silence', 'mean_silence', 'more3word', 'num_word', 'whisperX_transcription', 'delivery_vec']
        # HI 
        # remove_columns=['mean_pitch', 'mean_intensity', 'duration', 'localJitter', 'localShimmer', 'rapJitter', 'long_silence', 'silence', 'long_silence_num', 'silence_num', 'std_energy', 'avg_spectral', 'avg_energy_entropy', 'zero_cross_num', 'v_to_uv_ratio', 'voice_count', 'unvoice_count', 'more3word', 'num_word', 'whisperX_transcription', 'delivery_vec']
    )
    '''
    '''
    dev_df = dev_df.map(
        preprocess,
        batch_size=2,
        batched=True,
        # remove_columns=['speaker_id', 'score', 'classification', 'description', 'example', 'confidence_score', 'acoustic_score', 'lm_score', 'pitch', 'intensity', 'pause(sec)', 'silence(sec)', 'duration(sec)', 'asr_transcrip', 'wav_path']
        # remove_columns=['speaker_id', 'form_id', 'grade', 'wav_path', 'asr_transcription', 'prompt', 'confidence_score', 'acoustic_score', 'lm_score', 'pitch', 'intensity', 'pause(sec)', 'silence(sec)', 'duration(sec)']
        # WhisperX
        remove_columns=['example', 'wpm', 'total_num_word', 'level_1', 'level_2', 'level_3', 'seq_len', 'key1', 'key2', 'key3', 'key4', 'avg', 'all_avg_score', 'Threshold_Count', 'mean_pitch', 'mean_intensity', 'duration', 'localJitter', 'localShimmer', 'rapJitter', 'long_silence', 'silence', 'long_silence_num', 'silence_num', 'std_energy', 'avg_spectral', 'avg_energy_entropy', 'zero_cross_num', 'v_to_uv_ratio', 'voice_count', 'unvoice_count', 'mean_long_silence', 'mean_silence', 'more3word', 'num_word', 'whisperX_transcription', 'delivery_vec']
        # HI
        # remove_columns=['mean_pitch', 'mean_intensity', 'duration', 'localJitter', 'localShimmer', 'rapJitter', 'long_silence', 'silence', 'long_silence_num', 'silence_num', 'std_energy', 'avg_spectral', 'avg_energy_entropy', 'zero_cross_num', 'v_to_uv_ratio', 'voice_count', 'unvoice_count', 'more3word', 'num_word', 'whisperX_transcription', 'delivery_vec']

    )
    '''
    # print the labels of the first 5 examples in the dev_df
    #print(dev_df['labels'][:5])
    #dev_df.save_to_disk('/datas/store162/howard/samad/dataframe/dev_sort_df')
    #train_df.save_to_disk('/datas/store162/howard/samad/dataframe/train_sort_df')
    train_df = Dataset.load_from_disk('/datas/store162/howard/samad/dataframe/train_sort_df')
    dev_df = Dataset.load_from_disk('/datas/store162/howard/samad/dataframe/dev_sort_df')
    # 讓沒有 delivery,relevance,language 的資料放在最前面，不要用 sort
    def move_missing_columns_to_front(dataset, columns):
        def has_missing_columns(example):
            return any(pd.isna(example[col]) for col in columns)

        missing_indices = [i for i, example in enumerate(dataset) if has_missing_columns(example)]
        non_missing_indices = [i for i in range(len(dataset)) if i not in missing_indices]

        new_order = missing_indices + non_missing_indices
        return dataset.select(new_order)

    columns_to_check = ['delivery', 'relevance', 'language']
    #dev_df = move_missing_columns_to_front(dev_df, columns_to_check)
    #train_df = move_missing_columns_to_front(train_df, columns_to_check)

    # save to disk
    #dev_df.save_to_disk('/datas/store162/howard/samad/dataframe/dev_sort_df')
    #train_df.save_to_disk('/datas/store162/howard/samad/dataframe/train_sort_df')
    # show the first 5 rows of the train_df and dev_df, show only delivery, relevance, language
    #print(train_df.select(['delivery', 'relevance', 'language']).to_pandas().head())
    #print(dev_df.select_columns(['delivery']).to_pandas())
    #print(dev_df.select_columns(['relevance']).to_pandas())
    #print(dev_df.select_columns(['language']).to_pandas())


    #print('train_df', train_df)
    #print("dev_df", dev_df)

    # save the processed data to /datas/store162/howard/samad/dataframe/
    #train_df.save_to_disk('/datas/store162/howard/samad/dataframe/train_df')
    #dev_df.save_to_disk('/datas/store162/howard/samad/dataframe/dev_df')

    #train_df = Dataset.load_from_disk('/datas/store162/howard/samad/dataframe/train_df')
    #dev_df = Dataset.load_from_disk('/datas/store162/howard/samad/dataframe/dev_df')

    

    # ------------------------Model------------------------------
    from multi_subModule import *
    
    model = MultiFeatureModel(device, num_labels=num_labels)
    model.to(device)
    cal_parameter()

    # ======================= Trainer ===================================
    from transformers import TrainingArguments, Trainer
    import wandb
    wandb.init()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.train_batch,   # 32
        per_device_eval_batch_size=args.eval_batch,     # 32
        gradient_accumulation_steps=args.grad_acc,
        evaluation_strategy="steps",
        save_steps=10,
        eval_steps=10,
        num_train_epochs=args.train_epochs,
        logging_steps=8,
        learning_rate=args.learning_rate,
        save_total_limit=5,    
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_df,
        eval_dataset=dev_df,
        compute_loss_func=compute_loss_func,
    )
    trainer.train()
