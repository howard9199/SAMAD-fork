import pandas as pd
from datasets import Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk
import numpy as np
from constants import POS, morph, DEP
import spacy
from transformers.modeling_outputs import TokenClassifierOutput
from transformers import BertTokenizer, BertModel
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2Model, Wav2Vec2Processor
# from transformers import WhisperProcessor, WhisperModel
import os
import re
import warnings
warnings.filterwarnings('ignore')

# ======================= Attention =====================================
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.fc = nn.Linear(hidden_dim, 1)
        # self.fc2 = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        scores = self.fc(x) 
        attention_weights = F.softmax(scores, dim=1)
        attention = x * attention_weights
        representation = attention.sum(dim=1)
        subscore = self.fc(representation)

        return representation, subscore


# ======================= BERT =====================================
class content_BLSTM(nn.Module):
    def __init__(self, hidden_dim=768*2):
        super(content_BLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.bert_model = BertModel.from_pretrained("bert-base-uncased")
        self.attention = Attention(hidden_dim)

    def bert_freeze_feature_extractor(self):
        for param in self.bert_model.parameters():
            param.requires_grad = False
    
    def forward(self, 
                padded_res_emb, 
                padded_pro_emb,
                attention_mask_res, 
                attention_mask_pro): # labels=None

        # 1. v^p
        x_prompt  = self.bert_model(padded_pro_emb, attention_mask_pro).last_hidden_state # [batch_size, length, 768]
        cls_prompt = x_prompt[:, 0, :] # [batch_size, 768]
        cls_prompt_expanded = cls_prompt.unsqueeze(1).expand(-1, 256, -1)  # [batch_size, 256, 768]
        
        # 2. Each "word" need to concatenate with v^p
        res_emb = self.bert_model(padded_res_emb, attention_mask_res).last_hidden_state # [2, 256, 768]
        res_emb_with_cls = torch.cat([res_emb, cls_prompt_expanded], dim=2)  # [batch_size, 256, 768*2=1536]

        # 3. Attention
        representation, subscore = self.attention(res_emb_with_cls)
        
        return representation, subscore


# ======================= Delivery Use =====================================
class Delivery_BLSTM(nn.Module):
    def __init__(self, hidden_dim=128, n_layers=2):
        super(Delivery_BLSTM, self).__init__()
        self.embed_dim = 256
        self.blstm = nn.LSTM(1, hidden_dim, n_layers, bidirectional=True, batch_first=True)
        self.attention = Attention(hidden_dim*2)

    def forward(self, input_emb):
        input_emb = input_emb.to(device)

        lstm_out, _ = self.blstm(input_emb.unsqueeze(2))    # [64, 8, 256]
        representation, subscore = self.attention(lstm_out) # [64, 256]

        return representation, subscore

# ======================= Language Use =====================================
nlp = spacy.load("en_core_web_sm")
class Language_BLSTM(nn.Module):
    def __init__(self, pos=POS, morph=morph, dep=DEP, hidden_dim=128):
        super(Language_BLSTM, self).__init__()
        self.pos = pos
        self.morph = morph
        self.dep = dep
        self.embedding_dim = len(self.pos) + len(self.morph) + len(self.dep)
        self.blstm = nn.LSTM(self.embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.attention = Attention(hidden_dim*2)


    def forward(self, 
                padded_seq,
                masked_seq):

        # 1. Transfer to tensor
        padded_seq_tensor = padded_seq.clone().detach().to(dtype=torch.float32).requires_grad_(True) # [batch, 600, 256]
        masks_tensor = masked_seq.clone().detach().to(dtype=torch.float32).requires_grad_(True) # [batch, 600]
        
        # 2. BLSTM
        lstm_out, _ = self.blstm(padded_seq_tensor)             # [batch, 600]
        
        # 3. Masking
        lstm_out_masked = lstm_out * masks_tensor.unsqueeze(-1) # [batch, 600] * [batch, 600, 1]
        
        representation, subscore = self.attention(lstm_out_masked)

        return representation, subscore

# ==============================================================================
class MultiFeatureModel(nn.Module):
    def __init__(self, device, embed_dim=128, num_labels=5, hidden_dim=128, pos=POS, morph=morph, dep=DEP):
        super(MultiFeatureModel, self).__init__()
        self.device = device
        self.num_labels = num_labels
        self.embed_dim = embed_dim
        self.content_BLSTM = content_BLSTM().to(device)
        self.content_BLSTM.bert_freeze_feature_extractor()
        self.Delivery_BLSTM = Delivery_BLSTM()
        # self.Delivery_wav2vec = Delivery_wav2vec().to(device)
        # self.Delivery_wav2vec.wav2vec_freeze_feature_extractor()
        # self.Delivery_wav2vec = Delivery_wav2vec().to(device)
        # self.Delivery_wav2vec.whisper_freeze_feature_extractor()
        self.Language_BLSTM = Language_BLSTM(POS, morph, DEP, hidden_dim).to(device)
        # self.classifier_content = nn.Sequential(
        #     nn.Linear(1, self.embed_dim),
        #     nn.ReLU(),
        #     nn.Linear(self.embed_dim, self.embed_dim),
        #     nn.ReLU(),
        #     nn.Linear(self.embed_dim, self.embed_dim),
        #     nn.ReLU(),
        #     nn.LayerNorm(self.embed_dim),
        #     nn.Dropout(p=0.2),
        #     nn.Linear(self.embed_dim, self.num_labels),                        
        # )
        # self.classifier_delivery = nn.Sequential(
        #     nn.Linear(1, self.embed_dim),
        #     nn.ReLU(),
        #     nn.Linear(self.embed_dim, self.embed_dim),
        #     nn.ReLU(),
        #     nn.Linear(self.embed_dim, self.embed_dim),
        #     nn.ReLU(),
        #     nn.LayerNorm(self.embed_dim),
        #     nn.Dropout(p=0.2),
        #     nn.Linear(self.embed_dim, self.num_labels),                        
        # )
        # self.classifier_delivery_wav2vec = nn.Sequential(
        #     nn.Linear(1, self.embed_dim),
        #     nn.ReLU(),
        #     nn.Linear(self.embed_dim, self.embed_dim),
        #     nn.ReLU(),
        #     nn.Linear(self.embed_dim, self.embed_dim),
        #     nn.ReLU(),
        #     nn.LayerNorm(self.embed_dim),
        #     nn.Dropout(p=0.2),
        #     nn.Linear(self.embed_dim, self.num_labels),                        
        # )
        # self.classifier_langUse = nn.Sequential(
        #     nn.Linear(1, self.embed_dim),
        #     nn.ReLU(),
        #     nn.Linear(self.embed_dim, self.embed_dim),
        #     nn.ReLU(),
        #     nn.Linear(self.embed_dim, self.embed_dim),
        #     nn.ReLU(),
        #     nn.LayerNorm(self.embed_dim),
        #     nn.Dropout(p=0.2),
        #     nn.Linear(self.embed_dim, self.num_labels),                        
        # )
        self.dense = nn.Sequential(
            nn.Linear(3, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(nn.LayerNorm(self.embed_dim),
                                        nn.Dropout(p=0.2),
                                        nn.Linear(self.embed_dim, self.num_labels),
                                        nn.ReLU()
                                    )
    def cal_parameter(self):
        total_params = sum(
	        param.numel() for param in model.parameters()
        )
        trainable_params = sum(
        	p.numel() for p in model.parameters() if p.requires_grad
        )

        print(f'Total: {total_params}')
        print(f'Trainable: {trainable_params}')

    def forward(self,
                padded_res_emb,
                attention_mask_res,
                padded_pro_emb,
                attention_mask_pro,
                padded_seq,
                masked_seq,
                delivery_tra,
                labels=None):
        
        # 1. Get each aspect subscore
        content_rep, content_subscore = self.content_BLSTM(padded_res_emb, padded_pro_emb, attention_mask_res, attention_mask_pro) # [8, 768], [8, 1]
        del_tra_rep, delivery_subscore = self.Delivery_BLSTM(delivery_tra)           # [8, 256], [batch_size, 1]
        lang_rep, langUse_subscore = self.Language_BLSTM(padded_seq, masked_seq) # [8, 256], []

        # 2. Concatenate three of them
        concat = torch.cat((content_subscore, delivery_subscore, langUse_subscore), dim=1) #[batch, 2]

        # 3. Input to Classifier
        logits = self.classifier(self.dense(concat))

        loss_holistic = None
        if labels is not None:
            labels = labels.long()
            loss_fct = nn.CrossEntropyLoss()
            loss_holistic = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            
        return TokenClassifierOutput(loss=loss_holistic, logits=logits) 

import librosa
max_audio_length = 16000 * 60  # 假設音訊的最大長度是16000個樣本點
target_sampling_rate = 16000
def speech_file_to_array_fn(path):
    speech_array, sampling_rate = librosa.load(path, sr=target_sampling_rate)
    mono_waveform = librosa.resample(speech_array, orig_sr=sampling_rate, target_sr=target_sampling_rate)

    return mono_waveform
# =======================Language Use Preprocessing============================
max_length = 700
def text2BinVec(sentence):
    morph_features = []
    dep_feature = []
    pos_feature = []
    bin_vec = []

    # Each word -> [1 * 247], 10 words -> [10 * 247]
    for token in nlp(sentence):
        # print(f'{token}: {token.dep_}, {token.pos_}, {token.morph}')
        dep_feature.append(token.dep_)
        pos_feature.append(token.pos_)
        morph_features.extend(str(token.morph).split('|'))
        
        # 轉換成one-hot encoding
        token_pos = [1 if feature in pos_feature else 0 for feature in POS]
        token_dep = [1 if feature in dep_feature else 0 for feature in DEP]
        token_morph = [1 if feature in morph_features else 0 for feature in morph]
        token_bin_vector = token_pos + token_dep + token_morph
        bin_vec.append(token_bin_vector)

    return np.array(bin_vec)

def pad_sequence(one_hot_encoding):
    padded_sequences = []
    for seq in one_hot_encoding:
        padded_seq = np.pad(seq, pad_width=((0, max_length-seq.shape[0]),(0, 0)), mode='constant', constant_values=0)
        padded_sequences.append(padded_seq)

    return np.array(padded_sequences)

def mask_seq(one_hot_encoding):
    original_lengths = [seq.shape[0] for seq in one_hot_encoding]
    # initialize to zero
    mask = np.zeros((len(one_hot_encoding), max_length))

    # 根據每個序列的實際長度設置mask
    for i, length in enumerate(original_lengths):
        mask[i, :length] = 1

    return np.array(mask)

def label_to_id(label, label_list):
    if len(label_list) > 0:
        return float(label_list.index(label)) if label in label_list else -1.0
    return label


chars_to_ignore_regex = '[0-9,?!.;:"-]'
def combineContext(text):
    text = (re.sub(chars_to_ignore_regex, '', text.replace('\n', '')) + " ").strip()
    return text


# =======================Preprocessing===============================
def preprocess(examples):

    target_list = [label_to_id(label, label_list) for label in examples[args.score]]
    
    # =============== Handcraft Delivery ===================
    vec = []
    for i in range(len(examples[args.score])):
        single_vec = [
             round(examples['confidence_score'][i], 4),
             round(examples['acoustic_score'][i], 4),
             round(examples['lm_score'][i], 4),
             round(examples['pitch'][i], 4),
             round(examples['intensity'][i], 4),
             round(examples['pause(sec)'][i], 4),
             round(examples['silence(sec)'][i], 4),
             round(examples['duration(sec)'][i], 4)
        ]
        vec.append(np.array(single_vec))

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

    return result

from sklearn.metrics import accuracy_score
def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    references = eval_pred.label_ids
    acc = accuracy_score(eval_pred.label_ids, predictions)
    return {"accuracy": float(acc)}

import math
def dataset_prepare(file_path):
    df = pd.read_csv(file_path)
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
    parser.add_argument('--asr_transcrip',    default='asr_transcription', help='text_value')
    parser.add_argument('--description',    default='prompt', help='description')
    # parser.add_argument('--score',   default='score', help='score_value')
    parser.add_argument('--score',   default='grade', help='score_value')
    parser.add_argument('--path_column',    default='wav_path', help='path_value')
    parser.add_argument('--train_epochs',   type=int, default=8, help='epochs')
    parser.add_argument('--train_batch',    type=int, default=2, help='epochs')
    parser.add_argument('--eval_batch',     type=int, default=2, help='batch')
    parser.add_argument('--grad_acc',       type=int, default=4, help='grad_acc')
    parser.add_argument('--learning_rate',  type=float, default=1e-3, help='grad_acc')

    args = parser.parse_args()
    # ------------------------Cuda Device------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    # processor = WhisperProcessor.from_pretrained("openai/whisper-base")
    # --------------------------Data Preparation---------------------------
    train_df = train_df.map(
        preprocess,
        batch_size=100,
        batched=True,
        # remove_columns=['speaker_id', 'score', 'classification', 'description', 'example', 'confidence_score', 'acoustic_score', 'lm_score', 'pitch', 'intensity', 'pause(sec)', 'silence(sec)', 'duration(sec)', 'asr_transcrip', 'wav_path']
        remove_columns=['speaker_id', 'form_id', 'grade', 'wav_path', 'asr_transcription', 'prompt', 'confidence_score', 'acoustic_score', 'lm_score', 'pitch', 'intensity', 'pause(sec)', 'silence(sec)', 'duration(sec)']

    )
    dev_df = dev_df.map(
        preprocess,
        batch_size=100,
        batched=True,
        # remove_columns=['speaker_id', 'score', 'classification', 'description', 'example', 'confidence_score', 'acoustic_score', 'lm_score', 'pitch', 'intensity', 'pause(sec)', 'silence(sec)', 'duration(sec)', 'asr_transcrip', 'wav_path']
        remove_columns=['speaker_id', 'form_id', 'grade', 'wav_path', 'asr_transcription', 'prompt', 'confidence_score', 'acoustic_score', 'lm_score', 'pitch', 'intensity', 'pause(sec)', 'silence(sec)', 'duration(sec)']
    )
    print('train_df', train_df)
    print("dev_df", dev_df)

    # ------------------------Model------------------------------
    model = MultiFeatureModel(device, pos=POS, morph=morph, dep=DEP, num_labels=num_labels)
    model.to(device)
    model.cal_parameter()
    # ======================= Trainer ===================================
    from transformers import TrainingArguments, Trainer
    # import wandb
    # wandb.init(mode="disabled")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.train_batch,   # 32
        per_device_eval_batch_size=args.eval_batch,     # 32
        # gradient_accumulation_steps=args.grad_acc,
        evaluation_strategy="steps",
        save_steps=10,
        eval_steps=10,
        num_train_epochs=args.train_epochs,
        logging_steps=8,
        learning_rate=args.learning_rate,
        save_total_limit=5, # 限制保存的模型检查点的总数，即限制保存模型的历史版本数量
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_df,
        eval_dataset=dev_df,
    )
    trainer.train()