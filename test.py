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
import os
import re
import warnings
warnings.filterwarnings('ignore')
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# device = torch.device("cuda")

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
                                     # nn.Linear(hidden_dim*2, 1))    
    
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


# ======================= Delivery ================================
class Delivery_BLSTM(nn.Module):
    def __init__(self, hidden_dim=128, num_labels=5, n_layers=2):
        super(Delivery_BLSTM, self).__init__()
        self.embed_dim = 256
        self.num_labels = num_labels
        self.blstm = nn.LSTM(1, hidden_dim, n_layers, bidirectional=True, batch_first=True)
        self.attention = Attention(hidden_dim*2)

    def forward(self, input_emb):
        input_emb = input_emb.clone().detach().to(dtype=torch.float32).requires_grad_(True)

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
content_array = []
langUse_array = []
delivery_array = []
logits_array = []

class MultiFeatureModel(nn.Module):
    def __init__(self, device, embed_dim=128, num_labels=5, hidden_dim=128, pos=POS, morph=morph, dep=DEP):
        super(MultiFeatureModel, self).__init__()
        self.device = device
        self.num_labels = num_labels
        self.embed_dim = embed_dim
        self.content_BLSTM = content_BLSTM().to(device)
        self.content_BLSTM.bert_freeze_feature_extractor()
        self.Delivery_BLSTM = Delivery_BLSTM()
        self.Language_BLSTM = Language_BLSTM(POS, morph, DEP, hidden_dim).to(device)
        self.dense = nn.Sequential(
            nn.Linear(3, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU()
        )

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
   
    def forward(self,
                padded_res_emb,
                attention_mask_res,
                padded_pro_emb,
                attention_mask_pro,
                padded_seq,
                masked_seq,
                # delivery_emb,
                delivery_tra,
                labels=None):

        # 1. Get each aspect subscore
        content_rep, content_subscore = self.content_BLSTM(padded_res_emb, padded_pro_emb, attention_mask_res, attention_mask_pro) # [8, 768], [8, 1]
        del_tra_rep, delivery_subscore = self.Delivery_BLSTM(delivery_tra)           # [8, 256], [batch_size, 1]
        # del_rep, delivery_wav2vec_subscore = self.Delivery_wav2vec(delivery_emb)           # [8, 256], [batch_size, 1]
        lang_rep, langUse_subscore = self.Language_BLSTM(padded_seq, masked_seq) # [8, 256], []

        # 2. Concatenate three of them
        # content_weighted = 0.6
        # langUse_weighted = 0.4
        # delivery_weighted = 0.2
        # print(content_subscore)
        # print(delivery_subscore)
        # print(langUse_subscore)

        # global content_array
        # global langUse_array
        # global delivery_array
        # global logits_array

        # content_array.append(content_subscore)
        # langUse_array.append(delivery_subscore)
        # delivery_array.append(langUse_subscore)

        concat = torch.cat((content_subscore, delivery_subscore, langUse_subscore), dim=1) #[batch, 2]
        # concat = torch.cat((content_subscore, delivery_subscore), dim=1) #[batch, 2]

        # concat = torch.cat((content_subscore, delivery_subscore, delivery_wav2vec_subscore, langUse_subscore), dim=1) #[batch, 2]

        # 3. Input to Classifier
        logits = self.classifier(self.dense(concat))

        loss_holistic = None
        if labels is not None:
            labels = labels.long()
            loss_fct = nn.CrossEntropyLoss()
            loss_holistic = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            
        return TokenClassifierOutput(loss=loss_holistic, logits=logits) 

import librosa
max_audio_length = 16000 * 85  # 假設音訊的最大長度是16000個樣本點
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
    # initial to zeor
    mask = np.zeros((len(one_hot_encoding), max_length))

    # 根據每個序列的實際長度設置mask
    for i, length in enumerate(original_lengths):
        mask[i, :length] = 1

    return np.array(mask)

def label_to_id(label, label_list):
    if len(label_list) > 0:
        return float(label_list.index(label)) if label in label_list else -1.0
    return label

def phi(task_label, true_label):
    if task_label == true_label:
        return 0
    elif abs(task_label - true_label) == 1:
        return 1
    else:
        return 5    

def func_softLabels(label):
    exp_terms = torch.exp(-torch.tensor([phi(i, label) for i in range(num_labels)]))
    soft_label = []
    for i in range(num_labels):
        k = torch.exp(-torch.tensor(phi(i, label))) / exp_terms.sum()
        soft_label.append(k.item())

    return soft_label

chars_to_ignore_regex = '[0-9,?!.;:"-]'
def combineContext(text):
    text = (re.sub(chars_to_ignore_regex, '', text.replace('\n', '')) + " ").strip()
    return text

# =======================Preprocessing===============================
def preprocess(examples):

    # softlabel
    # target_list = [func_softLabels(label_to_id(label, label_list)) for label in examples[args.score]]
    # print(examples['speaker_id'])
    # Label
    target_list = [label_to_id(label, label_list) for label in examples[args.score]]

    vec = []
    for i in range(len(examples[args.score])):  # 假設 examples_size為4，那麼這個迴圈會運行4次
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

    # speech_list = [speech_file_to_array_fn(path) for path in examples[args.path_column]]

    # ==================Language=================================
    texts = [text2BinVec(txt) for txt in examples[args.asr_transcrip]]
    padded_seq = torch.from_numpy(pad_sequence(texts))
    masked_seq = torch.from_numpy(mask_seq(texts))

    # ==================Content=================================
    padded_response_emb = [combineContext(res) for res in examples[args.asr_transcrip]]
    padded_prompt_emb = [combineContext(prompt) for prompt in examples[args.description]]

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
    
    # ========= Wav2vec ============ Delivery =================
    # result['delivery_emb'] = processor(speech_list, 
    #                                 sampling_rate=target_sampling_rate,
    #                                 max_length=max_audio_length,
    #                                 truncation=True,
    #                                 padding='max_length',
    #                                 return_tensors="pt"
    #                             )['input_values']

    with torch.no_grad():
        logits = model(        
            padded_res_emb=result['padded_res_emb'].to(device),
            attention_mask_res=result['attention_mask_res'].to(device),
            padded_pro_emb=result['padded_pro_emb'].to(device),
            attention_mask_pro=result['attention_mask_pro'].to(device),
            padded_seq=result['padded_seq'].to(device),
            masked_seq=result['masked_seq'].to(device),
            # delivery_emb=torch.tensor(result['delivery_emb']).to(device),
            delivery_tra=torch.tensor(result['delivery_tra']).to(device)
            ).logits
    # print(logits)
    # pred_ids = logits.detach().cpu().numpy().flatten()
    pred_ids = torch.argmax(logits, dim=-1).detach().cpu().numpy()
    # target_list = torch.argmax(target_list, dim=-1).detach().cpu().numpy()
    # print("Target List", target_list)
    # print("Predicted", pred_ids)
    examples["y_predicted"] = pred_ids
    examples['y_true'] = target_list

    return examples


import math
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler() # 標準化
columns_to_scale = ['confidence_score', 'acoustic_score', 'lm_score', 'pitch', 'intensity', 'pause(sec)', 'silence(sec)', 'duration(sec)']

def dataset_prepare(file_path):
    df = pd.read_csv(file_path)
    # df['score'] = df['score'].apply(math.floor)     # 3.5 -> 3.0
    df['grade'] = df['grade'].apply(math.floor)
    df = Dataset.from_dict(df)
    
    return df


# ------------------------Confusion Matrix------------------------------
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
def compute_pearson_correlation(y_true, y_pred):
    correlation, _ = pearsonr(y_true, y_pred)

    # Plotting
    plt.figure(figsize=(8, 5))
    plt.scatter(y_true, y_pred, color='blue')
    plt.xlim(1.5, 5.5)
    plt.ylim(1.5, 5.5)
    plt.title(f'Scatter Plot with PCC: {correlation:.2f}')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.grid(True)
    plt.savefig(args.pic_name)

    return {"pearson_correlation": correlation}

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

# ------------------------ Plot ------------------------------
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
def compute_pearson_correlation(y_pred, y_true):

    df_record = {'true':y_true,  'predict':y_pred}
    df_record = pd.DataFrame(df_record)

    correlation, p_value = pearsonr(df_record['predict'], df_record['true'])
    # print(correlation, p_value)
    # Plotting
    # plt.figure(figsize=(8, 5))
    # plt.scatter(y_true, y_pred, color='blue')
    # plt.title(f'Scatter Plot with PCC: {correlation:.2f}')
    # plt.xlabel('Actual Values')
    # plt.ylabel('Predicted Values')
    # plt.grid(True)
    # plt.savefig(args.bin_pic_name)


import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda_id", type=int, default=1, help="cuda device")
    parser.add_argument('--test_file',     help='path')
    parser.add_argument('--model_path',     help='path')
    parser.add_argument('--output_path',     help='path')
    parser.add_argument('--pic_name',     help='path')
    parser.add_argument('--bin_pic_name', help='path')
    parser.add_argument('--asr_transcrip',    default='asr_transcription', help='text_value')
    parser.add_argument('--path_column',    default='wav_path', help='path_value')
    parser.add_argument('--example_column',    default='example', help='text_value')
    parser.add_argument('--description',    default='prompt', help='description')    
    parser.add_argument('--score',   default='grade', help='score_value')
    # parser.add_argument('--ques_id',        type=int, default=100, help='ques_id')

    args = parser.parse_args()

    # ------------------------Cuda Device------------------------------
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(args.cuda_id, device)

    # --------------------------- Loading Data------------------------------------
    test_df = dataset_prepare(args.test_file)

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
    model = MultiFeatureModel(device, pos=POS, morph=morph, dep=DEP, num_labels=num_labels)
    finetune_model = torch.load(args.model_path + '/pytorch_model.bin')
    model.load_state_dict(finetune_model, strict=False)
    model.to(device)
    model.eval()

    result = test_df.map(
        preprocess,
        batch_size=4,
        batched=True,
        # remove_columns=['score', 'classification', 'description', 'example', 'confidence_score', 'acoustic_score', 'lm_score', 'pitch', 'intensity', 'pause(sec)', 'silence(sec)', 'duration(sec)', 'asr_transcrip']
        remove_columns=['form_id', 'grade', 'asr_transcription', 'prompt', 'confidence_score', 'acoustic_score', 'lm_score', 'pitch', 'intensity', 'pause(sec)', 'silence(sec)', 'duration(sec)']
    
    )

    df = {
        'speaker_id': result['speaker_id'],
        'y_true':result['y_true'],
        'y_predicted':result['y_predicted']
    }


    # speaker_id
    col_order = ['speaker_id', 'y_predicted', 'y_true', 'wav_path']
    output_df = pd.DataFrame.from_dict(result)
    output_df = output_df[col_order]
    output_df = output_df.rename_axis('index')
    output_df.to_csv(args.output_path)

    plot_confusion_matrix(y_true=result['y_true'], y_pred=result['y_predicted'])
    plot_binary(y_true=result['y_true'], y_pred=result['y_predicted'])
