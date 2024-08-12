import numpy as np
import sys
sys.path.append('/share/nas165/peng/thesis_project/ablation_0417/model')
from constants import POS, morph, DEP
import spacy
import re

def label_to_id(label, label_list):
    if len(label_list) > 0:
        return float(label_list.index(label)) if label in label_list else -1.0
    return label


# ======================= Content ============================
chars_to_ignore_regex = '[0-9,?!.;:"-]'
def combineContext(text):
    text = (re.sub(chars_to_ignore_regex, '', text.replace('\n', '')) + " ").strip()
    return text

# ======================= Language Use Preprocessing ============================
nlp = spacy.load("en_core_web_sm")
max_length = 700
def text2BinVec(sentence):
    bin_vec = []
    # Each word -> [1 * 247], 10 words -> [10 * 247]
    for token in nlp(sentence):
        dep_feature = []
        pos_feature = []
        morph_features = []

        # print(f'{token}: {token.dep_}, {token.pos_}, {token.morph}')
        dep_feature.append(token.dep_)
        pos_feature.append(token.pos_)
        morph_features.extend(str(token.morph).split('|'))

        # 轉換成one-hot encoding
        token_pos = [1 if feature in pos_feature else 0 for feature in POS]
        token_dep = [1 if feature in dep_feature else 0 for feature in DEP]
        token_morph = [1 if feature in morph_features else 0 for feature in morph]
        token_bin_vector = token_pos + token_dep + token_morph + [0] * (256 - len(token_pos + token_dep + token_morph))
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


def pad_delivery_sequences(examples):
    # 计算最大长度
    max_length = 260
    # 创建一个列表来存储填充后的数组
    padded_arrays = []
    # 对每个数组进行填充
    for array in examples:
        array = np.array(array)
        padding_length = max_length - array.shape[0]
        # 创建一个形状为 (padding_length, feature_size) 的零数组
        padding = np.zeros((padding_length, array.shape[1]))
        # 将原始数组和填充数组垂直堆叠起来
        padded_array = np.vstack((array, padding))
        # 将填充后的数组添加到列表中
        padded_arrays.append(padded_array)
    
    # 返回填充后的数组列表
    return np.array(padded_arrays)
