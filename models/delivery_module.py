import torch
import torch.nn as nn
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2Model, Wav2Vec2Processor
# from content_module import Attention
from transformer_model import *

# 定义一个简单的CNN模型
class AudioCNN(nn.Module):
    def __init__(self):
        super(AudioCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=768, out_channels=128, kernel_size=5, stride=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        # self.fc = nn.Linear(128, 10)  # 假设有10个类别

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        # x = x.view(x.size(0), -1)  # 扁平化
        # x = self.fc(x)
        return x


# ======================= Delivery =====================================
class Delivery_wav2vec(nn.Module):
    def __init__(self, hidden_dim=768, n_layers=2):
        super(Delivery_wav2vec, self).__init__()
        # self.attention = Attention(hidden_dim)
        self.wav2vecModel = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base')#, config=self.wav2vecConfig)
        self.audioCnn = AudioCNN()


    def wav2vec_freeze_feature_extractor(self):
        for param in self.wav2vecModel.parameters():
            param.requires_grad = False

    def forward(self, input_emb):
        audio_encode = self.wav2vecModel(input_emb).last_hidden_state # [batch, 4499, 768]
        
        # Into CNN layer 
        # cnn_input = audio_encode.permute(0, 2, 1)
        # cnn_output = self.audioCnn(cnn_input) # (batch_size, out_channels, new_sequence_length)
        # print(f"Delivery CNN output {cnn_output.size()}")

        return audio_encode

# ======================= Delivery Use =====================================
class Delivery_BLSTM(nn.Module):
    def __init__(self, hidden_dim=14, num_features=256, n_layers=2):
        super(Delivery_BLSTM, self).__init__()
        self.conv1d_delivery = nn.Conv1d(in_channels=hidden_dim, out_channels=num_features, kernel_size=1, padding=0, bias=False) # audio
        self.transformer_encode_delivery = Transformer(3, 256, 4) # num_layers, d_model, num_heads


    def forward(self, input_emb):
        input_emb = input_emb.clone().detach().to(dtype=torch.float32)
        
        # 1. Conv1D
        cnn_input = input_emb.permute(0, 2, 1)
        cnn_delivery = self.conv1d_delivery(cnn_input).transpose(1, 2) # [batch, length, 256]

        # 2. Multi-head self attention
        x_delivery = self.transformer_encode_delivery(cnn_delivery) # [8, 260, 256]

        # 3. 
        # print(f'Delivery: {delivery.size()}')

        return x_delivery