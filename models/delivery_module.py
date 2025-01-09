import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2Model, Wav2Vec2Processor
from transformers.modeling_outputs import TokenClassifierOutput
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
        self.num_labels = 5


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
        
        # Projection layers
        self.num_features = num_features
        self.proj1 = nn.Linear(num_features*1, num_features*2)
        self.proj2 = nn.Linear(num_features*2, num_features*1)
        self.out_layer = nn.Linear(num_features*1, 5)      

        self.trainable = 1  # Default state is 0 (trainable)
        self.num_labels = 5

    def set_state(self, trainable):
        self.trainable = trainable
        if trainable == 0:
            # Freeze all parameters
            for param in self.parameters():
                param.requires_grad = False
            # Remove projection layers
            '''
            self.proj1 = None
            self.proj2 = None
            self.out_layer = None
            '''
        else:
            # Unfreeze all parameters
            for param in self.parameters():
                param.requires_grad = True
            # Recreate projection layers
            self.proj1 = nn.Linear(self.num_features*1, self.num_features*2)
            self.proj2 = nn.Linear(self.num_features*2, self.num_features*1)
            self.out_layer = nn.Linear(self.num_features*1, 5)

    def forward(self, input_emb, labels=None):
        input_emb = input_emb.clone().detach().to(dtype=torch.float32)
        
        # 1. Conv1D
        cnn_input = input_emb.permute(0, 2, 1)
        cnn_delivery = self.conv1d_delivery(cnn_input).transpose(1, 2) # [batch, length, 256]

        # 2. Multi-head self attention
        x_delivery = self.transformer_encode_delivery(cnn_delivery) # [8, 260, 256]

        # 3. 
        # print(f'Delivery: {delivery.size()}')
        '''
        last_hs = x_delivery[:, -1, :]
        last_hs = last_hs.to('cpu')
        self.proj1 = self.proj1.to('cpu')
        self.proj2 = self.proj2.to('cpu')
        self.out_layer = self.out_layer.to('cpu')
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs))))
        last_hs_proj += last_hs
        output = self.out_layer(last_hs_proj)
        '''
        #return output, x_delivery
        return x_delivery
        '''
        if self.trainable == 1:
            # Apply projection layers if trainable == 1
            last_hs = x_delivery[:, -1, :]
            last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs))))
            last_hs_proj += last_hs
            output = self.out_layer(last_hs_proj)
            # 計算損失
            loss = None
            if labels is not None:
                labels = labels.long().to('cpu')
                loss_fn = nn.CrossEntropyLoss()
                loss = loss_fn(output.view(-1, self.num_labels), labels.view(-1))
                loss = loss.to('cpu')

            # 返回包含 loss 和 logits 的輸出
            return TokenClassifierOutput(loss=loss, logits=output)
        else:
            return x_delivery
        '''
        '''
        last_hs = x_delivery[:, -1, :]
        last_hs = last_hs.to('cpu')
        self.proj1 = self.proj1.to('cpu')
        self.proj2 = self.proj2.to('cpu')
        self.out_layer = self.out_layer.to('cpu')
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs))))
        last_hs_proj += last_hs
        output = self.out_layer(last_hs_proj)
        return output
        '''