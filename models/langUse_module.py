import torch
import torch.nn as nn
import torch.nn.functional as F
from constants import POS, morph, DEP
from transformer_model import *
from transformers.modeling_outputs import TokenClassifierOutput

class Language_BLSTM(nn.Module):
    def __init__(self, pos=POS, morph=morph, dep=DEP, hidden_dim=128, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(Language_BLSTM, self).__init__()
        self.pos = pos
        self.morph = morph
        self.dep = dep
        self.embedding_dim = len(self.pos) + len(self.morph) + len(self.dep)
        # self.blstm = nn.LSTM(self.embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
        # self.attention = Attention(hidden_dim*2)
        self.conv1d_lang = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, padding=0, bias=False)        
        self.transformer_encode_lang = Transformer(3, 256, 4)

        # Projection layers
        self.num_features = 256
        self.proj1 = nn.Linear(self.num_features*1, self.num_features*2)
        self.proj2 = nn.Linear(self.num_features*2, self.num_features*1)
        self.out_layer = nn.Linear(self.num_features*1, 5)      

        # self.trainable = 1  # Default state is 0 (trainable)
        self.num_labels = 5
        self.device = device

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

    def forward(self, 
                padded_seq,
                masked_seq, labels=None):

        # 1. Transfer to tensor
        padded_seq_tensor = padded_seq.clone().detach().to(dtype=torch.float32).requires_grad_(True) # [batch, 600, 247]
        masks_tensor = masked_seq.clone().detach().to(dtype=torch.float32).requires_grad_(True) # [batch, 600]
        # print(padded_seq_tensor.size()) # [2, 700, 256]
        # print(masks_tensor.size())
        # 2. CNN
        cnn_input = padded_seq_tensor.permute(0, 2, 1)
        cnn_lang = self.conv1d_lang(cnn_input).transpose(1, 2)

        # 3. Multi-head self attention
        lang = self.transformer_encode_lang(cnn_lang)
        # 4. Masking
        lang = lang.to(self.device)
        masks_tensor = masks_tensor.to(self.device)
        x_lang = lang * masks_tensor.unsqueeze(-1)
        # print(f'langUse: {lang.size()}')

        # 2. BLSTM
        # lstm_out = self.transformer_encode(padded_seq_tensor)
        # # lstm_out, _ = self.blstm(padded_seq_tensor)             # [batch, 600]
        # # print(lstm_out.size())
        # # 3. Masking
        # lstm_out_masked = lstm_out * masks_tensor.unsqueeze(-1) # [batch, 600] * [batch, 600, 1]
        
        # representation, subscore = self.attention(lstm_out_masked)
        # print(f'langUse: {padded_seq_tensor.size()}')
        '''
        last_hs = x_lang[:, -1, :]
        last_hs = last_hs.to('cpu')
        self.proj1 = self.proj1.to('cpu')
        self.proj2 = self.proj2.to('cpu')
        self.out_layer = self.out_layer.to('cpu')
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs))))
        last_hs_proj += last_hs
        output = self.out_layer(last_hs_proj)
        '''
        #return output, x_lang
        return x_lang
        '''
        if self.trainable == 1:
            # Apply projection layers if trainable == 1
            last_hs = x_lang[:, -1, :]
            last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs))))
            last_hs_proj += last_hs
            output = self.out_layer(last_hs_proj)
            # 計算損失
            loss = None
            if labels is not None:
                labels = labels.long().to('cpu')
                loss_fn = nn.CrossEntropyLoss()
                loss = loss_fn(output.view(-1, self.num_labels), labels.view(-1))
                loss = loss.to(self.device)

            # 返回包含 loss 和 logits 的輸出
            return TokenClassifierOutput(loss=loss, logits=output)
        else:
            return x_lang
        '''
        '''
        last_hs = x_lang[:, -1, :]
        last_hs = last_hs.to('cpu')
        self.proj1 = self.proj1.to('cpu')
        self.proj2 = self.proj2.to('cpu')
        self.out_layer = self.out_layer.to('cpu')
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs))))
        last_hs_proj += last_hs
        output = self.out_layer(last_hs_proj)
        return output
        '''