import torch
import torch.nn as nn
import sys
sys.path.append('/share/nas165/peng/thesis_project/SAMAD_05/models')
from constants import POS, morph, DEP
from transformer_model import *

class Language_BLSTM(nn.Module):
    def __init__(self, pos=POS, morph=morph, dep=DEP, hidden_dim=128):
        super(Language_BLSTM, self).__init__()
        self.pos = pos
        self.morph = morph
        self.dep = dep
        self.embedding_dim = len(self.pos) + len(self.morph) + len(self.dep)
        # self.blstm = nn.LSTM(self.embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
        # self.attention = Attention(hidden_dim*2)
        self.conv1d_lang = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, padding=0, bias=False)        
        self.transformer_encode_lang = Transformer(3, 256, 4)

    def forward(self, 
                padded_seq,
                masked_seq):

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
        return x_lang