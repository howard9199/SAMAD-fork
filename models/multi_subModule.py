import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from transformers.modeling_outputs import TokenClassifierOutput
from transformer_model import *

from content_module import content_BLSTM
from delivery_module import *
from langUse_module import *

from safetensors.torch import load_file

class MultiFeatureModel(nn.Module):
    def __init__(self, device, embed_dim=256, num_labels=5, hidden_dim=128):
        super(MultiFeatureModel, self).__init__()
        self.device = device
        self.num_labels = num_labels
        self.embed_dim = embed_dim
        self.content_BLSTM = content_BLSTM()
        # load from output_content
        # 載入模型參數
        #state_dict = load_file('/datas/store163/howard/samad/SAMAD/output_content_dense/checkpoint-510/model.safetensors')
        #self.content_BLSTM.load_state_dict(state_dict,strict=False)
        #self.content_BLSTM.set_state(0)
        self.content_BLSTM = self.content_BLSTM.to(device)

        self.Delivery_BLSTM = Delivery_BLSTM()
        #state_dict = load_file('/datas/store163/howard/samad/SAMAD/output_delivery_dense/checkpoint-110/model.safetensors')
        #self.Delivery_BLSTM.load_state_dict(state_dict,strict=False)
        #self.Delivery_BLSTM.set_state(0)
        self.Delivery_BLSTM = self.Delivery_BLSTM.to(device)

        self.Language_BLSTM = Language_BLSTM(POS, morph, DEP, hidden_dim)
        #state_dict = load_file('/datas/store163/howard/samad/SAMAD/output_language_dense/checkpoint-490/model.safetensors')
        #self.Language_BLSTM.load_state_dict(state_dict,strict=False)
        #self.Language_BLSTM.set_state(0)
        self.Language_BLSTM = self.Language_BLSTM.to(device)
        self.cross_attention_cd = QKVTransformer(3, 256, 4) # num_layers, d_model, num_heads
        self.cross_attention_cl = QKVTransformer(3, 256, 4) # num_layers, d_model, num_heads

        self.self_attention_content = Transformer(3, 512, 4) # num_layers, d_model, num_heads

        self.num_features = 256

        # Projection layers
        self.proj1 = nn.Linear(self.num_features*2, self.num_features*4).to(device)
        self.proj2 = nn.Linear(self.num_features*4, self.num_features*2).to(device)
        self.out_layer = nn.Linear(self.num_features*2, num_labels).to(device)

        # MLP layer to combine the features
        '''
        num_dimention = 3
        self.mlp_layer = nn.Sequential(
            nn.Linear(5 * num_dimention, 5 * num_dimention * 2),
            nn.ReLU(),
            nn.Linear(5 * num_dimention * 2, 5 * num_dimention * 2),
            nn.ReLU(),
            nn.Linear(5 * num_dimention * 2, 5),
        ).to(self.device)
        '''

    def kappa_loss(self, p, y, n_classes=5, eps=1e-10, device='cuda'):
        """
        QWK loss function
        Arguments:
            p: probability predictions [batch_size, n_classes]
            y: one-hot encoded labels [batch_size, n_classes]
        """
        
        # 建立 weight matrix
        W = torch.zeros((n_classes, n_classes), device=device)
        for i in range(n_classes):
            for j in range(n_classes):
                W[i,j] = (i-j)**2
        
        # Change the W to float32
        W = W.float()

        # 計算 observed 和 expected matrices
        O = torch.matmul(y.t(), p)
        E = torch.matmul(y.sum(dim=0).view(-1,1), p.sum(dim=0).view(1,-1)) / O.sum()
        
        return (W*O).sum() / ((W*E).sum() + eps)

    def forward(self,
                padded_res_emb,
                attention_mask_res,
                padded_pro_emb,
                attention_mask_pro,
                padded_seq,
                masked_seq,
                delivery_tra,
                labels=None):
        
        # 1. Get the embedding

        #subscore_content, x_content = self.content_BLSTM(padded_res_emb, padded_pro_emb, attention_mask_res, attention_mask_pro) # [8, 768], [8, 1]
        #subscore_delivery, x_delivery = self.Delivery_BLSTM(delivery_tra)           # [8, 256], [batch_size, 1]
        #subscore_langUse, x_langUse = self.Language_BLSTM(padded_seq, masked_seq)
        x_content = self.content_BLSTM(padded_res_emb, padded_pro_emb, attention_mask_res, attention_mask_pro)
        x_delivery = self.Delivery_BLSTM(delivery_tra) 
        x_langUse = self.Language_BLSTM(padded_seq, masked_seq)

        # print(f'Content: {x_content.size()}, Delivery: {x_delivery.size()}, LangUse: {x_langUse.size()}')        
        # 2. Cross-attention 
        trans_c_with_d = self.cross_attention_cd(x_content, x_delivery, x_delivery) # [2, 254, 256]
        trans_c_with_l = self.cross_attention_cl(x_content, x_langUse, x_langUse) # 2, 254, 256]
        # print(f'C->D: {trans_c_with_d.size()}, C-L: {trans_c_with_l.size()}')     # [batch, 256, 256]
        # 3.1 concate through feature size
        h_content_states = torch.cat([trans_c_with_d, trans_c_with_l], dim=2) # 2, 254, 512] 用feature size連在一起
        # print(f'Content states: {h_content_states.size()}') # [B, length, 512]
        # 3.2 self-attention + Take the last output for prediction
        h_content = self.self_attention_content(h_content_states) # [2, 256, 512]
        # print(f'After self attention states: {h_content.size()}')
        

        # 6. projection
        # A residual block
        last_hs = h_content[:,-1,:]
        # print('Last Hidden', last_hs.size())
        last_hs = last_hs.to(self.device)
        self.proj1 = self.proj1.to(self.device)
        self.proj2 = self.proj2.to(self.device)
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs))))
        last_hs_proj += last_hs
        # print(f'After Residual: {last_hs_proj.size()}') # [2, 4751, 256]

        # 7. predictor
        self.out_layer = self.out_layer.to(self.device)
        output = self.out_layer(last_hs_proj)
        # print(f'output layer: {output.size()}')

        # Combine the outputs of x_content, x_delivery, and x_langUse
        '''
        combined_features = torch.cat([x_content, x_delivery, x_langUse], dim=1)  # Concatenate along the feature dimension


        # Final prediction
        self.mlp_layer = self.mlp_layer.to(self.device)
        combined_features = combined_features.to(self.device)
        output = self.mlp_layer(combined_features)

        '''
        overall_loss = None
        if labels is not None:
            # 將 output 轉換為機率分布
            probs = F.softmax(output, dim=1)
            
            # 將 labels 轉換為 one-hot 編碼並移至正確的設備
            labels_one_hot = F.one_hot(labels.long(), num_classes=self.num_labels).float().to(self.device)
            
            # 計算 kappa loss
            loss_holistic = self.kappa_loss(probs, labels_one_hot, n_classes=self.num_labels, device=self.device)
            overall_loss = loss_holistic
            
        return TokenClassifierOutput(loss=overall_loss, logits=output)
        #return output, subscore_content, subscore_delivery, subscore_langUse
