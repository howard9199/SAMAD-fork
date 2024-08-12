import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
from transformer_model import *

# class TextCNN(nn.Module):
#     def __init__(self, embed_dim):
#         super(TextCNN, self).__init__()
#         self.conv1 = nn.Conv1d(in_channels=embed_dim, out_channels=100, kernel_size=2)
#         self.relu = nn.ReLU()
#         self.pool = nn.MaxPool1d(kernel_size=2)
#         # self.fc = nn.Linear(100, 1)
    
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.relu(x)
#         x = self.pool(x)
#         # x = x.view(x.size(0), -1)  # flatten
#         # x = self.fc(x)
#         return x

# ======================= BERT =====================================
class content_BLSTM(nn.Module):
    def __init__(self, hidden_dim=768*2, num_features=256):
        super(content_BLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.bert_model = BertModel.from_pretrained("bert-base-uncased")
        # self.attention = Attention(hidden_dim)
        self.conv1d_content = nn.Conv1d(in_channels=self.hidden_dim, out_channels=num_features, kernel_size=3, padding=0, bias=False) # text 
        # self.conv1d_content = nn.Conv1d(in_channels=768, out_channels=num_features, kernel_size=3, padding=0, bias=False) # text 
        self.transformer_encode_content = Transformer(3, 256, 4) # num_layers, d_model, num_heads
        self.transformer_encode_prompt_content = Transformer(3, 768, 4)    
    
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

        # 2.5. [prompt;response] -> self-attention
        # cls_prompt_exp = cls_prompt.unsqueeze(1)
        # prompt_res = torch.cat([x_prompt, res_emb], dim=1) # 以length去擴增 #[batch, 256+, 768]
        # res_emb_with_cls = self.transformer_encode_prompt_content(prompt_res)
        
        # 3. CNN layer
        cnn_input = res_emb_with_cls.permute(0, 2, 1)
        cnn_output = self.conv1d_content(cnn_input).transpose(1, 2) # (batch_size, out_channels, new_sequence_length)
        # print(f"Content Output: {cnn_output.size()}")
        
        # 4. Self-Attention layer
        x_content = self.transformer_encode_content(cnn_output) # [batch, 254, 256]
        # print(f"Content Output: {x_content.size()}")

        return x_content
        # return res_emb_with_cls


# ======================= BERT =====================================
# class content_BLSTM(nn.Module):
#     def __init__(self, hidden_dim=768):
#         super(content_BLSTM, self).__init__()
#         self.hidden_dim = hidden_dim
#         self.bert_model = BertModel.from_pretrained("bert-base-uncased")
#         self.attention = Attention(self.hidden_dim*2)
#         self.fc = nn.Linear(self.hidden_dim*2, 5)
#         self.softmax = nn.Softmax(dim=1) 

#     def bert_freeze_feature_extractor(self):
#         for param in self.bert_model.parameters():
#             param.requires_grad = False
    
#     def forward(self, 
#                 padded_res_emb, 
#                 padded_pro_emb,
#                 attention_mask_res, 
#                 attention_mask_pro): # labels=None

#         # 1. v^p
#         x_prompt  = self.bert_model(padded_pro_emb, attention_mask_pro).last_hidden_state # [batch_size, length, 768]
#         cls_prompt = x_prompt[:, 0, :] # [batch_size, 768]
        
#         # 2. Each "word" need to concatenate with v^p
#         res_emb = self.bert_model(padded_res_emb, attention_mask_res).last_hidden_state # [2, 256, 768]

#         # ==========================================================
#         tmp_list = []
#         for index, cls_id in enumerate(padded_res_emb):
#             cls_indices = (cls_id == 101).nonzero(as_tuple=True)[0]
#             cls_token = res_emb[index, cls_indices, :]

#             # 3. max pooling
#             max_pooling = torch.max(cls_token, dim=0)[0]
#             # 3. mean pooling
#             mean_pooling = torch.mean(cls_token, dim=0)
            
#             # 5. mix pooling
#             lambda_param = 0.5
#             mix_pooling = (lambda_param * max_pooling) + ((1 - lambda_param) * mean_pooling)
#             tmp_list.append(mix_pooling)
#         combined_tensor = torch.cat(tmp_list, dim=0).view(-1, 768)  # convert to tensor        

#         res_emb_with_cls = torch.cat([combined_tensor, cls_prompt], dim=1)  # [batch_size, 256, 768*2=1536]
#         # print(res_emb_with_cls.size())
#         subscore = self.softmax(self.fc(res_emb_with_cls))
#         # print(subscore.size())
#         # representation, subscore = self.attention(res_emb_with_cls)
#         return res_emb_with_cls, subscore

