import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from transformers.modeling_outputs import TokenClassifierOutput
from transformer_model import Transformer

class content_BLSTM(nn.Module):
    def __init__(self, hidden_dim=768*2, num_features=256):
        super(content_BLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.bert_model = BertModel.from_pretrained("bert-base-uncased")
        self.conv1d_content = nn.Conv1d(
            in_channels=self.hidden_dim, 
            out_channels=num_features, 
            kernel_size=3, 
            padding=0, 
            bias=False
        )
        self.transformer_encode_content = Transformer(3, 256, 4)
        self.transformer_encode_prompt_content = Transformer(3, 768, 4)
        
        # Projection layers
        self.num_features = num_features
        self.proj1 = nn.Linear(num_features*1, num_features*2)
        self.proj2 = nn.Linear(num_features*2, num_features*1)
        self.out_layer = nn.Linear(num_features*1, 5)    

        self.trainable = 1  # Default state is 1 (trainable)
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
            # Recreate projection layers and移動到設備
            self.proj1 = nn.Linear(self.num_features*1, self.num_features*2)
            self.proj2 = nn.Linear(self.num_features*2, self.num_features*1)
            self.out_layer = nn.Linear(self.num_features*1, 5)

    def bert_freeze_feature_extractor(self):
        for param in self.bert_model.parameters():
            param.requires_grad = False
    
    def forward(self, 
                padded_res_emb, 
                padded_pro_emb,
                attention_mask_res, 
                attention_mask_pro,
                labels=None):

        # 將輸入移動到指定設備
        padded_res_emb = padded_res_emb#cpu
        padded_pro_emb = padded_pro_emb#cpu
        attention_mask_res = attention_mask_res#cpu
        attention_mask_pro = attention_mask_pro#cpu
        

        # 1. v^p
        self.bert_model = self.bert_model#cpu
        x_prompt  = self.bert_model(
            input_ids=padded_pro_emb, 
            attention_mask=attention_mask_pro
        ).last_hidden_state  # [batch_size, length, 768]
        cls_prompt = x_prompt[:, 0, :]  # [batch_size, 768]
        cls_prompt_expanded = cls_prompt.unsqueeze(1).expand(-1, 256, -1)  # [batch_size, 256, 768]
        
        # 2. Concatenate with v^p
        res_emb = self.bert_model(
            input_ids=padded_res_emb, 
            attention_mask=attention_mask_res
        ).last_hidden_state  # [batch_size, 256, 768]
        res_emb_with_cls = torch.cat([res_emb, cls_prompt_expanded], dim=2)  # [batch_size, 256, 1536]

        # 3. CNN layer
        cnn_input = res_emb_with_cls.permute(0, 2, 1)  # [batch_size, 1536, 256]
        self.conv1d_content = self.conv1d_content#cpu
        cnn_output = self.conv1d_content(cnn_input).transpose(1, 2)  # [batch_size, new_seq_len, num_features]
        
        # 4. Self-Attention layer
        x_content = self.transformer_encode_content(cnn_output)  # [batch_size, seq_len, num_features]
        '''
        last_hs = x_content[:, -1, :]
        last_hs = last_hs.to('cpu')
        self.proj1 = self.proj1.to('cpu')
        self.proj2 = self.proj2.to('cpu')
        self.out_layer = self.out_layer.to('cpu')
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs))))
        last_hs_proj += last_hs
        output = self.out_layer(last_hs)     
        '''
        #return output, x_content   
        return x_content   
        '''
        if self.trainable == 1:
            # Apply projection layers
            last_hs = x_content[:, -1, :]
            last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs))))
            last_hs_proj += last_hs
            output = self.out_layer(last_hs)
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
            return x_content
        '''
        '''
        # following is MLP exp.
        last_hs = x_content[:, -1, :]
        last_hs = last_hs.to('cpu')
        self.proj1 = self.proj1.to('cpu')
        self.proj2 = self.proj2.to('cpu')
        self.out_layer = self.out_layer.to('cpu')
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs))))
        last_hs_proj += last_hs
        output = self.out_layer(last_hs_proj)
        return output
        '''