import torch
import torch.nn as nn
import math

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, device=torch.device('cuda')):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0).to(device)

    def forward(self, x):
        return x + self.encoding[:, :x.size(1)].detach()

# Self-Attention(Multi-head)
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        self.dense = nn.Linear(d_model, d_model)

    def split_into_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, q, k, v, mask):
        batch_size = q.size(0)

        q = self.split_into_heads(self.wq(q), batch_size)
        k = self.split_into_heads(self.wk(k), batch_size)
        v = self.split_into_heads(self.wv(v), batch_size)

        # Scaled dot product attention
        matmul_qk = torch.matmul(q, k.transpose(-2, -1))  # (..., seq_len_q, seq_len_k)
        dk = torch.tensor(self.depth).float()
        scaled_attention_logits = matmul_qk / math.sqrt(dk)

        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        attention_weights = torch.softmax(scaled_attention_logits, dim=-1)
        output = torch.matmul(attention_weights, v)

        output = output.permute(0, 2, 1, 3).contiguous()
        output = output.view(batch_size, -1, self.d_model)
        output = self.dense(output)

        return output, attention_weights


# Encoding Layer
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads):
        super(TransformerEncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )

        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(0.1)

    def forward(self, x, mask):
        attn_output, _ = self.mha(x, x, x, mask)  # Self-attention
        out1 = self.layernorm1(x + self.dropout(attn_output))
        ffn_output = self.ffn(out1)
        out2 = self.layernorm2(out1 + self.dropout(ffn_output))
        return out2

class Transformer(nn.Module):
    def __init__(self, num_layers, d_model, num_heads):
        super(Transformer, self).__init__()
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads) for _ in range(num_layers)
        ])
        self.pos_encoder = PositionalEncoding(d_model)

    def forward(self, src, mask=None):
        src = self.pos_encoder(src)
        for layer in self.encoder_layers:
            src = layer(src, mask)
        return src

class QKVTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads):
        super(QKVTransformerEncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )

        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, query, key, value, mask):
        attn_output, _ = self.mha(query, key, value, mask)
        out1 = self.layernorm1(query + self.dropout(attn_output))
        ffn_output = self.ffn(out1)
        out2 = self.layernorm2(out1 + self.dropout(ffn_output))
        return out2

        
class QKVTransformer(nn.Module):
    def __init__(self, num_layers, d_model, num_heads):
        super(QKVTransformer, self).__init__()
        self.encoder_layers = nn.ModuleList([
            QKVTransformerEncoderLayer(d_model, num_heads) for _ in range(num_layers)
        ])
        self.pos_encoder = PositionalEncoding(d_model)

    def forward(self, query, key, value, mask=None):
        query = key = value = self.pos_encoder(query)  # Assuming all are initially the same but could be different
        for layer in self.encoder_layers:
            query = layer(query, key, value, mask)
        return query

    