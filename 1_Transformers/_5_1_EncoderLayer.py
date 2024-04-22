# 5_1_编码器
import torch.nn as nn

from _1_PositionalEncoder import PositionalEncoder
from _2_MultiHeadAttention import MultiHeadAttention
from _3_FeedForward import FeedForward
from _4_Norm import Norm
from tools._0_1_Embedder import Embedder
from tools._0_2_get_clones import get_clones


class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_output = self.attn(x, x, x, mask)
        attn_output = self.dropout_1(attn_output)
        x = x + attn_output
        x = self.norm_1(x)
        ff_output = self.ff(x)
        ff_output = self.dropout_2(ff_output)
        x = x + ff_output
        x = self.norm_2(x)
        return x


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads,
                 dropout):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model)
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)

    def forward(self, src, mask):
        x = self.embed(src)  # ��Ƕ��
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm(x)
