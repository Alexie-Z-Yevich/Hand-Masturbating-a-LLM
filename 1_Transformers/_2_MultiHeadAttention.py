# 2_注意力层_Self_Attention
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_models, dropout=0.1):  # heads是多头注意力的头数, d_models是词向量的维度, dropout是dropout的概率
        super().__init__()

        self.d_model = d_models
        self.d_k = d_models // heads  # 每个头的词向量维度
        self.h = heads

        self.q_linear = nn.Linear(d_models, d_models)  # 创建Q、K、V的线性层
        self.v_linear = nn.Linear(d_models, d_models)
        self.k_linear = nn.Linear(d_models, d_models)
        self.dropout = nn.Dropout(dropout)  # dropout层
        self.out = nn.Linear(d_models, d_models)

    def attention(q, k, v, d_k, mask=None, dropout=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)  # 计算注意力分数

        # 掩盖那些为了填补长度而增加的单元，使其通过Softmax计算后为0
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)  # 用一个很大的负数填充

        scores = F.softmax(scores, dim=-1)  # 计算softmax

        if dropout is not None:
            scores = dropout(scores)

        output = torch.matmul(scores, v)  # 计算输出
        return output

    def forward(self, q, k, v, mask=None):

        bs = q.size(0)

        # 利用线性计算划分成h个头
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # 矩阵转置
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # 计算attention
        scores = self.attention(q, k, v, self.d_k, mask, self.dropout)

        # 连接多个头并输入最后的线性层
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)

        output = self.out(concat)

        return output
