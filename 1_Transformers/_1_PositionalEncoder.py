# 1_嵌入表示层_Positional_Encoding
import math

import torch
import torch.nn as nn
from torch.autograd import Variable


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=80):  # d_model是词向量的维度, max_seq_len是序列的最大长度
        super().__init__()
        self.d_model = d_model  # 词向量的维度

        # 根据pos和i创建一个常量PE矩阵
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** (i / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** (i / d_model)))

        pe = pe.unsqueeze(0)  # 在第0维增加一个维度
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 使得单词嵌入表示相对大一些
        x = x * math.sqrt(self.d_model)
        # 增加位置常量到单词嵌入表示中
        seq_len = x.size(1)
        x = x + Variable(self.pe[:, :seq_len], requires_grad=False).cuda()  # 从PE矩阵中取出对应位置的PE
        return x
