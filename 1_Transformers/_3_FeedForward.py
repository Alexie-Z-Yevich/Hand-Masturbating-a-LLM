# 3_前馈层
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):

    def __init__(self, d_model, d_ff=2048, dropout=0.1):  # d_model是词向量的维度, d_ff是前馈神经网络的维度, dropout是dropout的概率
        super().__init__()

        # d_ff默认设置为2048
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear1(x)))  # 使用relu激活函数
        x = self.linear2(x)
        return x
