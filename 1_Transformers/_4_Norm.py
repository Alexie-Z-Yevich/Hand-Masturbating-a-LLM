# 4_层归一化
import torch
import torch.nn as nn


class Norm(nn.Module):

    def __init__(self, d_model, eps=1e-6):  # d_model是词向量的维度, eps是一个很小的数，防止除以0
        super().__init__()

        self.size = d_model

        # 层归一化包含两个可学习的参数
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))

        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias  # 归一化
        return norm
