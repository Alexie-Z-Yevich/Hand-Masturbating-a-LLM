# 只能尽可能的优化代码结构，但是不保证能否运行，后续我会在深入学习之后，再来完善这个代码
# 目前仅作为模板思想参考
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from _5_3_Transformer import Transformer
from tools._0_3_torchtext import EN_TEXT, FR_TEXT, train_iter, input_pad
from tools._0_4_create_masks import create_masks
from tools._0_5_tokenize import tokenize_en

# 模型参数定义
d_model = 512
heads = 8
N = 6
src_vocab = len(EN_TEXT.vocab)
trg_vocab = len(FR_TEXT.vocab)
model = Transformer(src_vocab, trg_vocab, d_model, N, heads)
for p in model.parameters():  # 初始化模型参数
    if p.dim() > 1:  # 判断维度是否大于1
        nn.init.xavier_uniform_(p)  # 使用均匀分布初始化参数

# 损失函数和优化器
optim = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)


# 模型训练
def tarin_model(epochs, print_every=100):
    model.train()

    start = time.time()
    temp = start

    total_loss = 0

    for epoch in range(epochs):

        for i, batch in enumerate(train_iter):
            src = batch.English.transpose(0, 1)
            trg = batch.French.transpose(0, 1)

            trg_input = trg[:-1, :]

            target = trg[:, 1:].contiguous().view(-1)

            src_mask, trg_mask = create_masks(src, trg_input)

            preds = model(src, trg_input, src_mask, trg_mask)

            optim.zero_grad()

            loss = F.cross_entropy(preds.view(-1, preds.size(-1)))
            loss.backward()
            optim.step()

            total_loss += loss.data[0]
            if (i + 1) % print_every == 0:
                loss_avg = total_loss / print_every
                print("time = %dm, epoch %d, iter = %d, loss = %.3f, %ds per %d iters"
                      % ((time.time() - start) // 60, epoch + 1, i + 1, loss_avg, time.time() - temp,
                         print_every))
                total_loss = 0
                temp = time.time()


# 模型测试
def translate(model, src, max_len=80, custom_string=False):

    model.eval()
    sentence = None
    if custom_string:
        # Tokenize the input string
        tokenized_src = tokenize_en(src)

        # Convert tokenized source to tensor indices
        sentence = torch.tensor([[EN_TEXT.vocab.stoi[tok] for tok in tokenized_src]],
                                device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    # Create source mask
    src_mask = (sentence != input_pad).unsqueeze(-2)

    # Encoder output
    e_output = model.encoder(sentence, src_mask)

    # Prepare output tensor
    outputs = torch.zeros(max_len, dtype=torch.long, device=sentence.device)
    outputs[0] = FR_TEXT.vocab.stoi["<sos>"]

    for i in range(1, max_len):
        trg_mask = np.triu(np.ones((1, i, i), k=1).astype("uint8"))
        trg_mask = Variable(torch.from_numpy(trg_mask) == 0).cuda()

        out = model.out(model.decoder(outputs[:i].unsqueeze(0), e_output, src_mask, trg_mask))
        out = F.softmax(out, dim=-1)
        val, ix = out[:, -1].data.topk(1)

        outputs[i] = ix[0][0]
        if ix[0][0] == FR_TEXT.vocab.stoi["<eos>"]:
            break
    return ' '.join([FR_TEXT.vocab.itos[ix] for ix in outputs[:i]])
