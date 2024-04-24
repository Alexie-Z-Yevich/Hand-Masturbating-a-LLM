from _2_1_prepare_dataset import d
from _2_2_Word_Piece import tokenizer, truncation_longer_sample, max_length


def encode_with_truncation(example):
    """使用词元分析对句子进行处理并截断的映射函数（Mappiong function）"""
    return tokenizer(example['text'], truncation=True, padding='max_length', max_length=max_length,
                     return_special_tokens_mask=True)


def encode_without_truncation(example):
    """使用词元分析对句子进行处理且不截断的映射函数（Mapping function）"""
    return tokenizer(example['text'], return_special_tokens_mask=True)


# 编码函数将依赖于truncate_longer_samples变量
encode = encode_with_truncation if truncation_longer_sample else encode_without_truncation
# 对训练数据集进行分词处理
train_dataset = d["train"].map(encode, batched=True)
# 对测试数据集进行分词处理
test_dataset = d["test"].map(encode, batched=True)
if truncation_longer_sample:
    # 移除其他列，并将input_ids和attention_mask设置为Pytorch张量
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
else:
    # 移除其他列，将它们保存为Python列表
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'special_tokens_mask'])
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'special_tokens_mask'])
