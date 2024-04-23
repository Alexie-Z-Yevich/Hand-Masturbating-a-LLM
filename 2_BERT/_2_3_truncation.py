from _2_2_Word_Piece import tokenizer


def encode_with_truncation(example):
    """使用词元分析对句子进行处理并截断的映射函数（Mappiong function），该处为512"""
    return tokenizer(example['text'], truncation=True, padding='max_length', max_length=512,
                     return_special_tokens_mask=True)
