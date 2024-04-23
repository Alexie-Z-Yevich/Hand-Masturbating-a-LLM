# 训练词元分析器
import json
import os.path

from tokenizers.implementations import BertWordPieceTokenizer

special_tokens = [
    "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "<S>", "<T>"
]
# 如果根据训练和测试两个集合训练词元分析器，则需要修改files
# files = ["train.txt", "test.txt"]
# 仅根据训练集合训练词元分析器
files = ["train.txt"]
# BERT中采用的默认词表大小为30522，可以随意修改
vocab_size = 30_522
# 最大序列长度，该值越小，训练速度越快
max_length = 512
# 是否将长样本阶段
truncation_longer_sample = False

# 初始化WordPiece词元分析器
tokenizer = BertWordPieceTokenizer()
# 训练词元分析器
tokenizer.train(files, vocab_size=vocab_size, special_tokens=special_tokens)
# 允许截断达到最大512词元
tokenizer.enable_truncation(max_length=max_length)

model_path = "pretrained-bert"

# 如果文件夹不存在，则先创建文件夹
if not os.path.isdir(model_path):
    os.mkdir(model_path)
# 保存词元分析器
tokenizer.save_model(model_path)
# 将一些词元分析器中的配置保存到配置文件，包括特殊词元、转换为小写、最大序列长度等
with open(os.path.join(model_path, "config.json"), "w") as f:
    tokenizer_cfg = {
        "do_lower_case": True,
        "unk_token": "[UNK]",
        "sep_token": "[SEP]",
        "pad_token": "[PAD]",
        "cls_token": "[CLS]",
        "mask_token": "[MASK]",
        "model_max_length": max_length,
        "max_len": max_length,
    }
    json.dump(tokenizer_cfg, f)
# 当词元分析器进行训练和配置时，将其装在到BertTokenizerFast
tokenizer = BertWordPieceTokenizer(model_path)
