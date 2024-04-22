import torch
from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

# 定义分词器
# python -m spacy download en_core_web_sm
tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
# python -m spacy download fr_core_news_sm
french_tokenizer = get_tokenizer('spacy', language='fr_core_news_sm')


# 构建词汇表
def build_vocab(sentences, tokenizer_fn):
    def yield_tokens(data_iter):
        for text_entry in data_iter:
            for tok in tokenizer_fn(text_entry):
                yield tok

                # 注意：这里不需要再将[sentences]包装在生成器表达式中

    return build_vocab_from_iterator(yield_tokens(sentences), specials=[])


class TextData:
    def __init__(self, vocab):
        self.vocab = vocab
        self.pad_token = '<pad>'
        # 在新版本的 torchtext 中，你可以直接这样添加新的词汇
        self.vocab.insert_token(self.pad_token, 0)  # 插入在词汇表最前面，并分配索引 0

    def get_pad_index(self):
        # 使用 vocab.get_stoi() 方法或者 vocab[] 来获取词汇的索引
        # return self.vocab.get_stoi()[self.pad_token]  # 如果你确定你的 torchtext 版本有这个方法
        return self.vocab[self.pad_token]  # 推荐使用这种方式，因为它与版本兼容性更好


class ParallelTextDataset(Dataset):
    def __init__(self, english_sentences, french_sentences, english_vocab, french_vocab):
        self.english_sentences = english_sentences
        self.french_sentences = french_sentences
        self.english_vocab = english_vocab
        self.french_vocab = french_vocab

    def __len__(self):
        return len(self.english_sentences)

    def __getitem__(self, idx):
        english_sentence = self.english_sentences[idx]
        french_sentence = self.french_sentences[idx]

        # 使用在函数外部定义的 tokenizer 和 french_tokenizer
        english_tensor = torch.tensor([self.english_vocab[tok] for tok in tokenizer(english_sentence)],
                                      dtype=torch.long)
        french_tensor = torch.tensor([self.french_vocab[tok] for tok in french_tokenizer(french_sentence)],
                                     dtype=torch.long)

        return {'English': english_tensor, 'French': french_tensor}

    # 初始化函数


def initialize_data(english_sentences, french_sentences):
    english_vocab = build_vocab(english_sentences, tokenizer)
    french_vocab = build_vocab(french_sentences, french_tokenizer)

    EN_TEXT = TextData(english_vocab)
    FR_TEXT = TextData(french_vocab)

    dataset = ParallelTextDataset(english_sentences, french_sentences, english_vocab, french_vocab)
    train_iter = DataLoader(dataset, batch_size=64, shuffle=True)

    input_pad = EN_TEXT.get_pad_index()

    return EN_TEXT, FR_TEXT, train_iter, input_pad


# 示例数据
english_sentences = ["hello world", "this is a test", "good morning"]
french_sentences = ["bonjour le monde", "ceci est un test", "bonjour"]

# 初始化数据并获取所需对象
EN_TEXT, FR_TEXT, train_iter, input_pad = initialize_data(english_sentences, french_sentences)
