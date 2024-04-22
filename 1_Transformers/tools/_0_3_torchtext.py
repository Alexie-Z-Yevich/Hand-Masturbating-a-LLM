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
            yield tokenizer_fn(text_entry)

    return build_vocab_from_iterator(yield_tokens, sentences)


# 类似于旧版EN_TEXT和FR_TEXT的对象
class TextData:
    def __init__(self, vocab):
        self.vocab = vocab
        self.pad_token = '<pad>'
        self.vocab.add_token(self.pad_token, special_tokens=True)

    def get_pad_index(self):
        return self.vocab.stoi[self.pad_token]

    # 并行文本数据集


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
