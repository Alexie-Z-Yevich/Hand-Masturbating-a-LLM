import spacy

nlp = spacy.load('en_core_web_sm')


def tokenize_en(text):
    return [tok.text for tok in nlp.tokenizer(text)]
