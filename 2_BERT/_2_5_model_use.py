import os

from transformers import BertForMaskedLM, BertTokenizerFast, pipeline

from _2_2_Word_Piece import model_path

# 加载模型检查点
model = BertForMaskedLM.from_pretrained(os.path.join(model_path, "checkpoint-10000"))

# 加载分词器
tokenizer = BertTokenizerFast.from_pretrained(model_path)
fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer)

# 进行预测
examples = [
    "Today's most trending hashtag on [MASK] is Donald Trump.",
    "The [MASK] was cloudy yesterday, but today it's rainy.",
]
for example in examples:
    for prediction in fill_mask(example):
        print(f"{prediction['sequence']}, confidence: {prediction['score']}")
    print("=" * 50)
