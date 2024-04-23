# 两个数据集加起来大概有十几个G，这里不载入测试
from datasets import concatenate_datasets, load_dataset

bookcorpus = load_dataset('bookcorpus', split="train")
wiki = load_dataset('wikipedia', '20220301.en', split="train")
# 仅保留'text'列
wiki = wiki.remove_columns([col for col in wiki.column_names if col != 'text'])

dataset = concatenate_datasets([bookcorpus, wiki])

# 将数据切分成90%训练集和10%验证集
d = dataset.train_test_split(test_size=0.1)

def dataset_to_text(dataset, output_filename="data.txt"):
    """将数据集文本保存到磁盘的通用函数"""
    with open(output_filename, "w") as f:
        for t in dataset['text']:
            print(t, file=f)


# 将训练集保存为train.txt
dataset_to_text(d['train'], "train.txt")
# 将测试集保存为test.txt
dataset_to_text(d['test'], "test.txt")