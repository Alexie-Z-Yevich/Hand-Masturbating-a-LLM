def dataset_to_text(dataset, output_filename="data.txt"):
    """将数据集文本保存到磁盘的通用函数"""
    with open(output_filename, "w") as f:
        for t in dataset['text']:
            print(t, file=f)


# 将训练集保存为train.txt
dataset_to_text(d['train'], "train.txt")
# 将测试集保存为test.txt
dataset_to_text(d['test'], "test.txt")
