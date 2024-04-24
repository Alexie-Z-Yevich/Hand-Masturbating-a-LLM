from itertools import chain

# 主要数据处理函数，拼接数据集中的所有文本并生成最大序列长度的块
from _2_2_Word_Piece import max_length, truncation_longer_sample
from _2_3_1_truncation import train_dataset, test_dataset


def group_texts(examples):
    # 拼接所有文本
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # 舍弃了剩余部分，如果模型支持填充而不是舍弃，则可以根据需要自定义这部分
    if total_length >= max_length:
        total_length = (total_length // max_length) * max_length
    # 按照最大长度分割成块
    result = {
        k: [t[i:i + max_length] for i in range(0, total_length, max_length)]
        for k, t in concatenated_examples.items()
    }
    return result


# 请注意，使用batched=True时，此映射一次处理1000个文本
# 因此，group_text会为这1000个文本组抛弃不足的部分
# 可以在这里调整batch_size，但较高的值可能会使预处理速度变慢
#
# 为了加速这一部分，使用了多进程处理
# 请查看map方法的文档以了解更多信息
# https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map
if not truncation_longer_sample:
    train_dataset = train_dataset.map(
        group_texts,
        batched=True,
        desc=f"Grouping texts in chunks of {max_length}",
    )
    test_dataset = test_dataset.map(
        group_texts,
        batched=True,
        desc=f"Grouping texts in chunks of {max_length}",
    )
    # 将它们从列表转换为Pytorch张量
    train_dataset.set_format("train")
    test_dataset.set_format("test")
