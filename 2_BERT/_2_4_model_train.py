# 模型训练
from transformers import BertConfig, BertForMaskedLM, DataCollatorForLanguageModeling, TrainingArguments, Trainer

from _2_2_Word_Piece import vocab_size, max_length, tokenizer, model_path
from _2_3_1_truncation import train_dataset, test_dataset

# 使用配置文件初始化模型
model_cfg = BertConfig(vocab_size=vocab_size, max_position_embeddings=max_length)
model = BertForMaskedLM(config=model_cfg)

# 初始化数据整理器，随机屏蔽20%（默认为15%）的标记
# 用于掩盖语言建模（MLM）任务
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.2)

training_args = TrainingArguments(
    output_dir=model_path,  # 输出目录，用于保存模型检查点
    evaluation_strategy="steps",  # 每隔`logging_steps`步进行一次评估
    overwrite_output_dir=True,
    num_train_epochs=10,  # 训练时的轮数
    per_device_train_batch_size=10,  # 每个设备的训练批次大小
    gradient_accumulation_steps=8,  # 在更新权重之前累积梯度
    per_device_eval_batch_size=64,  # 评估批量大小
    logging_steps=1000,  # 每隔1000步进行一次评估，记录并保存模型检查点
    save_steps=1000,
    # load_best_model_at_end=True,  # 是否在训练结束时加载最佳模型（根据损失）
    # save_total_limit=3,  # 如果磁盘空间有限，可以限制保存的模型数量
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)
