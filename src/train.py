import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW  # ✅ 正确导入方式
from transformers import (
    XLMRobertaForSequenceClassification,
    get_linear_schedule_with_warmup,
    default_data_collator
)
from datasets import load_from_disk
from datetime import datetime

DATE = datetime.now().strftime("%y%m%d")
def train_model():
    # 加载预处理后的 tokenized 数据集
    tokenized_dataset = load_from_disk("../data/tokenized_data")

    # 加载预训练模型（XLM-R）并指定分类数为2（诱导/非诱导）
    model = XLMRobertaForSequenceClassification.from_pretrained(
        "xlm-roberta-base",
        num_labels=2
    )

    # 设置优化器
    optimizer = AdamW(model.parameters(), lr=2e-5)

    # 训练配置
    num_epochs = 3
    batch_size = 16
    total_steps = len(tokenized_dataset["train"]) // batch_size * num_epochs
    warmup_steps = int(total_steps * 0.1)

    # 设置学习率调度器
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # 设置设备（GPU / CPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 创建 PyTorch DataLoader，并使用 collate_fn 自动张量化
    train_dataloader = DataLoader(
        tokenized_dataset["train"],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=default_data_collator
    )

    # 开始训练循环
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in train_dataloader:
            print(batch)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # 前向传播
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}")

    # 保存模型参数
    save_path = f"../models/trained_model_{DATE}.pt"
    torch.save(model.state_dict(), save_path)
    print(f"模型已保存至 {save_path}")

if __name__ == "__main__":
    train_model()
