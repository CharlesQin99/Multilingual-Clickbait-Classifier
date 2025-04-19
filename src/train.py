import time
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    XLMRobertaForSequenceClassification,
    get_linear_schedule_with_warmup,
    default_data_collator
)
from datasets import load_from_disk
from datetime import datetime

DATE = datetime.now().strftime("%y%m%d")

def evaluate_model(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=-1)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_samples

    return avg_loss, accuracy


def train_model():
    start_time = time.time()

    tokenized_dataset = load_from_disk("../data/tokenized_data")

    # 加载预训练好的 XLM-Roberta 模型，加上一个分类层，用于二分类
    model = XLMRobertaForSequenceClassification.from_pretrained(
        "xlm-roberta-base",
        num_labels=2
    )

    # 优化器，lr=2e-5 是学习率，表示模型参数每次更新时步子多大，步子太大容易过，太小又慢
    optimizer = AdamW(model.parameters(), lr=2e-5)

    # 设置训练轮数与 scheduler：整个训练集会训练 3遍，每次送进模型 16 条样本，一起计算梯度，加速训练
    num_epochs = 3  
    batch_size = 16
    total_steps = len(tokenized_dataset["train"]) // batch_size * num_epochs
    warmup_steps = int(total_steps * 0.1)   # 缓慢提高学习率，防止不稳定

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Device in use: {device}")

    train_dataloader = DataLoader(
        tokenized_dataset["train"],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=default_data_collator
    )

    val_dataloader = DataLoader(
        tokenized_dataset["val"],
        batch_size=batch_size,
        shuffle=False,
        collate_fn=default_data_collator
    )

    best_val_loss = float("inf")
    patience = 2
    patience_counter = 0

    # 进入训练主循环
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # 设置为训练模式
        model.train()
        total_loss = 0

        for batch in train_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward() # 反向传播, 计算梯度
            optimizer.step() # 根据梯度更新模型参数
            scheduler.step() # 动态调整学习率

            total_loss += loss.item()  # 累计整个 epoch 的损失

        avg_train_loss = total_loss / len(train_dataloader)
        print(f"[Epoch {epoch + 1}] Training Loss: {avg_train_loss:.4f}")

        # 不更新权重，只是验证当前模型在验证集上的效果,返回:验证集平均损失（val_loss）和 验证集准确率（val_accuracy）
        val_loss, val_accuracy = evaluate_model(model, val_dataloader, device)
        print(f"[Epoch {epoch + 1}] Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        epoch_time = time.time() - epoch_start
        print(f"[Epoch {epoch + 1}] Duration: {epoch_time:.2f} seconds")

        # 保存最优模型 + 早停策略        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_path = f"../models/trained_model_{DATE}.pt"
            torch.save(model.state_dict(), save_path)
            print(f"Best model saved to {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping: Validation loss did not improve.")
                break

    print(f"Training complete. Best model is saved at {save_path}")
    total_time = time.time() - start_time
    print(f"Total training time: {total_time:.2f} seconds")

if __name__ == "__main__":
    train_model()