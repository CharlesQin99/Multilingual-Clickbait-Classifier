import torch
import numpy as np 
import json
from transformers import XLMRobertaForSequenceClassification, TrainingArguments, Trainer
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from log import Log

logger = Log()

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)  # 获取预测的标签
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def evaluate_model():
    # 加载预处理后的数据集
    tokenized_dataset = load_from_disk("../data/tokenized_data")
    
    # 加载模型和训练好的权重
    model = XLMRobertaForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=2)
    model.load_state_dict(torch.load("../models/trained_model_250418.pt"))
    model.eval()  # 设置为评估模式
    
    # 定义评估参数
    evaluation_args = TrainingArguments(
        per_device_eval_batch_size=64,
        no_cuda=False,
        output_dir="./results"
    )

    # 定义 Trainer 对象
    trainer = Trainer(
        model=model,
        args=evaluation_args,
        eval_dataset=tokenized_dataset["test"],
        compute_metrics=compute_metrics
    )

    # ---------- 原始评估部分 ----------
    eval_results = trainer.evaluate()
    logger.info(f"Evaluation results: {eval_results}")

    accuracy = eval_results["eval_accuracy"]
    precision = eval_results["eval_precision"]
    recall = eval_results["eval_recall"]
    f1 = eval_results["eval_f1"]

    logger.info(f"Accuracy: {accuracy}")
    logger.info(f"Precision: {precision}")
    logger.info(f"Recall: {recall}")
    logger.info(f"F1-score: {f1}")


    # 获取原始文本（需保留原始文本字段，如 'text'）
    if "text" not in tokenized_dataset["test"].column_names:
        logger.warning("Warning: 'text' column not found in test set. Skipping wrong prediction export.")
        return

    # 使用 predict 获取 logits
    predict_output = trainer.predict(tokenized_dataset["test"])
    logits = predict_output.predictions
    labels = predict_output.label_ids
    predictions = np.argmax(logits, axis=1)

    wrong_samples = []
    for i, (pred, label) in enumerate(zip(predictions, labels)):
        if pred != label:
            sample = {
                "text": tokenized_dataset["test"][i]["text"],
                "true_label": int(label),
                "predicted_label": int(pred)
            }
            wrong_samples.append(sample)

    # 保存错误预测样本
    with open("./results/wrong_predictions.json", "w", encoding="utf-8") as f:
        json.dump(wrong_samples, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(wrong_samples)} wrong predictions to './results/wrong_predictions.json'")


if __name__ == "__main__":
    evaluate_model()
