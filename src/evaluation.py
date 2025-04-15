import torch
from transformers import XLMRobertaForSequenceClassification, TrainingArguments, Trainer
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from utils.logging_utils import setup_logging

logger = setup_logging()

def evaluate_model():
    # 加载预处理后的数据集
    tokenized_dataset = load_from_disk("data/processed/tokenized_data")
    
    # 加载训练好的模型
    model = XLMRobertaForSequenceClassification.from_pretrained("./models/trained_model")
    
    # 定义评估参数
    evaluation_args = TrainingArguments(
        output_dir="../models/trained_model",
        per_device_eval_batch_size=64
    )
    
    # 定义 Trainer 对象进行评估
    trainer = Trainer(
        model=model,
        args=evaluation_args,
        eval_dataset=tokenized_dataset["val"]
    )
    
    # 进行评估
    eval_results = trainer.evaluate()
    logger.info(f"Evaluation results: {eval_results}")
    
    # 进行预测
    test_preds = trainer.predict(tokenized_dataset["val"])
    predicted_labels = torch.argmax(torch.tensor(test_preds.predictions), dim=1)
    true_labels = tokenized_dataset["val"]["label"]
    
    # 计算评估指标
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='weighted')
    
    logger.info(f"Accuracy: {accuracy}")
    logger.info(f"Precision: {precision}")
    logger.info(f"Recall: {recall}")
    logger.info(f"F1-score: {f1}")

if __name__ == "__main__":
    evaluate_model()    