from datasets import load_dataset, DatasetDict
from transformers import XLMRobertaTokenizer
import csv

all_labeled_data_path = r"../data/all_labeled_data.csv"
clickbait_path = r"../data/clickbait/optimized_expanded_clickbait.txt"
non_clickbait_path =r"../data/non-clickbait/extracted_texts.txt"

def read_and_label_data():
    labeled_data = []
    # 标记为 1（诱导文本）
    with open(clickbait_path, 'r', encoding='utf-8') as file1:
        for line in file1.readlines():
            text = line.strip()
            labeled_data.append((text, 1))
    # 标记为 0（非诱导文本）
    with open(non_clickbait_path, 'r', encoding='utf-8') as file2:
        for line in file2.readlines():
            text = line.strip()
            labeled_data.append((text, 0))

    with open(all_labeled_data_path, 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["text", "label"])  # 写入表头
        for data in labeled_data:
            writer.writerow(data)

def preprocess_data(tokenizer_name: str = "xlm-roberta-base"):
    # 加载原始数据（CSV文件包含text和label列）
    dataset = load_dataset("csv", data_files=all_labeled_data_path)

    # 划分训练集和验证集（80%训练，20%验证）
    split_dataset = dataset["train"].train_test_split(test_size=0.2)
    dataset = DatasetDict({
        "train": split_dataset["train"],
        "val": split_dataset["test"]
    })

    # 加载XLM - Roberta的分词器
    tokenizer = XLMRobertaTokenizer.from_pretrained(tokenizer_name)
    
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",  # 填充到最大长度128
            truncation=True,  # 超过128则截断
            max_length=128  # 统一文本长度
        )

    # 定义分词函数（自动添加[CLS]、[SEP]、padding和截断）并应用分词处理到整个数据集
    tokenized_dataset = dataset.map(tokenize_fn, batched=True)
    tokenized_dataset.save_to_disk("../data/tokenized_data")
    return tokenized_dataset


if __name__ == "__main__":
    read_and_label_data()
    preprocess_data()