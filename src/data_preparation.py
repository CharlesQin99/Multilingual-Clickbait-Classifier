from datasets import load_dataset, DatasetDict
from transformers import XLMRobertaTokenizer
import csv

all_labeled_data_path = r"../data/all_labeled_data_r2.csv"
clickbait_path = r"../data/clickbait/r2/positive_list.txt"
non_clickbait_path =r"../data/non-clickbait/negative_list.txt"

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
    dataset = load_dataset("csv", data_files=all_labeled_data_path)

    # 划分数据集：80%训练，10%验证，10%测试
    split_dataset = dataset["train"].train_test_split(test_size=0.2)  # 80% 训练, 20% 用作验证+测试
    val_test_split = split_dataset["test"].train_test_split(test_size=0.5)  # 将 20% 划分为 10% 验证集和 10% 测试集

    dataset = DatasetDict({
        "train": split_dataset["train"],
        "val": val_test_split["train"],  # 10% 验证集
        "test": val_test_split["test"]   # 10% 测试集
    })

    # 加载 XLM-Roberta 的分词器
    tokenizer = XLMRobertaTokenizer.from_pretrained(tokenizer_name)
    
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=128 
        )

    # 定义分词函数（自动添加[CLS]、[SEP]、padding和截断）并应用分词处理到整个数据集
    tokenized_dataset = dataset.map(tokenize_fn, batched=True)
    tokenized_dataset.save_to_disk("../data/tokenized_data")
    return tokenized_dataset


if __name__ == "__main__":
    read_and_label_data()
    preprocess_data()
