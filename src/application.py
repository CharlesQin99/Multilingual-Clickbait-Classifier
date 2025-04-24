import torch
import urllib
from transformers import XLMRobertaForSequenceClassification, XLMRobertaTokenizer

file_path = "../data/test/test_cases.txt"
output_file_path = "../data/test/test_results.txt"  # 修改为正确的输出文件路径


# 获取设备（自动选择 GPU 或 CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def download_model(model_path):
    model_url = ""
    try:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        logger.info("Model Downloading...")
        urllib.request.urlretrieve(model_url, model_path)
        logger.info("Model Download Done!")
        return True
    except Exception as e:
        print(f"Model Downloading Error：{e}")
        return False

# 加载模型和分词器
def load_model_and_tokenizer(tokenizer_name: str = "xlm-roberta-base"):
    # 加载 xlm-roberta-base 模型，指定分类数量为 2（表示二分类）
    model = XLMRobertaForSequenceClassification.from_pretrained(tokenizer_name, num_labels=2)
    
    # 使用 map_location 将模型加载到适当的设备（GPU 或 CPU）

    model_path = "../models/trained_model_250423.pt"  # 模型路径
    
    if not os.path.exists(model_path) and not download_model(model_path):
        return None, None

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # 设置为评估模式
    
    tokenizer = XLMRobertaTokenizer.from_pretrained(tokenizer_name)
    return model, tokenizer


# 对输入文本进行预测
def predict_from_text(text: str, tokenizer_name: str = "xlm-roberta-base"):
    # 加载模型和分词器
    model, tokenizer = load_model_and_tokenizer(tokenizer_name)
    # 用 tokenizer 把 text 编码成模型输入格式：转成 token id（input_ids）补全到固定长度（padding）超过最大长度的截断（max_length=128）
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    # 把张量送到 GPU 或 CPU，和模型保持一致。
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    # 在维度 1 上取最大值的索引，也就是预测为哪一类
    predicted_class = torch.argmax(logits, dim=1).item()
    return predicted_class


def predict_from_file(tokenizer_name: str = "xlm-roberta-base"):
    model, tokenizer = load_model_and_tokenizer(tokenizer_name)
    
    if model and tokenizer:
        # 打开输出文件准备写入
        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            with open(file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    line = line.strip()
                
                    inputs = tokenizer(line, return_tensors="pt", padding=True, truncation=True, max_length=128)
                    inputs = {key: value.to(device) for key, value in inputs.items()}  # 将输入数据转移到正确的设备
                    
                    with torch.no_grad():
                        outputs = model(**inputs)
                    
                    logits = outputs.logits
                    predicted_class = torch.argmax(logits, dim=1).item()

                    output_file.write(f"Text: {line} => Predicted Class: {predicted_class}\n")

        print(f"Prediction results have been saved to {output_file_path}")  # 修改这里为文件路径
    else:
        print(f"Model Loading Error")


def test():
    # text = "K789pLmN654;点击得手办"
    # predicted_class = predict_from_text(text)
    # print(f"Prediction: {predicted_class}")
    
    predict_from_file()


if __name__ == "__main__":
    test()
