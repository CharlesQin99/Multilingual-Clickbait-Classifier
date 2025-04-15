import os
import json
import regex as re


# STORE_PATH = r"/home/shared/minigame/paper_evaluation/ISSTA/Ablation_123"
collect_file = r"E:\Personal\Master\Grade2\2024-autumn\Multilingual-Clickbait-Classifier\data\raw\collect_words.json"

chinese_output_file = '../data/chinese_words.txt'
other_output_file = '../data/other_words.txt'


def flatten_list(lst):
    flat_list = []
    for item in lst:
        if isinstance(item, list):
            flat_list.extend(flatten_list(item))
        else:
            flat_list.append(item)
    return flat_list


def collect_words(store_path):
    result_dict = {}
    for name in os.listdir(store_path):
        if name == "unzip":
            continue
        button_text_json = os.path.join(store_path, name, "button_text.json")
        if os.path.exists(button_text_json):
            try:
                with open(button_text_json, "r", encoding='utf-8') as file:
                    data = json.load(file)
                    text_dicts = []
                    for appid_data in data.values():
                        for item in appid_data:
                            if isinstance(item, dict) and 'text' in item:
                                text_dict = item['text']
                                new_text_dict = {}
                                for key, values in text_dict.items():
                                    all_values = []
                                    for value in values:
                                        if isinstance(value, str):
                                            all_values.append(value)
                                        elif isinstance(value, list):
                                            for sub_value in flatten_list(value):
                                                if isinstance(sub_value, str):
                                                    all_values.append(sub_value)
                                    # 去重
                                    unique_values = list(set(all_values))
                                    new_text_dict[key] = unique_values
                                text_dicts.append(new_text_dict)
                    result_dict[name] = text_dicts
            except Exception as e:
                print(f"处理 {button_text_json} 时出错: {e}")

    # 追加模式写入文件
    collect_file = "../data/collect_words.json"
    try:
        if os.path.exists(collect_file):
            with open(collect_file, 'r', encoding='utf-8') as f:
                try:
                    existing_data = json.load(f)
                except json.JSONDecodeError:
                    existing_data = []
        else:
            existing_data = []

        existing_data.append(result_dict)

        with open(collect_file, 'w', encoding='utf-8') as out_file:
            json.dump(existing_data, out_file, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"写入文件 {collect_file} 时出错: {e}")


def extract_chinese_others():
    try:
        # 读取 collect_words.json 文件
        with open(collect_file, "r", encoding="utf-8") as file:
            data = json.load(file)

            chinese_words = []
        other_words = []
        chinese_pattern = re.compile(r'[\u4e00-\u9fa5]')
        # 正则表达式，用于去除空格、换行符等无关符号
        clean_pattern = re.compile(r'[\s\n]')
        # 正则表达式，用于匹配只包含标点、特殊字符和数字的文本
        invalid_pattern = re.compile(r'^[^\p{L}]+$')   

        for sub_dict in data:
            for values in sub_dict.values():
                for value in values:
                    if isinstance(value, str):
                        # 去除无关符号
                        cleaned_value = clean_pattern.sub('', value)
                        if not cleaned_value:
                            continue
                        # 过滤掉只包含标点、特殊字符和数字的文本
                        if invalid_pattern.match(cleaned_value):
                            continue
                        if chinese_pattern.search(cleaned_value):
                            chinese_words.append(cleaned_value)
                        else:
                            other_words.append(cleaned_value)

        # 去重
        chinese_words = list(set(chinese_words))
        other_words = list(set(other_words))

        # 保存中文词语到文件
        with open(chinese_output_file, "w", encoding="utf-8") as out_file:
            for word in chinese_words:
                out_file.write(word + '\n')

        # 保存其他语言词语到文件
        with open(other_output_file, "w", encoding="utf-8") as out_file:
            for word in other_words:
                out_file.write(word + '\n')
    except Exception as e:
        print(f"Error: {e}")



def is_valid(value):
    # 去除首尾空白字符
    value = value.strip()
    # 检查是否为空字符串
    if not value:
        return False
    # 检查是否为纯数字
    if value.isdigit():
        return False
    # 检查是否为纯特殊字符
    pattern = r'^[\d\W_]+$'
    if re.fullmatch(pattern, value) and not re.search(r'[a-zA-Z\u4e00-\u9fa5]', value):
        return False
    return True



def collect_button_info(store_path):
    result_dict = []

    for name in os.listdir(store_path):
        if name == "unzip":
            continue
        button_text_json = os.path.join(store_path, name, "button_text.json")
        if os.path.exists(button_text_json):
            try:
                with open(button_text_json, "r", encoding='utf-8') as file:
                    data = json.load(file)
                    for appid_data in data.values():
                        for item in appid_data:
                            if isinstance(item, dict) and 'text' in item:
                                text_dict = item['text']
                                new_text_dict = {}
                                for key, values in text_dict.items():
                                    all_values = []
                                    for value in values:
                                        if isinstance(value, str):
                                            if is_valid(value):
                                                all_values.append(value)
                                        elif isinstance(value, list):
                                            for sub_value in flatten_list(value):
                                                if isinstance(sub_value, str) and is_valid(sub_value):
                                                    all_values.append(sub_value)
                                    # 去重
                                    unique_values = list(set(all_values))
                                    if unique_values:
                                        new_text_dict[key] = unique_values
                                result_dict.append(new_text_dict)
            except Exception as e:
                print(f"处理 {button_text_json} 时出错: {e}")

    # 追加模式写入文件
    try:
        if os.path.exists(collect_file):
            with open(collect_file, 'r', encoding='utf-8') as f:
                try:
                    existing_data = json.load(f)
                except json.JSONDecodeError:
                    existing_data = []
        else:
            existing_data = []

        print(f"写入文件: {collect_file}, 数据长度: {len(existing_data)}")
        existing_data.extend(result_dict)

        with open(collect_file, 'w', encoding='utf-8') as out_file:
            json.dump(existing_data, out_file, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"写入文件 {collect_file} 时出错: {e}")


def flatten_and_join_dict_values(dictionary):
    all_strings = []
    for values in dictionary.values():
        for value in values:
            # 去除换行符
            clean_value = value.replace('\n', '')
            all_strings.append(clean_value)
    return ";".join(all_strings)


def process_json_file():
    input_file_path = collect_file
    output_file_path = r"E:\Personal\Master\Grade2\2024-autumn\Multilingual-Clickbait-Classifier\data\raw\extracted_button_text.txt"
    try:
        with open(input_file_path, 'r', encoding='utf-8') as infile:
            data = json.load(infile)

        with open(output_file_path, 'w', encoding='utf-8') as outfile:
            for item in data:
                result = flatten_and_join_dict_values(item)
                outfile.write(result + '\n')
        print(f"处理完成，结果已写入 {output_file_path}")
    except Exception as e:
        print(f"处理文件时出错: {e}")


if __name__ == "__main__":
    WX_DIR = r"Z:\miniapp_dataset\WX_minigame_result"
    FB_DIR = r"Z:\miniapp_dataset\FB_minigame_result"
    VK_DIR = r"Z:\miniapp_dataset\VK_minigame_result"
    VIVO_100_DIR = r"Z:\miniapp_dataset\VIVO_minigame_result_100"
    # collect_words(WX_DIR)
    # collect_words(FB_DIR)
    # collect_words(VK_DIR)   
    # extract_chinese_others()

    # collect_button_info(WX_DIR)
    # collect_button_info(FB_DIR)
    # collect_button_info(VK_DIR)
    # collect_button_info(VIVO_100_DIR)
    
    process_json_file()