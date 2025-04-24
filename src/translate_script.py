import os
import re
import time
from pygtrans import Translate
from concurrent.futures import ThreadPoolExecutor, as_completed

# ========== 配置路径 ==========
positive_path = r"E:\Personal\Master\Grade2\2024-autumn\Multilingual-Clickbait-Classifier\data\clickbait\r2\positive_list.txt"
negative_path = r"E:\Personal\Master\Grade2\2024-autumn\Multilingual-Clickbait-Classifier\data\non-clickbait\negative_list.txt"

output_positive_path = r"E:\Personal\Master\Grade2\2024-autumn\Multilingual-Clickbait-Classifier\data\clickbait\r2\positive_list_translated.txt"
output_negative_path = r"E:\Personal\Master\Grade2\2024-autumn\Multilingual-Clickbait-Classifier\data\non-clickbait\negative_list_translated.txt"

client = Translate(proxies={'https': 'http://localhost:7890'})


def translate_chinese_in_text(text, retry=3, delay=1):
    chinese_texts = re.findall(r'[\u4e00-\u9fff，。！？；：“”‘’（）《》【】]+', text)
    for chinese_text in chinese_texts:
        attempt = 0
        while attempt < retry:
            try:
                translated_obj = client.translate(chinese_text, source='zh', target='en')
                translated = translated_obj.translatedText if translated_obj and hasattr(translated_obj, 'translatedText') else None
                if translated:
                    text = text.replace(chinese_text, translated)
                    print(f"原始中文: {chinese_text}, 翻译后: {translated}")
                    break
                else:
                    raise ValueError("翻译结果为空或无效")
            except Exception as e:
                print(f"[警告] 翻译失败（{attempt+1}/{retry}）: {chinese_text} -> 错误: {e}")
                attempt += 1
                time.sleep(delay)
    return text


def process_line(line):
    parts = line.strip().split(';')
    translated_parts = [translate_chinese_in_text(part) for part in parts]
    return ';'.join(translated_parts)


# ========== 主流程函数 ==========
def process_file_parallel_with_checkpoint(input_path, output_path, max_workers=8):
    # 读取原始文件和已翻译文件（用于断点续翻）
    with open(input_path, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()

    translated_lines = []
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as outfile:
            translated_lines = outfile.readlines()

    start_index = len(translated_lines)
    total_lines = len(lines)
    print(f"共有 {total_lines} 行，已完成 {start_index} 行，剩余 {total_lines - start_index} 行待翻译。")

    with open(output_path, 'a', encoding='utf-8') as outfile:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_line, lines[i]): i
                for i in range(start_index, total_lines)
            }
            future_results = [None] * (total_lines - start_index)
            for future in as_completed(futures):
                i = futures[future]
                rel_index = i - start_index
                try:
                    result = future.result()
                    future_results[rel_index] = result
                except Exception as e:
                    print(f"[错误] 第 {i} 行处理失败: {e}")
                    future_results[rel_index] = lines[i].strip()

            for line in future_results:
                outfile.write(line + '\n')

def deduplicate_file_lines(file_path):
    if not os.path.exists(file_path):
        print(f"[跳过] 文件不存在: {file_path}")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 去除前后空格后去重
    unique_lines = list(dict.fromkeys([line.strip() for line in lines]))

    with open(file_path, 'w', encoding='utf-8') as f:
        for line in unique_lines:
            f.write(line + '\n')

    print(f"[去重完成] 文件: {file_path}，原始行数: {len(lines)}，去重后行数: {len(unique_lines)}")

if __name__ == "__main__":
    # process_file_parallel_with_checkpoint(negative_path, output_negative_path)
    # process_file_parallel_with_checkpoint(positive_path, output_positive_path)

    deduplicate_file_lines(output_negative_path)
    deduplicate_file_lines(output_positive_path)
