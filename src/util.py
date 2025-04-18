import os
import re
import json


optimized_expanded_file = r"../data/clickbait/r2/clickbait.txt"

def deduplicate_file():
    with open(optimized_expanded_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    lines = [line.strip() for line in lines]
    
    unique_lines = list(set(lines))
    with open(optimized_expanded_file, 'w', encoding='utf-8') as file:
        for line in unique_lines:
            file.write(line + '\n')
    print("文件去重完成。")

deduplicate_file()