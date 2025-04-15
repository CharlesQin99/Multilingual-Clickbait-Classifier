from openai import OpenAI
import os
import ast
import re
import logging
import random

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

input_file = r"E:\Personal\Master\Grade2\2024-autumn\Multilingual-Clickbait-Classifier\data\raw\extracted_button_text.txt"

clickbait_output_file = r"../data/clickbait/extracted_clickbait.txt"
non_clickbait_output_file =r"../data/non-clickbait/extracted_texts.txt"

index_file = r"../data/raw/processing_index.txt"
words_file = r"../data/raw/all_words.txt"

expanded_file = r"../data/clickbait/expanded_clickbait.txt"
optimized_expanded_file = r"../data/clickbait/optimized_expanded_clickbait.txt"

sample = "点击宝箱领取奖励;每局游戏获得VIP专属宝箱；签到3倍奖励 ；免除开始、复活、钻石奖励的广告;领取奖励;立即复活;直接开始;去购买;解锁"

# 配置OpenAI客户端
openai_api_key = "EMPTY"
openai_api_base = "http://mml.yumeeu.com:6007/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

def extract_inductive_text_LLM(text):
    # 优化后的Prompt
    prompt_template = f"请仔细分析以下文本，从中提取所有包含诱导玩家点击按钮语义的文本。示例诱导文本如：{sample}。请严格以只包含诱导文本的数组形式返回结果，如果未提取到诱导文本，请返回空数组。文本：\n{{}}"
    prompt = prompt_template.format(text)
    try:
        chat_response = client.chat.completions.create(
            model="Qwen2.5-32B-Instruct",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        result = chat_response.choices[0].message.content
        logging.info(f"原始返回结果: {result}")

        # 更精确地提取列表字符串
        pattern = re.compile(r'\[.*?\]', re.DOTALL)
        matches = pattern.findall(result)
        non_empty_matches = [match for match in matches if match.strip() != "[]"]

        valid_results = []
        for match in non_empty_matches:
            match = match.replace('\n', '')
            try:
                valid_results.extend(ast.literal_eval(match))
            except (SyntaxError, ValueError):
                logging.warning(f"无法解析匹配项: {match}")
        return valid_results
    except Exception as e:
        logging.error(f"请求出错: {e}")
        return []


def read_start_index():
    if os.path.exists(index_file):
        with open(index_file, 'r', encoding='utf-8') as f:
            try:
                return int(f.read().strip())
            except ValueError:
                logging.warning("索引文件内容格式错误，将从文件开头开始处理。")
    return 0


def read_written_texts():
    written_texts = set()
    if os.path.exists(clickbait_output_file):
        with open(clickbait_output_file, 'r', encoding='utf-8') as f:
            for line in f:
                written_texts.add(line.strip())
    return written_texts


def extract_clickbait_text():
    start_index = read_start_index()
    written_texts = read_written_texts()

    with open(input_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        batch_size = 100
        for i in range(start_index, len(lines), batch_size):
            batch = lines[i:i + batch_size]
            text = ''.join(batch)
            inductive_texts = extract_inductive_text_LLM(text)
            print("***\n",inductive_texts)
            logging.info(f"提取到的诱导文本: {inductive_texts}")

            with open(clickbait_output_file, 'a', encoding='utf-8') as out_file:
                for text in inductive_texts:
                    if text not in written_texts:
                        out_file.write(text + '\n')
                        written_texts.add(text)
            logging.info(f"已处理到第 {i + batch_size} 行")

            # 记录当前处理的索引
            with open(index_file, 'w', encoding='utf-8') as f:
                f.write(str(i + batch_size))


def expand_inductive_texts():
    if not os.path.exists(clickbait_output_file):
        logging.warning("诱导文本文件不存在，无法进行扩充。")
        return

    with open(clickbait_output_file, 'r', encoding='utf-8') as f:
        inductive_texts = [line.strip() for line in f.readlines()]

    clickbait_sample = "1.XyZ321;wupin;Info;Cover;Label; 点击获奖品；speedRacer;reward\n2.导航栏；nav_bar;btn_explore;NavLink; 这里赢奖励；skyJump;prize\n3.ScrollView;pop_up;steps;PrizePool;btn_forward; 完成领金币；fireBall;coins\n4.BaseLayer;title_bar;AddFeature;CloseBtn;ShadowBg; 添加拿宝石；cloudNine;gem\n5.Carousel;btn_claim;pop_up;steps;PrizePool; 领充值福利；redPanda;bonus\n6.btn_claim;pop_up;PrizePool;pop_up_bg;btn_join; 免费获取金币；flashLight;gold"

    for i in range(0, len(inductive_texts), 10):
        batch = inductive_texts[i:i + 10]
        print(batch)
        prompt = rf"请将以下给出的 10 个初始诱导文本扩充为 100 条互相不同的包含诱导词的文本组合,在这100个生成的数据集中不许出现重复的。最后组合文本格式用分号分隔，例如：{clickbait_sample}。要求生成的诱导文本组合尽可能随机且格式多样(英文词语，中文词语，拼音的出现顺序，长度，内容都尽可能模拟真实提取文本的场景)，诱导词语可以使用同义词转换或者中英文，或者中文拼音进行扩充，无诱导含义的词语尽量使用真实游戏场景内可能出现的内容(如上述样例给出的)，最好不要第一个单词就是中文或者诱导词，顺序尽量随机。请严格按照此格式返回结果。初始诱导文本：{batch}"

        try:
            chat_response = client.chat.completions.create(
                model="Qwen2.5-32B-Instruct",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            result = chat_response.choices[0].message.content
            processed_result = re.sub(r'^\s*(?:\d+[:.])?\s*', '', result, flags=re.M)
            logging.info(f"扩充结果: {processed_result}")

            with open(expanded_file, 'a', encoding='utf-8') as out_file:
                out_file.write(processed_result + '\n')
        except Exception as e:
            logging.error(f"扩充出错: {e}")


def random_select():
    try:
        with open(words_file, 'r', encoding='utf-8') as f:
            all_words = [line.strip() for line in f.readlines()]
        if len(all_words) >= 20:
            selected_words = random.sample(all_words, 20)
        else:
            selected_words = all_words
        return selected_words
    except Exception as e:
        logging.error(f"执行 random_select 函数时出现错误: {e}")
        return []


def extract_all_words():
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    all_words = set()
    for i in range(0, len(lines), 10):
        batch = lines[i:i + 10]
        for line in batch:
            line = line.strip()
            words = line.split(';')
            all_words.update(words)

    with open(words_file, 'w', encoding='utf-8') as f:
        for word in all_words:
            f.write(word + '\n')


def optimize_clickbait_result():
    with open(expanded_file, "r", encoding = "utf-8") as f:
        inductive_texts = [line.strip() for line in f.readlines()]
        
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    extract_all_words()
    print("extract done!")
            
    for i in range(0, len(inductive_texts), 15):
        batch = inductive_texts[i:i + 15]
        print(batch)
        random_select_words = random_select()
        print("***",random_select_words)
        prompt = rf"现在请优化以下 15 条初始诱导文本组合:{batch}。优化要求：15个文本组合中中文诱导点击词语保留或者使用同语义表达进行替换，如果有重复的非诱导词语可使用以下提供的文本样例中随机选取并替换，要求替换的语言不变：{random_select_words}。不允许出现文本组合中某个固定位置(可在开头，中间，末尾)是诱导词;将单一文本组合中排列顺序打乱，保持组合的长度不变，且以分号隔开。最后输出优化后的15条文本组合数组。"
        try:
            chat_response = client.chat.completions.create(
                model="Qwen2.5-32B-Instruct",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            result = chat_response.choices[0].message.content
            processed_result = re.sub(r'^\s*(?:\d+[:.])?\s*', '', result, flags=re.M)
            logging.info(f"优化后的扩充结果: {processed_result}")

            with open(optimized_expanded_file, 'a', encoding='utf-8') as out_file:
                out_file.write(processed_result + '\n')
        except Exception as e:
                logging.error(f"扩充出错: {e}")


def formalize_clickbait_result():
    try:
        with open(input_file, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        # 过滤掉空行
        non_empty_lines = [line for line in lines if line.strip()]
        with open(input_file, 'w', encoding='utf-8') as file:
            file.writelines(non_empty_lines)

        with open(optimized_expanded_file, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        processed_lines = []
        for line in lines:
            line = line.strip()
            # 删除开头是中文、[、]、空行的行
            if not line or line[0].isascii() is False or line[0] in ['[', ']']:
                continue
            # 删除开头的引号
            if line.startswith(("'", '"',"`")):
                line = line[1:]
            # 删除结尾的逗号
            if line.endswith(','):
                line = line.rstrip(',')
            # 删除结尾的引号
            if line.endswith(("'", '"',"`")):
                line = line[:-1]
            if line:
                processed_lines.append(line)

        # 将处理后的内容写回文件
        with open(optimized_expanded_file, 'w', encoding='utf-8') as file:
            for line in processed_lines:
                file.write(line + '\n')
    except Exception as e:
        print(f"发生未知错误：{e}")


def generate_clickbait_dataset():
    extract_clickbait_text()
    expand_inductive_texts()
    optimize_clickbait_result()
    formalize_clickbait_result()


def extract_non_clickbait_text_LLM(text):
    clickbait_samples = "点击,领取,奖励;VIP专属宝箱,免除,复活,钻石奖励,购买,解锁,惊喜,免费,获取,立即领取"
    prompt_template = rf"请仔细分析以下给出的文本:{text}。其中每一行是一个已有文本组合.请筛选出其中完全不包含任何诱导玩家点击语义的词语的文本组合，如果文本组合中某个文本包含诱导玩家点击语义的词语或类似同义词的话，，删除该文本组合中的所有包含诱导点击语义的文本。例如：{clickbait_samples}。请严格以数组形式返回所有符合条件的非诱导文本组合。"
    prompt = prompt_template.format(text)
   
    try:
        chat_response = client.chat.completions.create(
            model="Qwen2.5-32B-Instruct",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        result = chat_response.choices[0].message.content
        logging.info(f"原始返回结果: {result}")

        # 使用栈来匹配最外层的 []
        stack = []
        start_index = -1
        valid_results = []
        for i, char in enumerate(result):
            if char == '[':
                if not stack:
                    start_index = i
                stack.append(char)
            elif char == ']':
                stack.pop()
                if not stack:
                    end_index = i
                    match = result[start_index:end_index + 1]
                    # 去除换行符
                    match = match.replace('\n', '')
                    try:
                        valid_results.extend(ast.literal_eval(match))
                    except (SyntaxError, ValueError):
                        logging.warning(f"无法解析匹配项: {match}")

        return valid_results
    except Exception as e:
        logging.error(f"发生错误: {e}")
        return []


def extract_non_clickbait_task():
    start_index = read_start_index()
    written_texts = read_written_texts()

    with open(input_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        batch_size = 50
        for i in range(start_index, len(lines), batch_size):
            batch = lines[i:i + batch_size]
            text = '\n'.join(batch)
            inductive_texts = extract_non_clickbait_text_LLM(text)
            print("***\n",inductive_texts)
            logging.info(f"提取到的非诱导文本: {inductive_texts}")

            with open(non_clickbait_output_file, 'a', encoding='utf-8') as out_file:
                for text in inductive_texts:
                    if text not in written_texts:
                        out_file.write(text + '\n')
                        written_texts.add(text)
            logging.info(f"已处理到第 {i + batch_size} 行")

            # 记录当前处理的索引
            with open(index_file, 'w', encoding='utf-8') as f:
                f.write(str(i + batch_size))


def filter_non_clickbait_text():
    clickbait_samples = "点击,领取,领奖,试用,礼包,点这里,翻倍,奖励,VIP,专属,宝箱,免除,复活,钻石奖励,购买,解锁,惊喜,免费,获取,立即领取,领 取,金币,无限,抽奖,获赠,加速,一键"
    clickbait_list = [word for word in clickbait_samples.split(',')]
    print(clickbait_list)

    try:
        with open(input_file, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            with open(non_clickbait_output_file, 'a', encoding='utf-8') as out_file:
                for line in lines:
                    texts = line.strip().split(';')
                    non_clickbait_texts = []
                    for text in texts:
                        is_clickbait = False
                        for keyword in clickbait_list:
                            if keyword in text:
                                is_clickbait = True
                                break
                        if not is_clickbait:
                            non_clickbait_texts.append(text)
                    if non_clickbait_texts:
                        non_clickbait_line = ';'.join(non_clickbait_texts) + '\n'
                        out_file.write(non_clickbait_line)
    except Exception as e:
        print(f"Error: {e}")



def generate_non_clickbait_dataset():
    # extract_non_clickbait_task()
    filter_non_clickbait_text()


if __name__ == "__main__":
    # generate_clickbait_dataset()
    generate_non_clickbait_dataset()
    