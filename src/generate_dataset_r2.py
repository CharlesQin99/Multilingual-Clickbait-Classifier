import random
from itertools import product

clickbait_output_file = r"../data/clickbait/r2/all_clickbait_list.txt"
non_clickbait_output_file = r"../data/non-clickbait/non_clickbait_list.txt"

negative_dataset = r"../data/non-clickbait/negative_list.txt"
positive_dataset = r"../data/clickbait/r2/positive_list.txt"

adverbs = ["快速", "快来", "马上", "立刻", "立马", "速度", "现在", "限时", "轻松", "再次", "自动"]
verbs = ["点击","领取", "复活", "收集", "抽奖", "购买", "解锁", "提现", "获取", "得到", "获得", "分享"]
nouns = ["宝箱", "奖励", "优惠", "惊喜", "代金券", "礼物", "礼品", "奖品", "红包", "财富", "福利", "大礼", "金币", "装备", "现金", "机会", "收益"]
adjectives = ["免费", "翻倍", "海量", "大量", "更多", "2倍", "双倍", "限量"]
pointers = ["这里", "这边", "此处"]
commands = ["点击","点我", "快点我", "点点我"]


def generate_positive_samples(num_per_pattern=50):
    data = []

    data += [f"{random.choice(adverbs)}{random.choice(verbs)}" for _ in range(num_per_pattern)]
    data += [f"{random.choice(adverbs)}{random.choice(verbs)}{random.choice(nouns)}" for _ in range(num_per_pattern)]
    data += [f"{random.choice(verbs)}{random.choice(nouns)}" for _ in range(num_per_pattern)]
    data += [f"{random.choice(adjectives)}{random.choice(nouns)}" for _ in range(num_per_pattern)]
    data += [f"{random.choice(adverbs)}{random.choice(verbs)}{random.choice(adjectives)}{random.choice(nouns)}" for _ in range(num_per_pattern)]
    data += [f"{random.choice(pointers)}有{random.choice(nouns)}" for _ in range(num_per_pattern)]
    data += [f"{random.choice(pointers)}{random.choice(verbs)}{random.choice(nouns)}" for _ in range(num_per_pattern)]
    data += [f"{random.choice(verbs)}{random.choice(pointers)}" for _ in range(num_per_pattern)]
    data += [f"{random.choice(nouns)}在{random.choice(pointers)}" for _ in range(num_per_pattern)]

    non_point_verbs = [v for v in verbs if "点" not in v]
    data += [f"{random.choice(commands)}{random.choice(non_point_verbs)}{random.choice(nouns)}" for _ in range(num_per_pattern)]

    return data


def generate_full_combinations():
    all_samples = []

    def sample_combinations(parts):
        return ["".join(c) for c in product(*parts)]

    all_samples += sample_combinations([adverbs, verbs])
    all_samples += sample_combinations([adverbs, verbs, nouns])
    all_samples += sample_combinations([verbs, nouns])
    all_samples += sample_combinations([adjectives, nouns])
    all_samples += sample_combinations([adverbs, verbs, adjectives, nouns])
    all_samples += sample_combinations([pointers, ["有"], nouns])
    all_samples += sample_combinations([pointers, verbs, nouns])
    all_samples += sample_combinations([verbs, pointers])
    all_samples += sample_combinations([nouns, ["在"], pointers])

    non_point_verbs = [v for v in verbs if "点" not in v]
    all_samples += sample_combinations([commands, non_point_verbs, nouns])

    return all_samples


def generate_clickbait():
    # positive_samples = generate_positive_samples(num_per_pattern=50)
    positive_samples = generate_full_combinations()
    positive_samples = list(set(positive_samples))

    with open(clickbait_output_file, "w", encoding="utf-8") as f:
        for sample in positive_samples:
            f.write(sample + "\n")

    generate_positive(positive_samples)


def generate_positive(positive_samples, short_sample_num=12000, long_sample_num=3000, single_sample_num=4000):

    with open(non_clickbait_output_file, "r", encoding="utf-8") as f:
        non_clickbait_list = [line.strip() for line in f if line.strip()]

    single_samples = random.sample(positive_samples, min(single_sample_num, len(positive_samples)))

    generated_set = set()
    generated_set.update(single_samples)

    def create_mixed_samples(num_samples, min_parts, max_parts):
        samples = []
        while len(samples) < num_samples:
            parts_count = random.randint(min_parts, max_parts)

            # 选1~2个诱导短句
            pos_parts = random.sample(positive_samples, random.randint(1, min(2, parts_count)))
            remain = parts_count - len(pos_parts)

            # 选剩余数量的非诱导短句
            neg_parts = random.sample(non_clickbait_list, remain) if remain > 0 else []
            all_parts = pos_parts + neg_parts
            random.shuffle(all_parts)

            sample = ";".join(all_parts).strip()
            if sample not in generated_set:
                samples.append(sample)
                generated_set.add(sample)
        return samples

    short_mixed_samples = create_mixed_samples(short_sample_num, 2, 3)
    long_mixed_samples = create_mixed_samples(long_sample_num, 7, 8)

    all_samples = single_samples + short_mixed_samples + long_mixed_samples
    random.shuffle(all_samples)

    with open(positive_dataset, "w", encoding="utf-8") as f:
        for sample in all_samples:
            f.write(sample + "\n")



def generate_negative(non_clickbait_set, short_sample_num=12000, long_sample_num=3000):
    non_clickbait_list = list(non_clickbait_set)
    generated_set = set()

    def create_samples(num_samples, min_parts, max_parts):
        samples = []
        while len(samples) < num_samples:
            parts_count = random.randint(min_parts, max_parts)
            parts = random.sample(non_clickbait_list, parts_count)
            sample = ";".join(parts).strip()
            if sample not in generated_set:
                samples.append(sample)
                generated_set.add(sample)
        return samples

    short_samples = create_samples(short_sample_num, 1, 3)
    long_samples = create_samples(long_sample_num, 7, 8)

    all_samples = short_samples + long_samples
    random.shuffle(all_samples)

    with open(negative_dataset, "w", encoding="utf-8") as f:
        for sample in all_samples:
            f.write(sample + "\n")


def generate_non_clickbait():
    non_clickbait_set = set()
    with open(r"E:\Personal\Master\Grade2\2024-autumn\Multilingual-Clickbait-Classifier\data\non-clickbait\extracted_texts.txt", "r",encoding="utf-8") as file:
        line_list = [line.strip() for line in file.readlines()]
        for line in line_list:
            for part in line.split(";"):
                part = part.strip()
                if part:
                    non_clickbait_set.add(part)
    
    with open(non_clickbait_output_file, "w",encoding="utf-8") as f:
        for sample in non_clickbait_set:
            f.write(sample + "\n")

    generate_negative(non_clickbait_set)



if __name__ == "__main__":
    generate_non_clickbait()
    generate_clickbait()
    
