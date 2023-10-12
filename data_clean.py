import re
from tqdm import tqdm

f = open('dataset/wiki_weapon1.txt', encoding='utf-8', errors='ignore')


def remove_non_standard_characters(input_string):
    # 定义允许的中英文标点符号的正则表达式模式
    allowed_characters_pattern = r'[a-zA-Z0-9\u4e00-\u9fa5。，？！、,?!./\\ \{ \} \[\]（）()@#$%^&*~`·￥……——\-+=|;；‘\'"<>《》\n]'

    # 使用正则表达式将所有非允许字符替换为空字符串
    result = re.sub(f'[^{allowed_characters_pattern}]+', '', input_string)
    result = re.sub(' ', '', result)
    result = re.sub(r'。{2,}', '。', result)
    result = re.sub(f'\t', '', result)

    return result


f1 = open('dataset/cleaned_data.txt', 'w', encoding='utf-8')
for line in tqdm(f):
    line = remove_non_standard_characters(line)
    # print(type(line))
    if(len(line)):
        f1.write(line)
    # print(line)
