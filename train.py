from transformers import AutoTokenizer
from tqdm import tqdm

tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')


def tokenize_function(examples):
    return tokenizer(examples, padding="max_length", truncation=True)


tokenized_dataset = []
with open('dataset/cleaned_data.txt', 'r') as f:
    for i, line in tqdm(enumerate(f)):
        temp = tokenize_function(line)
        print(temp)
        break
        temp['labels'] = i % 5
        tokenized_dataset.append(temp)
