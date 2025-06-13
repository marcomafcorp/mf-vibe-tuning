import json
import random
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), 'datasets')
RAW_FILE = os.path.join(DATA_PATH, 'sample_raw.jsonl')
CLEAN_FILE = os.path.join(DATA_PATH, 'sample_clean.jsonl')
TRAIN_FILE = os.path.join(DATA_PATH, 'train.jsonl')
VAL_FILE = os.path.join(DATA_PATH, 'val.jsonl')
TEST_FILE = os.path.join(DATA_PATH, 'test.jsonl')

# 1. Load raw data
with open(RAW_FILE, 'r', encoding='utf-8') as f:
    raw_data = [json.loads(line) for line in f]

# 2. Clean data (remove duplicates, trivial cleaning)
seen = set()
cleaned = []
for entry in raw_data:
    text = entry['text'].strip()
    if text and text not in seen:
        cleaned.append({'text': text})
        seen.add(text)

with open(CLEAN_FILE, 'w', encoding='utf-8') as f:
    for entry in cleaned:
        f.write(json.dumps(entry) + '\n')

# 3. Tokenization (simulate by splitting to words, for demo)
tokenized = [{'tokens': entry['text'].split()} for entry in cleaned]

# 4. Split into train/val/test (60/20/20)
random.shuffle(tokenized)
n = len(tokenized)
train, val, test = tokenized[:int(0.6*n)], tokenized[int(0.6*n):int(0.8*n)], tokenized[int(0.8*n):]

for fname, subset in zip([TRAIN_FILE, VAL_FILE, TEST_FILE], [train, val, test]):
    with open(fname, 'w', encoding='utf-8') as f:
        for entry in subset:
            f.write(json.dumps(entry) + '\n')

print(f"Raw: {len(raw_data)}, Clean: {len(cleaned)}, Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
