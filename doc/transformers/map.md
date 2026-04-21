# Dataset Map 方法詳解

## 概述

`map()` 是 Hugging Face Datasets 中最重要的方法之一，用於**對資料集中的每筆資料進行轉換處理**。它支援單筆處理和批次處理，並且可以自動利用多進程加速。

---

## 基本用法

### 單筆處理

```python
from datasets import load_dataset

dataset = load_dataset("text", data_files={"train": "./data.txt"})

# 定義處理函數
def process_example(example):
    example["text"] = example["text"].lower()  # 轉小寫
    return example

# 應用 map
processed_dataset = dataset["train"].map(process_example)
```

### 批次處理（batched=True）⭐

```python
def process_batch(examples):
    # examples["text"] 是一個 list
    examples["text"] = [text.lower() for text in examples["text"]]
    return examples

# 批次處理（更高效）
processed_dataset = dataset["train"].map(process_batch, batched=True)
```

---

## 常用參數

| 參數 | 說明 | 預設值 |
|------|------|--------|
| `function` | 處理函數 | 必填 |
| `batched` | 是否批次處理 | `False` |
| `batch_size` | 批次大小 | `1000` |
| `remove_columns` | 移除指定欄位 | `None` |
| `num_proc` | 多進程數量 | `None` |
| `load_from_cache_file` | 是否使用緩存 | `True` |

---

## 實際應用場景

### 1. Tokenization（最常見用法）⭐

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=128,
    )

# 使用 batched=True 可以大幅提升速度
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]  # 移除原始文本欄位
)
```

**處理前後對比**：
```python
# 處理前
{"text": "Hello world!"}

# 處理後
{
    "input_ids": [101, 7592, 2088, 999, 102],
    "attention_mask": [1, 1, 1, 1, 1]
}
```

### 2. 新增欄位

```python
def add_length(example):
    example["text_length"] = len(example["text"])
    return example

dataset = dataset.map(add_length)
# 現在資料有 "text" 和 "text_length" 兩個欄位
```

### 3. 過濾與清理資料

```python
def clean_text(example):
    # 移除多餘空白
    example["text"] = " ".join(example["text"].split())
    # 移除特殊字符
    example["text"] = example["text"].replace("\n", " ")
    return example

cleaned_dataset = dataset.map(clean_text)
```

### 4. 資料增強

```python
import random

def augment_data(example):
    words = example["text"].split()
    # 隨機刪除一個詞
    if len(words) > 5:
        idx = random.randint(0, len(words) - 1)
        words.pop(idx)
    example["augmented_text"] = " ".join(words)
    return example

augmented_dataset = dataset.map(augment_data)
```

### 5. 標籤處理

```python
label2id = {"positive": 1, "negative": 0, "neutral": 2}

def encode_labels(example):
    example["label"] = label2id[example["label_text"]]
    return example

dataset = dataset.map(encode_labels)
```

### 6. 批次處理多個輸出

```python
def create_pairs(examples):
    # 輸入多個欄位，輸出多個欄位
    results = {
        "combined": [],
        "word_count": []
    }
    for text in examples["text"]:
        results["combined"].append(f"[START] {text} [END]")
        results["word_count"].append(len(text.split()))
    return results

dataset = dataset.map(create_pairs, batched=True)
```

---

## 進階用法

### 多進程處理

```python
# 使用 4 個進程並行處理
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    num_proc=4  # 使用 4 個 CPU 核心
)
```

> ⚠️ **注意**: 使用 `num_proc` 時，處理函數不能使用 lambda 或閉包

### 控制批次大小

```python
# 較小的 batch_size 可減少記憶體使用
dataset = dataset.map(
    process_batch,
    batched=True,
    batch_size=100  # 每次處理 100 筆
)
```

### 移除欄位

```python
# 方法 1: 在 map 中移除
dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text", "label_text"]  # 移除多個欄位
)

# 方法 2: 移除所有原始欄位
dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset.column_names  # 移除所有原始欄位
)
```

### 禁用緩存

```python
# 每次都重新處理，不使用緩存
dataset = dataset.map(
    process_function,
    load_from_cache_file=False
)
```

### 帶索引的處理

```python
def process_with_index(example, idx):
    example["id"] = idx
    return example

dataset = dataset.map(process_with_index, with_indices=True)
```

---

## 完整 Tokenization 範例

```python
from datasets import load_dataset
from transformers import AutoTokenizer

# 載入資料集
dataset = load_dataset(
    "text",
    data_files={
        "train": "./dataset/train.txt",
        "test": "./dataset/test.txt"
    }
)

# 載入 tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# 定義 tokenize 函數
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=128,
        # padding 不在這裡做，交給 DataCollator
    )

# 對資料集進行 tokenize
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    num_proc=4,
    remove_columns=["text"],
    desc="Tokenizing"  # 顯示進度條說明
)

print(tokenized_dataset)
# DatasetDict({
#     train: Dataset({
#         features: ['input_ids', 'attention_mask'],
#         num_rows: 10000
#     })
#     test: Dataset({
#         features: ['input_ids', 'attention_mask'],
#         num_rows: 1000
#     })
# })

# 查看處理後的資料
print(tokenized_dataset["train"][0])
# {'input_ids': [15496, 995, 0, 0, ...], 'attention_mask': [1, 1, 0, 0, ...]}
```

---

## 處理流程示意

```text
原始資料集
┌────────────────────────┐
│ {"text": "Hello world"}│
│ {"text": "Hi there"}   │
│ {"text": "Good morning"}│
└────────────────────────┘
           ↓
    map(tokenize_function, batched=True)
           ↓
Tokenized 資料集
┌─────────────────────────────────────────┐
│ {"input_ids": [101, 7592, ...],         │
│  "attention_mask": [1, 1, ...]}         │
│ {"input_ids": [101, 7632, ...],         │
│  "attention_mask": [1, 1, ...]}         │
│ {"input_ids": [101, 2204, ...],         │
│  "attention_mask": [1, 1, ...]}         │
└─────────────────────────────────────────┘
```

---

## batched=True vs batched=False

### batched=False（預設）

```python
def process_single(example):
    # example 是單筆資料 (dict)
    # example["text"] 是一個字串
    return {"processed": example["text"].upper()}

dataset.map(process_single, batched=False)
```

### batched=True（推薦）⭐

```python
def process_batch(examples):
    # examples 是多筆資料 (dict of lists)
    # examples["text"] 是一個字串列表
    return {"processed": [text.upper() for text in examples["text"]]}

dataset.map(process_batch, batched=True)
```

**比較表**：

| 特性 | `batched=False` | `batched=True` |
|------|----------------|----------------|
| 輸入格式 | `{"text": "Hello"}` | `{"text": ["Hello", "Hi", ...]}` |
| 處理方式 | 逐筆處理 | 批次處理 |
| 速度 | 較慢 | 較快 ⭐ |
| 適用場景 | 簡單轉換 | Tokenization、向量化操作 |

---

## 常見錯誤

### 錯誤 1: batched 模式下返回值格式錯誤

```python
# ❌ 錯誤：batched=True 時返回單一值
def wrong_batch(examples):
    return {"text": examples["text"][0].upper()}  # 只返回一個值

# ✅ 正確：返回列表
def correct_batch(examples):
    return {"text": [t.upper() for t in examples["text"]]}  # 返回列表
```

### 錯誤 2: 忘記返回值

```python
# ❌ 錯誤：沒有 return
def no_return(example):
    example["text"] = example["text"].upper()
    # 忘記 return example

# ✅ 正確
def with_return(example):
    example["text"] = example["text"].upper()
    return example
```

### 錯誤 3: num_proc 與 lambda 衝突

```python
# ❌ 錯誤：lambda 無法 pickle
dataset.map(lambda x: {"text": x["text"].upper()}, num_proc=4)

# ✅ 正確：使用具名函數
def uppercase(x):
    return {"text": x["text"].upper()}
dataset.map(uppercase, num_proc=4)
```

---

## 效能優化建議

1. **使用 `batched=True`**: 批次處理比逐筆處理快很多
2. **使用 `num_proc`**: 利用多核 CPU 並行處理
3. **合理設定 `batch_size`**: 太大可能 OOM，太小效率低
4. **移除不需要的欄位**: 減少記憶體使用
5. **善用緩存**: 避免重複處理相同資料

---

## 參考資源

- [Hugging Face Datasets - Map](https://huggingface.co/docs/datasets/process#map)
- [Datasets 官方教程](https://huggingface.co/docs/datasets/tutorial)
