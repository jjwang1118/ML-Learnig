# Load Dataset

## 使用 Hugging Face Datasets 載入本地文本文件

### 基本用法

```python
from datasets import load_dataset

dataset_name = "your_dataset_name"

dataset = load_dataset(
    "text",
    data_files={
        "train": f"./dataset/{dataset_name}-cleaned-Train.txt",
        "test": f"./dataset/{dataset_name}-cleaned-Test.txt"
    }
)
```

### 資料集格式

載入後的資料集結構如下：

```python
dataset ≈ {
    'train': [
        {'text': 'sample text 1'},
        {'text': 'sample text 2'},
        {'text': 'sample text 3'}
    ],
    'test': [
        {'text': 'sample text 1'},
        {'text': 'sample text 2'},
        {'text': 'sample text 3'}
    ]
}


dataset=DatasetDict({
    'train': Dataset({
        features: ['text'],
        num_rows: 3
    }),
    'test': Dataset({
        features: ['text'],
        num_rows: 1
    })
})
 
# 取出全部訓練集資料
dataset['train']['text']
# 取出第一筆
dataset['train'][0]['text']
```

### 參數說明

| 參數 | 說明 |
|------|------|
| `"text"` | 指定載入文本格式的資料 |
| `data_files` | 字典格式，指定不同分割（split）的文件路徑 |
| `"train"` | 訓練集的鍵名 |
| `"test"` | 測試集的鍵名 |

### 其他載入方式

#### 1. 載入單一文件

```python
dataset = load_dataset("text", data_files="./dataset/data.txt")
# 結果會放在 dataset["train"] 中
```

#### 2. 載入多個文件到同一個 split

```python
dataset = load_dataset(
    "text",
    data_files={
        "train": ["./dataset/file1.txt", "./dataset/file2.txt", "./dataset/file3.txt"]
    }
)
```

#### 3. 使用通配符（wildcard）

```python
dataset = load_dataset(
    "text",
    data_files={
        "train": "./dataset/train_*.txt",
        "test": "./dataset/test_*.txt"
    }
)
```

#### 4. 載入 CSV 格式

```python
dataset = load_dataset(
    "csv",
    data_files={
        "train": "./dataset/train.csv",
        "test": "./dataset/test.csv"
    }
)
```

#### 5. 載入 JSON 格式

```python
dataset = load_dataset(
    "json",
    data_files={
        "train": "./dataset/train.json",
        "test": "./dataset/test.json"
    }
)
```

### 訪問資料

載入後可以這樣訪問資料：

```python
# 查看訓練集大小
print(len(dataset["train"]))

# 查看第一筆資料
print(dataset["train"][0])
# 輸出: {'text': 'sample text 1'}

# 查看前 5 筆資料
print(dataset["train"][:5])

# 遍歷資料
for example in dataset["train"]:
    print(example["text"])
```

### 資料集操作

#### 查看資料集資訊

```python
# 查看資料集結構
print(dataset)

# 查看欄位名稱
print(dataset["train"].column_names)
# 輸出: ['text']

# 查看資料集特徵
print(dataset["train"].features)
```

#### 分割資料集

```python
# 將訓練集分割為訓練集和驗證集
train_test_split = dataset["train"].train_test_split(test_size=0.1, seed=42)

train_dataset = train_test_split["train"]
val_dataset = train_test_split["test"]

print(f"Train size: {len(train_dataset)}")
print(f"Validation size: {len(val_dataset)}")
```

#### 過濾資料

```python
# 過濾掉空白或過短的文本
filtered_dataset = dataset["train"].filter(
    lambda example: len(example["text"].strip()) > 10
)
```

#### 映射（Map）處理

```python
# 對每筆資料進行處理
def process_text(example):
    example["text"] = example["text"].lower()  # 轉小寫
    return example

processed_dataset = dataset["train"].map(process_text)
```

### 完整範例

```python
from datasets import load_dataset

# 載入資料集
dataset_name = "rockyou"
dataset = load_dataset(
    "text",
    data_files={
        "train": f"./dataset/{dataset_name}-cleaned-Train.txt",
        "test": f"./dataset/{dataset_name}-cleaned-Test.txt"
    }
)

print(f"Dataset structure: {dataset}")
print(f"Train samples: {len(dataset['train'])}")
print(f"Test samples: {len(dataset['test'])}")
print(f"First example: {dataset['train'][0]}")

# 分割訓練集為訓練集和驗證集
train_val_split = dataset["train"].train_test_split(test_size=0.1, seed=42)
train_dataset = train_val_split["train"]
val_dataset = train_val_split["test"]

print(f"\nAfter split:")
print(f"Train: {len(train_dataset)} samples")
print(f"Validation: {len(val_dataset)} samples")
print(f"Test: {len(dataset['test'])} samples")
```

### 注意事項

1. **文件編碼**: 確保文本文件使用 UTF-8 編碼
2. **文件格式**: 對於 `"text"` 格式，每行會被當作一筆資料
3. **記憶體管理**: 大型資料集會使用 Arrow 格式在磁碟上緩存，不會全部載入記憶體
4. **路徑問題**: 確保文件路徑正確，支援相對路徑和絕對路徑

### 常見錯誤

#### 錯誤 1: 文件不存在
```python
FileNotFoundError: [Errno 2] No such file or directory: './dataset/...'
```
**解決**: 檢查文件路徑是否正確

#### 錯誤 2: 編碼問題
```python
UnicodeDecodeError: 'utf-8' codec can't decode byte...
```
**解決**: 指定正確的編碼
```python
dataset = load_dataset("text", data_files={"train": "file.txt"}, encoding="latin-1")
```

### 參考資源

- [Hugging Face Datasets 官方文檔](https://huggingface.co/docs/datasets)
- [Load Dataset 指南](https://huggingface.co/docs/datasets/loading)
