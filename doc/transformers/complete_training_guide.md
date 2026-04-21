# Hugging Face 模型完整訓練指南

## 目錄
1. [Transformer 與 Pipeline 介紹](#transformer-與-pipeline-介紹)
2. [Tokenizer 詳解](#tokenizer-詳解)
3. [模型架構與使用](#模型架構與使用)
4. [完整訓練流程](#完整訓練流程)

---

## Transformer 與 Pipeline 介紹

### Pipeline 快速上手

Pipeline 是 Hugging Face 提供的高階 API，可以快速執行各種 NLP 任務，無需手動處理 tokenization 和模型推理。

#### 支援的任務類型
- 文本分類（Text Classification）
- 命名實體識別（Named Entity Recognition）
- 情感分析（Sentiment Analysis）
- 文本生成（Text Generation）
- 文本摘要（Text Summarization）
- 問答系統（Question-Answering）
- 語言翻譯（Translation）
- 文本匹配（Text Matching）
- 語音識別（Speech Recognition）

#### 使用範例

```python
from transformers import pipeline

# 建立情感分析器
classifier = pipeline("sentiment-analysis")

# 進行預測
results = classifier([
    "Your kindness and generosity are truly heartwarming.",
    "Their rude behavior and disrespect left a sour taste in my mouth."
])

print(results)
# 輸出:
# [{'label': 'POSITIVE', 'score': 0.9598047137260437},
#  {'label': 'NEGATIVE', 'score': 0.9994558095932007}]
```

### Pipeline 處理流程

```text
原始文本
    ↓
1. Tokenizer 預處理
   ├─ 將輸入拆分為 tokens
   ├─ 將每個 token 映射至整數（tokenization）
   └─ 添加特殊標記（如 [CLS], [SEP]）
    ↓
2. Model 傳遞輸入
   └─ 將數字化的輸入傳入模型，獲得 logits
    ↓
3. Post-processing
   └─ 將 logits 轉換為標籤和分數
```

---

## Tokenizer 詳解

### 為什麼需要 Tokenizer？

**模型無法直接理解文字，需要將文字轉換為數字。** Tokenizer 負責將原始文本轉換為模型可以理解的數字序列（token IDs）。

> ⚠️ **重要**: 所有的預處理與模型預訓練需要以相同的方法完成，因為模型的權重是在「特定 tokenizer 規則」下學出來的。

### Tokenizer 完整處理流程

#### 方法一：逐步處理（教學用）

```python
from transformers import AutoTokenizer

# 0. 載入 tokenizer
checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

sequence = "Using a Transformer network is simple"

# 1. 將 sequence 切割成 tokens
tokens = tokenizer.tokenize(sequence)
print(tokens)
# 輸出: ['Using', 'a', 'Trans', '##former', 'network', 'is', 'simple']

# 2. 將 token 轉換成 index
ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids)
# 輸出: [7993, 170, 11303, 1200, 2443, 1110, 3014]

# 3. 將 index 轉換為輸入格式（添加特殊標記）
final_input = tokenizer.prepare_for_model(ids)
print(final_input['input_ids'])
# 輸出: [101, 7993, 170, 11303, 1200, 2443, 1110, 3014, 102]
# 101 是 [CLS]，102 是 [SEP]
```

#### 方法二：一步到位（實際使用）⭐

```python
from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# 一次處理多個句子
raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]

# 直接呼叫 tokenizer
inputs = tokenizer(
    raw_inputs,
    padding=True,        # 自動補齊長度
    truncation=True,     # 超出長度自動截斷
    return_tensors="pt"  # 返回 PyTorch tensor
)

print(inputs)
```

**輸出結果**:
```python
{
    'input_ids': tensor([
        [  101,  1045,  1005,  2310,  2042,  3403,  2005,  1037, 17662, 12172, 2607,  2026,  2878,  2166,  1012,   102],
        [  101,  1045,  5223,  2023,  2061,  2172,   999,   102,     0,     0,     0,     0,     0,     0,     0,     0]
    ]), 
    'attention_mask': tensor([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])
    # attention_mask: 1 代表真實內容；0 代表補齊後的 padding
}
```

### 關鍵參數說明

| 參數 | 說明 |
|------|------|
| `padding=True` | 將短句補 0 到與 batch 中最長句子相同長度 |
| `truncation=True` | 超出模型最大長度的部分會被截斷 |
| `return_tensors="pt"` | 返回 PyTorch tensor（`"tf"` 為 TensorFlow，`"np"` 為 NumPy） |

### ⚠️ 常見錯誤：維度問題

在將 token IDs 轉換為 tensor 時，**必須加上 batch 維度**，否則模型會報錯。

```python
# ❌ 錯誤寫法（缺少 batch 維度）
ids = [1045, 1005, 2310, 2042]
input_ids = torch.tensor(ids)  # shape: [4]
# 會導致: IndexError: Dimension out of range

# ✅ 正確寫法 1
input_ids = torch.tensor([ids])  # shape: [1, 4]

# ✅ 正確寫法 2
input_ids = torch.tensor([[1045, 1005, 2310, 2042]])
```

### 完整範例（含維度修正）

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequence = "I've been waiting for a HuggingFace course my whole life."

# 逐步處理
tokens = tokenizer.tokenize(sequence)
ids = tokenizer.convert_tokens_to_ids(tokens)

# ✅ 加上 batch 維度
input_ids = torch.tensor([ids])
print("Input IDs:", input_ids)

# 模型推理
output = model(input_ids)
print("Logits:", output.logits)
```

---

## 模型架構與使用

### AutoModel 系列

Hugging Face 提供 `AutoModel` 系列，可以根據 checkpoint 自動載入對應的模型架構。

```python
from transformers import AutoModel

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModel.from_pretrained(checkpoint)
```

### 模型架構概念圖

```text
原始文本
    ↓
Tokenizer（轉換為 input_ids, attention_mask）
    ↓
Transformer Backbone（多層 encoder/decoder）
    ├─ Layer 1: hidden_states
    ├─ Layer 2: hidden_states
    ├─ ...
    └─ Layer N: last_hidden_state
    ↓
Task-Specific Head（依任務不同）
    ├─ 分類頭
    ├─ 生成頭
    └─ 其他
    ↓
輸出（logits / 機率）
```

### 模型輸出維度

模型輸出的維度為: **[batch_size, sequence_length, hidden_size]**

- **batch_size**: 一次處理的序列數
- **sequence_length**: 序列的 token 數量
- **hidden_size**: 每個 token 的向量維度（如 BERT-base 是 768）

```python
# 獲取模型輸出
outputs = model(
    inputs,
    output_hidden_states=True  # 開啟後可獲取所有層的輸出
)

# 取得最後一層的輸出
last_hidden_state = outputs.last_hidden_state  # shape: [batch_size, seq_len, hidden_size]

# 如果開啟 output_hidden_states=True，可以取得所有層
all_hidden_states = outputs.hidden_states  # tuple，每層一個 tensor
```

### AutoModel 不同類型的 Head

根據任務不同，需要選擇不同的 `AutoModel` 類型：

#### 1. `AutoModel` - 純 Backbone
返回 hidden states，不附加任務頭。適合：
- 抽取 embedding
- 自己接下游任務的 head

```python
from transformers import AutoModel

model = AutoModel.from_pretrained(checkpoint)
outputs = model(inputs)
# 輸出: last_hidden_state, (可選) pooler_output, hidden_states
```

#### 2. `AutoModelForSequenceClassification` - 序列分類
用於文本分類、情感分析等。

```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
outputs = model(inputs)
logits = outputs.logits  # shape: [batch_size, num_labels]

# 轉換為機率
import torch.nn.functional as F
probs = F.softmax(logits, dim=-1)
```

#### 3. `AutoModelForCausalLM` - 自回歸語言模型
用於文本生成（GPT 類模型）。

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(checkpoint)
outputs = model(inputs)
logits = outputs.logits  # shape: [batch_size, seq_len, vocab_size]

# 生成文本
generated = model.generate(input_ids, max_length=50)
```

#### 4. `AutoModelForMaskedLM` - 遮罩語言模型
用於填空任務（BERT 類模型）。

```python
from transformers import AutoModelForMaskedLM

model = AutoModelForMaskedLM.from_pretrained(checkpoint)
# 適用於 MLM 預訓練或填空
```

#### 5. 其他類型
- `AutoModelForQuestionAnswering` - 問答系統
- `AutoModelForTokenClassification` - 序列標註（NER）
- `AutoModelForMultipleChoice` - 多選題

### Logits 與機率轉換

模型輸出的 `logits` 是未標準化的分數，需要轉換為機率：

```python
import torch.nn.functional as F

# 獲取 logits
outputs = model(inputs)
logits = outputs.logits  # shape: [batch_size, num_classes]

# 轉換為機率
probs = F.softmax(logits, dim=-1)

# 查看標籤對應
print(model.config.id2label)
# 輸出: {0: 'NEGATIVE', 1: 'POSITIVE'}
```

### 建置與保存模型

#### 從零建置模型

```python
from transformers import BertConfig, BertModel

# 建立配置
config = BertConfig(
    hidden_size=256,
    num_hidden_layers=4,
    num_attention_heads=4,
    intermediate_size=1024
)

# 建立模型
model = BertModel(config)
```

#### 載入預訓練模型

```python
model = AutoModel.from_pretrained("bert-base-uncased")
```

#### 保存模型

```python
model.save_pretrained("./my_model")
tokenizer.save_pretrained("./my_model")
```

---

## 完整訓練流程

### 訓練流程圖

```text
Dataset（原始資料）
    ↓
Tokenization（使用 tokenizer 處理）
    ↓
DataCollator（動態 padding + 生成 labels）
    ↓
Model（載入預訓練模型）
    ↓
LoraConfig（可選，提升訓練效率）
    ↓
TrainingArguments（設定訓練參數）
    ↓
Trainer（整合所有組件）
    ↓
train() / evaluate()（開始訓練與評估）
    ↓
保存模型
```

### 步驟 1：準備資料集與 Tokenization

> 載入資料集的詳細用法（格式、過濾、分割等）請參考 [load_dataset.md](load_dataset.md)；`map()` 的詳細用法請參考 [map.md](map.md)。

```python
from datasets import load_dataset
from transformers import AutoTokenizer

# 載入資料集
dataset = load_dataset("your_dataset")

# 載入 tokenizer（GPT2 預設沒有 pad_token，需要手動設定）
checkpoint = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenizer.pad_token = tokenizer.eos_token

# Tokenize 並套用到資料集（padding 交給 DataCollator 動態處理，節省記憶體）
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=128)

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
```

### 步驟 2：DataCollator（動態 Padding + Labels）

`DataCollatorForLanguageModeling` 負責：
1. **動態 Padding**: 將同一 batch 內的序列補齊到相同長度
2. **自動生成 Labels**: 對語言模型，`labels = input_ids`，padding 位置設為 `-100`
3. **生成 Attention Mask**: 真實 token 為 1，padding 為 0

```python
from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # GPT2 是 Causal LM，不是 Masked LM
)
```

**DataCollator 的作用示意**:
```python
# 輸入（batch 內兩個樣本，長度不同）
[
    {"input_ids": [7, 34, 22, 56]},
    {"input_ids": [12, 45]}
]
    ↓ 經過 DataCollator
# 輸出（補齊到相同長度）
{
    "input_ids": tensor([
        [7, 34, 22, 56],
        [12, 45, 50256, 50256]  # 補 EOS token
    ]),
    "attention_mask": tensor([
        [1, 1, 1, 1],
        [1, 1, 0, 0]  # padding 位置為 0
    ]),
    "labels": tensor([
        [7, 34, 22, 56],
        [12, 45, -100, -100]  # padding 位置為 -100（不計算 loss）
    ])
}
```

### 步驟 3：LoraConfig（可選，提升訓練效率）⭐

**LoRA (Low-Rank Adaptation)** 是一種高效微調技術：
- **凍結原始模型權重**（不更新）
- **只訓練少量低秩矩陣**（插入到指定層）
- 大幅減少顯存需求與訓練時間

```python
from peft import LoraConfig, TaskType, get_peft_model

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,        # GPT 類自回歸模型
    r=16,                                 # 低秩矩陣的秩（越小參數越少）
    target_modules=['c_attn', 'c_proj'],  # 只在這些層插入 LoRA
    lora_alpha=32,                        # 縮放係數（控制 LoRA 的影響力）
    lora_dropout=0.2,                     # LoRA 分支的 dropout
    bias='none',                          # 不訓練 bias
)

# 將 LoRA 應用到模型
model = get_peft_model(model, lora_config)
```

**關鍵參數**:
| 參數 | 說明 |
|------|------|
| `task_type` | 任務類型（`CAUSAL_LM` / `SEQ_2_SEQ_LM` / `SEQ_CLS` 等） |
| `r` | 低秩矩陣的秩，越小參數越少（常用 8, 16, 32） |
| `target_modules` | 要插入 LoRA 的層（需根據模型架構設定） |
| `lora_alpha` | 縮放係數，控制 LoRA 輸出強度（常設為 `2*r`） |
| `lora_dropout` | 防止過擬合 |

### 步驟 4：TrainingArguments（設定訓練參數）

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    # 基本設定
    output_dir="./checkpoint",              # checkpoint 保存路徑
    
    # 訓練設定
    num_train_epochs=3,                      # 訓練 epoch 數
    per_device_train_batch_size=8,           # 每張 GPU 的 batch size
    gradient_accumulation_steps=64,          # 梯度累積（模擬更大 batch）
    
    # 學習率與優化器
    learning_rate=5e-4,                      # 學習率
    weight_decay=0.01,                       # 權重衰減（防止過擬合）
    warmup_ratio=0.1,                        # warmup 比例
    optim="adamw_torch",                     # 優化器
    
    # 保存策略
    save_strategy="steps",                   # 按 steps 保存
    save_steps=500,                          # 每 500 步保存一次
    save_total_limit=3,                      # 最多保留 3 個 checkpoint
    
    # 評估策略
    evaluation_strategy="steps",             # 按 steps 評估
    eval_steps=500,                          # 每 500 步評估一次
    
    # 日誌設定
    logging_dir="./logs",                   # TensorBoard 日誌路徑
    logging_steps=100,                       # 每 100 步記錄一次
    report_to="tensorboard",                 # 使用 TensorBoard
    
    # 精度設定
    bf16=True,                               # 使用 BF16（節省顯存）
    # fp16=True,                             # 或使用 FP16（只能選一個）
    
    # 其他
    seed=42,                                 # 隨機種子
)
```

**關鍵參數說明**:

| 類別 | 參數 | 說明 |
|------|------|------|
| **訓練** | `num_train_epochs` | 訓練幾個 epoch |
| | `per_device_train_batch_size` | 每張 GPU 的 batch size |
| | `gradient_accumulation_steps` | 梯度累積，模擬更大的 batch size |
| **保存** | `save_strategy` | 保存策略（`steps` / `epoch` / `no`） |
| | `save_steps` | 每幾步保存一次 checkpoint |
| | `save_total_limit` | 最多保留幾個 checkpoint |
| **評估** | `evaluation_strategy` | 評估策略（`steps` / `epoch` / `no`） |
| | `eval_steps` | 每幾步評估一次 |
| **優化** | `learning_rate` | 學習率 |
| | `weight_decay` | 權重衰減（L2 正則化） |
| | `warmup_ratio` | warmup 階段佔總訓練步數的比例 |
| **日誌** | `logging_steps` | 每幾步記錄一次 |
| | `report_to` | 日誌工具（`tensorboard` / `wandb` / `none`） |
| **精度** | `fp16` / `bf16` | 半精度訓練（節省顯存，加速訓練） |

### 步驟 5：Trainer（整合所有組件）

```python
from transformers import Trainer

trainer = Trainer(
    model=model,                             # 模型
    args=training_args,                      # 訓練參數
    train_dataset=train_dataset,             # 訓練集
    eval_dataset=val_dataset,                # 驗證集
    tokenizer=tokenizer,                     # tokenizer
    data_collator=data_collator,             # DataCollator
)
```

### 步驟 6：開始訓練

```python
# 開始訓練
trainer.train()

# 保存最終模型
model.save_pretrained("./trained_model/my-model")
tokenizer.save_pretrained("./trained_model/my-model")
```

### 完整訓練範例

以下是一個完整的 GPT2 密碼生成模型訓練範例：

```python
import torch
from transformers import (
    Trainer,
    TrainingArguments,
    GPT2Tokenizer,
    GPT2LMHeadModel,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model

# ========== 1. 載入資料集 ==========
dataset = load_dataset("your_dataset")

# ========== 2. 載入 Tokenizer 與 Model ==========
checkpoint = "gpt2"
model = GPT2LMHeadModel.from_pretrained(checkpoint)
tokenizer = GPT2Tokenizer.from_pretrained(checkpoint)

# 設定 pad token（GPT2 預設沒有）
tokenizer.pad_token = tokenizer.eos_token

# 將模型移到 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"Using device: {device}")

# ========== 3. Tokenize 資料集 ==========
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=128,
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# ========== 4. DataCollator ==========
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # Causal LM
)

# ========== 5. LoRA 配置（可選）==========
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    target_modules=['c_attn', 'c_proj'],
    lora_alpha=32,
    lora_dropout=0.2,
    bias='none',
)
model = get_peft_model(model, lora_config)

# ========== 6. 分割訓練集與驗證集 ==========
train_dataset = tokenized_dataset["train"].train_test_split(test_size=0.1)
train_ds = train_dataset["train"]
val_ds = train_dataset["test"]

# ========== 7. Training Arguments ==========
training_args = TrainingArguments(
    output_dir="./checkpoint",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=64,
    learning_rate=5e-4,
    weight_decay=0.01,
    warmup_ratio=0.1,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=3,
    evaluation_strategy="steps",
    eval_steps=500,
    logging_dir="./logs",
    logging_steps=100,
    report_to="tensorboard",
    bf16=True,
    optim="adamw_torch",
    seed=42,
)

# ========== 8. Trainer ==========
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# ========== 9. 開始訓練 ==========
trainer.train()

# ========== 10. 保存模型 ==========
model.save_pretrained("./trained_model/my-gpt2-model")
tokenizer.save_pretrained("./trained_model/my-gpt2-model")

print("訓練完成！")
```

---

## 總結

### 關鍵概念回顧

1. **Pipeline**: 快速使用預訓練模型的高階 API
2. **Tokenizer**: 將文本轉換為模型可理解的數字
3. **AutoModel**: 根據任務選擇合適的模型頭
4. **DataCollator**: 動態 padding 與 labels 生成
5. **LoRA**: 高效微調技術，節省顯存
6. **TrainingArguments**: 設定訓練參數
7. **Trainer**: 整合所有組件，簡化訓練流程

### 訓練流程檢查清單

- [ ] 載入並 tokenize 資料集
- [ ] 設定 DataCollator（處理 padding 與 labels）
- [ ] 載入預訓練模型
- [ ] （可選）應用 LoRA 配置
- [ ] 設定 TrainingArguments
- [ ] 建立 Trainer
- [ ] 開始訓練（`trainer.train()`）
- [ ] 保存模型與 tokenizer

### 常見問題

**Q1: 維度錯誤 `Dimension out of range`**
- **原因**: 缺少 batch 維度
- **解決**: `torch.tensor([ids])` 而非 `torch.tensor(ids)`

**Q2: 顯存不足 (OOM)**
- **解決方法**:
  - 減少 `per_device_train_batch_size`
  - 增加 `gradient_accumulation_steps`
  - 使用 `fp16` 或 `bf16`
  - 使用 LoRA

**Q3: 訓練太慢**
- **解決方法**:
  - 使用半精度訓練（`fp16` / `bf16`）
  - 使用 LoRA（減少訓練參數）
  - 增加 batch size（需有足夠顯存）

**Q4: Loss 不下降**
- **檢查項目**:
  - 學習率是否合適（嘗試 1e-5 ~ 5e-4）
  - 資料集是否正確處理
  - 是否需要 warmup
  - 模型是否凍結了不該凍結的層

---

## 參考資源

- [Hugging Face Transformers 官方文檔](https://huggingface.co/docs/transformers)
- [PEFT (LoRA) 官方文檔](https://huggingface.co/docs/peft)
- [Datasets 官方文檔](https://huggingface.co/docs/datasets)

---

**文檔更新日期**: 2026-01-28
