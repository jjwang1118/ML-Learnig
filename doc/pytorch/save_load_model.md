# 模型儲存與載入：state_dict vs config

## 一、`state_dict()` / `load_state_dict()` — PyTorch 原生

### 說明
`state_dict` 是一個 Python 字典，儲存模型所有**可學習參數**（權重 weights 和偏差 biases）的當前數值。它**不包含**模型的架構（class 定義）。

### 常見用法

```python
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)
    
    def forward(self, x):
        return self.fc(x)

model = MyModel()

# --- 儲存 ---
torch.save(model.state_dict(), "model_weights.pth")

# --- 載入 ---
model2 = MyModel()                              # 必須先重建相同架構
state = torch.load("model_weights.pth")
model2.load_state_dict(state)
model2.eval()                                   # 切換到推論模式
```

### 注意
- `load_state_dict()` 要求模型架構與存檔的 key/shape **完全一致**，否則會報錯。
- 加上 `strict=False` 可允許部分匹配（遷移學習常用）：
  ```python
  model.load_state_dict(state, strict=False)
  ```

---

## 二、`model.config` — HuggingFace Transformers

### 說明
`config` 是 HuggingFace `PretrainedConfig` 的物件，儲存的是模型的**架構超參數**，例如層數、hidden size、attention heads 數量等。它**不包含**訓練後的參數數值。

```python
from transformers import BertConfig, BertModel

config = BertConfig(
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
)

model = BertModel(config)
print(model.config.hidden_size)   # 768
```

### 儲存與載入

```python
# 儲存 config（產生 config.json）
config.save_pretrained("./my_model_dir/")

# 載入 config
config2 = BertConfig.from_pretrained("./my_model_dir/")
```

---

## 三、差異比較

| 比較項目 | `state_dict` / `load_state_dict` | `model.config` |
|---|---|---|
| 儲存內容 | 模型參數（weights, biases）數值 | 模型架構超參數 |
| 框架 | PyTorch 原生 | HuggingFace Transformers |
| 載入後能做什麼 | 恢復訓練好的權重 | 重新建立相同架構的模型 |
| 單獨使用能否推論 | 否（需搭配架構定義） | 否（需搭配 weights） |
| 格式 | `.pth` / `.pt`（Python dict） | `config.json` |

---

## 四、HuggingFace 完整儲存 / 載入（config + weights 一起）

HuggingFace 提供 `save_pretrained()` / `from_pretrained()` 同時處理 config 和 weights：

```python
# 儲存（同時寫出 config.json 和 pytorch_model.bin）
model.save_pretrained("./my_model_dir/")

# 載入
from transformers import BertModel
model = BertModel.from_pretrained("./my_model_dir/")
```

這等同於同時執行：
1. `config.save_pretrained()` — 儲存架構
2. `torch.save(model.state_dict(), ...)` — 儲存權重

---

## 五、PTH 檔案的本質

PTH 檔案是用 Python `pickle` 序列化的檔案，保存 `state_dict()` 時，**內容就是一個字典**：

```python
# state_dict() 的字典結構
{
    'fc1.weight': tensor([[...]]),
    'fc1.bias':   tensor([...]),
    'fc2.weight': tensor([[...]]),
    'fc2.bias':   tensor([...]),
}
```

### 查看與操作 PTH 內容

```python
import torch

# 查看 state_dict 類型與結構
state_dict = model.state_dict()
print(type(state_dict))               # <class 'collections.OrderedDict'>
print(list(state_dict.keys()))        # ['fc1.weight', 'fc1.bias', ...]

# 遍歷每個參數
for name, tensor in state_dict.items():
    print(f"{name}: {tensor.shape}")

# 載入後同樣可以查看
saved = torch.load('model.pth')
print(saved['fc1.weight'].shape)

# 部分載入（允許 key 不完全匹配）
model.load_state_dict(saved, strict=False)
```

### 三種保存方式的 PTH 內容比較

| 保存方式 | PTH 內容 |
|---------|---------|
| `torch.save(model.state_dict(), ...)` | 字典（只含參數） |
| `torch.save(model, ...)` | 完整模型物件 |
| `torch.save(checkpoint, ...)` | 自定義字典（含參數、優化器、epoch 等） |

---

## 六、總結

```
config         → 描述「模型長什麼樣子」（架構）
state_dict     → 描述「模型學到了什麼」（參數數值）

兩者缺一不可，才能完整還原一個訓練好的模型。
```
