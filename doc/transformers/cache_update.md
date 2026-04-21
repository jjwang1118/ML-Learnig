# KV Cache：新版 vs 舊版 `cache.update()` 用法

## 背景

Transformer 解碼（auto-regressive generation）時，每個 token 都需要計算 Key / Value，  
用 KV Cache 把歷史 K/V 存起來，避免重複計算。

HuggingFace Transformers **≥ 4.38** 引入了新的 `Cache` 物件體系（`DynamicCache`、`StaticCache` 等），  
並提供統一的 `cache.update()` 介面，取代了舊版直接操作 tuple 的方式。

---

## 一、舊版（< 4.38）：past_key_values 是 tuple

舊版的 KV Cache 是一個巢狀 tuple，格式為：

```
past_key_values: Tuple[Tuple[Tensor]]
# past_key_values[layer_idx] = (key, value)
```

### 手動更新方式

在自定義模型的 attention 層裡，通常這樣寫：

```python
def forward(self, hidden_states, past_key_value=None, ...):
    key = self.k_proj(hidden_states)     # (batch, heads, seq, dim)
    value = self.v_proj(hidden_states)

    if past_key_value is not None:
        # 舊版：手動 cat 拼接
        key = torch.cat([past_key_value[0], key], dim=2)
        value = torch.cat([past_key_value[1], value], dim=2)

    present_key_value = (key, value)     # 回傳新的 cache tuple
    ...
    return attn_output, present_key_value
```

### 呼叫模型時傳入 / 取出

```python
outputs = model(input_ids, past_key_values=past_key_values)
new_past = outputs.past_key_values      # tuple of tuples
```

---

## 二、新版（≥ 4.38）：Cache 物件 + `cache.update()`

新版引入 `transformers.cache_utils.DynamicCache`，提供統一介面：

```python
from transformers.cache_utils import DynamicCache

cache = DynamicCache()
```

### `cache.update()` 方法簽名

```python
key_states, value_states = cache.update(
    key_states,    # 當前 step 的 Key tensor
    value_states,  # 當前 step 的 Value tensor
    layer_idx,     # 當前是第幾層（int）
)
```

回傳的是**已拼接好歷史的完整 Key / Value**，可直接送入 attention 計算。

### 在 Attention 層裡的使用

```python
from transformers.cache_utils import DynamicCache

def forward(self, hidden_states, past_key_value: DynamicCache = None, layer_idx: int = 0, ...):
    key = self.k_proj(hidden_states)
    value = self.v_proj(hidden_states)

    if past_key_value is not None:
        # 新版：交給 cache 物件管理，自動拼接並寫入
        key, value = past_key_value.update(key, value, layer_idx)

    # key, value 已包含所有歷史 token
    attn_output = self.attention(query, key, value)
    return attn_output
```

### 呼叫模型時

```python
from transformers.cache_utils import DynamicCache

cache = DynamicCache()

outputs = model(input_ids, past_key_values=cache, use_cache=True)
# outputs.past_key_values 就是同一個 DynamicCache 物件（已就地更新）
```

---

## 三、差異比較

| 比較項目 | 舊版（tuple） | 新版（DynamicCache） |
|---|---|---|
| 資料格式 | `Tuple[Tuple[Tensor]]` | `DynamicCache` 物件 |
| 更新 Cache | 手動 `torch.cat` | `cache.update(k, v, layer_idx)` |
| 回傳值 | 新的 tuple | 同一個 cache 物件（就地更新） |
| 取得累積 K/V | `past[layer_idx][0]` / `[1]` | `cache.key_cache[layer_idx]` |
| 序列長度查詢 | `past[0][0].shape[2]` | `cache.get_seq_length()` |
| 支援靜態形狀 | 否 | 是（`StaticCache`） |
| 支援版本 | 全版本 | Transformers ≥ 4.38 |

---

## 四、其他常用 Cache 類型（新版）

| 類別 | 說明 |
|---|---|
| `DynamicCache` | 動態增長，最常用 |
| `StaticCache` | 預先分配固定大小，適合 `torch.compile` |
| `SlidingWindowCache` | 僅保留最近 N 個 token（Mistral 等模型） |
| `QuantizedCache` | 對 K/V 做量化以節省記憶體 |

載入特定 Cache 類型：

```python
from transformers.cache_utils import StaticCache

cache = StaticCache(config=model.config, max_batch_size=1, max_cache_len=512, device="cuda", dtype=torch.float16)
outputs = model(input_ids, past_key_values=cache)
```

---

## 五、取得 Cache 內容

`cache.key_cache` 和 `cache.value_cache` 都是 Python list，用 index 取值，就像 array 一樣：

```python
# 內部結構
cache.key_cache   = [ layer0_K, layer1_K, layer2_K, ... ]
cache.value_cache = [ layer0_V, layer1_V, layer2_V, ... ]
# 每個元素 shape: (batch_size, num_heads, seq_len, head_dim)
#                                           ↑ 隨生成步數增長
```

### 常用存取方式

```python
# 取得特定層
cache.key_cache[0]      # 第 0 層的所有 K，shape: (batch, heads, seq_len, dim)
cache.value_cache[0]    # 第 0 層的所有 V
cache.key_cache[-1]     # 最後一層的 K

# 取得某層的 (K, V) tuple（類似舊版 past[layer_idx]）
cache[0]                # = (key_cache[0], value_cache[0])

# 總共有幾層
len(cache)              # = num_layers

# 目前累積了幾個 token
cache.get_seq_length()  # = seq_len

# 遍歷所有層
for layer_idx in range(len(cache)):
    k = cache.key_cache[layer_idx]    # (batch, heads, seq_len, head_dim)
    v = cache.value_cache[layer_idx]
```

### Tensor 各維度說明

| 維度 | 意義 | 說明 |
|---|---|---|
| `batch_size` | 批次大小 | 同時推論幾個序列 |
| `num_heads` | attention head 數 | 模型架構決定，固定值 |
| `seq_len` | 序列長度 | **每生成一個 token 就 +1** |
| `head_dim` | 每個 head 的維度 | `hidden_size / num_heads`，固定值 |

---

## 六、向後相容

新版模型仍可接受舊版 tuple 格式作為輸入，HuggingFace 內部會自動轉換：

```python
# 傳入舊版 tuple，內部自動包成 DynamicCache
outputs = model(input_ids, past_key_values=old_tuple_cache)
```

但建議新專案統一使用 `DynamicCache` 以獲得完整功能支援。
