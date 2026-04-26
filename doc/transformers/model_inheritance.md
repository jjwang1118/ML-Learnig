# 繼承 Transformers 模型的寫法與呼叫差異

## 目錄
1. [為什麼需要繼承模型](#為什麼需要繼承模型)
2. [繼承 PreTrainedModel（最底層基底）](#繼承-pretrainedmodel最底層基底)
3. [繼承具體模型（加自訂 Head）](#繼承具體模型加自訂-head)
4. [其他常用 PreTrainedModel 子類](#其他常用-pretrainedmodel-子類)
   - [CausalLM（GPT2 / LLaMA）](#causallmgpt2--llama)
   - [MaskedLM（BERT / RoBERTa）](#maskedlmbert--roberta)
   - [Seq2SeqLM（T5 / BART）](#seq2seqlmt5--bart)
5. [覆寫 forward() 的注意事項](#覆寫-forward-的注意事項)
6. [呼叫差異對照](#呼叫差異對照)
7. [常見錯誤與排查](#常見錯誤與排查)

---

## 為什麼需要繼承模型

Hugging Face 提供的模型（如 `BertModel`、`GPT2Model`）只輸出 hidden states，若要：
- 接自訂分類頭（classifier head）
- 改變 loss 計算方式
- 新增額外輸入（如知識圖譜 embedding）

就需要繼承既有模型來擴充。

---

## 繼承 PreTrainedModel（最底層基底）

`PreTrainedModel` 提供權重載入、儲存、`.from_pretrained()` 等共用功能。

```python
from transformers import PreTrainedModel, BertConfig, BertModel

class MyBertClassifier(PreTrainedModel):
    config_class = BertConfig  # 告訴 HF 這個模型對應哪個 config

    def __init__(self, config):
        super().__init__(config)           # 必須呼叫，初始化 HF 基礎功能
        self.bert = BertModel(config)      # 建立 backbone
        self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()                   # 必須呼叫，初始化權重 & 梯度 checkpoint

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output           # [batch, hidden_size]
        logits = self.classifier(pooled)         # [batch, num_labels]

        loss = None
        if labels is not None:
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        # 回傳格式建議用 ModelOutput 或 tuple
        return (loss, logits) if loss is not None else logits
```

### 載入預訓練權重

```python
from transformers import BertConfig

config = BertConfig.from_pretrained("bert-base-uncased", num_labels=3)
model = MyBertClassifier(config)

# 只載入 BERT backbone 部分的預訓練權重
# （classifier 因為形狀不同，會被忽略並隨機初始化）
model.bert = BertModel.from_pretrained("bert-base-uncased")
```

或直接用 `.from_pretrained()` 整包載入（適合已 fine-tune 完的模型）：

```python
model = MyBertClassifier.from_pretrained("path/to/saved/model")
```

---

## 繼承具體模型（加自訂 Head）

若不需要實作 `from_pretrained` 等功能，直接繼承具體模型是最快的方式：

```python
import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel

class BertForCustomTask(BertPreTrainedModel):
    """
    繼承 BertPreTrainedModel 而非 PreTrainedModel，
    可以直接繼承 Bert 系列的初始化邏輯。
    """
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=True)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        # 會加在原模型的後面
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        **kwargs,       # 繼承後保留 **kwargs，避免多餘參數報錯
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        pooled_output = self.dropout(outputs.pooler_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return {"loss": loss, "logits": logits}
```

### 用 `.from_pretrained()` 載入

```python
model = BertForCustomTask.from_pretrained(
    "bert-base-uncased",
    num_labels=3,
    ignore_mismatched_sizes=True,  # classifier 形狀不同時不報錯
)
```

---

## 其他常用 PreTrainedModel 子類

Hugging Face 依任務類型提供多種 `PreTrainedModel` 子類，繼承對應的子類可直接獲得該任務的預設行為。

### CausalLM（GPT2 / LLaMA）

CausalLM 是自回歸語言模型（Autoregressive LM），適合文字生成、chat 等任務。  
繼承 `GPT2PreTrainedModel` 或 `LlamaPreTrainedModel`，並自訂 language modeling head。

```python
import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2PreTrainedModel, GPT2Config
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

class GPT2ForCustomCausalLM(GPT2PreTrainedModel):
    """
    繼承 GPT2PreTrainedModel，加上自訂 lm_head。
    原本 GPT2LMHeadModel 也是這樣實作的。
    """
    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPT2Model(config)           # backbone（只輸出 hidden states）
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        **kwargs,
    ):
        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
        )
        hidden_states = outputs.last_hidden_state       # [batch, seq_len, n_embd]
        logits = self.lm_head(hidden_states)            # [batch, seq_len, vocab_size]

        loss = None
        if labels is not None:
            # CausalLM：將 logits 左移一格對齊 labels
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.CrossEntropyLoss()(shift_logits.view(-1, shift_logits.size(-1)),
                                         shift_labels.view(-1))

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
        )
```

**載入 GPT2 backbone 預訓練權重：**

```python
config = GPT2Config.from_pretrained("gpt2")
model = GPT2ForCustomCausalLM(config)

# 方式一：只載入 transformer backbone
model.transformer = GPT2Model.from_pretrained("gpt2")

# 方式二：整包 from_pretrained（lm_head 不匹配時用 ignore_mismatched_sizes）
model = GPT2ForCustomCausalLM.from_pretrained("gpt2", ignore_mismatched_sizes=True)
```

**生成文字（CausalLM 特有）：**

```python
# CausalLM 繼承後自動獲得 .generate() 方法
tokenizer = AutoTokenizer.from_pretrained("gpt2")
inputs = tokenizer("Hello, my name is", return_tensors="pt")

generated = model.generate(**inputs, max_new_tokens=20, do_sample=True)
print(tokenizer.decode(generated[0]))
```

> **與 BertPreTrainedModel 的關鍵差異**：CausalLM 使用因果注意力遮罩（causal mask），  
> 每個 token 只能看到左側的 token；BERT 則是雙向注意力。

---

### MaskedLM（BERT / RoBERTa）

MaskedLM 用於填充遮蓋詞（`[MASK]`），適合 fine-tune 預訓練、語言理解任務。

```python
from transformers import BertModel, BertPreTrainedModel, BertConfig
from transformers.modeling_outputs import MaskedLMOutput

class BertForCustomMaskedLM(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=False)  # MLM 不需要 pooler
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state          # [batch, seq_len, hidden]
        logits = self.lm_head(sequence_output)               # [batch, seq_len, vocab]

        loss = None
        if labels is not None:
            # labels 中 -100 的位置會被 CrossEntropyLoss 忽略（非 MASK 位置）
            loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), labels.view(-1))

        return MaskedLMOutput(loss=loss, logits=logits)
```

---

### Seq2SeqLM（T5 / BART）

Seq2SeqLM 有 encoder 和 decoder，適合翻譯、摘要、問答生成。

```python
from transformers import T5ForConditionalGeneration, T5PreTrainedModel, T5Config
from transformers.modeling_outputs import Seq2SeqLMOutput

class T5ForCustomSeq2Seq(T5PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # T5 的 encoder-decoder 包在 T5Model 裡
        from transformers import T5Model
        self.model = T5Model(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,   # Seq2Seq 需要明確傳入 decoder 的輸入
        labels=None,
        **kwargs,
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
        )
        sequence_output = outputs.last_hidden_state
        logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss(ignore_index=-100)(
                logits.view(-1, logits.size(-1)), labels.view(-1)
            )

        return Seq2SeqLMOutput(loss=loss, logits=logits)
```

**Seq2SeqLM 呼叫時需多傳 `decoder_input_ids`：**

```python
model = T5ForCustomSeq2Seq.from_pretrained("t5-small", ignore_mismatched_sizes=True)
tokenizer = AutoTokenizer.from_pretrained("t5-small")

src = tokenizer("translate English to French: Hello", return_tensors="pt")
tgt = tokenizer("Bonjour", return_tensors="pt")

outputs = model(
    input_ids=src["input_ids"],
    attention_mask=src["attention_mask"],
    decoder_input_ids=tgt["input_ids"],
    labels=tgt["input_ids"],
)
print(outputs.loss)
```

---

### 三種子類快速對照

| | `BertPreTrainedModel` | `GPT2PreTrainedModel` | `T5PreTrainedModel` |
|---|---|---|---|
| 注意力方向 | 雙向（Masked Self-Attention） | 單向（Causal） | Encoder 雙向 / Decoder 單向 |
| 常見任務 | 分類、NER、MLM | 生成、Chat | 翻譯、摘要 |
| decoder_input_ids | 不需要 | 不需要 | **需要** |
| `.generate()` 支援 | ❌ | ✅ | ✅ |
| labels 對齊方式 | 直接對齊 | **左移一格** | 直接對齊 |

---

## 覆寫 forward() 的注意事項

| 項目 | 說明 |
|------|------|
| 必須接受 `**kwargs` | Trainer / Pipeline 可能傳額外參數 |
| `labels` 放最後 | HF Trainer 會自動注入 labels |
| 回傳格式 | 推薦回傳 `dict` 或 `ModelOutput`，含 `loss`、`logits` |
| `post_init()` | 初始化後必須呼叫，負責權重初始化與 gradient checkpointing |

---

## 呼叫差異對照

### 原始模型（未繼承）

```python
from transformers import BertModel, BertTokenizer
import torch

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

inputs = tokenizer("Hello world", return_tensors="pt")
outputs = model(**inputs)

# outputs 是 BaseModelOutputWithPoolingAndCrossAttentions
print(outputs.last_hidden_state.shape)   # [1, seq_len, 768]
print(outputs.pooler_output.shape)       # [1, 768]
# 沒有 logits，沒有 loss
```

### 繼承後的自訂模型

```python
model = BertForCustomTask.from_pretrained("bert-base-uncased", num_labels=3)

inputs = tokenizer("Hello world", return_tensors="pt")
labels = torch.tensor([1])

# 不傳 labels → 只拿 logits
outputs = model(**inputs)
print(outputs["logits"].shape)           # [1, 3]

# 傳入 labels → 自動計算 loss
outputs = model(**inputs, labels=labels)
print(outputs["loss"])                   # tensor(1.09...)
print(outputs["logits"].shape)           # [1, 3]
```

### 差異總結

| 比較點 | 原始 `BertModel` | 繼承後 `BertForCustomTask` |
|--------|-----------------|--------------------------|
| 輸出內容 | `last_hidden_state`, `pooler_output` | `logits`, `loss` |
| 是否自動算 loss | ❌ 需手動 | ✅ 傳入 `labels` 自動計算 |
| 能否直接接 Trainer | ❌ 需要包裝 | ✅ 直接使用 |
| `.from_pretrained()` | ✅ | ✅（繼承後仍保留） |
| 輸出型別 | `ModelOutput` 物件 | `dict`（或自訂 `ModelOutput`） |

---

## 常見錯誤與排查

### 1. `post_init()` 忘記呼叫

```python
# 錯誤：新增的 Linear 層沒有正確初始化
class Bad(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.head = nn.Linear(768, 2)
        # 忘記 self.post_init()
```

**解法**：`__init__` 最後一行加 `self.post_init()`。

---

### 2. `ignore_mismatched_sizes` 未設定

```
RuntimeError: Error(s) in loading state_dict for BertForCustomTask:
    size mismatch for classifier.weight: ...
```

**解法**：

```python
model = BertForCustomTask.from_pretrained(
    "bert-base-uncased",
    num_labels=5,
    ignore_mismatched_sizes=True,
)
```

---

### 3. forward() 沒有接 `**kwargs`，Trainer 報錯

```
TypeError: forward() got an unexpected keyword argument 'return_loss'
```

**解法**：`forward` 簽名改為 `def forward(self, ..., **kwargs)`。

---

### 4. 回傳 tuple 導致 Trainer 取不到 loss

Trainer 預期回傳的第一個元素是 loss：

```python
# 正確：loss 放第一個
return (loss, logits)

# 也可以用 dict，key 必須是 "loss"
return {"loss": loss, "logits": logits}
```
