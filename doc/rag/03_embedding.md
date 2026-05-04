# 向量嵌入（Embedding）模型

Embedding 將文字轉換成高維向量，使語意相近的文字在向量空間中距離更近。

---

## 常用 Embedding 模型

### 1. OpenAI Embeddings

```python
from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
# 也可用 "text-embedding-3-large"（維度更高、效果更好）

vector = embeddings.embed_query("什麼是 RAG？")
print(len(vector))  # text-embedding-3-small: 1536 維
```

| 模型 | 維度 | 特點 |
|------|------|------|
| `text-embedding-3-small` | 1536 | 便宜快速 |
| `text-embedding-3-large` | 3072 | 效果最好 |
| `text-embedding-ada-002` | 1536 | 舊版，仍廣泛使用 |

---

### 2. HuggingFace 本地模型

不需要 API，完全在本地運行。

```python
from langchain.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",           # 多語言，支援中文
    model_kwargs={"device": "cuda"},    # 使用 GPU
    encode_kwargs={"normalize_embeddings": True}  # 建議開啟，利於餘弦相似度
)

vector = embeddings.embed_query("什麼是 RAG？")
```

**推薦模型：**

| 模型 | 語言 | 維度 | 特點 |
|------|------|------|------|
| `BAAI/bge-m3` | 多語言 | 1024 | 中英文皆強，主流選擇 |
| `BAAI/bge-large-zh-v1.5` | 中文 | 1024 | 中文優化 |
| `intfloat/multilingual-e5-large` | 多語言 | 1024 | 多語言表現穩定 |
| `sentence-transformers/all-MiniLM-L6-v2` | 英文 | 384 | 輕量快速 |

---

### 3. 直接使用 sentence-transformers

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-m3")

sentences = ["什麼是 RAG？", "RAG 是一種生成方法"]
vectors = model.encode(sentences, normalize_embeddings=True)
print(vectors.shape)  # (2, 1024)
```

---

## 相似度計算

```python
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

query_vec = embeddings.embed_query("什麼是 Transformer？")
doc_vec = embeddings.embed_query("Transformer 是一種注意力機制模型")

score = cosine_similarity(query_vec, doc_vec)
print(f"相似度：{score:.4f}")
```

---

## 批次嵌入（效率優化）

```python
# 批次嵌入文件，避免逐一呼叫 API
texts = [chunk.page_content for chunk in chunks]
vectors = embeddings.embed_documents(texts)
# 返回 List[List[float]]，長度為文件數量
```

---

## 選擇建議

| 需求 | 推薦 |
|------|------|
| 快速原型、中文 | `BAAI/bge-m3`（本地）|
| 生產環境、最佳效果 | `text-embedding-3-large`（OpenAI）|
| 資源有限 | `all-MiniLM-L6-v2` |
| 嚴格隱私（不可上雲） | 任何 HuggingFace 本地模型 |
