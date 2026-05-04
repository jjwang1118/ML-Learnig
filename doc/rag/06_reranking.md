# 重排序（Reranking）

Reranking 是在初步檢索（召回 Top-K）之後，用更精準的模型對結果重新排序，篩選出最相關的文件。

---

## 為什麼需要 Reranking？

向量搜尋速度快但精度有限；Cross-encoder 精度高但速度慢。  
標準做法：**Bi-encoder 粗排（大量召回）→ Cross-encoder 精排（精選）**

```
向量搜尋召回 Top-50 → Reranker 精排 → 取 Top-3 → 送給 LLM
```

---

## 1. Cross-Encoder Reranker（本地模型）

```python
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# 載入 Cross-encoder 模型
model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")

# 設定 Reranker（保留 top 3）
reranker = CrossEncoderReranker(model=model, top_n=3)

# 包裝成 Retriever
compression_retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=vectorstore.as_retriever(search_kwargs={"k": 20})
)

docs = compression_retriever.get_relevant_documents("查詢")
```

**推薦模型：**

| 模型 | 語言 | 特點 |
|------|------|------|
| `BAAI/bge-reranker-v2-m3` | 多語言 | 中英文皆強，主流選擇 |
| `BAAI/bge-reranker-large` | 中英文 | 效果好 |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | 英文 | 輕量快速 |

---

## 2. Cohere Reranker（API）

```python
from langchain.retrievers.document_compressors import CohereRerank
from langchain.retrievers import ContextualCompressionRetriever

reranker = CohereRerank(
    cohere_api_key="YOUR_API_KEY",
    top_n=3,
    model="rerank-multilingual-v3.0"  # 支援中文
)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=vectorstore.as_retriever(search_kwargs={"k": 20})
)
```

---

## 3. 手動使用 sentence-transformers 排序

```python
from sentence_transformers import CrossEncoder

model = CrossEncoder("BAAI/bge-reranker-v2-m3")

query = "什麼是 RAG？"
# pairs: [(query, doc1), (query, doc2), ...]
pairs = [(query, doc.page_content) for doc in retrieved_docs]
scores = model.predict(pairs)

# 排序
ranked = sorted(zip(scores, retrieved_docs), key=lambda x: x[0], reverse=True)
top_docs = [doc for score, doc in ranked[:3]]
```

---

## Bi-encoder vs Cross-encoder

| 比較項目 | Bi-encoder | Cross-encoder |
|---------|-----------|--------------|
| 工作原理 | 分別編碼 query 和 doc，計算向量相似度 | 將 query + doc 拼接後一起編碼 |
| 速度 | 快（可預先計算文件向量） | 慢（每次查詢都需配對計算） |
| 精度 | 中等 | 高 |
| 用途 | 初步召回（ANN 搜尋） | 精排（Reranking）|

---

## 完整流程範例

```python
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers import BM25Retriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# 1. 混合檢索（召回 Top-20）
dense = vectorstore.as_retriever(search_kwargs={"k": 20})
sparse = BM25Retriever.from_documents(chunks, k=20)
hybrid = EnsembleRetriever(retrievers=[sparse, dense], weights=[0.3, 0.7])

# 2. Reranking（精選 Top-3）
reranker = CrossEncoderReranker(
    model=HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3"),
    top_n=3
)

final_retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=hybrid
)

docs = final_retriever.get_relevant_documents("查詢問題")
```
