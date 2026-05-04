# 檢索策略（Retrieval）

如何從向量資料庫中找到最相關的文件片段，是 RAG 效果的核心。

---

## 1. Dense Retrieval（稠密檢索）

最基本的方式：用 embedding 向量做餘弦相似度搜尋。

```python
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)
docs = retriever.get_relevant_documents("什麼是 attention？")
```

**優點：** 語意理解強，能處理同義詞  
**缺點：** 對精確關鍵字（如專有名詞、型號）效果較差

---

## 2. Sparse Retrieval（稀疏檢索）— BM25

基於關鍵字頻率的傳統方法（TF-IDF 的改良版）。

```python
from langchain.retrievers import BM25Retriever

bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 5

docs = bm25_retriever.get_relevant_documents("GPT-4 參數量")
```

**優點：** 對精確關鍵字、專有名詞效果好  
**缺點：** 無法理解語意，不能處理同義詞

---

## 3. Hybrid Retrieval（混合檢索）

結合 Dense + Sparse 的優點，是目前生產環境最推薦的方式。

```python
from langchain.retrievers import EnsembleRetriever

dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
sparse_retriever = BM25Retriever.from_documents(chunks, k=5)

# weights 控制兩者的權重（需加總為 1.0）
ensemble_retriever = EnsembleRetriever(
    retrievers=[sparse_retriever, dense_retriever],
    weights=[0.4, 0.6]
)

docs = ensemble_retriever.get_relevant_documents("查詢")
```

---

## 4. Multi-Query Retrieval（多查詢擴展）

讓 LLM 從不同角度生成多個查詢，再合併結果，解決單一查詢表達不準確的問題。

```python
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")
retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(),
    llm=llm
)

# 內部會自動生成 3 個不同角度的查詢，再合併去重
docs = retriever.get_relevant_documents("如何提升 RAG 效果？")
```

**原理：** 對於查詢 Q，LLM 生成 Q1, Q2, Q3，分別檢索後取聯集。

---

## 5. Contextual Compression（上下文壓縮）

先取回大塊文件，再讓 LLM 萃取出真正相關的部分，減少噪音。

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# 壓縮器：讓 LLM 只保留與查詢相關的句子
compressor = LLMChainExtractor.from_llm(llm)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectorstore.as_retriever()
)

docs = compression_retriever.get_relevant_documents("查詢")
```

---

## 6. Self-Query Retrieval（自生成過濾條件）

讓 LLM 從自然語言查詢中提取結構化的 metadata 過濾條件。

```python
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo

metadata_field_info = [
    AttributeInfo(name="source", description="文件來源", type="string"),
    AttributeInfo(name="year", description="發布年份", type="integer"),
]

self_query_retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vectorstore,
    document_contents="ML 論文內容",
    metadata_field_info=metadata_field_info,
)

# 查詢 "2023年後的 RAG 論文" 會自動生成 filter: year >= 2023
docs = self_query_retriever.get_relevant_documents("2023年後的 RAG 論文")
```

---

## 策略選擇建議

| 場景 | 推薦策略 |
|------|---------|
| 一般文件問答 | Dense（基礎）|
| 含大量專有名詞 | Hybrid（BM25 + Dense）|
| 查詢表達不清晰 | Multi-Query |
| 文件含大量噪音 | Contextual Compression |
| 文件有豐富 metadata | Self-Query |
