# 向量資料庫（Vector Store）

向量資料庫負責儲存 embedding 向量，並提供高效的近似最近鄰（ANN）搜尋。

---

## FAISS（本地，Facebook AI）

適合原型開發與中小規模資料集。

```python
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

# 從文件建立
vectorstore = FAISS.from_documents(chunks, embeddings)

# 儲存到本地
vectorstore.save_local("faiss_index")

# 載入
vectorstore = FAISS.load_local("faiss_index", embeddings,
                                allow_dangerous_deserialization=True)

# 搜尋
results = vectorstore.similarity_search("查詢", k=3)
results_with_scores = vectorstore.similarity_search_with_score("查詢", k=3)
```

**特點：**
- 純 CPU/GPU 計算，不需要額外服務
- 不支援持久化後的增量更新（需重建）
- 支援 IndexFlatL2、IndexIVFFlat 等多種索引類型

---

## Chroma（本地持久化）

輕量級，支援本地持久化，適合開發階段。

```python
from langchain.vectorstores import Chroma

# 建立（自動持久化）
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db",
    collection_name="my_collection"
)

# 載入已有資料庫
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
    collection_name="my_collection"
)

# 新增文件（支援增量更新）
vectorstore.add_documents(new_chunks)

# 刪除
vectorstore.delete(ids=["id1", "id2"])
```

---

## Pinecone（雲端託管）

生產環境首選，全託管、自動擴展。

```python
from langchain.vectorstores import Pinecone
import pinecone

pinecone.init(api_key="YOUR_API_KEY", environment="us-east-1-aws")

# 建立 index（只需一次）
pinecone.create_index("rag-index", dimension=1536, metric="cosine")

# 從文件建立
vectorstore = Pinecone.from_documents(
    chunks,
    embeddings,
    index_name="rag-index"
)

# 連接已有 index
vectorstore = Pinecone.from_existing_index("rag-index", embeddings)
```

---

## Qdrant（本地或雲端）

功能豐富，支援 payload 過濾，適合需要條件過濾的場景。

```python
from langchain.vectorstores import Qdrant
from qdrant_client import QdrantClient

# 本地模式
client = QdrantClient(path="./qdrant_local")

vectorstore = Qdrant.from_documents(
    chunks,
    embeddings,
    path="./qdrant_local",
    collection_name="my_collection"
)

# 帶 metadata 過濾的搜尋
from qdrant_client.models import Filter, FieldCondition, MatchValue

results = vectorstore.similarity_search(
    "查詢",
    k=3,
    filter=Filter(
        must=[FieldCondition(key="source", match=MatchValue(value="doc1.pdf"))]
    )
)
```

---

## 搜尋模式比較

```python
# 1. 相似度搜尋（最常用）
docs = vectorstore.similarity_search(query, k=3)

# 2. 帶分數的搜尋
docs_scores = vectorstore.similarity_search_with_score(query, k=3)
for doc, score in docs_scores:
    print(f"Score: {score:.4f} | {doc.page_content[:50]}")

# 3. MMR（最大邊際相關性）——增加多樣性，避免重複內容
docs = vectorstore.max_marginal_relevance_search(
    query, k=3, fetch_k=10, lambda_mult=0.5
)

# 4. 轉換為 Retriever
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "fetch_k": 10}
)
```

---

## 選擇建議

| 場景 | 推薦 |
|------|------|
| 快速原型 / 小資料 | FAISS |
| 本地開發需持久化 | Chroma |
| 生產環境（需擴展） | Pinecone / Qdrant Cloud |
| 需要複雜過濾條件 | Qdrant |
| 自建基礎設施 | Weaviate / Milvus |
