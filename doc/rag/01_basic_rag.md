# 基礎 RAG 架構與流程

## 什麼是 RAG？

RAG（Retrieval-Augmented Generation）是一種將**資訊檢索**與**語言模型生成**結合的技術。它讓 LLM 在回答問題時，能夠從外部知識庫中動態取得相關資料，而不只依賴訓練時的參數記憶。

**解決的問題：**
- LLM 知識截止日期限制
- Hallucination（幻覺）問題
- 私有資料無法訓練進模型

---

## 基礎 RAG 三大階段

### 1. 索引階段（Indexing / Offline）

```python
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# 1. 載入文件
loader = PyPDFLoader("document.pdf")
docs = loader.load()

# 2. 切割文件
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# 3. 嵌入 + 存入向量資料庫
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("faiss_index")
```

### 2. 檢索階段（Retrieval / Online）

```python
# 載入向量資料庫
vectorstore = FAISS.load_local("faiss_index", embeddings)

# 使用者查詢
query = "什麼是 Transformer？"

# 相似度搜尋，取前 k 個
retrieved_docs = vectorstore.similarity_search(query, k=3)

for doc in retrieved_docs:
    print(doc.page_content)
```

### 3. 生成階段（Generation）

```python
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# 組合 Prompt
context = "\n\n".join([doc.page_content for doc in retrieved_docs])

prompt = ChatPromptTemplate.from_template("""
你是一個知識問答助手。根據以下資料回答問題。

資料：
{context}

問題：{question}

回答：
""")

llm = ChatOpenAI(model="gpt-4o")
chain = prompt | llm

response = chain.invoke({"context": context, "question": query})
print(response.content)
```

---

## 用 LangChain 快速建立完整 RAG Pipeline

```python
from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4o"),
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True
)

result = qa_chain.invoke({"query": "什麼是 Transformer？"})
print(result["result"])
print(result["source_documents"])
```

---

## 評估 RAG 品質的指標

| 指標 | 說明 |
|------|------|
| **Faithfulness** | 回答是否忠實於檢索到的文件 |
| **Answer Relevancy** | 回答是否切題 |
| **Context Precision** | 檢索到的文件是否相關 |
| **Context Recall** | 所有相關資料是否都被檢索到 |

常用評估工具：[RAGAS](https://github.com/explodinggradients/ragas)

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

result = evaluate(dataset, metrics=[faithfulness, answer_relevancy])
print(result)
```

---

## 相關筆記

- [02_chunking.md](02_chunking.md) — 如何切割文件
- [03_embedding.md](03_embedding.md) — 嵌入模型選擇
- [08_advanced_rag.md](08_advanced_rag.md) — 進階技術
