# 文件切割策略（Chunking）

Chunking 是 RAG 最關鍵的前處理步驟。切太大會讓 LLM 上下文過長；切太小則語意不完整。

---

## 常見切割方法

### 1. Fixed-Size Chunking（固定大小）

最簡單的方式，按字元數或 token 數切割。

```python
from langchain.text_splitter import CharacterTextSplitter

splitter = CharacterTextSplitter(
    chunk_size=500,      # 每塊最大字元數
    chunk_overlap=50,    # 前後重疊字元數（保留上下文連貫性）
    separator="\n"
)
chunks = splitter.split_text(text)
```

**優點：** 實作簡單、速度快  
**缺點：** 可能在句子中間切斷，語意不完整

---

### 2. Recursive Character Text Splitter（遞迴切割）

LangChain 最推薦的通用切割器，依序嘗試 `\n\n` → `\n` → ` ` → `` 切割，盡量保留語意邊界。

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", ".", " ", ""]
)
chunks = splitter.split_documents(docs)
```

**優點：** 語意較完整，適合大多數文件  
**缺點：** 對結構化文件（表格、程式碼）效果一般

---

### 3. Semantic Chunking（語意切割）

根據句子間的語意相似度動態決定切割點，當相鄰句子差異大時才切割。

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain.embeddings import OpenAIEmbeddings

splitter = SemanticChunker(
    embeddings=OpenAIEmbeddings(),
    breakpoint_threshold_type="percentile",  # 或 "standard_deviation"
    breakpoint_threshold_amount=95
)
chunks = splitter.split_text(text)
```

**優點：** 語意最完整，適合長文件  
**缺點：** 需要呼叫 embedding 模型，速度較慢、成本較高

---

### 4. Document-Structure-Aware Chunking（結構感知）

針對 Markdown、HTML、程式碼等有結構的文件，按標題、段落、函式邊界切割。

```python
from langchain.text_splitter import MarkdownHeaderTextSplitter

headers_to_split_on = [
    ("#", "H1"),
    ("##", "H2"),
    ("###", "H3"),
]

splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
chunks = splitter.split_text(markdown_text)

# 每個 chunk 會帶有 metadata，例如 {"H1": "介紹", "H2": "安裝"}
```

---

### 5. Parent-Child Chunking（父子切割）

大塊（Parent）用於保留完整語意，小塊（Child）用於精準搜尋。

```python
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 父塊：大（用於最終傳給 LLM）
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
# 子塊：小（用於向量搜尋）
child_splitter = RecursiveCharacterTextSplitter(chunk_size=200)

store = InMemoryStore()
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

retriever.add_documents(docs)
results = retriever.get_relevant_documents("查詢問題")
# 搜尋時用小塊，但返回對應的大塊給 LLM
```

**優點：** 兼顧搜尋精準度與語意完整性

---

## 選擇策略參考

| 場景 | 推薦方法 |
|------|---------|
| 一般文件（PDF、TXT） | `RecursiveCharacterTextSplitter` |
| Markdown / HTML 結構文件 | `MarkdownHeaderTextSplitter` |
| 需要高語意完整性 | `SemanticChunker` |
| 追求搜尋精準 + 完整上下文 | `ParentDocumentRetriever` |
| 程式碼文件 | `Language` splitter（按函式切割） |

---

## Chunk 大小建議

| 模型上下文 | 建議 chunk_size | chunk_overlap |
|-----------|----------------|---------------|
| GPT-3.5（4K） | 300–500 tokens | 30–50 |
| GPT-4（8K–128K） | 500–1500 tokens | 50–100 |
| Claude（200K） | 1000–3000 tokens | 100–200 |

> **經驗法則：** chunk_overlap 約為 chunk_size 的 10%
