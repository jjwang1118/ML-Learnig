# 進階 RAG 技術

超越基礎 RAG 的進階方法，解決更複雜的檢索與生成問題。

---

## 1. HyDE（Hypothetical Document Embeddings）

**問題：** 查詢（query）和答案文件（document）的語意空間差距大。  
**方法：** 讓 LLM 先生成一個假設性的答案，用這個假設答案去搜尋，而非用原始查詢。

```python
from langchain.chains import HypotheticalDocumentEmbedder
from langchain.prompts import PromptTemplate

# HyDE Prompt：讓 LLM 寫一段假設性的文件
hyde_prompt = PromptTemplate.from_template("""
請寫一段簡短的文章段落來回答以下問題（即使你不確定答案也沒關係）：

問題：{question}

段落：
""")

hyde_embeddings = HypotheticalDocumentEmbedder.from_llm(
    llm=llm,
    base_embeddings=embeddings,
    prompt=hyde_prompt
)

# 用假設文件的向量做搜尋
vectorstore_hyde = FAISS.from_documents(chunks, hyde_embeddings)
retriever = vectorstore_hyde.as_retriever()
```

**適用場景：** 查詢是問題形式，文件是陳述形式時效果顯著提升。

---

## 2. RAPTOR（遞迴抽象處理）

**問題：** 無法跨越多個 chunk 回答需要整體理解的問題。  
**方法：** 對文件進行多層摘要，建立樹狀結構，同時索引原文和摘要。

```python
# 簡化版 RAPTOR 流程
from langchain.chat_models import ChatOpenAI

def summarize_cluster(texts, llm):
    combined = "\n\n".join(texts)
    response = llm.invoke(f"請摘要以下內容：\n\n{combined}")
    return response.content

# 1. 嵌入所有 chunks
# 2. 用 UMAP 降維 + GMM 分群
# 3. 每群生成摘要
# 4. 對摘要再次嵌入、分群、摘要（遞迴）
# 5. 所有層級的文件都加入索引
```

**適用場景：** 需要理解整份文件大意的問題（如「這份報告的主要結論是？」）。

---

## 3. Self-RAG

**方法：** 訓練一個特殊的 LLM，能夠：
1. 判斷是否需要檢索（`[Retrieve]` token）
2. 評估檢索結果是否相關（`[ISREL]` token）
3. 評估生成內容是否有根據（`[ISSUP]` token）
4. 評估整體回答品質（`[ISUSE]` token）

```python
# 用 LangGraph 實作 Self-RAG 邏輯
from langgraph.graph import StateGraph

def grade_retrieval(state):
    """評估文件是否相關"""
    question = state["question"]
    docs = state["documents"]
    
    grade_prompt = f"""
    問題：{question}
    文件：{docs[0].page_content}
    
    這份文件對回答問題有幫助嗎？請只回答 yes 或 no。
    """
    score = llm.invoke(grade_prompt).content.strip().lower()
    return "generate" if score == "yes" else "retrieve_again"

def check_hallucination(state):
    """檢查答案是否有文件支撐"""
    answer = state["answer"]
    docs = state["documents"]
    
    check_prompt = f"""
    文件：{docs[0].page_content}
    答案：{answer}
    
    答案是否完全基於文件內容？請只回答 yes 或 no。
    """
    score = llm.invoke(check_prompt).content.strip().lower()
    return "end" if score == "yes" else "regenerate"
```

---

## 4. Corrective RAG（CRAG）

**方法：** 加入文件品質評估器，當檢索品質差時自動切換到 Web 搜尋。

```python
from langgraph.graph import StateGraph, END
from langchain_community.tools.tavily_search import TavilySearchResults

web_search = TavilySearchResults(k=3)

def grade_and_correct(state):
    question = state["question"]
    docs = state["documents"]
    
    # 評估文件相關性
    scores = []
    for doc in docs:
        score = grade_document(question, doc)
        scores.append(score)
    
    if all(s == "no" for s in scores):
        # 所有文件都不相關，切換到 Web 搜尋
        web_results = web_search.invoke({"query": question})
        return {"documents": web_results, "source": "web"}
    else:
        # 過濾掉不相關的文件
        filtered = [d for d, s in zip(docs, scores) if s == "yes"]
        return {"documents": filtered, "source": "vectorstore"}
```

---

## 5. RAG-Fusion

**方法：** 生成多個查詢 → 分別檢索 → 用 RRF（倒數排名融合）合併結果。

```python
def reciprocal_rank_fusion(results_list, k=60):
    """
    results_list: List[List[Document]]，每個子列表是一次查詢的結果
    """
    fused_scores = {}
    
    for docs in results_list:
        for rank, doc in enumerate(docs):
            doc_id = doc.page_content  # 用內容作為 ID
            if doc_id not in fused_scores:
                fused_scores[doc_id] = 0
            fused_scores[doc_id] += 1 / (rank + k)
    
    # 按分數排序
    reranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return reranked

# 使用
queries = generate_multiple_queries(original_query, n=4)  # 用 LLM 生成 4 個查詢
all_results = [retriever.get_relevant_documents(q) for q in queries]
fused_results = reciprocal_rank_fusion(all_results)
```

---

## 6. GraphRAG

**方法：** 將文件解析成知識圖譜（實體 + 關係），結合圖搜尋與向量搜尋。

```python
# 使用 Microsoft GraphRAG（需要額外安裝）
# pip install graphrag

# 或使用 LlamaIndex 的 KnowledgeGraphIndex
from llama_index.core import KnowledgeGraphIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader("./docs").load_data()
index = KnowledgeGraphIndex.from_documents(
    documents,
    max_triplets_per_chunk=5
)

query_engine = index.as_query_engine(
    include_text=True,
    response_mode="tree_summarize"
)
response = query_engine.query("公司的主要競爭對手有哪些關係？")
```

**適用場景：** 實體關係複雜、需要多跳推理的問題。

---

## 方法選擇指南

| 問題類型 | 推薦方法 |
|---------|---------|
| 查詢是問題、文件是陳述 | HyDE |
| 需要全文理解 | RAPTOR |
| 需要自我評估品質 | Self-RAG |
| 文件庫不完整、可能需要網路 | CRAG |
| 單一查詢表達不足 | RAG-Fusion |
| 需要多跳推理 | GraphRAG |
