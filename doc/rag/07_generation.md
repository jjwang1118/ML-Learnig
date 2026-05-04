# 生成階段（Generation）— Prompt 設計與 LLM 整合

檢索到相關文件後，如何設計 Prompt 並整合 LLM 輸出最終答案。

---

## 基本 Prompt 結構

```python
from langchain.prompts import ChatPromptTemplate

RAG_PROMPT = ChatPromptTemplate.from_template("""
你是一個專業的知識問答助手。請根據以下提供的資料來回答問題。

規則：
1. 只根據提供的資料回答，不要使用外部知識
2. 如果資料中沒有相關資訊，請明確說明「根據現有資料無法回答」
3. 回答要簡潔、清晰、有條理

參考資料：
{context}

問題：{question}

回答：
""")
```

---

## 基本 RAG Chain

```python
from langchain.chat_models import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

llm = ChatOpenAI(model="gpt-4o", temperature=0)

def format_docs(docs):
    return "\n\n---\n\n".join([
        f"[來源: {doc.metadata.get('source', '未知')}]\n{doc.page_content}"
        for doc in docs
    ])

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | RAG_PROMPT
    | llm
    | StrOutputParser()
)

answer = rag_chain.invoke("什麼是 Transformer？")
print(answer)
```

---

## 帶來源引用的 RAG Chain

```python
from langchain.schema.runnable import RunnableParallel

# 同時返回答案和來源文件
rag_chain_with_source = RunnableParallel(
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
).assign(answer=RAG_PROMPT | llm | StrOutputParser())

result = rag_chain_with_source.invoke("查詢問題")
print("答案:", result["answer"])
print("來源:", result["context"])
```

---

## Conversational RAG（帶對話歷史）

```python
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage

# Step 1: 將歷史對話整合到查詢中
contextualize_prompt = ChatPromptTemplate.from_messages([
    ("system", "根據對話歷史，將最新問題重寫為獨立的查詢（不依賴歷史）。"),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
])

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_prompt
)

# Step 2: 問答 Chain
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "根據以下資料回答問題：\n\n{context}"),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# 使用
chat_history = []
question = "什麼是 RAG？"
response = rag_chain.invoke({"input": question, "chat_history": chat_history})

chat_history.extend([
    HumanMessage(content=question),
    AIMessage(content=response["answer"])
])

# 後續問題可以利用歷史
follow_up = rag_chain.invoke({
    "input": "它和微調有什麼差別？",
    "chat_history": chat_history
})
```

---

## 串流輸出（Streaming）

```python
for chunk in rag_chain.stream("什麼是 RAG？"):
    print(chunk, end="", flush=True)
```

---

## Prompt 設計技巧

### 指定引用格式
```
回答時請在每個陳述後標注來源，格式為 [來源1]、[來源2]。
最後列出參考文獻列表。
```

### 控制幻覺
```
重要：如果提供的資料不足以回答問題，請直接說「根據現有資料，我無法回答此問題」，
不要猜測或補充資料以外的內容。
```

### 格式化輸出
```
請以以下格式回答：
**直接答案：** （一句話）
**詳細說明：** （2-3 段）
**注意事項：** （如有）
```

---

## 溫度設定建議

| 任務 | Temperature |
|------|------------|
| 事實問答（RAG） | 0 — 0.2 |
| 摘要 | 0.3 — 0.5 |
| 創意生成 | 0.7 — 1.0 |
