# RAG（Retrieval-Augmented Generation）詳細方法

本資料夾整理 RAG 相關的技術細節與實作方法。

## 目錄

| 檔案 | 主題 |
|------|------|
| `01_basic_rag.md` | 基礎 RAG 架構與流程 |
| `02_chunking.md` | 文件切割策略（Chunking） |
| `03_embedding.md` | 向量嵌入（Embedding）模型與技巧 |
| `04_vector_store.md` | 向量資料庫（FAISS、Chroma、Pinecone 等） |
| `05_retrieval.md` | 檢索策略（Dense、Sparse、Hybrid） |
| `06_reranking.md` | 重排序（Reranking） |
| `07_generation.md` | 生成階段（Prompt 設計與 LLM 整合） |
| `08_advanced_rag.md` | 進階 RAG（HyDE、RAPTOR、Self-RAG 等） |

## RAG 整體流程

```
原始文件
   ↓ 切割（Chunking）
文字片段（Chunks）
   ↓ 嵌入（Embedding）
向量（Vectors）
   ↓ 儲存
向量資料庫（Vector Store）

使用者查詢
   ↓ 嵌入
查詢向量
   ↓ 相似度搜尋
Top-K 相關片段
   ↓ 組合成 Prompt
LLM 生成回答
```
