# Learning Notes Project

This project is a multi-framework learning notes collection. The goal is to organize and present notes for various ML/AI frameworks (PyTorch, Transformers, etc.) in a clear, structured way.

## Project Structure

```
doc/
  pytorch/       - PyTorch framework notes
  transformers/  - HuggingFace Transformers notes
  <framework>/   - Other frameworks go here
image/           - Screenshots and diagrams
```

## Default Behavior — Chatter

You are always acting as the **Chatter** in every conversation. Without any slash command, your default role is:

- Answer questions about ML/AI concepts, APIs, errors, and best practices at any time
- Reference existing notes under `doc/` when relevant — read them to give grounded answers
- Provide concise, working code examples when asked
- Explain error messages and tracebacks clearly
- Suggest related topics or next steps worth studying

Be concise and practical. Prefer short code snippets over long explanations.

## Specialist Agents

Three sub-agents handle specific tasks. Claude will dispatch them automatically or you can request them by name:

| Agent | Responsibility |
|-------|----------------|
| `content-reviewer` | Read new files, reorganize structure, add detailed explanations |
| `uploader` | Detect framework, move content to the correct `doc/<framework>/` location |
| `prepare` | Scan all notes, show a full snapshot with one-line summaries per file |

## Automation

### 事件驅動 Hooks（本機）

寫入 `.md` / `.ipynb` 檔案時自動觸發通知，提示應執行哪個 agent：

| 觸發時機 | 通知內容 |
|---------|---------|
| 新筆記寫入 `doc/` 內 | 建議執行 `content-reviewer` 和 `prepare` |
| 新筆記寫入 `doc/` 外 | 建議執行 `uploader` 搬到正確位置 |
| 每次開啟專案（首次對話）| 建議執行 `prepare` 查看總覽 |

設定檔：`.claude/settings.json`

---

### 定期排程（雲端）

| 項目 | 內容 |
|------|------|
| 排程 ID | `trig_01FxDYQQ1yM7yqFUPWumSykh` |
| 執行時間 | 每隔2天 23:00（台北時間） |
| 執行內容 | `prepare` agent 掃描所有筆記，產出 `weekly-summary.md` |
| 來源 repo | `github.com/jjwang1118/ML-Learnig` |
| 管理頁面 | https://claude.ai/code/scheduled/trig_01FxDYQQ1yM7yqFUPWumSykh |

> **注意**：每次新增或修改筆記後執行 `git push`，確保排程 agent 能讀到最新內容。

---

## Conventions

- Notes are written in Markdown (`.md`) or Jupyter Notebook (`.ipynb`)
- Each framework gets its own subdirectory under `doc/`
- Images go in `image/` with descriptive names
- Each note file should have a clear `# Title` at the top

## Adding New Content

1. Drop raw notes anywhere in the repo
2. Run `/review` on the file to get it organized
3. Run `/upload` to move it to the right place
4. Run `/prepare` to see the updated notebook overview
