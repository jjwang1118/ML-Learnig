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
