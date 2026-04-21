# Repair Log

---

## [2026-04-21] marked.js renderer.image API 不相容

**錯誤訊息：** `href.replace is not a function. Please report this to https://github.com/markedjs/marked`

**原因：**
marked.js v5+ 將 `renderer.image` 的參數從 `(href, title, text)` 改為單一 token 物件 `{ href, title, text }`，舊寫法直接對 token 物件呼叫 `.replace()` 導致 TypeError。

**修補方式：**
```js
// 舊（v4）
renderer.image = (href, title, text) => {
  const normalized = href.replace(...);
};

// 新（v5+）
renderer.image = (token) => {
  const href = (token.href || token).replace(...);
  const text = token.text || '';
  const title = token.title || '';
};
```

---

## [2026-04-21] highlight.js CDN 載入失敗

**錯誤訊息：** `hljs is not defined`

**原因：**
`jsdelivr` CDN 偶發性載入失敗，導致 `hljs` 全域變數未定義，後續呼叫 `hljs.highlightElement()` 直接拋出 ReferenceError。

**修補方式：**
1. 將 CDN 來源從 `jsdelivr` 改為 `cdnjs (Cloudflare)`，穩定性較高
2. 在所有呼叫 `hljs` 前加上防衛性檢查：
```js
if (typeof hljs !== 'undefined') {
  contentEl.querySelectorAll('pre code').forEach(codeEl => hljs.highlightElement(codeEl));
}
```

---

## [2026-04-21] HTTPServer 單執行緒導致 SSE 阻塞

**錯誤訊息：** 無明顯錯誤訊息，伺服器停止回應所有請求

**原因：**
Python 內建 `HTTPServer` 為單執行緒，瀏覽器建立 SSE 長連線（`/api/watch`）後佔用唯一的請求處理執行緒，導致後續所有 `/api/tree`、`/api/content` 請求全部卡死。

**修補方式：**
將 `HTTPServer` 改為 `ThreadingHTTPServer`：
```python
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
```

---
