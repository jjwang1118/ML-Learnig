#!/usr/bin/env python3
"""Notes server — file tree, content, images, AI chat, and SSE file-watch."""

import json
import os
import queue
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import anthropic
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

ROOT     = Path(__file__).parent
DOC_ROOT = ROOT / "doc"
IMG_ROOT = ROOT / "image"
WEB_ROOT = ROOT / "web"

MIME = {
    ".html": "text/html; charset=utf-8",
    ".js":   "application/javascript",
    ".css":  "text/css",
    ".png":  "image/png",
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif":  "image/gif",
    ".svg":  "image/svg+xml",
    ".webp": "image/webp",
}

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))

# ── SSE subscriber registry ──
_subscribers: list[queue.Queue] = []
_subscribers_lock = threading.Lock()


def _broadcast(event_type: str, data: dict):
    payload = f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
    with _subscribers_lock:
        dead = []
        for q in _subscribers:
            try:
                q.put_nowait(payload)
            except queue.Full:
                dead.append(q)
        for q in dead:
            _subscribers.remove(q)


# ── Watchdog ──
class DocWatcher(FileSystemEventHandler):
    def _notify(self, event):
        if event.is_directory:
            return
        p = Path(event.src_path)
        if p.suffix not in (".md", ".ipynb"):
            return
        _broadcast("tree-changed", {"path": str(p.relative_to(ROOT))})

    on_created  = _notify
    on_deleted  = _notify
    on_moved    = _notify
    on_modified = _notify


def build_tree(base: Path) -> list:
    result = []
    for item in sorted(base.iterdir()):
        if item.name.startswith("."):
            continue
        if item.is_dir():
            result.append({
                "type": "dir",
                "name": item.name,
                "path": str(item.relative_to(ROOT)),
                "children": build_tree(item),
            })
        elif item.suffix in (".md", ".ipynb"):
            result.append({
                "type": "file",
                "name": item.name,
                "path": str(item.relative_to(ROOT)),
            })
    return result


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(f"  {self.address_string()} {fmt % args}")

    def send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", len(body))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def send_text(self, text, status=200):
        body = text.encode()
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", len(body))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def send_bytes(self, data, mime, cache=True, status=200):
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", len(data))
        self.send_header("Cache-Control", "max-age=3600" if cache else "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path   = parsed.path
        qs     = parse_qs(parsed.query)

        if path == "/api/tree":
            self.send_json(build_tree(DOC_ROOT) if DOC_ROOT.exists() else [])
            return

        if path == "/api/content":
            rel = qs.get("path", [None])[0]
            if not rel:
                self.send_text("missing path", 400)
                return
            target = (ROOT / rel).resolve()
            if not str(target).startswith(str(ROOT)):
                self.send_text("forbidden", 403)
                return
            if not target.exists():
                self.send_text("not found", 404)
                return
            self.send_text(target.read_text(encoding="utf-8"))
            return

        # SSE: real-time file-system events
        if path == "/api/watch":
            q: queue.Queue = queue.Queue(maxsize=50)
            with _subscribers_lock:
                _subscribers.append(q)

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()

            # Send initial heartbeat so browser confirms connection
            try:
                self.wfile.write(b": connected\n\n")
                self.wfile.flush()
            except Exception:
                with _subscribers_lock:
                    _subscribers.remove(q)
                return

            while True:
                try:
                    msg = q.get(timeout=20)
                    self.wfile.write(msg.encode())
                    self.wfile.flush()
                except queue.Empty:
                    # heartbeat to keep connection alive
                    try:
                        self.wfile.write(b": heartbeat\n\n")
                        self.wfile.flush()
                    except Exception:
                        break
                except Exception:
                    break

            with _subscribers_lock:
                if q in _subscribers:
                    _subscribers.remove(q)
            return

        if path.startswith("/image/"):
            img_name = path[len("/image/"):]
            img_path = (IMG_ROOT / img_name).resolve()
            if not str(img_path).startswith(str(IMG_ROOT)):
                self.send_text("forbidden", 403)
                return
            if img_path.exists():
                mime = MIME.get(img_path.suffix.lower(), "application/octet-stream")
                self.send_bytes(img_path.read_bytes(), mime)
            else:
                self.send_text("not found", 404)
            return

        if path == "/":
            path = "/index.html"
        file_path = (WEB_ROOT / path.lstrip("/")).resolve()
        if not str(file_path).startswith(str(WEB_ROOT)):
            self.send_text("forbidden", 403)
            return
        if file_path.exists() and file_path.is_file():
            mime = MIME.get(file_path.suffix, "application/octet-stream")
            cacheable = file_path.suffix not in (".html", ".js", ".css")
            self.send_bytes(file_path.read_bytes(), mime, cache=cacheable)
        else:
            self.send_text("not found", 404)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/api/chat":
            self.send_text("not found", 404)
            return

        length = int(self.headers.get("Content-Length", 0))
        body   = json.loads(self.rfile.read(length))
        message = body.get("message", "")
        context = body.get("context", "")

        if not client.api_key:
            self.send_json({"reply": "請設定 ANTHROPIC_API_KEY 環境變數後重啟伺服器。"})
            return

        system = (
            "你是一個 ML/AI 學習筆記助理。使用者正在閱讀學習筆記，你可以回答關於 "
            "PyTorch、HuggingFace Transformers 及其他 ML 框架的問題。\n"
            "回答要簡潔，優先給程式碼範例。"
        )
        if context:
            system += f"\n\n目前開啟的筆記內容：\n```\n{context[:3000]}\n```"

        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": message}],
        )
        self.send_json({"reply": resp.content[0].text})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))

    # Start watchdog observer
    DOC_ROOT.mkdir(parents=True, exist_ok=True)
    observer = Observer()
    observer.schedule(DocWatcher(), str(DOC_ROOT), recursive=True)
    observer.start()
    print(f"  Watching {DOC_ROOT}")

    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    server.allow_reuse_address = True
    print(f"  Notes server running → http://localhost:{port}\n")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("  ⚠  ANTHROPIC_API_KEY 未設定，AI 對話功能將停用\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Stopped.")
    finally:
        observer.stop()
        observer.join()
