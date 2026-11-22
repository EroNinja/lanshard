"""Minimal OpenAI-compatible HTTP front end (stdlib only).

POST /v1/chat/completions   {"messages": [...], "max_tokens", "temperature", "top_p", "stream"}
POST /v1/completions        {"prompt": "...", ...}
GET  /status                cluster, plan and node statistics
"""
from __future__ import annotations

import json
import queue
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def serve_api(coord, host: str = "0.0.0.0", port: int = 8000):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _json(self, code, obj):
            body = json.dumps(obj, default=str).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.startswith("/status"):
                return self._json(200, coord.status())
            if self.path.startswith("/v1/models"):
                return self._json(200, {"object": "list", "data": [{"id": coord.model, "object": "model"}]})
            self._json(404, {"error": "not found"})

        def do_POST(self):
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            except ValueError:
                return self._json(400, {"error": "bad json"})
            chat = self.path.startswith("/v1/chat/completions")
            if not chat and not self.path.startswith("/v1/completions"):
                return self._json(404, {"error": "not found"})
            ids = coord.encode_chat(req["messages"]) if chat else coord.tokenizer(req["prompt"])["input_ids"]
            kw = dict(max_new=int(req.get("max_tokens", 256)), temperature=float(req.get("temperature", 0.0)),
                      top_p=float(req.get("top_p", 1.0)), seed=req.get("seed"))
            cid = f"{'chatcmpl' if chat else 'cmpl'}-{uuid.uuid4().hex[:12]}"
            if not req.get("stream"):
                r = coord.submit(ids, **kw)
                out = r.future.result()
                text = coord.tokenizer.decode(out, skip_special_tokens=True)
                choice = {"index": 0, "finish_reason": r.finish_reason}
                choice.update({"message": {"role": "assistant", "content": text}} if chat else {"text": text})
                return self._json(200, {"id": cid, "object": "chat.completion" if chat else "text_completion",
                                        "created": int(time.time()), "model": coord.model, "choices": [choice],
                                        "usage": {"prompt_tokens": len(ids), "completion_tokens": len(out),
                                                  "total_tokens": len(ids) + len(out)}})
            q: "queue.Queue" = queue.Queue()
            r = coord.submit(ids, on_token=q.put, **kw)
            r.future.add_done_callback(lambda _: q.put(None))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            prev, toks = "", []
            while True:
                t = q.get()
                if t is None:
                    break
                toks.append(t)
                text = coord.tokenizer.decode(toks, skip_special_tokens=True)
                delta, prev = text[len(prev):], text
                if not delta:
                    continue
                chunk = {"id": cid, "object": "chat.completion.chunk", "model": coord.model,
                         "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}]}
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.flush()
            end = {"id": cid, "object": "chat.completion.chunk", "model": coord.model,
                   "choices": [{"index": 0, "delta": {}, "finish_reason": r.finish_reason}]}
            self.wfile.write(f"data: {json.dumps(end)}\n\ndata: [DONE]\n\n".encode())
            self.wfile.flush()
            self.close_connection = True

    srv = ThreadingHTTPServer((host, port), H)
    print(f"OpenAI-compatible API on http://{host}:{port}/v1/chat/completions", flush=True)
    srv.serve_forever()
