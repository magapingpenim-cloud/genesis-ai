"""Small web app: serves the chat UI and forwards messages to the configured AI engine."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
import json
import os

ROOT = Path(__file__).resolve().parent
PORT = int(os.environ.get("PORT", "80"))
ENGINE_URL = os.environ.get("ENGINE_URL", "").strip()
ENGINE_API_KEY = os.environ.get("ENGINE_API_KEY", "").strip()


class Handler(BaseHTTPRequestHandler):
    def send_json(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/api/health":
            return self.send_json(200, {"status": "ok", "engine_configured": bool(ENGINE_URL)})
        if path == "/":
            path = "/index.html"
        target = (ROOT / path.lstrip("/")).resolve()
        if ROOT not in target.parents and target != ROOT:
            return self.send_error(403)
        if not target.is_file():
            return self.send_error(404)
        content_type = "text/html; charset=utf-8" if target.suffix == ".html" else "text/css; charset=utf-8" if target.suffix == ".css" else "application/javascript; charset=utf-8" if target.suffix == ".js" else "application/octet-stream"
        data = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        if self.path != "/api/chat":
            return self.send_error(404)
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 1 or length > 64_000:
                return self.send_json(413, {"error": "Сообщение слишком большое или пустое."})
            payload = json.loads(self.rfile.read(length))
            message = str(payload.get("message", "")).strip()
            if not message:
                return self.send_json(400, {"error": "Напиши сообщение."})
            if not ENGINE_URL:
                return self.send_json(503, {"error": "Движок GENESIS пока не подключён. Сообщение дошло до сервера чата; добавим адрес движка в переменную ENGINE_URL, когда подготовим его API."})
            outgoing = json.dumps({
                "message": message,
                "conversation_id": payload.get("conversation_id")
            }).encode("utf-8")
            headers = {"Content-Type": "application/json", "Accept": "application/json"}
            if ENGINE_API_KEY:
                headers["Authorization"] = "Bearer " + ENGINE_API_KEY
            request = Request(ENGINE_URL, data=outgoing, headers=headers, method="POST")
            with urlopen(request, timeout=120) as response:
                result = json.loads(response.read().decode("utf-8"))
            reply = result.get("reply") or result.get("response") or result.get("message")
            if not isinstance(reply, str):
                return self.send_json(502, {"error": "Движок ответил в неизвестном формате. Ожидается поле reply или response."})
            return self.send_json(200, {"reply": reply})
        except HTTPError as error:
            detail = error.read(2000).decode("utf-8", "replace")
            return self.send_json(502, {"error": f"Ошибка движка ({error.code}): {detail}"})
        except (URLError, TimeoutError) as error:
            return self.send_json(502, {"error": f"Не удалось связаться с движком: {error}"})
        except (ValueError, json.JSONDecodeError):
            return self.send_json(400, {"error": "Некорректный JSON-запрос."})
        except Exception as error:
            print("GENESIS server error:", repr(error), flush=True)
            return self.send_json(500, {"error": "Внутренняя ошибка сервера чата."})

    def log_message(self, fmt, *args):
        print("[GENESIS]", fmt % args, flush=True)


if __name__ == "__main__":
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"GENESIS chat listening on 0.0.0.0:{PORT}; engine configured: {bool(ENGINE_URL)}", flush=True)
    server.serve_forever()
