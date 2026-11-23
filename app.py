"""GENESIS chat backend: Ollama-compatible answers plus DuckDuckGo web search."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from html.parser import HTMLParser
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse, parse_qs
import json
import os

ROOT = Path(__file__).resolve().parent
PORT = int(os.environ.get("PORT", "80"))
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "https://ollama.com").rstrip("/")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma4:31b")
OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY", "").strip()
MAX_HISTORY = 12


class SearchParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.results = []
        self.current = None
        self.capture = None
        self.depth = 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        classes = attrs.get("class", "").split()
        if tag == "a" and "result__a" in classes:
            self.current = {"title": "", "url": attrs.get("href", "")}
            self.capture = "title"
        elif self.current and tag in ("a", "div") and "result__snippet" in classes:
            self.capture = "snippet"
            self.depth = 1
        elif self.capture == "snippet" and tag == "div":
            self.depth += 1

    def handle_endtag(self, tag):
        if self.capture == "snippet" and tag == "div":
            self.depth -= 1
            if self.depth <= 0:
                self.capture = None
        elif self.capture == "title" and tag == "a":
            self.capture = None
            if self.current and self.current["title"].strip():
                self.results.append(self.current)
                self.current = None

    def handle_data(self, data):
        if self.current and self.capture in ("title", "snippet"):
            key = self.capture
            self.current[key] = (self.current.get(key, "") + " " + data).strip()


def search_web(query, limit=5):
    url = "https://html.duckduckgo.com/html/?" + urlencode({"q": query})
    request = Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; GENESIS-AI/0.1)"})
    with urlopen(request, timeout=15) as response:
        page = response.read().decode("utf-8", "replace")
    parser = SearchParser()
    parser.feed(page)
    results = []
    for item in parser.results:
        link = item.get("url", "")
        if link.startswith("//"):
            link = "https:" + link
        if "duckduckgo.com/l/?" in link:
            link = parse_qs(urlparse(link).query).get("uddg", [link])[0]
        if link.startswith("http") and item.get("title"):
            results.append({"title": item["title"], "url": link, "snippet": item.get("snippet", "")})
        if len(results) >= limit:
            break
    return results


def ask_model(messages, sources):
    if not OLLAMA_API_KEY:
        raise RuntimeError("Не настроен OLLAMA_API_KEY. Добавь ключ Ollama в переменные приложения Amvera.")
    system = (
        "Ты GENESIS AI — полезный русскоязычный ассистент. Отвечай прямо, понятно и честно. "
        "Если ниже даны результаты веб-поиска, используй их как источники фактов, не выдумывай "
        "сведения и указывай, если источники не дают ответа. В конце ответа кратко упомяни "
        "источники по названиям, если поиск использовался. Не утверждай, что искал в интернете, "
        "если результатов поиска нет."
    )
    if sources:
        context = "\n\nРЕЗУЛЬТАТЫ ВЕБ-ПОИСКА:\n" + "\n".join(
            f"{i+1}. {s['title']}\nURL: {s['url']}\nОписание: {s['snippet']}"
            for i, s in enumerate(sources)
        )
        system += context
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "system", "content": system}] + messages[-MAX_HISTORY:],
        "stream": False,
        "options": {"temperature": 0.4}
    }
    headers = {"Content-Type": "application/json", "Accept": "application/json",
               "Authorization": "Bearer " + OLLAMA_API_KEY}
    request = Request(OLLAMA_HOST + "/api/chat", data=json.dumps(payload).encode("utf-8"),
                      headers=headers, method="POST")
    with urlopen(request, timeout=180) as response:
        result = json.loads(response.read().decode("utf-8"))
    answer = result.get("message", {}).get("content", "").strip()
    if not answer:
        raise RuntimeError("Модель вернула пустой ответ.")
    return answer


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
            return self.send_json(200, {"status": "ok", "model_configured": bool(OLLAMA_API_KEY),
                                        "model": OLLAMA_MODEL, "web_search": True})
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
            history = payload.get("history", [])
            if not isinstance(history, list):
                history = []
            clean_history = []
            for item in history[-MAX_HISTORY:]:
                if isinstance(item, dict) and item.get("role") in ("user", "assistant") and isinstance(item.get("content"), str):
                    clean_history.append({"role": item["role"], "content": item["content"][:12000]})
            clean_history.append({"role": "user", "content": message})
            sources = []
            if payload.get("web_search") is True:
                try:
                    sources = search_web(message)
                except (HTTPError, URLError, TimeoutError) as error:
                    print("Web search failed:", repr(error), flush=True)
            answer = ask_model(clean_history, sources)
            return self.send_json(200, {"reply": answer, "sources": sources, "model": OLLAMA_MODEL})
        except HTTPError as error:
            detail = error.read(1500).decode("utf-8", "replace")
            return self.send_json(502, {"error": f"Ошибка AI-провайдера ({error.code}): {detail}"})
        except (URLError, TimeoutError) as error:
            return self.send_json(502, {"error": f"Не удалось связаться с AI-моделью: {error}"})
        except RuntimeError as error:
            return self.send_json(503, {"error": str(error)})
        except (ValueError, json.JSONDecodeError):
            return self.send_json(400, {"error": "Некорректный JSON-запрос."})
        except Exception as error:
            print("GENESIS server error:", repr(error), flush=True)
            return self.send_json(500, {"error": "Внутренняя ошибка сервера GENESIS."})

    def log_message(self, fmt, *args):
        print("[GENESIS]", fmt % args, flush=True)


if __name__ == "__main__":
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"GENESIS ready on port {PORT}; model configured: {bool(OLLAMA_API_KEY)}", flush=True)
    server.serve_forever()
