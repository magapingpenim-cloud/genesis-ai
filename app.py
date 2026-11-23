"""GENESIS v0.1 — self-written rule engine and web-search answer composer.
No hosted language-model API and no third-party Python packages.
"""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from html.parser import HTMLParser
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse, parse_qs
import ast
import operator
import json
import os
import re
import html

ROOT = Path(__file__).resolve().parent
PORT = int(os.environ.get("PORT", "80"))
MAX_HISTORY = 12
OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def safe_calculate(text):
    """Evaluate only basic arithmetic; never execute arbitrary Python."""
    candidate = text.strip().replace(",", ".").replace("×", "*").replace("÷", "/").replace("^", "**")
    candidate = re.sub(r"(?i)^(сколько будет|посчитай|вычисли|реши)\s*", "", candidate).strip(" ?=")
    if not candidate or len(candidate) > 100 or not re.fullmatch(r"[\d\s.+*/()%\-]+", candidate):
        return None
    try:
        tree = ast.parse(candidate, mode="eval")
        def visit(node):
            if isinstance(node, ast.Expression):
                return visit(node.body)
            if isinstance(node, ast.Constant) and type(node.value) in (int, float):
                return node.value
            if isinstance(node, ast.BinOp) and type(node.op) in OPS:
                left, right = visit(node.left), visit(node.right)
                if isinstance(node.op, ast.Pow) and abs(right) > 8:
                    raise ValueError("exponent too large")
                value = OPS[type(node.op)](left, right)
                if abs(value) > 10**15:
                    raise ValueError("result too large")
                return value
            if isinstance(node, ast.UnaryOp) and type(node.op) in OPS:
                return OPS[type(node.op)](visit(node.operand))
            raise ValueError("unsupported expression")
        result = visit(tree)
        return str(int(result)) if isinstance(result, float) and result.is_integer() else str(result)
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError):
        return None


class DuckParser(HTMLParser):
    """Extract result titles and snippets from DuckDuckGo HTML."""
    def __init__(self):
        super().__init__()
        self.items = []
        self.current = None
        self.capture = None
        self.snippet_depth = 0

    def _finish_current(self):
        if self.current and self.current.get("title", "").strip():
            self.items.append(self.current)
        self.current = None
        self.capture = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        classes = attrs.get("class", "").split()
        if tag == "a" and "result__a" in classes:
            self._finish_current()
            self.current = {"title": "", "url": attrs.get("href", ""), "snippet": ""}
            self.capture = "title"
        elif self.current and "result__snippet" in classes:
            self.capture = "snippet"
            self.snippet_depth = 1
        elif self.capture == "snippet" and tag in ("div", "span"):
            self.snippet_depth += 1

    def handle_endtag(self, tag):
        if self.capture == "title" and tag == "a":
            self.capture = None
        elif self.capture == "snippet" and tag in ("div", "span"):
            self.snippet_depth -= 1
            if self.snippet_depth <= 0:
                self.capture = None

    def handle_data(self, data):
        if self.current and self.capture in ("title", "snippet"):
            key = self.capture
            self.current[key] = (self.current[key] + " " + data).strip()

    def close(self):
        super().close()
        self._finish_current()


def search_web(query, limit=5):
    url = "https://html.duckduckgo.com/html/?" + urlencode({"q": query})
    request = Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; GENESIS-AI/0.1)"})
    with urlopen(request, timeout=18) as response:
        page = response.read().decode("utf-8", "replace")
    parser = DuckParser()
    parser.feed(page)
    results, seen = [], set()
    for item in parser.items:
        link = html.unescape(item["url"])
        if link.startswith("//"):
            link = "https:" + link
        if "duckduckgo.com/l/?" in link:
            link = parse_qs(urlparse(link).query).get("uddg", [link])[0]
        if link.startswith("http") and link not in seen and item["title"]:
            seen.add(link)
            results.append({
                "title": re.sub(r"\s+", " ", item["title"]).strip(),
                "url": link,
                "snippet": re.sub(r"\s+", " ", item["snippet"]).strip()
            })
        if len(results) >= limit:
            break
    return results


def compose_search_answer(query, sources):
    """Compose a transparent extractive response from search snippets, not an LLM."""
    if not sources:
        return ("Я выполнил поиск, но не получил подходящих результатов. "
                "Попробуй переформулировать запрос или повторить позже.")
    lines = ["Вот что удалось найти по запросу «" + query + "»:"]
    for item in sources[:4]:
        snippet = item["snippet"] or "Открой источник, чтобы прочитать подробности."
        lines.append("• " + item["title"] + " — " + snippet)
    lines.append("\nЭто краткая сводка найденных описаний, а не ответ обученной нейросети. "
                 "Открой источники ниже, чтобы проверить детали.")
    return "\n".join(lines)


def answer_locally(message, history=None):
    """Conversational rules; unknown factual questions are routed to web search."""
    text = message.strip()
    lower = re.sub(r"\s+", " ", text.lower()).strip(" .,!?:;")
    calculation = safe_calculate(text)
    if calculation is not None:
        return "Результат: " + calculation, "calculator"

    if re.search(r"\b(привет|здравствуй|здравствуйте|хай|hello|hi)\b", lower):
        return "Привет! 👋 Рад тебя видеть. Что будем делать?", "conversation"
    if re.search(r"\b(как дела|как ты|как поживаешь|как жизнь|ты как)\b", lower):
        return "У меня всё отлично — я на связи и готов помогать! А у тебя как дела?", "conversation"
    if re.search(r"\b(что нового|что новенького)\b", lower):
        return "Пока развиваюсь: учусь лучше обрабатывать сообщения, искать информацию и отвечать точнее. А у тебя что нового?", "conversation"
    if re.search(r"\b(доброе утро|добрый день|добрый вечер|спокойной ночи)\b", lower):
        return "И тебе хорошего дня! 😊 Чем займёмся?", "conversation"
    if re.search(r"\b(пока|до свидания|увидимся|бай)\b", lower):
        return "Пока! Буду здесь, когда понадоблюсь. 👋", "conversation"
    if re.search(r"\b(спасибо|благодарю|thanks)\b", lower):
        return "Пожалуйста! Рад помочь 🙂", "conversation"
    if re.search(r"\b(кто ты|что ты умеешь|о себе|что такое genesis|как тебя зовут)\b", lower):
        return ("Я GENESIS — экспериментальный ассистент с собственным кодом. "
                "Сейчас умею поддерживать базовый диалог, считать и искать фактическую информацию в интернете. "
                "Собственную обучаемую языковую модель мы ещё создаём."), "conversation"
    if re.search(r"\b(классно|круто|молодец|супер)\b", lower):
        return "Спасибо! 😄 Продолжаем?", "conversation"
    return None, "needs_search"

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
            return self.send_json(200, {"status": "ok", "engine": "GENESIS-native-v0.2",
                                        "model": "none", "web_search": True})
        if path == "/":
            path = "/index.html"
        target = (ROOT / path.lstrip("/")).resolve()
        if ROOT not in target.parents and target != ROOT:
            return self.send_error(403)
        if not target.is_file():
            return self.send_error(404)
        content_type = ("text/html; charset=utf-8" if target.suffix == ".html" else
                        "text/css; charset=utf-8" if target.suffix == ".css" else
                        "application/javascript; charset=utf-8" if target.suffix == ".js" else
                        "application/octet-stream")
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
            answer, engine = answer_locally(message)
            sources = []
            # Local skills (greetings, identity, calculator) always run first.
            # Web search is only used when local rules cannot answer the question.
            if answer is None and payload.get("web_search", True) is True:
                try:
                    sources = search_web(message)
                    answer = compose_search_answer(message, sources)
                    engine = "GENESIS-search-v0.1"
                except (HTTPError, URLError, TimeoutError) as error:
                    print("Search failed:", repr(error), flush=True)
                    answer = ("Не удалось подключиться к поиску прямо сейчас. "
                              "Проверь соединение сервера и попробуй ещё раз.")
                    engine, sources = "GENESIS-rules-v0.1", []
            elif answer is None:
                answer = ("Я пока не знаю ответа на этот вопрос. "
                          "Включи поиск в интернете — попробую найти материалы.")
                engine = "GENESIS-rules-v0.1"
            return self.send_json(200, {"reply": answer, "sources": sources, "engine": engine})
        except (ValueError, json.JSONDecodeError):
            return self.send_json(400, {"error": "Некорректный JSON-запрос."})
        except Exception as error:
            print("GENESIS server error:", repr(error), flush=True)
            return self.send_json(500, {"error": "Внутренняя ошибка сервера GENESIS."})

    def log_message(self, fmt, *args):
        print("[GENESIS]", fmt % args, flush=True)


if __name__ == "__main__":
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"GENESIS native engine v0.1 listening on port {PORT}", flush=True)
    server.serve_forever()
