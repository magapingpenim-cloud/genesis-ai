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
    # Deterministic FAQ skills: fixed, reviewed answers; no neural training involved.
    faq = [
        (r"\b(что такое )?бокс\b|\b(правила|суть) бокса\b",
         "Бокс — вид единоборства, в котором два спортсмена в перчатках наносят удары руками по правилам. Бой проходит раундами; победу определяют по очкам судей или досрочно."),
        (r"\b(что такое )?кикбоксинг\b",
         "Кикбоксинг — единоборство, в котором разрешены удары руками и ногами. Точный набор разрешённых приёмов зависит от правил конкретной дисциплины."),
        (r"\b(что такое )?(искусственный интеллект|ии)\b",
         "Искусственный интеллект — область информатики, в которой создают программы для задач вроде распознавания изображений, обработки языка, прогнозирования и планирования."),
        (r"\b(что такое )?(нейросеть|нейронная сеть)\b",
         "Нейросеть — математическая модель из связанных слоёв вычислений. Во время обучения она настраивает числовые параметры на примерах, чтобы находить закономерности."),
        (r"\b(что такое )?python\b|\bязык python\b",
         "Python — язык программирования с понятным синтаксисом. Его используют для автоматизации, веб-разработки, анализа данных и машинного обучения."),
        (r"\b(что такое )?(видеокарта|gpu)\b|\bдля чего нужна видеокарта\b",
         "Видеокарта обрабатывает графику. Её графический процессор (GPU) также умеет выполнять много вычислений параллельно, поэтому его используют для нейросетей."),
        (r"\b(что такое )?(процессор|cpu)\b",
         "Процессор (CPU) выполняет инструкции программ и управляет вычислениями компьютера. Он хорошо подходит для широкого круга последовательных и управляющих задач."),
        (r"\b(что такое )?интернет\b",
         "Интернет — глобальная система соединённых компьютерных сетей, которые обмениваются данными по общим протоколам."),
        (r"\b(что такое )?алгоритм\b",
         "Алгоритм — точная последовательность шагов, которая приводит к решению задачи. Например, рецепт — бытовой пример алгоритма."),
        (r"\b(что такое )?git(hub)?\b|\bзачем нужен github\b",
         "Git — система контроля версий, которая хранит историю изменений кода. GitHub — платформа для размещения Git-репозиториев и совместной разработки."),
        (r"\b(что такое )?фотосинтез\b",
         "Фотосинтез — процесс, при котором растения и некоторые другие организмы используют энергию света, чтобы создавать органические вещества из воды и углекислого газа."),
        (r"\b(что такое )?база данных\b",
         "База данных — организованное хранилище информации, в котором можно добавлять, изменять, искать и связывать данные."),
        (r"\b(что такое )?солнечная система\b",
         "Солнечная система включает Солнце и связанные с ним гравитацией объекты: восемь планет, их спутники, карликовые планеты, астероиды и кометы."),
    ]
    for pattern, response in faq:
        if re.search(pattern, lower):
            return response, "GENESIS-scripted-faq"

    # Identity and practical-help responses for common everyday questions.
    if re.search(r"\b(сколько тебе лет|твой возраст|какого ты возраста)\b", lower):
        return "У меня нет человеческого возраста. Я — программа GENESIS, и мой возраст не измеряется годами, как у человека.", "conversation"
    if re.search(r"\b(как починить кран|кран протекает|течет кран|капает кран|починить смеситель)\b", lower):
        return ("Если кран течёт: 1) перекрой воду вентилями под раковиной; "
                "2) открой кран, чтобы сбросить давление; 3) определи, откуда течёт — из излива, "
                "под ручкой или у основания. Часто помогает замена аэратора, прокладки или картриджа. "
                "Не разбирай смеситель, пока вода не перекрыта. Если течь сильная или вентили не закрываются, "
                "лучше вызвать сантехника. Напиши, откуда именно капает, и я подскажу точнее."), "GENESIS-scripted-help"
    if re.search(r"\b(кто тебя создал|кто разработал genesis|кто тебя разработал)\b", lower):
        return "GENESIS — наш собственный экспериментальный проект. Его программную логику и интерфейс мы разрабатываем сами.", "conversation"
    if re.search(r"\b(ты настоящий|ты живой|у тебя есть чувства)\b", lower):
        return "Я не человек и не обладаю сознанием или чувствами. Я программа, которая обрабатывает сообщения и формирует ответы.", "conversation"

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
            return self.send_json(200, {"status": "ok", "engine": "GENESIS-native-v0.3",
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
            if answer is None:
                try:
                    sources = search_web(message)
                    answer = compose_search_answer(message, sources)
                    engine = "GENESIS-search-v0.1"
                except (HTTPError, URLError, TimeoutError) as error:
                    print("Search failed:", repr(error), flush=True)
                    answer = ("Не удалось подключиться к поиску прямо сейчас. "
                              "Проверь соединение сервера и попробуй ещё раз.")
                    engine, sources = "GENESIS-rules-v0.1", []
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
