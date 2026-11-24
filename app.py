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
import ipaddress
import socket
import time
from genesis.knowledge import answer_from_knowledge

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


class PageTextParser(HTMLParser):
    """Extract readable page text while ignoring scripts and page chrome."""
    SKIP = {"script", "style", "noscript", "svg", "nav", "footer", "header", "form"}
    BLOCK = {"p", "div", "article", "section", "h1", "h2", "h3", "li", "blockquote", "br"}
    def __init__(self):
        super().__init__()
        self.skip_depth = 0
        self.parts = []
    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip_depth += 1
        elif not self.skip_depth and tag in self.BLOCK:
            self.parts.append("\n")
    def handle_endtag(self, tag):
        if tag in self.SKIP and self.skip_depth:
            self.skip_depth -= 1
        elif not self.skip_depth and tag in self.BLOCK:
            self.parts.append("\n")
    def handle_data(self, data):
        if not self.skip_depth:
            clean = re.sub(r"\s+", " ", data).strip()
            if clean:
                self.parts.append(clean)
    def text(self):
        return re.sub(r"\n{3,}", "\n\n", " ".join(self.parts)).strip()


def _public_http_url(url):
    """Allow only public HTTP(S) destinations; reject local/private hosts."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return False
    host = parsed.hostname.lower()
    if host in ("localhost",) or host.endswith((".local", ".internal")):
        return False
    try:
        addresses = {item[4][0] for item in socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80))}
        return bool(addresses) and all(ipaddress.ip_address(addr).is_global for addr in addresses)
    except (OSError, ValueError):
        return False


def read_webpage(url, max_chars=9000):
    """Fetch a public web page and return readable text; never fetch private-network URLs."""
    if not _public_http_url(url):
        return ""
    request = Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; GENESIS-AI/0.2)", "Accept": "text/html"})
    try:
        with urlopen(request, timeout=12) as response:
            content_type = response.headers.get("Content-Type", "")
            if "text/html" not in content_type and "application/xhtml" not in content_type:
                return ""
            raw = response.read(1_000_000).decode("utf-8", "replace")
        parser = PageTextParser()
        parser.feed(raw)
        return parser.text()[:max_chars]
    except (HTTPError, URLError, TimeoutError, OSError, ValueError):
        return ""


def search_web(query, limit=5):
    url = "https://html.duckduckgo.com/html/?" + urlencode({"q": query})
    request = Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; GENESIS-AI/0.2)"})
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
        if link.startswith("http") and link not in seen and item["title"] and _public_http_url(link):
            seen.add(link)
            results.append({
                "title": re.sub(r"\s+", " ", item["title"]).strip(),
                "url": link,
                "snippet": re.sub(r"\s+", " ", item["snippet"]).strip(),
                "content": ""
            })
        if len(results) >= limit:
            break
    # Read the actual pages, not just the search-engine snippets.
    for item in results[:3]:
        item["content"] = read_webpage(item["url"])
    return results


def compose_search_answer(query, sources):
    """Create a longer, query-relevant extractive brief from retrieved pages."""
    if not sources:
        return ("Поиск не вернул подходящих результатов. Попробуй уточнить запрос "
                "или повторить его чуть позже.")
    usable = [item for item in sources if item.get("content") or item.get("snippet")]
    if not usable:
        return ("Поиск нашёл страницы, но сайты не разрешили прочитать их содержимое. "
                "Открой ссылки на источники ниже.")
    query_terms = set(re.findall(r"[a-zа-яё0-9]{3,}", query.lower()))
    lines = ["## Подробный ответ", "Я проверил найденные страницы и собрал наиболее полезные фрагменты."]
    included = 0
    for item in usable[:3]:
        body = item.get("content") or item.get("snippet", "")
        sentences = [re.sub(r"\s+", " ", s).strip() for s in re.split(r"(?<=[.!?])\s+", body)]
        candidates = []
        for sentence in sentences:
            if len(sentence) < 45:
                continue
            words = set(re.findall(r"[a-zа-яё0-9]{3,}", sentence.lower()))
            candidates.append((len(words & query_terms), sentence))
        candidates.sort(key=lambda pair: pair[0], reverse=True)
        selected, seen = [], set()
        for _, sentence in candidates:
            key = sentence[:100].lower()
            if key not in seen:
                seen.add(key)
                selected.append(sentence)
            if len(selected) >= 4:
                break
        if not selected and body:
            selected = [body[:700]]
        if selected:
            lines.append("### " + item["title"])
            lines.extend("- " + sentence[:650] for sentence in selected)
            included += len(selected)
    if not included:
        return "Я нашёл страницы, но не смог извлечь из них достаточно содержательного текста. Проверь источники ниже."
    lines.extend([
        "## Важный момент",
        "Это подробная выжимка из доступных страниц, а не независимое подтверждение каждого утверждения. "
        "Если источники расходятся или тема быстро меняется, сверяй даты и первоисточники.",
        "## Источники",
        "Ссылки на страницы, которые удалось найти, находятся под ответом."
    ])
    return "\n\n".join(lines)


def generate_with_comfyui(prompt):
    """Generate one image through a local ComfyUI server; no cloud key required."""
    prompt = str(prompt or "").strip()
    if not prompt or len(prompt) > 1800:
        raise ValueError("Опиши изображение (до 1800 символов).")
    base = os.environ.get("GENESIS_COMFYUI_URL", "http://127.0.0.1:8188").rstrip("/")
    checkpoint = os.environ.get("GENESIS_SD_CHECKPOINT", "v1-5-pruned-emaonly.safetensors")
    workflow = {
        "1": {"class_type":"CheckpointLoaderSimple","inputs":{"ckpt_name":checkpoint}},
        "2": {"class_type":"CLIPTextEncode","inputs":{"text":prompt,"clip":["1",1]}},
        "3": {"class_type":"CLIPTextEncode","inputs":{"text":"low quality, blurry, distorted, bad anatomy, text, watermark","clip":["1",1]}},
        "4": {"class_type":"EmptyLatentImage","inputs":{"width":512,"height":512,"batch_size":1}},
        "5": {"class_type":"KSampler","inputs":{"seed":int(time.time()*1000)%2147483647,"steps":20,"cfg":7.0,"sampler_name":"euler","scheduler":"normal","denoise":1.0,"model":["1",0],"positive":["2",0],"negative":["3",0],"latent_image":["4",0]}},
        "6": {"class_type":"VAEDecode","inputs":{"samples":["5",0],"vae":["1",2]}},
        "7": {"class_type":"SaveImage","inputs":{"filename_prefix":"genesis","images":["6",0]}}
    }
    def request_json(url, data=None, timeout=15):
        request = Request(url, data=data, headers={"Content-Type":"application/json"} if data is not None else {})
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    try:
        queued = request_json(base + "/prompt", json.dumps({"prompt":workflow}).encode("utf-8"))
    except (OSError, URLError, HTTPError, TimeoutError) as error:
        raise RuntimeError("Не удалось подключиться к ComfyUI. Запусти ComfyUI на этом компьютере (порт 8188).") from error
    prompt_id = queued.get("prompt_id")
    if not prompt_id:
        raise RuntimeError("ComfyUI не принял задачу: " + str(queued.get("error", queued))[:500])
    deadline = time.time() + 180
    while time.time() < deadline:
        time.sleep(1)
        try:
            history = request_json(base + "/history/" + prompt_id, timeout=10)
        except (OSError, ValueError, URLError):
            continue
        record = history.get(prompt_id, {})
        images = record.get("outputs", {}).get("7", {}).get("images", [])
        if images:
            image = images[0]
            return {"image_url":"/api/comfy-image?"+urlencode({
                "filename":image.get("filename",""),"subfolder":image.get("subfolder",""),
                "type":image.get("type","output")}),"prompt":prompt}
        if record.get("status", {}).get("status_str") == "error":
            raise RuntimeError("ComfyUI завершил генерацию с ошибкой. Проверь модель и консоль ComfyUI.")
    raise RuntimeError("Генерация заняла слишком много времени. Проверь очередь ComfyUI.")


def is_image_request(message):
    text = message.lower().strip()
    return bool(re.search(
        r"\\b(нарисуй|создай (?:картинк|изображени)|сгенерируй (?:картинк|изображени)|сделай (?:картинк|изображени)|"
        r"изобрази|сделай арт|нарисовать|генерируй картинку)\\b", text
    ))


def extract_image_prompt(message):
    text = re.sub(
        r"(?i)^\\s*(?:пожалуйста\\s*)?(?:нарисуй|изобрази|нарисовать|сделай арт|"
        r"создай (?:мне )?(?:картинку|изображение)|сгенерируй (?:мне )?(?:картинку|изображение)|"
        r"сделай (?:мне )?(?:картинку|изображение)|генерируй картинку)\\s*",
        "", str(message)
    ).strip(" .,!?:;")
    return text or str(message).strip()


def answer_locally(message, history=None):
    """Conversational rules; unknown factual questions are routed to web search."""
    text = message.strip()
    lower = re.sub(r"\s+", " ", text.lower()).strip(" .,!?:;")
    calculation = safe_calculate(text)
    if calculation is not None:
        return "Результат: " + calculation, "calculator"

    # Common Russian informal greetings and chat openers should never trigger web search.
    if re.search(r"\b(привет|приветик|прив|здравствуй|здравствуйте|здорово|здарова|здаров|салам|ассалам|хай|хелло|hello|hi|йо|йоу|ку)\b", lower):
        return "Привет! 👋 Рад тебя видеть. Как сам? Что будем делать?", "conversation"
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

    # Search the local curated knowledge base before sending unknown questions to web search.
    local_knowledge = answer_from_knowledge(text)
    if local_knowledge is not None:
        return local_knowledge

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
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/api/comfy-image":
            params = parse_qs(parsed.query)
            filename = params.get("filename", [""])[0]
            subfolder = params.get("subfolder", [""])[0]
            image_type = params.get("type", ["output"])[0]
            if not filename or filename != os.path.basename(filename) or subfolder or image_type != "output":
                return self.send_error(400)
            base = os.environ.get("GENESIS_COMFYUI_URL", "http://127.0.0.1:8188").rstrip("/")
            try:
                url = base + "/view?" + urlencode({"filename":filename,"subfolder":"","type":"output"})
                with urlopen(Request(url), timeout=20) as response:
                    data = response.read(12_000_000)
                    content_type = response.headers.get("Content-Type", "image/png")
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)
                return
            except (OSError, URLError, TimeoutError):
                return self.send_error(502)
        if path == "/api/health":
            return self.send_json(200, {"status": "ok", "engine": "GENESIS-native-v0.4",
                                        "model": "none", "web_search": True,
                                        "image_generation": "ComfyUI (optional, local)"})
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
        if self.path == "/api/generate-image":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length < 1 or length > 8000:
                    return self.send_json(413, {"error":"Запрос слишком большой или пустой."})
                payload = json.loads(self.rfile.read(length))
                result = generate_with_comfyui(payload.get("prompt", ""))
                return self.send_json(200, result)
            except ValueError as error:
                return self.send_json(400, {"error":str(error)})
            except RuntimeError as error:
                return self.send_json(503, {"error":str(error)})
            except Exception as error:
                print("GENESIS image error:", repr(error), flush=True)
                return self.send_json(500, {"error":"Не удалось создать изображение."})
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
            history = history if isinstance(history, list) else []
            previous_user = next((item.get("content", "")[:500]
                                  for item in reversed(history[-12:])
                                  if isinstance(item, dict) and item.get("role") == "user"
                                  and isinstance(item.get("content"), str)), "")
            follow_up = bool(re.match(
                r"^(почему|зачем|подробнее|расскажи подробнее|что дальше|и что|а потом|что это значит|продолжай)\b",
                message.lower().strip()
            ))
            contextual_query = (previous_user + " " + message).strip() if follow_up and previous_user else message
            # Natural-language image requests work directly in the chat; no separate button required.
            if is_image_request(message):
                image_prompt = extract_image_prompt(message)
                try:
                    result = generate_with_comfyui(image_prompt)
                except RuntimeError as error:
                    return self.send_json(503, {"error":str(error)})
                return self.send_json(200, {"reply":"Изображение готово!\\n\\n**Описание:** " + image_prompt,
                                            "image":result.get("image_url"),"sources":[],"engine":"GENESIS-image"})
            sources = []
            answer, engine = answer_locally(message, history)
            # Our own conversation rules and knowledge base run first.
            # Unknown questions use web search; no external language model is called.
            if answer is None:
                try:
                    sources = search_web(contextual_query, limit=6)
                    answer = compose_search_answer(contextual_query, sources)
                    engine = "GENESIS-search"
                except (HTTPError, URLError, TimeoutError, OSError) as error:
                    print("Search failed:", repr(error), flush=True)
                    answer = ("Сейчас не удалось получить данные из интернета. "
                              "Попробуй повторить запрос позже или переформулировать его. "
                              "Я не буду придумывать ответ без источников.")
                    engine = "GENESIS-search-unavailable"
            public_sources = [{"title": item.get("title", ""), "url": item.get("url", "")} for item in sources if item.get("title") and item.get("url")]
            return self.send_json(200, {"reply": answer, "sources": public_sources, "image":None, "engine": engine})
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
