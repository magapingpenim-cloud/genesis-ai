const form = document.getElementById("chatForm");
const input = document.getElementById("messageInput");
const messages = document.getElementById("messages");
const welcome = document.getElementById("welcome");
const sendButton = document.getElementById("sendButton");
const clearButton = document.getElementById("clearChat");
const newChatButton = document.getElementById("newChat");
const sidebar = document.getElementById("sidebar");
const mobileMenu = document.getElementById("mobileMenu");

function addMessage(role, text) {
  welcome.hidden = true;
  const row = document.createElement("article");
  row.className = `message ${role}`;
  if (role === "assistant") {
    const avatar = document.createElement("div");
    avatar.className = "avatar";
    avatar.textContent = "G";
    row.appendChild(avatar);
  }
  const body = document.createElement("div");
  body.className = "message-body";
  if (role === "assistant") {
    const label = document.createElement("span");
    label.className = "message-label";
    label.textContent = "GENESIS AI";
    body.appendChild(label);
  }
  const content = document.createElement("div");
  content.textContent = text;
  body.appendChild(content);
  row.appendChild(body);
  messages.appendChild(row);
  document.getElementById("conversation").scrollTop = document.getElementById("conversation").scrollHeight;
}

function sendMessage(value) {
  const text = value.trim();
  if (!text) return;
  addMessage("user", text);
  input.value = "";
  input.style.height = "auto";
  // The interface is intentionally frontend-only until a real inference API is connected.
  window.setTimeout(() => {
    addMessage("assistant", "Я получил твоё сообщение. Интерфейс GENESIS готов, но языковая модель пока не подключена. Следующий этап — связать этот чат с работающей моделью.");
  }, 350);
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  sendMessage(input.value);
});
input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 160) + "px";
  sendButton.disabled = !input.value.trim();
});
input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});
document.querySelectorAll(".suggestion").forEach((button) => {
  button.addEventListener("click", () => sendMessage(button.dataset.prompt || ""));
});
function resetChat() {
  messages.replaceChildren();
  welcome.hidden = false;
  input.value = "";
  input.style.height = "auto";
  sendButton.disabled = true;
  sidebar.classList.remove("open");
}
clearButton.addEventListener("click", resetChat);
newChatButton.addEventListener("click", resetChat);
mobileMenu.addEventListener("click", () => sidebar.classList.toggle("open"));
sendButton.disabled = true;
