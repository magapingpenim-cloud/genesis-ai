const form=document.querySelector("#chatForm");
const input=document.querySelector("#prompt");
const send=document.querySelector("#send");
const messages=document.querySelector("#messages");
const welcome=document.querySelector("#welcome");
const newChat=document.querySelector("#newChat");
let busy=false;
const conversationId=()=>window.crypto?.randomUUID?.()||String(Date.now());

function addMessage(role,text){
  welcome?.remove();
  const row=document.createElement("article");
  row.className="message "+role;
  if(role==="assistant"||role==="error"){
    const avatar=document.createElement("div");avatar.className="avatar";avatar.textContent="G";row.append(avatar);
  }
  const bubble=document.createElement("div");bubble.className="bubble";bubble.textContent=text;row.append(bubble);
  messages.append(row);messages.scrollTop=messages.scrollHeight;return bubble;
}
async function submitMessage(text){
  if(!text||busy)return;
  addMessage("user",text);input.value="";input.style.height="auto";busy=true;send.disabled=true;
  const pending=addMessage("assistant","Передаю сообщение движку…");pending.classList.add("typing");
  try{
    const response=await fetch("/api/chat",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({message:text,conversation_id:window.genesisConversationId})});
    const data=await response.json().catch(()=>({error:"Сервер вернул некорректный ответ."}));
    if(!response.ok)throw new Error(data.error||("Ошибка сервера: "+response.status));
    pending.classList.remove("typing");pending.textContent=data.reply||data.response||"Движок получил запрос, но не вернул текстовый ответ.";
  }catch(error){
    const row=pending.closest(".message");row.classList.add("error");pending.classList.remove("typing");pending.textContent=error.message||"Не удалось связаться с движком.";
  }finally{busy=false;send.disabled=!input.value.trim();messages.scrollTop=messages.scrollHeight;input.focus();}
}
window.genesisConversationId=conversationId();
form.addEventListener("submit",e=>{e.preventDefault();submitMessage(input.value.trim());});
input.addEventListener("input",()=>{input.style.height="auto";input.style.height=Math.min(input.scrollHeight,180)+"px";send.disabled=busy||!input.value.trim();});
input.addEventListener("keydown",e=>{if(e.key==="Enter"&&!e.shiftKey){e.preventDefault();form.requestSubmit();}});
newChat.addEventListener("click",()=>{messages.replaceChildren();const w=document.createElement("div");w.id="welcome";w.className="welcome";w.innerHTML='<div class="hero-mark">G</div><div class="eyebrow">GENESIS AI · CONVERSATION</div><h1>Давай начнём.</h1><p>Напиши сообщение — чат передаст его подключённому движку GENESIS.</p>';messages.append(w);window.genesisConversationId=conversationId();input.focus();});
send.disabled=true;
