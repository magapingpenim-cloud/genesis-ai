const form=document.querySelector("#chatForm");
const input=document.querySelector("#prompt");
const send=document.querySelector("#send");
const messages=document.querySelector("#messages");
const newChat=document.querySelector("#newChat");
let busy=false;
let history=[];
const conversationId=()=>window.crypto?.randomUUID?.()||String(Date.now());

function addMessage(role,text,sources=[]){
  document.querySelector("#welcome")?.remove();
  const row=document.createElement("article");row.className="message "+role;
  if(role==="assistant"||role==="error"){const avatar=document.createElement("div");avatar.className="avatar";avatar.textContent="G";row.append(avatar);}
  const bubble=document.createElement("div");bubble.className="bubble";bubble.textContent=text;row.append(bubble);
  if(sources.length){
    const box=document.createElement("div");box.className="sources";
    const heading=document.createElement("div");heading.className="sources-title";heading.textContent="ИСТОЧНИКИ";box.append(heading);
    sources.forEach(source=>{const a=document.createElement("a");a.href=source.url;a.target="_blank";a.rel="noopener noreferrer";a.textContent=source.title;a.className="source-link";box.append(a);});
    bubble.append(box);
  }
  messages.append(row);messages.scrollTop=messages.scrollHeight;return bubble;
}
async function submitMessage(text){
  if(!text||busy)return;
  addMessage("user",text);input.value="";input.style.height="auto";busy=true;send.disabled=true;
  const pending=addMessage("assistant","GENESIS обрабатывает сообщение…");pending.classList.add("typing");
  try{
    const response=await fetch("/api/chat",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({message:text,history,conversation_id:window.genesisConversationId})});
    const data=await response.json().catch(()=>({error:"Сервер вернул некорректный ответ."}));
    if(!response.ok)throw new Error(data.error||("Ошибка сервера: "+response.status));
    pending.classList.remove("typing");pending.textContent=data.reply||"Модель не вернула текстовый ответ.";
    if(data.sources?.length){const box=document.createElement("div");box.className="sources";const heading=document.createElement("div");heading.className="sources-title";heading.textContent="ИСТОЧНИКИ";box.append(heading);data.sources.forEach(source=>{const a=document.createElement("a");a.href=source.url;a.target="_blank";a.rel="noopener noreferrer";a.textContent=source.title;a.className="source-link";box.append(a);});pending.append(box);}
    history.push({role:"user",content:text},{role:"assistant",content:data.reply||""});history=history.slice(-12);
  }catch(error){pending.closest(".message").classList.add("error");pending.classList.remove("typing");pending.textContent=error.message||"Не удалось связаться с движком.";}
  finally{busy=false;send.disabled=!input.value.trim();messages.scrollTop=messages.scrollHeight;input.focus();}
}
window.genesisConversationId=conversationId();
form.addEventListener("submit",e=>{e.preventDefault();submitMessage(input.value.trim());});
input.addEventListener("input",()=>{input.style.height="auto";input.style.height=Math.min(input.scrollHeight,180)+"px";send.disabled=busy||!input.value.trim();});
input.addEventListener("keydown",e=>{if(e.key==="Enter"&&!e.shiftKey){e.preventDefault();form.requestSubmit();}});
newChat.addEventListener("click",()=>{messages.replaceChildren();history=[];window.genesisConversationId=conversationId();const w=document.createElement("div");w.id="welcome";w.className="welcome";w.innerHTML='<div class="hero-mark">G</div><div class="eyebrow">GENESIS AI · CONVERSATION</div><h1>Давай начнём.</h1><p>Просто напиши мне — я сам решу, когда искать информацию.</p>';messages.append(w);input.focus();});
send.disabled=true;
