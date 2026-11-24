const form=document.querySelector("#chatForm");
const input=document.querySelector("#prompt");
const send=document.querySelector("#send");
const messages=document.querySelector("#messages");
const newChat=document.querySelector("#newChat");
let busy=false;
let history=[];
const conversationId=()=>window.crypto?.randomUUID?.()||String(Date.now());

function renderFormattedText(target, text){
  // Small safe Markdown renderer: builds DOM nodes and never injects HTML.
  const lines=String(text||"").replace(/\r/g,"").split("\n");
  let list=null, paragraph=[];
  const flushParagraph=()=>{
    if(!paragraph.length)return;
    const p=document.createElement("p");
    appendInline(p,paragraph.join(" "));
    target.append(p);paragraph=[];
  };
  const closeList=()=>{if(list){target.append(list);list=null;}};
  for(const line of lines){
    const trimmed=line.trim();
    if(!trimmed){flushParagraph();closeList();continue;}
    const heading=trimmed.match(/^(#{1,3})\s+(.+)$/);
    if(heading){flushParagraph();closeList();const h=document.createElement("h"+Math.min(heading[1].length+1,4));appendInline(h,heading[2]);target.append(h);continue;}
    const item=trimmed.match(/^(?:[-*•]|\d+[.)])\s+(.+)$/);
    if(item){flushParagraph();if(!list){list=document.createElement(trimmed.match(/^\d+[.)]/)?"ol":"ul");}const li=document.createElement("li");appendInline(li,item[1]);list.append(li);continue;}
    closeList();paragraph.push(trimmed);
  }
  flushParagraph();closeList();
}
function appendInline(parent,text){
  const pattern=/\*\*(.+?)\*\*|\*(.+?)\*|\x60([^\x60]+)\x60/g;
  let last=0,match;
  while((match=pattern.exec(text))){
    parent.append(document.createTextNode(text.slice(last,match.index)));
    const node=document.createElement(match[1]?"strong":match[2]?"em":"code");
    node.textContent=match[1]||match[2]||match[3];parent.append(node);last=pattern.lastIndex;
  }
  parent.append(document.createTextNode(text.slice(last)));
}
function addMessage(role,text,sources=[]){
  document.querySelector("#welcome")?.remove();
  const row=document.createElement("article");row.className="message "+role;
  if(role==="assistant"||role==="error"){const avatar=document.createElement("div");avatar.className="avatar";avatar.textContent="G";row.append(avatar);}
  const bubble=document.createElement("div");bubble.className="bubble";renderFormattedText(bubble,text);row.append(bubble);
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
    pending.classList.remove("typing");pending.replaceChildren();renderFormattedText(pending,data.reply||"Модель не вернула текстовый ответ.");
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
