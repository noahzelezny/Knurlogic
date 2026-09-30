// ---------------------------------------------------------------------------
// Chat tab. Talks to the same served model over the same /v1/chat/completions
// the bench uses, but the opposite way on all three axes the bench deliberately
// defeats: full history (so the model actually has a conversation), a stable
// prompt prefix (no nonce -- that is what lets the server's prefix cache reuse
// an unchanged image across turns), and multimodal content.
//
// Reasoning is never parsed out of text here: the server sends it as
// `reasoning_content`, because only the server knows each template's
// thinking syntax (gemma's is not <think>).

import {OVL} from '../ui/overlay.js';
import {$, esc} from '../format.js';
import {httpWhy} from '../api.js';

// --- tabs -------------------------------------------------------------
window.ACTIVETAB='chat';
function switchTab(t){
  window.ACTIVETAB=t;
  $('ptabs').querySelectorAll('button').forEach(b=>
    b.classList.toggle('on', b.dataset.tab===t));
  $('chat').hidden = t!=='chat';
  $('bench').hidden = t!=='bench';
  document.body.classList.toggle('bench', t==='bench');
  $('newchat').textContent = t==='chat' ? '+ New chat' : '+ New run';
  if(t==='chat') renderChats(); else if(window.renderRunsBench) window.renderRunsBench();
  layout();
}
// HOME vs TALK. The page is in TALK while the chat tab shows a conversation
// with messages, unless the dashboard was asked for; opening a chat or
// sending one asks for the conversation again.
let HOME=false;
function layout(){
  const c=typeof curChat==='function' ? curChat() : null;
  const has=!!(c && c.msgs && c.msgs.length);
  $('chat').classList.toggle('empty', !has);
  const talk=window.ACTIVETAB==='chat' && has && !HOME;
  document.body.classList.toggle('talk', talk);
  // The memory panel is one element that moves: to the top of the right-hand
  // column in TALK, back above the chat in HOME. Moved rather than drawn
  // twice, so there is one set of machines for tick() to fill.
  const mem=$('memory'), home=document.querySelector('main');
  if(talk && mem.parentElement!==document.querySelector('.right'))
    document.querySelector('.right').prepend(mem);
  else if(!talk && mem.parentElement!==home) home.prepend(mem);
}
// The logo is the way home (the overlays float over the page, so a Home tab
// had nothing left to do): close whatever is floating, show the chat tab,
// and put the dashboard back.
function goHome(){ OVL.close(); HOME=true;
  if(window.ACTIVETAB!=='chat') switchTab('chat'); else layout();
  window.scrollTo(0,0) }
$('mark').onclick=goHome;
window.switchTab=switchTab;
$('ptabs').querySelectorAll('button').forEach(b=>b.onclick=()=>switchTab(b.dataset.tab));

// --- IndexedDB image store ----------------------------------------------
// Images are content-addressed by the SHA-256 of the FINAL data URL (the
// one downscaled/re-encoded once at attach time -- §3.2/3.4). Kept out of
// localStorage on purpose: inline data URLs would blow past its 5 MB quota
// in a couple of turns of screenshots.
let IDB=null, IDBFAIL=false;
function openIDB(){
  return new Promise((res)=>{
    if(IDB){ res(IDB); return }
    if(IDBFAIL || !window.indexedDB){ res(null); return }
    try{
      const rq=indexedDB.open('kn-img', 1);
      rq.onupgradeneeded=()=>{ rq.result.createObjectStore('blobs', {keyPath:'hash'}) };
      rq.onsuccess=()=>{ IDB=rq.result; res(IDB) };
      rq.onerror=()=>{ IDBFAIL=true; res(null) };
    }catch(e){ IDBFAIL=true; res(null) }
  });
}
const MEMIMG={}; // session-only fallback when IndexedDB is unavailable
async function idbPut(rec){
  const db=await openIDB();
  if(!db){ MEMIMG[rec.hash]=rec; return }
  try{ await new Promise((res,rej)=>{
    const tx=db.transaction('blobs','readwrite');
    tx.objectStore('blobs').put(rec);
    tx.oncomplete=res; tx.onerror=()=>rej(tx.error);
  }) }catch(e){ MEMIMG[rec.hash]=rec }
}
async function idbGet(hash){
  const db=await openIDB();
  if(!db) return MEMIMG[hash]||null;
  try{ return await new Promise((res,rej)=>{
    const tx=db.transaction('blobs','readonly');
    const rq=tx.objectStore('blobs').get(hash);
    rq.onsuccess=()=>res(rq.result||null); rq.onerror=()=>rej(rq.error);
  }) }catch(e){ return MEMIMG[hash]||null }
}
async function sha256(str){
  const buf=await crypto.subtle.digest('SHA-256', new TextEncoder().encode(str));
  return [...new Uint8Array(buf)].map(b=>b.toString(16).padStart(2,'0')).join('');
}

// --- chat store (localStorage kn.chats) ----------------------------------
const CSTORE='kn.chats';
let CHATS=[], CCUR=null;
function loadChats(){ try{ CHATS=JSON.parse(localStorage.getItem(CSTORE)||'[]') }
  catch(e){ CHATS=[] } if(!Array.isArray(CHATS)) CHATS=[] }
function saveChats(){ try{ localStorage.setItem(CSTORE, JSON.stringify(CHATS.slice(0,100))) }
  catch(e){} }
function curChat(){ return CHATS.find(c=>c.id===CCUR)||null }

// A model as the chat shows it: no path, no org. Model directories are
// org--name, so everything up to the last `--` is the org.
function shortModel(name){
  const s=String(name||'').replace(/\/+$/,'');
  const b=s.slice(s.lastIndexOf('/')+1);
  return b.includes('--') ? b.slice(b.lastIndexOf('--')+2) : b;
}
// The chat's speeds are the MEAN over its replies that measured them: one
// reply is one sample, and a chat is usually several runs of one setup.
function chatSpeeds(c){
  const ms=(c.msgs||[]).map(x=>x.met).filter(Boolean);
  const avg=k=>{ const v=ms.map(m=>m[k]).filter(x=>x>0);
    return v.length ? v.reduce((a,b)=>a+b,0)/v.length : 0 };
  const tt=avg('ttft_ms'), pf=avg('prefill_tps'), dc=avg('decode_tps');
  if(!tt && !pf && !dc) return '';
  // One line, the labels in the tooltip; a speed not measured is a dash.
  const bits=[tt ? 'TTFT '+(tt<1000?`${Math.round(tt)} ms`:`${(tt/1000).toFixed(2)} s`) : 'TTFT —',
    pf?`${Math.round(pf)} tok/s`:'—', dc?`${dc.toFixed(1)} tok/s`:'—'];
  return `<div class="spd" title="time to first token • prefill • decode, mean over this chat's replies">${
    bits.join(' • ')}</div>`;
}
function renderChats(){
  const el=$('chats');
  if(!CHATS.length){ el.innerHTML='<div class="sidelab">no chats yet</div>'; return }
  el.innerHTML='<div class="sidelab">chats</div>'+CHATS.map((c,i)=>
    `<div class="chat" data-i="${i}" aria-current="${c.id===CCUR}">
       <span class="x" data-del="${i}" title="delete">×</span>
       <div class="t" data-rn="${i}" title="rename">${esc(c.title||'untitled')}</div>
       <div class="m" title="${esc(c.model||'')}">${esc(shortModel(c.model))}</div>
       ${chatSpeeds(c)}</div>`).join('');
  el.querySelectorAll('.chat').forEach(c=>c.onclick=e=>{
    if(e.target.dataset.del!==undefined){
      if(confirm('Delete this chat?')){
        CHATS.splice(+e.target.dataset.del,1); saveChats();
        if(!CHATS.some(x=>x.id===CCUR)) newChat(); else renderChats();
      }
      return;
    }
    if(e.target.dataset.rn!==undefined){
      e.stopPropagation();
      const idx=+e.target.dataset.rn, name=prompt('Rename chat', CHATS[idx].title||'');
      if(name!=null){ CHATS[idx].title=name.slice(0,80); saveChats(); renderChats() }
      return;
    }
    openChat(CHATS[+c.dataset.i]);
  });
}
function newChat(modelHint, where, vision){
  CCUR=String(Date.now())+Math.random().toString(36).slice(2,6);
  CHATS.unshift({id:CCUR, title:'new chat', model:modelHint||SERVEDNAME||'',
                 where:where||'', vision:vision===undefined?null:!!vision,
                 at:Date.now(), updated:Date.now(), msgs:[]});
  saveChats(); switchTab('chat'); renderChats(); renderClog();
  clearAttachments(); chatGate(); $('cq').focus();
}
// Vision for a running model, from /models.json (the server's answer). A
// model is named org/name; its directory, and so the model list, org--name.
function visionOf(name){
  const k=String(name||'').replace('/','--');
  const m=(window.ALLMODELS||[]).find(x=>x.name===k||x.name===name);
  return !!(m && m.vision);
}
window.visionOf=visionOf;
// Composer state follows the CURRENT chat's target: its own endpoint if it
// has one (a running model clicked on the control page), else this page's
// served model.
function chatGate(){
  const c=curChat(), running=window.CHATTABLE||[];
  // A chat whose model has stopped: an empty one moves to a running model;
  // one with messages keeps its model, and says to pick a running one (a
  // pick starts a new chat). Never a send that can only fail.
  let gone=!!(c && c.where && window.RUNNING && !window.RUNNING.has(c.where));
  if(gone && !(c.msgs||[]).length && running.length){
    const r=running[0];
    c.where=r.where; c.model=r.name; c.vision=visionOf(r.name);
    saveChats(); renderChats(); gone=false;
  }
  // A pick not yet sent, still running: what the bar shows and what Send
  // will actually go to -- the open chat's own stored model/where are
  // untouched until Send commits it (or creates a new chat for it).
  if(PENDMODEL && window.RUNNING && !window.RUNNING.has(PENDMODEL.where))
    PENDMODEL=null;
  const eff=PENDMODEL || (c && c.where ? {where:c.where, name:c.model,
    vision: c.vision} : null);
  const on=!gone && !!(eff || SERVED);
  const vision=eff ? !!(PENDMODEL ? visionOf(eff.name) : c.vision)
                    : !!(SERVED && SERVED.vision);
  // The box is always typeable, so the page never looks broken; only Send
  // waits for a model, and the placeholder and Send's title say which.
  const goneMsg=gone ? `${shortModel(c.model)||'this chat\'s model'} isn't running — pick a running model`+
    (running.length?' in Model:':'') : '';
  $('cq').placeholder=gone ? goneMsg : on ? 'Message'
                         : 'Type away; click a running model to send it';
  $('csend').disabled=!on || BUSYC;
  $('csend').title=gone ? goneMsg : on ? '' : 'click a running model to chat with it';
  $('cattach').disabled=!on || !vision;
  const opts=window.CHATTABLE||[], el=$('cmodel');
  const sig=opts.map(r=>r.where+'|'+r.name).join('\n');
  if(el.dataset.sig!==sig){
    el.dataset.sig=sig;
    el.innerHTML=`<option value="">${opts.length?'choose':'none running'}</option>`+
      opts.map(r=>`<option value="${esc(r.where)}">${esc(shortModel(r.name))}${
        r.cluster&&(r.cluster.machines||[]).length>1
          ?' · '+esc(r.cluster.machines.join(' + '))
          :r.machine?' · '+esc(r.machine):''}</option>`).join('');
  }
  el.disabled=!opts.length;
  el.value=eff && opts.some(r=>r.where===eff.where) ? eff.where : '';
  el.title=eff ? 'sending to '+eff.name+' at '+eff.where : 'the running model Send goes to';
  $('cattach').title=vision ? 'attach an image'
    : (on ? 'this model has no vision' : 'click a running model to chat with it');
  loadSampling(eff ? eff.where : (SERVED ? '' : null), eff ? eff.name : (c && c.model));
}
window.chatGate=chatGate;
// Picking in Model: (or clicking an instance card) only names what the
// NEXT Send goes to -- it does not touch the open chat's stored model or
// its list entry. Send itself decides: a chat with messages already in a
// different model gets a new chat; an empty chat, or the same model, just
// sends where it is.
let PENDMODEL=null;
$('cmodel').onchange=()=>{
  const r=(window.CHATTABLE||[]).find(x=>x.where===$('cmodel').value);
  useModel(r||null);
};
function useModel(r){
  PENDMODEL=r||null;
  chatGate(); $('cq').focus();
}
window.useModel=useModel;
window.curChatPub=()=>curChat();

// --- chat settings ---
// Per target, once: a model's listing does not change while it is loaded.
const SDEF={};
let SDEFAT;
async function loadSampling(where, model){
  const key=where===null ? null : where+'|'+(model||'');
  SDEFAT=key;
  if(where===null){ thinkLevels(undefined); return }
  if(!(key in SDEF)){
    // A clicked model is reached through this page's read-only /peek (a
    // browser will not call another port); the page's own model directly.
    // A peer's relay lists every model that machine runs: take this one's.
    const url=where ? '/peek?'+new URLSearchParams({where, path:'/v1/models'})
                    : '/v1/models';
    const k=String(model||''), alt=k.replace('/','--');
    SDEF[key]=fetch(url).then(r=>r.ok?r.json():null)
      .then(j=>{ const d=j&&Array.isArray(j.data)?j.data:[];
        return d.find(x=>x.id===k||x.id===alt)||d[0]; })
      .catch(()=>undefined);
  }
  const m=await SDEF[key];
  if(m===undefined) delete SDEF[key];      // ask again next time
  if(SDEFAT!==key) return;
  thinkLevels(m ? m.thinking : undefined);
  CTXLEN=m && m.context_length>0 ? m.context_length : 0; showCtx();
}
// Thinking: only the levels this model's template has, in its own names
// (/v1/models `thinking`: native [{level, name}]), plus its default. The
// value sent is the reasoning_effort ladder level that maps exactly onto
// that native level, so the server applies it as named. A server that does
// not say gets the plain levels; no model chosen, only its default.
const LADDER=['none','low','medium','high'];
let EFFWANT=null;                 // what the person picked, kept across models
function thinkLevels(t){
  const el=$('ceffort');
  if(EFFWANT===null) EFFWANT=el.value;
  const nat=t && Array.isArray(t.native) && t.native.length ? t.native : null;
  const opts=nat ? nat.map(n=>[n.level, n.name]) : t ? LADDER.map(l=>[l,l]) : [];
  const sig=JSON.stringify(opts)+(t&&t.default||'');
  if(el.dataset.sig===sig) return;
  el.dataset.sig=sig;
  el.innerHTML=`<option value="">model default${t&&t.default?' ('+esc(t.default)+')':''}</option>`+
    opts.map(([v,n])=>`<option value="${esc(v)}">${esc(n)}</option>`).join('');
  el.value=opts.some(o=>o[0]===EFFWANT) ? EFFWANT : '';
  el.title=nat ? `this model's thinking levels (${t.dialect}); sent as reasoning_effort, applied as named`
    : "reasoning_effort: knurlogic maps it onto this model's own controls and says what it applied";
}
$('ceffort').addEventListener('change',()=>{ EFFWANT=$('ceffort').value });
let CTXLEN=0;                   // the served model's window; 0 is unknown
// Used by showCtx: 1200 -> "1.2k".
const kfmt=v=>v<1000?String(v):v<1e6?(v/1000).toFixed(v<1e5?1:0).replace(/\.0$/,'')+'k'
  :(v/1e6).toFixed(1).replace(/\.0$/,'')+'M';
// The bench's context line: what the last reply used (its prompt plus its
// answer, as the server counted them) over the window. Unknown window: just
// the count, since a made-up denominator would read as a limit.
function showCtx(){
  const el=$('cctx'); if(!el) return;
  const c=curChat(), m=c&&(c.msgs||[]).map(x=>x.met).filter(Boolean).pop();
  const used=m?(m.prompt||0)+(m.completion||0):null;
  el.textContent=used==null ? '—'
    : CTXLEN ? `${kfmt(used)} / ${kfmt(CTXLEN)}` : kfmt(used);
  el.title=used==null?'no reply yet in this chat'
    :`${used.toLocaleString()} tokens`+(CTXLEN?` of ${CTXLEN.toLocaleString()}`:'');
}
// The system prompt and thinking level as last left, in this browser only: a
// convenience, not state.
(()=>{
  const ids=['csys','ceffort'];
  try{
    const v=JSON.parse(localStorage.getItem('kn.copts')||'{}');
    ids.forEach(id=>{ if(v[id]!=null) $(id).value=v[id] });
  }catch(e){}
  const save=()=>{ try{ localStorage.setItem('kn.copts', JSON.stringify(
    Object.fromEntries(ids.map(id=>[id,$(id).value])))) }catch(e){} };
  ids.forEach(id=>$(id).addEventListener('change',save));
})();
window.newChat=newChat;
function openChat(c){
  CCUR=c.id; HOME=false; renderChats(); renderClog(); chatGate();
}

// --- served-model / vision gating -----------------------------------------
let SERVED=null, SERVEDNAME=null;
function chatServedChanged(a){
  SERVED=a; SERVEDNAME=a?a.name:null;
  const on=!!a;
  chatGate();
  if(on && !curChat()) newChat(a.name);
  if(on && curChat() && !curChat().model) { curChat().model=a.name; saveChats(); renderChats() }
}
window.chatServedChanged=chatServedChanged;

// --- attachments ------------------------------------------------------
// Pending attachments for the message being composed. Images are downscaled
// and re-encoded exactly ONCE, here, at attach time -- never again -- which
// is what lets every resend be byte-identical and hit the server's cache
// (report-chat-ui.md §0.4, §3.4).
const MAXEDGE=1536, THUMBEDGE=160;
let PENDING=[]; // [{kind,name,hash,mime,w,h,bytes,text?}]

function clearAttachments(){ PENDING=[]; renderChips() }
function renderChips(){
  const el=$('chips');
  el.innerHTML=PENDING.map((a,i)=>{
    if(a.kind==='image'){
      return `<div class="chip"><img src="${a.thumb}"><span>${esc(a.name)}</span>
        <span class="ctok">≈${estTokens(a)} tok</span>
        <span class="cx" data-rm="${i}">×</span></div>`;
    }
    return `<div class="chip"><span>📄 ${esc(a.name)}</span>
      <span class="cx" data-rm="${i}">×</span></div>`;
  }).join('');
  el.querySelectorAll('[data-rm]').forEach(x=>x.onclick=()=>{
    PENDING.splice(+x.dataset.rm,1); renderChips();
  });
}
// A rough patch-count estimate (√-area / 28px patches, no family-specific
// merge factor -- the real number comes back in `usage` after the send;
// this is only for the chip, per §3.5). Not verified per family.
function estTokens(a){ return Math.max(64, Math.round((a.w*a.h)/(28*28))) }

function loadImage(dataURL){
  return new Promise((res,rej)=>{
    const im=new Image(); im.onload=()=>res(im); im.onerror=rej; im.src=dataURL;
  });
}
function fileToDataURL(f){
  return new Promise((res,rej)=>{
    const r=new FileReader(); r.onload=()=>res(r.result); r.onerror=rej; r.readAsDataURL(f);
  });
}
async function attachImage(file){
  const raw=await fileToDataURL(file);
  const im=await loadImage(raw);
  const scale=Math.min(1, MAXEDGE/Math.max(im.width, im.height));
  const w=Math.max(1,Math.round(im.width*scale)), h=Math.max(1,Math.round(im.height*scale));
  const cv=document.createElement('canvas'); cv.width=w; cv.height=h;
  cv.getContext('2d').drawImage(im, 0, 0, w, h);
  const mime = file.type==='image/png' ? 'image/png' : 'image/jpeg';
  const dataURL=cv.toDataURL(mime, 0.85);
  const hash=await sha256(dataURL);
  const tcv=document.createElement('canvas');
  const ts=Math.min(1, THUMBEDGE/Math.max(w,h));
  tcv.width=Math.max(1,Math.round(w*ts)); tcv.height=Math.max(1,Math.round(h*ts));
  tcv.getContext('2d').drawImage(cv, 0, 0, tcv.width, tcv.height);
  const thumb=tcv.toDataURL('image/jpeg', 0.7);
  await idbPut({hash, dataURL, thumbDataURL:thumb, w, h, bytes:dataURL.length});
  PENDING.push({kind:'image', name:file.name, hash, mime, w, h,
                bytes:dataURL.length, thumb});
  renderChips();
}
async function attachText(file){
  const text=await file.text();
  PENDING.push({kind:'text', name:file.name, text:text.slice(0,100000)});
  renderChips();
}
async function handleFiles(files){
  for(const f of files){
    try{
      if(f.type.startsWith('image/')) await attachImage(f);
      else if(f.type==='application/pdf' || /\.pdf$/i.test(f.name)){
        // PDFs are not supported: rendering them to page images needs
        // pdf.js, which the page does not bundle.
        alert(`${f.name}: PDF attachments are not supported yet. `+
          `Convert it to images first, or paste its text.`);
      } else await attachText(f);
    }catch(e){ alert(`Could not attach ${f.name}: ${e}`) }
  }
}
$('cfile').addEventListener('change', e=>{ handleFiles([...e.target.files]); e.target.value='' });
$('cattach').onclick=()=>$('cfile').click();
$('chat').addEventListener('dragover', e=>e.preventDefault());
$('chat').addEventListener('drop', e=>{
  e.preventDefault();
  if(!$('cattach').disabled && e.dataTransfer.files.length)
    handleFiles([...e.dataTransfer.files]);
});
$('cq').addEventListener('paste', e=>{
  const items=[...(e.clipboardData?.items||[])];
  const imgs=items.filter(i=>i.type.startsWith('image/'));
  if(imgs.length && !$('cattach').disabled){
    e.preventDefault();
    imgs.forEach(i=>attachImage(i.getAsFile()));
    return;
  }
  // A paste that would swamp the box (a log, a file's worth of code) goes in
  // as an attachment chip instead: many lines, or a lot of text either way.
  const text=e.clipboardData?.getData('text/plain')||'';
  const lines=text.split('\n').length;
  if(lines>60 || text.length>6000){
    e.preventDefault();
    const n=PENDING.filter(a=>a.name&&a.name.startsWith('paste')).length+1;
    PENDING.push({kind:'text', name:`paste-${n}.txt (${lines} lines)`,
                  text:text.slice(0,100000)});
    renderChips();
  }
});

// --- wire format ------------------------------------------------------
// One chat message as the OpenAI request carries it. A user turn with
// attachments becomes content parts: the attached images, then the typed text
// with any attached text files appended as fenced blocks. Built from the
// stored message alone, so the same turn is byte-identical on every resend --
// which is what lets the server's prompt cache reuse it, images included.
async function toWire(m){
  if(m.role==='assistant'){
    const o={role:'assistant', content:m.content||''};
    if(m.thinking) o.reasoning_content=m.thinking;
    return o;
  }
  const att=m.att||[];
  if(m.role!=='user' || !att.length) return {role:m.role, content:m.content||''};
  const images=[], files=[];
  for(const a of att){
    if(a.kind==='image'){
      const rec=await idbGet(a.hash);
      if(rec) images.push({type:'image_url', image_url:{url:rec.dataURL}});
    } else if(a.kind==='text'){
      files.push(`[attached: ${a.name}]\n\`\`\`\n${a.text}\n\`\`\``);
    }
  }
  const text=[m.content||'', ...files].filter(Boolean).join('\n\n');
  return {role:'user', content:[...images, {type:'text', text}]};
}

// --- minimal, safe markdown (escape first, §3.8 option B) -----------------
function mdRender(src){
  let s=esc(src);
  const codeBlocks=[];
  s=s.replace(/```(\w*)\n([\s\S]*?)```/g, (_,lang,code)=>{
    const i=codeBlocks.length;
    codeBlocks.push({lang,code});
    return `\u0000CB${i}\u0000`;
  });
  s=s.replace(/^######\s?(.*)$/gm,'<h6>$1</h6>').replace(/^#####\s?(.*)$/gm,'<h5>$1</h5>')
    .replace(/^####\s?(.*)$/gm,'<h4>$1</h4>').replace(/^###\s?(.*)$/gm,'<h3>$1</h3>')
    .replace(/^##\s?(.*)$/gm,'<h2>$1</h2>').replace(/^#\s?(.*)$/gm,'<h1>$1</h1>');
  s=s.replace(/`([^`\n]+)`/g,'<code>$1</code>');
  s=s.replace(/\*\*([^*]+)\*\*/g,'<b>$1</b>').replace(/\*([^*]+)\*/g,'<i>$1</i>');
  s=s.replace(/((?:^|\n)- .*(?:\n- .*)*)/g, m=>
    '\n<ul>'+m.trim().split('\n').map(l=>`<li>${l.replace(/^- /,'')}</li>`).join('')+'</ul>');
  // http(s) links only, per the spec's "links (http/https only)".
  s=s.replace(/(https?:\/\/[^\s<]+)/g,'<a href="$1" target="_blank" rel="noopener">$1</a>');
  s=s.replace(/\n{2,}/g,'</p><p>').replace(/\n/g,'<br>');
  s=`<p>${s}</p>`;
  s=s.replace(/\u0000CB(\d+)\u0000/g, (_,i)=>{
    const {lang,code}=codeBlocks[+i];
    const id='cb'+Math.random().toString(36).slice(2,8);
    return `<pre><div class="codehd"><span>${esc(lang||'')}</span>`+
      `<span data-copy="${id}">copy</span></div><code id="${id}">${code}</code></pre>`;
  });
  return s;
}
document.addEventListener('click', e=>{
  const t=e.target.closest('[data-copy]');
  if(t){ const code=document.getElementById(t.dataset.copy);
    if(code) navigator.clipboard?.writeText(code.textContent).catch(()=>{}) }
});

// --- prefill progress -------------------------------------------------------
// mlx-lm reports prompt processing as `: keepalive done/total` comments. The
// bar shows the fraction; the time left comes from a smoothed rate (an
// exponential average of tokens/s between reports), so one slow chunk --
// an image block, a long first chunk -- does not swing the estimate.
let pf=null;
function pfShow(done,total){
  if(!total) return;
  const now=performance.now();
  if(!pf) pf={t:now, done:0, rate:0};
  const dt=(now-pf.t)/1000;
  if(dt>0.05 && done>pf.done){
    const r=(done-pf.done)/dt;
    pf.rate = pf.rate ? 0.7*pf.rate+0.3*r : r;
    pf.t=now; pf.done=done;
  }
  const frac=Math.min(1, done/total);
  const el=$('pfbar'); el.hidden=false;
  el.querySelector('.pfbarfill').style.width=(100*frac).toFixed(1)+'%';
  const fmt=n=>n<1000 ? String(n) : (n/1000).toFixed(n<10000?1:0)+'k';
  const left = pf.rate>0 ? (total-done)/pf.rate : 0;
  el.querySelector('.pfbartxt').textContent =
    `prompt ${fmt(done)} / ${fmt(total)}` + (left>=1 ? ` · ${Math.ceil(left)}s left` : '');
}
function pfHide(){ $('pfbar').hidden=true; pf=null }

// --- lightbox -----------------------------------------------------------------
// The download keeps the image's own format: the extension comes from the
// data URL's MIME subtype, with the two whose subtype is not the usual
// extension spelled out.
const EXT_FOR={jpeg:'jpg', 'svg+xml':'svg'};
function openLightbox(dataURL){
  const sub=(/^data:image\/([^;,]+)/.exec(dataURL)||[])[1]||'png';
  $('lbimg').src=dataURL; $('lbdl').href=dataURL;
  $('lbdl').download='image.'+(EXT_FOR[sub]||sub);
  $('lightbox').hidden=false;
}
$('lbclose').onclick=()=>{ $('lightbox').hidden=true };
$('lightbox').onclick=e=>{ if(e.target.id==='lightbox') $('lightbox').hidden=true };
addEventListener('keydown', e=>{ if(e.key==='Escape' && !$('lightbox').hidden)
  $('lightbox').hidden=true });

// --- rendering the log --------------------------------------------------
// Under each reply: TTFT, prefill, decode. The server's numbers when it sent
// them; otherwise the page's clock, which also counts the network and proxy.
function replyMetHTML(m){
  const bits=[];
  const bit=(k,v)=>bits.push(`<span class="k">${k}</span> <b>${v}</b>`);
  if(m.ttft_ms) bit('TTFT', m.ttft_ms<1000?`${Math.round(m.ttft_ms)} ms`
                                           :`${(m.ttft_ms/1000).toFixed(2)} s`);
  if(m.prefill_tps) bit('Prefill', `${Math.round(m.prefill_tps)} tok/s`);
  if(m.decode_tps) bit('Decode', `${m.decode_tps.toFixed(1)} tok/s`);
  if(!bits.length) return '';
  return `<div class="rmet" title="${m.server?'measured by the server':
    'measured by this page'}">${bits.join(' · ')}</div>`;
}
// The page's own clock, when the server sent no timing. All times are
// performance.now() ms: t0 the send, ka the first prefill keepalive (0 if
// none), first/last the first and last GENERATED token (reasoning counts),
// not the stream's close. Prefill under ~256 new tokens or ~50 ms is too
// thin to be a rate, so it is left out (shown as a dash).
function pageMetrics(t){
  const fresh=Math.max((t.prompt||0)-(t.cached||0),0);
  const pft=t.first-(t.ka||t.t0), dt=(t.last||0)-t.first;
  return {ttft_ms:t.first-t.t0,
    prefill_tps: fresh>=256 && pft>=50 ? fresh/(pft/1000) : 0,
    decode_tps: t.completion>1 && dt>0 ? (t.completion-1)/(dt/1000) : 0};
}
window.pageMetrics=pageMetrics;
// The in-flight mark, drawn into the streaming reply until its answer
// starts. While waiting there's nothing to show yet, so it's a plain pill;
// once thinking begins the SAME gear mark becomes the one-and-only header
// of the thinking block (no separate inner "thinking" summary), and the
// text streams straight into its <pre> with no gap above it.
function waitHTML(w, think){
  const on=w.phase==='thinking';
  const word=on ? 'Thinking'+(w.pct!=null&&w.pct<100?` · ${w.pct}%`:'') : 'Waiting';
  const gear=`${window.GEAR||''}<span>${word}</span>`;
  if(on) return `<details class="think waitwrap" open><summary class="wait on" role="status">${gear}</summary><pre>${esc(think||'')}</pre></details>`;
  return `<div class="wait waitwrap" role="status">${gear}</div>`;
}
let WAIT=null;
function setWait(w, think){
  WAIT=w;
  const b=$('clog').lastElementChild?.querySelector('.bubble.assistant'); if(!b) return;
  let el=b.querySelector('.waitwrap');
  if(!w){ if(el) el.remove(); return }
  const d=document.createElement('div'); d.innerHTML=waitHTML(w, think);
  if(el) el.replaceWith(d.firstChild); else b.prepend(d.firstChild);
}
// Once the answer itself starts, the live gear stops turning -- but if
// there was thinking, its block (now the ordinary, static header) stays
// put rather than vanishing out from under the reply.
function freezeWait(){
  const b=$('clog').lastElementChild?.querySelector('.bubble.assistant'); if(!b) return;
  const w=b.querySelector('.waitwrap.think');
  if(w){ w.classList.remove('on'); w.querySelector('summary')?.classList.remove('on') }
  else b.querySelector('.waitwrap')?.remove();
}
// A message's actions as icons (stroke, like the rest of the marks), each
// named by its tooltip and aria-label.
const ICONS={
  cp:'<rect x="9" y="9" width="12" height="12" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>',
  ed:'<path d="M17 3a2.8 2.8 0 0 1 4 4L7.5 20.5 2 22l1.5-5.5z"/>',
  rg:'<path d="M21 12a9 9 0 0 1-15.5 6.2L3 16"/><path d="M3 21v-5h5"/><path d="M3 12a9 9 0 0 1 15.5-6.2L21 8"/><path d="M21 3v5h-5"/>',
  del2:'<path d="M3 6h18"/><path d="M8 6V4a1 1 0 0 1 1-1h6a1 1 0 0 1 1 1v2"/><path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/><path d="M10 11v6M14 11v6"/>'};
function actBtn(k, idx, name){
  return `<button type="button" data-${k}="${idx}" title="${name}" aria-label="${name}"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${ICONS[k]}</svg></button>`;
}
function bubbleFor(m, idx){
  const wrap=document.createElement('div');
  wrap.className='bubblewrap '+(m.role==='user'?'user':'assistant');
  let thumbs='';
  if(m.att&&m.att.length){
    thumbs=`<div class="thumbs">${m.att.filter(a=>a.kind==='image')
      .map(a=>`<img class="thumb" data-hash="${a.hash}" src="${a.thumb||''}">`).join('')}</div>`;
  }
  // One header for the thinking block, live or finished: the gear. While
  // still waiting on the answer, WAIT drives it (and carries the thinking
  // seen so far, if any); once content has started, the message's own
  // .thinking is all there is and the block is just a static, closed one.
  let thinkBlock='';
  if(m.role!=='user' && m._streaming && !m.content && WAIT) thinkBlock=waitHTML(WAIT, m.thinking);
  else if(m.thinking && m.thinking.trim())
    thinkBlock=`<details class="think waitwrap" ${m._streaming?'open':''}><summary class="wait" role="status">${window.GEAR||''}<span>Thinking</span></summary><pre>${esc(m.thinking.trim())}</pre></details>`;
  const body = m.role==='assistant' ? `<div class="md">${mdRender(m.content||'')}</div>`
                                     : esc(m.content||'');
  const bubble=document.createElement('div');
  bubble.className='bubble '+(m.role==='user'?'user':(m.err?'err assistant':'assistant'));
  bubble.innerHTML = (thumbs?thumbs:'') + (m.role==='user'?body:'');
  if(m.role!=='user') bubble.innerHTML = thinkBlock + `<div class="md">${mdRender(m.content||'')}</div>`;
  wrap.appendChild(bubble);
  const foot=document.createElement('div'); foot.className='bfoot';
  if(m.met){ const met=document.createElement('div'); met.innerHTML=replyMetHTML(m.met);
    if(met.firstChild) foot.appendChild(met.firstChild) }
  const act=document.createElement('div'); act.className='bactions';
  act.innerHTML = actBtn('cp',idx,'Copy')+
    (m.role==='user'?actBtn('ed',idx,'Edit & resend'):'')+
    (m.role==='assistant'?actBtn('rg',idx,'Regenerate'):'')+
    actBtn('del2',idx,'Delete');
  foot.appendChild(act); wrap.appendChild(foot);
  wrap.querySelectorAll('.thumb').forEach(async img=>{
    const rec=await idbGet(img.dataset.hash);
    if(rec) img.onclick=()=>openLightbox(rec.dataURL);
  });
  return wrap;
}
function renderClog(){
  const el=$('clog'); el.innerHTML='';
  const c=curChat();
  layout(); showCtx();
  if(!c){ el.innerHTML='<div class="msg">No chat open. Click a running model, '+
    'or "+ New chat".</div>'; return }
  (c.msgs||[]).forEach((m,i)=>el.appendChild(bubbleFor(m,i)));
  el.scrollTop=1e9;
  el.querySelectorAll('[data-cp]').forEach(x=>x.onclick=()=>{
    const m=c.msgs[+x.dataset.cp]; navigator.clipboard?.writeText(m.content||'').catch(()=>{});
  });
  el.querySelectorAll('[data-del2]').forEach(x=>x.onclick=()=>{
    c.msgs.splice(+x.dataset.del2,1); saveChats(); renderClog();
  });
  el.querySelectorAll('[data-ed]').forEach(x=>x.onclick=()=>{
    const i=+x.dataset.ed, m=c.msgs[i];
    $('cq').value=m.content||''; c.msgs.length=i; saveChats(); renderClog(); $('cq').focus();
  });
  el.querySelectorAll('[data-rg]').forEach(x=>x.onclick=()=>{
    const i=+x.dataset.rg;
    // Regenerate = drop this assistant turn and everything after it, then
    // resend using the user turn right before it.
    let j=i; while(j>0 && c.msgs[j-1].role!=='user') j--;
    c.msgs.length=j; saveChats(); renderClog(); runTurn();
  });
}

// --- send / stream ------------------------------------------------------
let BUSYC=false, ABORTC=null;
function autoGrow(){ const q=$('cq'); q.style.height='auto';
  q.style.height=Math.min(160,q.scrollHeight)+'px' }
$('cq').addEventListener('input', autoGrow);
// the system prompt grows the same way, up to a few lines
function sysGrow(){ const s=$('csys'); s.style.height='auto';
  s.style.height=Math.min(240,Math.max(s.scrollHeight,54))+'px' }
$('csys').addEventListener('input', sysGrow);
$('cq').addEventListener('keydown', e=>{
  if(e.key==='Enter' && !e.shiftKey && !e.isComposing){
    e.preventDefault(); $('cf').requestSubmit();
  }
});
$('cf').onsubmit=e=>{ e.preventDefault(); sendChat() };
$('cstop').onclick=()=>{ if(ABORTC) ABORTC.abort() };

function chatTargetOn(){
  const c=curChat(); return !!((c && c.where) || SERVED);
}
function sendChat(){
  if(BUSYC) return;
  const text=$('cq').value.trim();
  if(!text && !PENDING.length) return;
  let c=curChat();
  const pend=PENDMODEL;
  // newChat() clears PENDING, so take the attachments before it can run
  const att=PENDING.slice();
  if(pend){
    // Differs from the model this chat already has messages with: a NEW
    // chat, with the picked model, gets the message. Empty, or the same
    // model: the open chat just adopts the pick and sends.
    if(!c || ((c.msgs||[]).length && c.model!==pend.name)){
      newChat(pend.name, pend.where, visionOf(pend.name));
      c=curChat();
    } else {
      c.where=pend.where; c.model=pend.name; c.vision=visionOf(pend.name);
    }
  }
  if(!c){ if(!chatTargetOn()) return; newChat(); c=curChat() }
  if(!((c.where)||SERVED)) return;
  PENDMODEL=null;
  HOME=false;
  if(c.msgs.length===0) c.title=text.slice(0,52)||'(image)';
  c.msgs.push({role:'user', content:text, att});
  $('cq').value=''; autoGrow(); clearAttachments(); c.updated=Date.now();
  saveChats(); renderChats(); renderClog();
  runTurn();
}

// --- server-sent events ---------------------------------------------------------
// The response body as events, in order: {data: parsed JSON} for each `data:`
// line (the `[DONE]` sentinel is dropped) and {comment: text} for each `:`
// line. Bytes are decoded as a stream, so a multi-byte character split across
// network chunks survives, and a line split across chunks is held until its
// newline arrives.
async function* events(body){
  const rd=body.getReader(), dec=new TextDecoder();
  let pending='';
  for(;;){
    const {value, done}=await rd.read();
    pending += done ? dec.decode() : dec.decode(value, {stream:true});
    const lines=pending.split(/\r?\n/);
    pending = done ? '' : lines.pop();
    for(const line of lines){
      if(line.startsWith(':')){ yield {comment:line.slice(1)}; continue }
      if(!line.startsWith('data:')) continue;
      const payload=line.slice(5).trim();
      if(!payload || payload==='[DONE]') continue;
      try{ yield {data:JSON.parse(payload)} }catch(e){}
    }
    if(done) return;
  }
}

async function runTurn(){
  const c=curChat(); if(!c) return;
  BUSYC=true; $('csend').disabled=true; $('csend').hidden=true; $('cstop').hidden=false;
  const sys=$('csys').value.trim();
  const turns=[];
  for(const m of c.msgs) turns.push(await toWire(m));
  const wire=[];
  if(sys) wire.push({role:'system', content:sys});
  wire.push(...turns);
  const assistant={role:'assistant', content:'', thinking:'', _streaming:true};
  WAIT={phase:'waiting'};
  c.msgs.push(assistant); renderClog();
  const bubbles=$('clog').children; const wrap=bubbles[bubbles.length-1];
  const mdEl=()=>wrap.querySelector('.md');
  ABORTC=new AbortController();
  pf=null;
  // include_usage: the final chunk then carries the token counts and the
  // server's own timing (usage.knurlogic.timing). No sampling fields and no
  // max_tokens: the server applies the model's own defaults.
  const body={model:c.model||undefined,
    messages:wire, stream:true, stream_options:{include_usage:true}};
  const effort=$('ceffort').value;
  if(effort) body.reasoning_effort=effort;
  const t0=performance.now();
  let first=0, last=0, ka=0, usage=null, finishReason=null, streamErr=null, aborted=false;
  try{
    const url=c.where ? '/chat?where='+encodeURIComponent(c.where)
                      : '/v1/chat/completions';
    const r=await fetch(url,{method:'POST',
      headers:{'Content-Type':'application/json'}, body:JSON.stringify(body),
      signal:ABORTC.signal});
    if(!r.ok||!r.body) throw new Error(await httpWhy(r));
    for await (const ev of events(r.body)){
      if(ev.comment!==undefined){
        // mlx-lm's prefill signal: `: keepalive done/total`
        const mm=/^\s*keepalive\s+(\d+)\/(\d+)/.exec(ev.comment);
        if(mm){
          if(!ka) ka=performance.now();
          pfShow(+mm[1], +mm[2]);
          if(!first) setWait({phase:'thinking',
            pct:+mm[2] ? Math.floor(100*+mm[1]/+mm[2]) : null}, assistant.thinking.trim());
        }
        continue;
      }
      const j=ev.data; if(!j) continue;
      if(j.error){ streamErr=j.error.message||j.error; continue }
      if(j.usage) usage=j.usage;
      const fr=j.choices?.[0]?.finish_reason;
      if(fr) finishReason=fr;
      const delta=j.choices?.[0]?.delta||{};
      const thought=delta.reasoning_content||delta.reasoning;
      if(thought){
        last=performance.now(); if(!first) first=last;
        assistant.thinking+=thought; pfHide();
        // The gear (one header) is the live thinking block itself; once it
        // exists, stream straight into its <pre> instead of re-rendering.
        let th=$('clog').lastElementChild.querySelector('.think pre');
        if(!th){ setWait({phase:'thinking'}, assistant.thinking.trim());
          th=$('clog').lastElementChild.querySelector('.think pre') }
        else th.textContent=assistant.thinking.trim();
      }
      if(delta.content){
        last=performance.now(); if(!first) first=last;
        assistant.content+=delta.content; pfHide();
        if(WAIT){ WAIT=null; freezeWait() }
        const md=$('clog').lastElementChild.querySelector('.md');
        if(md) md.innerHTML=mdRender(assistant.content);
      }
      $('clog').scrollTop=1e9;
    }
  }catch(err){
    if(err.name!=='AbortError'){ assistant.content=(assistant.content||'')+`\n\n[error: ${err.message||err}]`;
      assistant.err=true }
    else aborted=true;
  }
  pfHide(); WAIT=null;
  assistant._streaming=false;
  // The stream can end without ever saying why: a hop between here and
  // rank 0 dropped the connection, or the server sent one last error
  // event instead of a finish_reason. Either way, say so visibly instead
  // of leaving a silent half sentence -- the speeds still shown below are
  // real, so a quiet cutoff reads as a finished answer otherwise.
  if(!aborted && !assistant.err && (assistant.content || assistant.thinking)){
    if(streamErr) assistant.content += `\n\n[stopped: ${streamErr}]`;
    else if(finishReason==='length') assistant.content += `\n\n[stopped: hit the max tokens limit]`;
    else if(!finishReason) assistant.content += `\n\n[stopped: stream ended early]`;
  }
  if(first){
    const prompt=usage?.prompt_tokens||0;
    const cached=usage?.prompt_tokens_details?.cached_tokens||0;
    const comp=usage?.completion_tokens||0;
    // Prefill starts at the first keepalive when there was one: before it
    // the request may only have been queued. Decode ends at the last token,
    // not when the stream closed (the usage chunk and the close can trail
    // it by seconds, which on a short reply read as 0.1 tok/s).
    assistant.met={prompt, cached, completion:comp,
      ...pageMetrics({t0, ka, first, last, prompt, cached, completion:comp})};
  }
  // The server's own measurement wins where it has one: it timed the steps
  // where they ran, while the page's clock also counts the proxy and the
  // queue. Kept whole (and with the message) so a later look can tell
  // which numbers were the server's.
  const kt=usage?.knurlogic?.timing;
  if(kt){
    const m=assistant.met||(assistant.met={prompt:usage.prompt_tokens||0,
      cached:usage.prompt_tokens_details?.cached_tokens||0,
      completion:usage.completion_tokens||0});
    m.server=kt;
    if(kt.ttft_s!=null) m.ttft_ms=kt.ttft_s*1000;
    m.prefill_tps=kt.prefill_tok_s||0;
    m.decode_tps=kt.decode_tok_s||0;
    if(kt.queue_s) m.queue_ms=kt.queue_s*1000;
  }
  const ks=usage?.knurlogic?.sampling;
  if(ks && assistant.met) assistant.met.sampling=ks;
  showCtx();
  if(!assistant.content && !assistant.thinking) assistant.content='(no content)';
  // renderChats too: the list carries the chat's speeds.
  c.updated=Date.now(); saveChats(); renderChats(); renderClog();
  BUSYC=false; $('csend').disabled=!chatTargetOn(); $('csend').hidden=false; $('cstop').hidden=true;
  ABORTC=null;
}

loadChats();
document.addEventListener('DOMContentLoaded', ()=>{});
switchTab('chat');
renderClog();
