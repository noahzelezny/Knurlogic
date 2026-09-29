import {$, esc} from '../format.js';
import {ASK} from './settings/cluster.js';
import {httpWhy} from '../api.js';

// --- try it, which is really "measure it" ---------------------------------
// This panel is a bench, not a chat window. The reason to send a prompt here
// is to turn a knob and send it again, so every reply carries three numbers:
//
//   TTFT        wall time to the first token. What a harness feels as lag.
//   prefill     tokens/s through the prompt -- the phase the memory knobs
//               actually govern, and the one that spikes.
//   decode      tokens/s after the first token.
//
// Measuring prefill at all requires STREAMING (a non-streaming call cannot
// tell you when the first token appeared) plus `stream_options.include_usage`,
// which is how mlx-lm reports the token counts.
//
// THE TRAP, and it would have made this instrument lie: `prompt_tokens`
// includes tokens served from the prompt cache, which were never prefilled.
// Send the same prompt twice while turning a knob -- exactly what this panel
// is for -- and the second run divides a full prompt by a near-zero prefill
// time. So the rate uses prompt_tokens MINUS cached_tokens, and when that is
// zero the rate is not shown at all: there is nothing to have a rate of.
const log=$('log');
function say(who,t){const p=document.createElement('p');p.innerHTML=`<b>${who}</b> `;
  p.appendChild(document.createTextNode(t));log.appendChild(p);log.scrollTop=1e9;return p}

// The point of running twice is the difference, so the page draws the
// difference. A line telling you to compare the two numbers is a line
// doing work the numbers should be doing themselves.
//
// Only against the run above it, and only when that run measured the same
// thing: a delta against a different prompt is not a delta.
// One sample per arm cannot support a small difference, so anything under
// 5% reads as "=" rather than as a change. Drawing a 2% wobble with an arrow
// on it is the same error as quoting a speed claim from a single run.
function delta(now, was){
  if(!(now>0&&was>0)) return '';
  const pc=100*(now-was)/was;
  if(Math.abs(pc)<5) return '<i class="d flat">=</i>';
  return `<i class="d ${pc>0?'up':'down'}">${pc>0?'▲':'▼'}${
    Math.abs(pc).toFixed(0)}%</i>`;
}
function metHTML(m, prev){
  if(!m) return '';
  const same=prev && prev.prompt===m.prompt;
  const bit=(k,v,warn)=>`<span class="${warn?'warnv':''}">${k} <b>${v}</b></span>`;
  let h=bit('ttft', m.ttft_ms<1000?`${Math.round(m.ttft_ms)} ms`
                                  :`${(m.ttft_ms/1000).toFixed(2)} s`);
  if(m.queue_ms>=50) h+=bit('queued', `${(m.queue_ms/1000).toFixed(2)} s`);
  if(m.prefill_tps) h+=bit('prefill', `${Math.round(m.prefill_tps)} tok/s`+
    (same?delta(m.prefill_tps, prev.prefill_tps):''));
  if(m.decode_tps) h+=bit('decode', `${m.decode_tps.toFixed(1)} tok/s`+
    (same?delta(m.decode_tps, prev.decode_tps):''));
  // Prefill is the rate of NEW prefill either way -- cached tokens are
  // already out of it -- so a cache hit is a smaller sample, not a wrong
  // number. Shown because it explains a thin measurement, not warned about.
  if(m.prompt) h+=bit('prompt', m.cached?`${m.prompt-m.cached} new of ${m.prompt}`
                                        :String(m.prompt));
  if(m.completion) h+=bit('out', String(m.completion));
  // Whose clock: the server's timing when it sent one, else the page's.
  return `<div class="met" title="${m.server?'measured by the server':
    'measured by this page (includes the network)'}">${h}</div>`;
}
function showMet(afterEl, m, prev){
  const d=document.createElement('div');
  d.innerHTML=metHTML(m, prev); const el=d.firstElementChild;
  if(el){ afterEl.after(el); log.scrollTop=1e9 }
}
function lastMet(){
  const r=RUNS.find(x=>x.id===CUR);
  return r ? (r.msgs||[]).map(x=>x.met).filter(Boolean).pop() : null;
}

// --- chat memory ----------------------------------------------------------
// Cheap on purpose: this browser's localStorage, no server state, no
// database. Losing it costs a list of old prompts, which is the right price
// for something that has to work the moment the page is opened. (The drawer
// state was removed from storage for a different reason -- it changed what
// the page DID. A remembered prompt does not.)
const STORE='kn.runs';
let RUNS=[], CUR=null;
function loadRuns(){ try{ RUNS=JSON.parse(localStorage.getItem(STORE)||'[]') }
  catch(e){ RUNS=[] } if(!Array.isArray(RUNS)) RUNS=[] }
function saveRuns(){ try{ localStorage.setItem(STORE,JSON.stringify(RUNS.slice(0,40))) }
  catch(e){} }
function best(run){
  // The list is for comparing runs, so the summary is the number being
  // tuned, not the time of day.
  const m=(run.msgs||[]).map(x=>x.met).filter(Boolean).pop();
  if(!m) return run.tune||'';
  const pf=m.prefill_tps?`${Math.round(m.prefill_tps)} pf`:'';
  const dc=m.decode_tps?`${m.decode_tps.toFixed(0)} tok/s`:'';
  return [run.tune,pf,dc].filter(Boolean).join(' · ');
}
function renderRuns(){
  const el=$('chats');
  if(!RUNS.length){ el.innerHTML=
    '<div class="sidelab">no runs yet</div>'; return }
  el.innerHTML='<div class="sidelab">runs</div>'+RUNS.map((r,i)=>
    `<div class="chat" data-i="${i}" aria-current="${r.id===CUR}">
       <span class="x" data-del="${i}" title="delete">×</span>
       <div class="t">${esc(r.title||'untitled')}</div>
       <div class="m">${esc(best(r))}</div></div>`).join('');
  el.querySelectorAll('.chat').forEach(c=>c.onclick=e=>{
    if(e.target.dataset.del!==undefined){
      RUNS.splice(+e.target.dataset.del,1); saveRuns();
      if(!RUNS.some(r=>r.id===CUR)) newRun(); else renderRuns();
      return;
    }
    openRun(RUNS[+c.dataset.i]);
  });
}
function newRun(){ CUR=null; log.innerHTML=''; renderRuns(); $('q').focus() }
function openRun(r){
  CUR=r.id; log.innerHTML='';
  let prev=null;
  (r.msgs||[]).forEach(m=>{
    const p=say(m.role==='user'?'you':'it', m.content);
    if(m.met){ showMet(p, m.met, prev); prev=m.met }
  });
  renderRuns();
}
function record(role, content, met){
  if(CUR===null){
    CUR=String(Date.now());
    RUNS.unshift({id:CUR, title:content.slice(0,52), at:Date.now(),
                  tune:(ASK&&ASK.tune)||'', msgs:[]});
  }
  const r=RUNS.find(x=>x.id===CUR);
  if(r){ r.msgs.push({role,content,met}); r.tune=(ASK&&ASK.tune)||r.tune }
  saveRuns(); renderRuns();
}
$('newchat').onclick=()=>{ if(window.ACTIVETAB==='chat') window.newChat(); else newRun() };
loadRuns();
window.renderRunsBench=renderRuns;
if(window.ACTIVETAB!=='chat') renderRuns();

// --- the timed request ----------------------------------------------------
// Enter sends, explicitly. A form with one text input and a submit button
// implicitly submits on Enter, and that is exactly the kind of "should work"
// worth nailing down on the one control the whole panel exists to use.
// Disabling the button does not stop implicit submission either, so the
// in-flight guard is here rather than on the button.
let BUSY=false;
$('q').addEventListener('keydown',e=>{
  if(e.key==='Enter'&&!e.shiftKey&&!e.isComposing){
    e.preventDefault(); $('f').requestSubmit();
  }
});
// "fresh prefill" is NOT a cache off-switch, and calling it one would be a
// claim this server cannot honour: mlx-lm has no per-request option for it
// (checked -- the body options are stream/temperature/penalties and no
// more), only the `--prompt-cache-size` flag it was started with. What it
// has is a PREFIX TRIE: a cached entry is reused only for a shared leading
// token run. So a unique marker at the FRONT of the prompt leaves nothing to
// share, and the prompt is genuinely pushed through the model again.
//
// It cannot reach zero cached tokens and does not pretend to -- the chat
// template's own preamble sits ahead of anything we write and stays shared.
// That is why the rate subtracts `cached_tokens` regardless of this box: the
// checkbox makes the measurement BIG, the subtraction makes it HONEST, and
// the two are independent.
let LAST='';
function withNonce(q){
  return $('fresh').checked
    ? `[bench ${Math.random().toString(36).slice(2,8)}] ${q}` : q;
}
$('again').onclick=()=>{ if(LAST && !BUSY) send(LAST) };

async function send(q){
  if(BUSY||!q) return;
  BUSY=true; LAST=q; $('again').disabled=true;
  const prev=lastMet();
  $('q').value=''; say('you',q); record('user',q,null);
  const out=say('it','…'); $('go').disabled=true;
  const sent=withNonce(q);
  const t0=performance.now();
  let first=0, text='', usage=null;
  try{
    const r=await fetch('/v1/chat/completions',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({messages:[{role:'user',content:sent}],max_tokens:256,
        stream:true, stream_options:{include_usage:true}})});
    if(!r.ok||!r.body) throw new Error(await httpWhy(r));
    const rd=r.body.getReader(), dec=new TextDecoder();
    let buf='';
    for(;;){
      const {value,done}=await rd.read(); if(done) break;
      buf+=dec.decode(value,{stream:true});
      let nl;
      while((nl=buf.indexOf('\n'))>=0){
        const line=buf.slice(0,nl).trim(); buf=buf.slice(nl+1);
        if(!line.startsWith('data:')) continue;
        const payload=line.slice(5).trim();
        if(payload==='[DONE]') continue;
        let j; try{ j=JSON.parse(payload) }catch(err){ continue }
        if(j.usage) usage=j.usage;
        const piece=j.choices?.[0]?.delta?.content;
        if(piece){
          if(!first) first=performance.now();
          text+=piece; out.lastChild.textContent=text; log.scrollTop=1e9;
        }
      }
    }
    const end=performance.now();
    let met=null;
    if(first){
      const ttft=first-t0;
      const prompt=usage?.prompt_tokens||0;
      const cached=usage?.prompt_tokens_details?.cached_tokens||0;
      const fresh=Math.max(prompt-cached,0);
      const comp=usage?.completion_tokens||0;
      met={ttft_ms:ttft, prompt, cached, completion:comp, fresh:$('fresh').checked,
           // Only tokens actually pushed through the model count toward a
           // prefill rate, and only when there were any.
           prefill_tps: fresh>0 ? fresh/(ttft/1000) : 0,
           decode_tps: (comp>1 && end>first) ? (comp-1)/((end-first)/1000) : 0};
    }
    if(!text) out.lastChild.textContent='(no content)';
    showMet(out, met, prev);
    record('assistant', text||'(no content)', met);
  }catch(err){ out.lastChild.textContent=String(err);
    record('assistant',String(err),null) }
  BUSY=false; $('go').disabled=false; $('again').disabled=!LAST;
}
$('f').onsubmit=e=>{ e.preventDefault(); send($('q').value.trim()) };
