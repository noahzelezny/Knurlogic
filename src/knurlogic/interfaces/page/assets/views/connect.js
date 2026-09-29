import {$, esc} from '../format.js';
import {getJSON} from '../api.js';

// --- connect ----------------------------------------------------------------
// The endpoint is worthless until something is pointed at it, and the
// pointing is a handful of lines nobody remembers. One way in at a time, for
// the model the chat is on (else the first one running). It was empty on the
// control page: its text came with a model server's settings, which that
// page never has.
let CONNDOC=null, CONNSEL='openai';
// Claude Code names a model per tier and takes one base URL: this page's,
// whose router hands each request to the server holding the model it names.
// Only a running model can be routed to, so the choices are those.
const CONNTIER={};
// OpenAI-compatible and curl name one model: picked the same way, and sent
// through the same router; that model's own server is shown second.
let CONNPICK='';
function connRows(){
  const d=window.RESIDENCY||{};
  return (d.resident||[]).concat(...(d.peers||[]).map(p=>p.resident||[]))
    .filter(r=>r.runtime==='knurlogic' && r.where);
}
function connTarget(){
  const rows=connRows();
  const c=window.curChatPub && window.curChatPub();
  if(c && c.where){ const r=rows.find(r=>r.where===c.where); if(r) return r }
  if(window.PAGEMODEL) return {name:window.PAGEMODEL.name, where:location.origin};
  return rows[0]||null;
}
async function showConnect(){
  if(!CONNDOC) CONNDOC=await getJSON('/connect.json');
  const eps=CONNDOC.endpoints||[];
  $('conntabs').innerHTML=eps.map(e=>`<div class="fam" data-e="${esc(e.id)}"
    aria-current="${e.id===CONNSEL}">${esc(e.name)}</div>`).join('');
  $('conntabs').querySelectorAll('.fam').forEach(f=>f.onclick=()=>{
    CONNSEL=f.dataset.e; showConnect() });
  const e=eps.find(x=>x.id===CONNSEL); if(!e){ $('conn').innerHTML=
    `<div class="msg">${esc(CONNDOC.error||'nothing to show')}</div>`; return }
  const t=connTarget();
  let h=`<div class="shd">${esc(e.name)}</div><div class="msg" style="margin-top:0">${esc(e.what)}</div>`;
  if(e.needs_model && !t){
    $('conn').innerHTML=h+`<div class="msg">Nothing is running, so there is no
      address to point at yet. Load a model, then come back.</div>`; return }
  if(e.tiers||e.pick_model) h+=`<div class="msg">through this page, <b style="color:var(--acc)">${
    esc(location.origin)}</b>, which sends each request to the running model it names</div>`;
  else if(e.needs_model) h+=`<div class="msg">for <b style="color:var(--acc)">${esc(t.name)}</b>
    at ${esc(t.where)}</div>`;
  let ids=[], direct=null;
  if(e.tiers||e.pick_model){
    const j=await getJSON('/v1/models');
    ids=(j.data||[]).map(m=>m.id).filter(Boolean);
    if(CONNSEL!==e.id) return;
    if(!ids.length){ $('conn').innerHTML=h+`<div class="msg">No running model
      answers to its name here yet; the router only reaches loaded models.</div>`; return }
    const pref=ids.includes(t&&t.name)?t.name:ids[0];
    if(e.pick_model){
      if(!ids.includes(CONNPICK)) CONNPICK=pref;
      // the chosen model's own server, when this page can see which it is
      const tail=n=>String(n||'').replace(/\/+$/,'').split('/').pop();
      direct=connRows().find(r=>r.name===CONNPICK||tail(r.name)===tail(CONNPICK))||null;
      h+=`<div class="ctl" style="gap:10px;margin-top:10px"><label class="cs" style="margin:0;gap:6px">model
        <select data-pick style="width:auto">${ids.map(id=>
          `<option${id===CONNPICK?' selected':''}>${esc(id)}</option>`).join('')}</select></label></div>`;
    }
    if(e.tiers){ e.tiers.forEach(x=>{ if(!ids.includes(CONNTIER[x])) CONNTIER[x]=pref });
    h+=`<div class="ctl conntiers" style="gap:10px;margin-top:10px;flex-wrap:wrap">${
      e.tiers.map(x=>`<label class="cs tier" style="margin:0;gap:6px">${esc(x)}
        <select data-tier="${esc(x)}" style="width:auto">${ids.map(id=>
          `<option${id===CONNTIER[x]?' selected':''}>${esc(id)}</option>`).join('')}</select></label>`).join('')}</div>`; }
  }
  const base=e.pick_model ? (direct?direct.where:'') : (t?t.where:'');
  const model=e.pick_model ? CONNPICK : (t?t.name:'');
  const blocks=e.blocks.filter(b=>!b.direct||base);
  const fill=s=>s.split('__BASE__').join(base.replace(/\/$/,''))
                 .split('__ROUTER__').join(location.origin)
                 .split('__OPUS__').join(CONNTIER.opus||'')
                 .split('__SONNET__').join(CONNTIER.sonnet||'')
                 .split('__HAIKU__').join(CONNTIER.haiku||'')
                 .split('__MODEL__').join(model);
  h+=blocks.map((b,i)=>`<div class="connblk"><div class="optlab"><span${b.client?` class="cl-${esc(b.client)}"`:''}>${esc(b.label)}</span>
    <button class="mini" data-cp="${i}">copy</button></div><pre${b.direct?' style="color:var(--dim)"':''}>${esc(fill(b.text))}</pre></div>`).join('');
  $('conn').innerHTML=h;
  $('conn').querySelectorAll('[data-tier]').forEach(x=>x.onchange=()=>{
    CONNTIER[x.dataset.tier]=x.value; showConnect() });
  $('conn').querySelectorAll('[data-pick]').forEach(x=>x.onchange=()=>{
    CONNPICK=x.value; showConnect() });
  $('conn').querySelectorAll('[data-cp]').forEach(x=>x.onclick=()=>{
    navigator.clipboard?.writeText(fill(blocks[+x.dataset.cp].text))
      .then(()=>{ x.textContent='copied'; setTimeout(()=>x.textContent='copy',1200) })
      .catch(()=>{});
  });
}

export {showConnect};
