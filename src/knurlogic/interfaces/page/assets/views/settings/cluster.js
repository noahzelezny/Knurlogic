import {$, esc, gb} from '../../format.js';
import {knobHTML} from './knobs.js';
import {SEQ, SMACH, keepSet, machines, nextSeq, setSMach} from './index.js';
import {getJSON, peekURL} from '../../api.js';

let DOC={}, ASK={};
// The wired limit: what the GPU may hold, which decides whether a rung
// loads at all. knurlogic NEVER runs the sysctl -- it needs sudo on the Mac
// it changes -- so it reads the current value, works out where the ceiling
// is, and hands over the exact line to run THERE. A soft gate, like
// everything else here: past what it recommends you get the command and a
// warning rather than a refusal. Every machine's field is live: this one's
// asks its own /settings.json, a peer's asks that peer's through /peek, so
// the command is worked out by the machine it is for.
function wiredHTML(doc, m){
  if(!doc || doc.error) return `<div class="wired"><div class="msg">${esc(doc&&doc.error||'no answer')}</div></div>`;
  const w=doc.wired||{}, own=!m||m.id==='local';
  if(!w.known) return '<div class="wired"><div class="msg">wired limit not known</div></div>';
  const cur=doc.wired_current_gib, rec=doc.wired_recommended_gib,
        inst=doc.wired_installed_gib, tgt=doc.wired_target_gib??cur;
  const host=own?'':(m.page||'').replace(/^https?:\/\//,'').replace(/:\d+$/,'');
  const run=own ? 'Run this in Terminal on this Mac. sudo asks for its admin password; it lasts until the Mac restarts.'
    : `Run this in Terminal on ${esc(m.name)} (on it, or over ssh ${esc(host)}). sudo asks for that Mac's admin password; knurlogic cannot run it for you, and it lasts until that Mac restarts.`;
  return `<div class="wired">
    <div class="sidelab" style="margin-top:0">wired limit · ${esc(w.key)}</div>
    <div class="wiredbar"><i style="width:${(100*tgt/inst).toFixed(1)}%"></i>
      <u style="left:${(100*rec/inst).toFixed(1)}%" title="recommended"></u></div>
    <div class="wiredrow">
      <label class="ro">set to <input type="number" class="wset" aria-label="wired limit" min="8"
        max="${Math.round(inst-4)}" step="1" value="${Math.round(tgt)}"> GiB</label>
      <span class="ro">now <b>${cur.toFixed(0)}</b> of ${inst.toFixed(0)} GiB ·
        knurlogic suggests <b>${rec.toFixed(0)}</b></span>
    </div>
    ${doc.wired_command?`<div class="run">${run}</div><pre>${esc(doc.wired_command)}</pre>`:''}
  </div>`;
}
// A number, not a slider: a slider spent the panel's width on one value.
// Asked on change (Enter or leaving the field): the machine it is for
// recomputes the command and the warning.
function wireWired(box, m){
  const sl=box&&box.querySelector('.wset'); if(!sl) return;
  if(!m || m.id==='local'){
    sl.onchange=()=>loadSettings({...(ASK.tune?{tune:ASK.tune}:{}), wired_gib:sl.value});
    return;
  }
  sl.onchange=async()=>{
    const doc=await getJSON(peekURL(m.page,'/settings.json',{wired_gib:sl.value}));
    box.innerHTML=wiredHTML(doc, m); wireWired(box, m);
  };
}
async function loadSettings(q){
  const qs=new URLSearchParams(q||{}).toString();
  try{ DOC=await (await fetch('/settings.json'+(qs?'?'+qs:''))).json() }
  catch(e){ DOC={} }
  ASK=DOC.asked||{};
  const w=document.querySelector('#machbody [data-m="local"] .mwired');
  if(w){ w.innerHTML=wiredHTML(DOC,null); wireWired(w,null) }
  const al=document.querySelector('#machbody [data-m="local"] .mallow');
  if(al) al.innerHTML=allowHTML(DOC.allowance, machines()[0], true);
}

// The facts beside the wired limit: what the Mac is, what it has, what the
// GPU may hold of it (from /status.json's node; left out when not known).
function factsHTML(n){
  const hw=n.machine||{}, nm=n.memory||{}, mm=n.memory_map||{};
  const inst=mm.installed_bytes||0, ws=nm.working_set_bytes||0;
  const f=[hw.chip&&(hw.model?hw.model+' · ':'')+hw.chip,
    inst&&`${gb(inst).replace(/\.\d+ /,' ')} installed`,
    ws&&`GPU working set ${gb(ws).replace(/\.\d+ /,' ')}`].filter(Boolean);
  return f.length ? `<div class="mfacts">${esc(f.join(' · '))}</div>` : '';
}
// The knurlogic allowance: the most memory knurlogic may use on the machine
// -- the ceiling for a load's fit check and a model server's memory guard.
// Staged like a knob and sent on close: this machine's to its own POST
// /allowance.json, a peer's to /machine.json?where=<its page>, which asks
// that peer's page to set its own (each Mac keeps its allowance itself).
function allowHTML(a, m, own){
  const head='<div class="sidelab">knurlogic allowance<i class="info down" tabindex="0">i<span class="bub">'
    +'The most memory knurlogic may use on this Mac. Models are loaded only if they fit under it; '
    +'the rest stays free for everything else on the machine.</span></i></div>';
  if(!a) return `<div class="allow">${head}<div class="msg">${own?'not read':
    'not reported: that machine\'s knurlogic predates the allowance'}</div></div>`;
  const none=!a.allowance_gib;
  const say=`${none?'none -- knurlogic takes the working set':
    'at most '+a.allowance_gib+' GiB'} · a load may have ${a.effective_gib} GiB`;
  const k={name:'knurlogic allowance', value:String(a.allowance_gib), reach:'live',
    unit:'GiB', what:'The most memory knurlogic may use on '+(own?'this machine':m.name)+': the '+
      'ceiling for a load\'s fit check and for the memory guard of a model '+
      'server it starts. 0 is none.',
    reach_why:'A load checks against it at once; a model already running '+
      'keeps the working set it started with. Kept in '+a.file+(own?'':' on '+m.name)};
  const c=own ? {g:'allowance', where:'', url:'/allowance.json', model:'', label:m.name+' · machine'}
    : {g:'allowance|'+m.id, where:'', url:'/machine.json?'+new URLSearchParams({where:m.page}),
       ukey:'allowance_gib', model:'', label:m.name+' · machine'};
  return `<div class="allow">${head}${knobHTML(k,c)}
    <div class="msg">${esc(say)}</div></div>`;
}
// Cluster: the machines down the side, this one first, then each answering
// peer, all drawn by the same code; only where the data comes from differs.
// A peer's settings are read through /peek (one GET) and set on its own page.
function showCluster(){
  const ms=machines();
  if(!ms.some(m=>m.id===SMACH)) setSMach(ms[0].id);
  $('setlist').innerHTML=ms.map(m=>`<div class="fam" data-m="${esc(m.id)}"
    aria-current="${m.id===SMACH}">${esc(m.name)}<small>${m.id==='local'?'this machine':
      esc(m.id)+(m.error?' · not answering':'')}</small></div>`).join('');
  $('setlist').querySelectorAll('[data-m]').forEach(v=>v.onclick=()=>{
    if(v.dataset.m===SMACH) return;
    setSMach(v.dataset.m); keepSet();
    $('setlist').querySelectorAll('[data-m]').forEach(x=>x.setAttribute('aria-current', x===v));
    showMachine(ms.find(m=>m.id===SMACH));
  });
  keepSet(); showMachine(ms.find(m=>m.id===SMACH));
}
async function showMachine(m){
  const el=$('machbody'), seq=nextSeq(), own=m.id==='local';
  const where=own?location.host:m.page.replace(/^https?:\/\//,'');
  el.innerHTML=`<div class="sgrp" data-m="${esc(m.id)}">
    <div class="chd">${esc(m.name)}<span class="ro">${esc(where)}${own?' · this machine':
      m.error?' · not answering':''}</span></div>
    ${factsHTML(m.node||{})}
    <div class="mwired"><div class="msg">reading…</div></div>
    <div class="mallow"></div>
  </div>`;
  const doc=own ? DOC : await getJSON(peekURL(m.page,'/settings.json'));
  const row=el.querySelector(`[data-m="${CSS.escape(m.id)}"]`);
  if(seq!==SEQ || !row) return;
  const wb=row.querySelector('.mwired');
  wb.innerHTML=wiredHTML(doc, m); wireWired(wb, m);
  row.querySelector('.mallow').innerHTML=allowHTML(doc&&doc.allowance, m, own);
}

export {ASK, DOC, loadSettings, showCluster};
