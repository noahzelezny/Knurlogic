import {OVL} from '../ui/overlay.js';
import {$, GIB, esc, gb} from '../format.js';
import {LASTNODES, fitWS, fitWhere, isLocal, selNodes} from '../nodes.js';
import {launchSets, migrateLaunchSets} from './settings/knobs.js';
import {getJSON, peekURL} from '../api.js';
import {tick} from './home.js';
import {act} from './memory.js';

// --- what else is here ----------------------------------------------------
// Shown, not offered: a loaded model is loaded, and a page that presents a
// list of models as if any of them could be clicked into service would be
// promising something the runtime cannot do. What it can hand over is the
// command, which is why the row opens one.
// Runs AFTER the first status, not beside it: "fits this box" needs the
// working set, and two fetches racing means the list is sorted against zero
// about half the time.
// --- load model -----------------------------------------------------------
// One picker, its options, and a launch -- exo's shape. The previous version
// listed seven models with a button on each and "+46 more", which is a
// catalogue pretending to be a control: the answer to fifty rungs is a
// picker, not a longer list.
let MODELS=[], GROUPS=[], FAM='All', SEL=null, QUERY='';
// a launch that went ahead clears the pick; loadModels must not re-pick
let SELCLEARED=false;
// Favorites/recents: a set of starred paths and a 20-entry most-recent list,
// both in localStorage (the idea is exo's picker; the code is ours).
function loadFav(){ try{ return new Set(JSON.parse(localStorage.getItem('kn.fav')||'[]')) }
  catch(e){ return new Set() } }
function isFav(p){ return loadFav().has(p) }
function toggleFav(p){ const s=loadFav();
  s.has(p)?s.delete(p):s.add(p);
  try{ localStorage.setItem('kn.fav', JSON.stringify([...s])) }catch(e){} }
function loadRecents(){ try{ return JSON.parse(localStorage.getItem('kn.recent')||'[]') }
  catch(e){ return [] } }
function pushRecent(path){
  let r=loadRecents().filter(x=>x.path!==path);
  r.unshift({path, at:Date.now()});
  try{ localStorage.setItem('kn.recent', JSON.stringify(r.slice(0,20))) }catch(e){}
}
// /models.json has no `vision` field (interfaces/page/documents.py), so
// this is a family-name heuristic matching the vision-capable families
// (gemma4, glm5, qwen-vl) until the server reports one.
function isVision(m){
  // The server's answer, never a guess from the name: /models.json carries
  // `vision` per model and /status.json the served VisionSpec. A name
  // heuristic missed every Qwen rung (no "vl" in the name).
  return !!(m && m.vision);
}
window.isVision=isVision;

// Grouping, tested against the real disk before any of this was drawn: 49
// artifacts collapse to 13 rows. Publishers name a rung by its quantisation
// -- `-8bit`, `-VQ-3.1bpw`, `-bf16` -- so stripping the publisher prefix and
// that tail leaves the model, and the rungs of one model gather under it.
const _PUB=/^[^/]*?(?:--|\/)/;
const _TAIL=/[-_]?(?:vq[-_])?(?:\d+(?:\.\d+)?bpw|\d+bit|bf16|fp16|mlx|vq[-_]base|base|q\d+|instruct|it)\b.*$/i;
// A name that parses to nothing shared and carries no publisher is a local
// build -- qwen4exp_vq_packed_mixL01p4q is not a model anybody publishes --
// and it groups under its architecture ("Qwen3.8 Flash-Next · VQ").
// Left ungrouped they were 14 rows of one, which is the wall again.
const ARCHLABEL={qwen4_exp_text:'Qwen3.8 Flash-Next', qwen3_5_moe_text:'Qwen3.5 MoE',
  qwen3_5_text:'Qwen3.8 27B', glm5_next_text:'GLM-5.3 Flash',
  gemma4_text:'Gemma 4', deepseek_v4:'DeepSeek V4'};
const FAMS=[[/glm/,'GLM'],[/deepseek/,'DeepSeek'],[/gemma/,'Gemma'],
            [/qwen/,'Qwen'],[/llama/,'Meta'],[/mistral/,'Mistral']];
function famOf(m){
  const hay=`${m.name} ${m.model_type||''}`.toLowerCase();
  for(const [re,name] of FAMS) if(re.test(hay)) return name;
  return 'Other';
}
function baseOf(n){ return n.replace(_PUB,'').replace(_TAIL,'')
  .replace(/^[-_\s]+|[-_\s]+$/g,'') || n }

// Each artifact's BASE MODEL: the picker's row, and the key its settings are
// kept under (Settings > Models, Launch). Worked out over every servable
// artifact on disk, the running ones too, so a model keeps its base while it
// runs and the picker hides it.
let BASEKEY={};
function baseKeys(ms){
  const bases={}, count={}, out={};
  ms.forEach(m=>{ const b=baseOf(m.name); bases[m.name]=b;
    count[b]=(count[b]||0)+1 });
  const lone=m=>count[bases[m.name]]===1
    && !m.name.includes('--') && !m.name.includes('/');
  const arch=m=>ARCHLABEL[m.model_type]||m.model_type||'unknown';
  // named by what they are: an architecture's VQ builds, when every one of
  // them is VQ, else just the architecture
  const allvq={};
  ms.filter(lone).forEach(m=>{ const a=arch(m);
    allvq[a]=(allvq[a]??true) && !!m.is_vq });
  // A published model's VQ builds are a model of their own beside its stock
  // quantisations -- different runtime, different knobs -- so they group
  // apart: "Qwen3.8-Flash-Next · VQ" holds the bpw rungs, the plain name the
  // bits. (Unpublished builds are not offered; they get a key only so one
  // that is running still has somewhere for its settings.)
  ms.forEach(m=>{
    out[m.name]=lone(m) ? (allvq[arch(m)] ? `${arch(m)} · VQ` : arch(m))
      : bases[m.name]+(m.is_vq?' · VQ':'');
  });
  return out;
}
// What Load model and Settings offer: finished, published models the engine
// runs -- an artifact under a publisher's namespace (Org--Name on disk,
// Org/Name in a hub cache) with an architecture it knows. Local and
// experimental builds (qwen4exp_*, 397b-v2-*, ...) are not something to pick.
const published=m=>m.servable && /--|\//.test(m.name)
  && !/-VQ-BASE$/i.test(m.name)
  && !!m.model_type && m.model_type!=='unknown';
// A running model's name as its runtime gives it: exo's org/name is the
// directory org--name; one not on this disk is named by the same rule.
function baseKey(name){
  const k=String(name||'').replace('/','--');
  return BASEKEY[k] || BASEKEY[name] || baseOf(k);
}
function regroup(){
  const by={};
  MODELS.forEach(m=>{
    const key=BASEKEY[m.name]||baseOf(m.name);
    (by[key]=by[key]||[]).push(m);
  });
  GROUPS=Object.entries(by).map(([name,ms])=>{
    ms.sort((a,b)=>b.size_bytes-a.size_bytes);
    return {name, ms, fam:famOf(ms[0]),
            lo:Math.min(...ms.map(m=>m.size_bytes)),
            hi:Math.max(...ms.map(m=>m.size_bytes))};
  }).sort((a,b)=>b.hi-a.hi);
}

function renderPicker(){
  const ws=fitWS();
  const q=QUERY.trim().toLowerCase();
  const famCount={};
  GROUPS.forEach(g=>famCount[g.fam]=(famCount[g.fam]||0)+1);
  $('pfams').innerHTML=['All',...Object.keys(famCount).sort()].map(f=>
    `<div class="fam" data-f="${esc(f)}" aria-current="${f===FAM}">${esc(f)}
      <i>${f==='All'?GROUPS.length:famCount[f]}</i></div>`).join('');
  $('pfams').querySelectorAll('.fam').forEach(e=>e.onclick=()=>{
    FAM=e.dataset.f; renderPicker() });

  const shown=GROUPS.filter(g=>(FAM==='All'||g.fam===FAM) &&
    (!q || g.name.toLowerCase().includes(q) ||
     g.ms.some(m=>m.name.toLowerCase().includes(q))));
  const okm=m=>(!ws||m.size_bytes<=ws) && onPeer(m);
  const fits=g=>g.ms.some(okm);
  const pn=launchPeer();
  const rng=g=>g.ms.length===1 ? gb(g.hi)
    : `${g.ms.length} variants (${(g.lo/GIB).toFixed(0)}–${(g.hi/GIB).toFixed(0)} GiB)`;
  const grp=g=>`<div class="grp${fits(g)?'':' no'}" data-g="${esc(g.name)}">
      <div class="grphd"><span class="cv">›</span>
        <span class="gn">${esc(g.name)}</span>
        <span class="gv">${rng(g)}</span></div>
      <div class="vars">${[...g.ms.filter(okm),...g.ms.filter(m=>!okm(m))].map(m=>{
        const ok=okm(m);
        return `<div class="var ${ok?'':'no'}" data-p="${esc(m.path)}"
          aria-current="${SEL&&SEL.path===m.path}">
          <span class="star" data-fav="${esc(m.path)}"
            title="favorite">${isFav(m.path)?'★':'☆'}</span>
          <span class="vn">${esc(m.name)}</span>
          ${onPeer(m)?'':`<span class="tag">not on ${esc(selNodes().filter(n=>!isLocal(n)&&!onNode(n,m)).map(n=>n.node).join(' + '))}</span>`}
          ${m.is_vq?'<span class="tag">VQ</span>':''}
          ${m.mtp?'<span class="tag">MTP</span>':''}
          ${isVision(m)?'<span class="tag">VISION</span>':''}
          <span class="vs${m.room&&m.room.small?' tight':''}"${m.room?` title="${
            esc(m.room.text)}"`:''}>${gb(m.size_bytes)}</span></div>`}).join('')}</div>
    </div>`;
  const a=shown.filter(fits), b=shown.filter(g=>!fits(g));
  // Favorites and recents, shown only on the unfiltered "All" view so they
  // never fight the search/family filters.
  let favHTML='';
  if(FAM==='All' && !q){
    const favMs=MODELS.filter(m=>isFav(m.path));
    const recMs=loadRecents().map(r=>MODELS.find(m=>m.path===r.path)).filter(Boolean);
    const oneRow=m=>`<div class="var" data-p="${esc(m.path)}"
        aria-current="${SEL&&SEL.path===m.path}">
        <span class="star" data-fav="${esc(m.path)}">${isFav(m.path)?'★':'☆'}</span>
        <span class="vn">${esc(m.name)}</span>
        ${isVision(m)?'<span class="tag">VISION</span>':''}
        <span class="vs">${gb(m.size_bytes)}</span></div>`;
    if(favMs.length) favHTML+=`<div class="sect">favorites</div>
      <div class="vars">${favMs.map(oneRow).join('')}</div>`;
    if(recMs.length) favHTML+=`<div class="sect">recent</div>
      <div class="vars">${recMs.map(oneRow).join('')}</div>`;
  }
  $('prows').innerHTML=favHTML+
    (a.length?`<div class="sect">for ${esc(fitWhere())}</div>`
      +a.map(grp).join(''):'')
    +(b.length?`<div class="sect no">${pn?`not on ${esc(pn.node)}, or `:``}needs more than ${gb(ws)}</div>`
      +b.map(grp).join(''):'')
    || '<div class="sect no">nothing matches</div>';

  $('prows').querySelectorAll('.grphd').forEach(h=>h.onclick=()=>
    h.parentElement.classList.toggle('open'));
  $('prows').querySelectorAll('[data-fav]').forEach(s=>s.onclick=e=>{
    e.stopPropagation(); toggleFav(s.dataset.fav); renderPicker();
  });
  $('prows').querySelectorAll('.var').forEach(v=>v.onclick=()=>{
    SEL=MODELS.find(m=>m.path===v.dataset.p); SELCLEARED=false;
    OVL.close();
    pickInfo();
  });
  // One group open is enough to show the shape without a wall of rungs.
  if(a.length===1||q) $('prows').querySelectorAll('.grp').forEach(
    g=>g.classList.add('open'));
  $('pbudget').innerHTML=ws?`<b>${gb(ws)}</b> to fill`:'';
}

async function loadModels(){
  let d; try{ d=await (await fetch('/models.json')).json() }catch(e){ return }
  window.ALLMODELS=d.models||[];
  MODELS=(d.models||[]).filter(m=>!m.serving && published(m))
    .sort((a,b)=>b.size_bytes-a.size_bytes);
  $('diskcount').textContent=`${(d.models||[]).length} models`;
  BASEKEY=baseKeys((d.models||[]).filter(m=>m.servable));
  migrateLaunchSets();
  regroup();
  const ws=fitWS();
  if(!SEL && !SELCLEARED) SEL=MODELS.find(m=>!ws||m.size_bytes<=ws)||null;
  pickInfo();
}
// What this artifact WOULD resolve to, fetched when the selection changes.
// What a model would launch with, asked for its room against the tune; the
// launch settings themselves (SETS) are edited in Settings.
let PREVIEW=null, SETS={};
// Settings re-reads them after an Apply (an imported binding is read-only there)
function setSets(v){ SETS=v }
async function loadPreview(){
  const m=SEL; if(!m){ PREVIEW=null; return }
  const q=new URLSearchParams({artifact:m.path,
                               tune:window.LOADTUNE||'balanced'});
  // the room is counted at the KV precision it would launch with
  const kvb=launchSets(baseKey(m.name)).KNURLOGIC_KV_BITS;
  if(kvb) q.set('kv_bits', kvb);
  // YaRN long context moves the context cap and what its KV needs
  const lc=launchSets(baseKey(m.name)).KNURLOGIC_LONG_CONTEXT;
  if(lc) q.set('long_context', lc);
  try{ PREVIEW=await (await fetch('/settings.json?'+q)).json(); PREVIEW.for=m.path }
  catch(e){ PREVIEW=null }
  if(SEL!==m) return;
  // the preview's room is asked against the tune picked, and fresher
  const pr=$('pickroom'); if(pr && PREVIEW && PREVIEW.room) pr.innerHTML=roomHTML(PREVIEW.room);
  mtpState(); launchGate();
}
function launchGate(){
  const why=SEL ? launchBlock(selNodes()) : 'choose a model';
  const b=$('launch'); if(b.textContent.trim()!=='Launch') return;
  b.disabled=!!why; b.title=why||'';
}

// What a model that fits leaves to talk in (the server's context_room):
// the working set, under this machine's allowance, less the weights and the
// scheduler's step margin -- in GiB and about how many tokens of context,
// shared by every conversation. Amber when it is small: GLM-5.3 "fit" the
// 128 GB M4 with 6 GiB left, and four long agent conversations never ran.
const ktok=n=>n>=1e6?(n/1e6).toFixed(1)+'M':n>=1e3?Math.round(n/1e3)+'k':String(n);
function roomHTML(r){
  if(!r) return '';
  // asked of THIS machine: a model picked for a bigger peer does not fit here
  if(!r.fits) return `<div class="room small">${gb(r.weights_bytes)} does not fit
    this machine's ${gb(r.working_set_bytes)} (its working set, under the
    knurlogic allowance)</div>`;
  return `<div class="room${r.small?' small':''}" title="${esc(
    `working set ${gb(r.working_set_bytes)} − weights ${gb(r.weights_bytes)} − `+
    `step margin ${gb(r.margin_bytes)}`+(r.kv_why?` · KV ${r.kv_why}`:'')+
    (r.window?` · its window is ${r.window.toLocaleString()} tokens`:''))}">${esc(r.text)}${
    r.small?' — little room for long conversations':''}</div>`;
}
function pickInfo(){
  const m=SEL, el=$('pickinfo');
  $('pickname').textContent=m?m.name:'choose a model';
  $('picksize').textContent=m?gb(m.size_bytes):'';
  whereLine(); splitState();
  if(!m){ el.innerHTML=''; $('launch').disabled=true; $('launch').title='choose a model';
    $('mtpopts').hidden=true; return }
  const ws=fitWS(), fits=!ws||m.size_bytes<=ws, ns=selNodes();
  const tags=[m.is_vq?'VQ':'', m.mtp?'MTP':''].filter(Boolean).join(' · ');
  // Launch loads on THIS machine and nowhere else: the page has no way yet
  // to start a model on a peer, or across several. Picking those still
  // answers the fit question; it does not pretend to launch.
  const blocked=launchBlock(ns);
  $('launch').disabled=!!blocked;
  $('launch').title=blocked||'';
  el.innerHTML=(tags?`<div>${tags}</div>`:'')+
    (!fits?`<div class="warn">more space required</div>`
     : ns.length===1&&isLocal(ns[0])
       ? `<div id="pickroom">${roomHTML(m.room)}</div>` : '')+
    (blocked && !/does not fit/.test(blocked)?`<div class="note">${esc(blocked)}</div>`:'');
  SETS=launchSets(baseKey(m.name)); mtpState(); loadPreview();
}
function launchBlock(ns){
  if(!ns.length) return 'pick the machines to run it on: click them in Memory';
  // the info pane's "it will not fit" is a reason Launch cannot go
  if(SEL){ const ws=fitWS();
    if(ws && SEL.size_bytes>ws) return `${gb(SEL.size_bytes)} does not fit ${fitWhere()} (${gb(ws)})`
    // on this machine: its own answer, under the knurlogic allowance
    const r=SEL.room || (PREVIEW && PREVIEW.for===SEL.path && PREVIEW.room);
    if(ns.length===1 && isLocal(ns[0]) && r && r.fits===false)
      return `${gb(r.weights_bytes)} does not fit this machine's ${gb(r.working_set_bytes)}` }
  if(ns.length>1){
    for(const n of ns){
      if(isLocal(n)) continue;
      if(!n.id || n.state!=='answering') return `${n.node} is not answering`;
      if(SEL && !onNode(n, SEL)) return `not on ${n.node}`;
    }
    if(SEL && Array.isArray(SEL.splits) && !SEL.splits.length)
      return `${SEL.model_type||'this model'} cannot be split across machines`;
    if(MULTI.link==='rdma'){ const why=rdmaWhy(ns); if(why) return why }
    return '';
  }
  if(ns.length===1 && !isLocal(ns[0])){
    const n=ns[0];
    if(!n.id || n.state!=='answering') return `${n.node} is not answering`;
    if(SEL && !onPeer(SEL)) return `not on ${n.node}`;
  }
  return '';
}
// --- one peer picked: Launch goes there -------------------------------------
// The peer's own /models.json (read through /peek) says what it has, by
// identity -- never by path; a model it does not have is greyed "not on".
// The load itself goes to this page's POST /loaded.json with `node`, which
// forwards it by identity; the peer runs its own checks.
const PEERMODELS={};   // node id -> Set of identities, once read
const launchPeer=()=>{ const ns=selNodes();
  return ns.length===1 && !isLocal(ns[0]) ? ns[0] : null };
function onNode(n, m){
  const have=PEERMODELS[n.id]; return !have || have.has(m.identity);
}
// every picked peer must hold the model: one of them missing it greys it
function onPeer(m){
  return selNodes().filter(n=>!isLocal(n)).every(n=>onNode(n, m));
}
async function loadPeerModels(){
  const todo=selNodes().filter(n=>!isLocal(n) && n.address && !PEERMODELS[n.id]);
  for(const n of todo){
    const d=await getJSON(peekURL('http://'+n.address,'/models.json'));
    if(d.error || !Array.isArray(d.models)) continue;
    PEERMODELS[n.id]=new Set(d.models.map(m=>m.identity).filter(Boolean));
  }
  if(todo.length){ pickInfo(); if(!$('picker').hidden) renderPicker() }
}
// RDMA (jaccl) needs it up on every picked machine: each machine's status
// says why not (cluster/links.rdma), and the button is greyed with that
function rdmaWhy(ns){
  if(ns.length!==2) return 'RDMA joins exactly two machines; use TCP/IP';
  for(const n of ns){
    const r=(n.cluster||{}).rdma;
    if(!r) return `${n.node} does not report RDMA (update knurlogic there)`;
    if(!r.available) return `RDMA on ${n.node}: ${r.reason}`;
  }
  return tb5Why(ns[0].cluster, ns[1].cluster);
}
// RDMA runs only over a Thunderbolt 5 (80 Gb/s) cable: over Thunderbolt 4
// the port still reads active and jaccl fails (cluster_jobs.rdma_pair_reason
// says the same on launch). A link an older knurlogic does not time passes.
function tb5Why(a, b){
  const net=ip=>String(ip).split('.').slice(0,3).join('.');
  const on=m=>{ const act=new Set(((m.rdma||{}).active)||[]), o={};
    for(const t of m.thunderbolt||[]) if(t.ip && act.has('rdma_'+t.iface))
      o[net(t.ip)]=t.gbps||0;
    return o };
  const A=on(a||{}), B=on(b||{}), slow=[];
  for(const k of Object.keys(A)) if(k in B){
    const g=Math.min(A[k]||1e9, B[k]||1e9);
    if(g>=80) return '';
    slow.push(`the ${k} link is Thunderbolt ${g>=40?4:3} (${g} Gb/s)`);
  }
  return slow.length ? 'RDMA needs a Thunderbolt 5 cable between these Macs; '
    + slow.join('; ') : '';
}
// A pick changed what "fits" means: the rail and an open picker follow.
// Sharding and interconnect for a multi-machine launch: page state only.
const MULTI={shard:'tensor', link:'tcp'};
$('multiopts').querySelectorAll('.seg').forEach(g=>
  g.querySelectorAll('button').forEach(b=>b.onclick=()=>{
    if(b.disabled) return;
    MULTI[g.dataset.k]=b.dataset.v;
    g.querySelectorAll('button').forEach(x=>x.setAttribute('aria-pressed', x===b));
    pickInfo();
  }));
// RDMA greyed, with the reason on hover, when a picked machine cannot
// Only the splits this model can take (the server runs the launch's own
// refusals, /models.json `splits`); one left is a fixed choice, none hides it
function splitState(){
  const ok=SEL && Array.isArray(SEL.splits) ? SEL.splits : ['tensor','pipeline'];
  const g=$('multiopts').querySelector('[data-k=shard]');
  if(!ok.includes(MULTI.shard) && ok.length) MULTI.shard=ok[0];
  g.querySelectorAll('button').forEach(b=>{
    b.hidden=!ok.includes(b.dataset.v);
    b.disabled=ok.length<2;
    b.setAttribute('aria-pressed', b.dataset.v===MULTI.shard);
  });
  g.closest('.opt').hidden=!ok.length;
}
function multiState(){
  const ns=selNodes(), b=$('multiopts').querySelector('[data-v=rdma]');
  const why=ns.length>1 ? rdmaWhy(ns) : '';
  b.disabled=!!why; b.title=why||'RDMA over Thunderbolt (jaccl)';
  if(why && MULTI.link==='rdma'){
    MULTI.link='tcp';
    b.parentNode.querySelectorAll('button').forEach(x=>
      x.setAttribute('aria-pressed', x.dataset.v==='tcp'));
  }
}
// What a cluster launch placed where, as the coordinator decided it
function placementHTML(p){
  if(!p) return '';
  return `<div class="room">leader ${esc(p.leader)} · ${esc(p.split)}</div>`+
    (p.shares||[]).map(s=>`<div class="note">rank ${s.rank} ${esc(s.machine)}: ${
      s.layers!=null?`${s.layers} layers (${s.bounds[0]}..${s.bounds[1]-1}), `:''}${gb(s.bytes)}</div>`).join('')+
    `<div class="note">${esc(p.reason)}</div>`;
}
function nodeSelChanged(){
  $('multiopts').hidden=selNodes().length<2;
  multiState();
  loadPeerModels();
  pickInfo();
  if(!$('picker').hidden) renderPicker();
}
$('openpick').onclick=()=>{
  OVL.open($('picker'), {face:$('picker').querySelector('.box')});
  renderPicker(); $('psearch').focus();
};
$('pclose').onclick=()=>OVL.close();
$('psearch').oninput=e=>{ QUERY=e.target.value; renderPicker() };

$('launch').onclick=async()=>{
  const m=SEL; if(!m || launchBlock(selNodes())) return;
  const b=$('launch'); b.disabled=true; b.textContent='Launching…';
  const pn=launchPeer(), ns=selNodes();
  const tune=window.LOADTUNE||'balanced';
  const sets=launchMTP(m);
  const t0=Date.now();
  const j=await act(ns.length>1
    ? {action:'load', identity:m.identity, nodes:ns.map(n=>n.id),
       split:MULTI.shard, link:MULTI.link, tune, sets}
    : pn ? {action:'load', node:pn.id, identity:m.identity, tune, sets}
    : {action:'load', target:m.path, tune, sets});
  b.textContent='Launch'; b.disabled=false;
  trackLaunch(m, ns, pn, j, t0);
  // across machines: where it went (or would have), from the coordinator
  if(j.placement) $('pickinfo').insertAdjacentHTML('beforeend',
    `<div class="placement">${placementHTML(j.placement)}</div>`);
  if(!j.error && !j.refused){
    // done with this pick: back to "choose a model"
    SEL=null; SELCLEARED=true; pickInfo();
    tick(); loadModels(); pushRecent(m.path);
    const st=(j.settings||{}).needs_restart||{};
    const n=Object.keys(st).length;
    if(n) alert(`Loaded. ${n} setting${n===1?'':'s'} could not be applied `+
      `without a restart:\n\n`+Object.entries(st).map(([k,v])=>
        `${k}  wants ${v.wanted}, running ${v.running}`).join('\n'));
  }
};
// the tune a launch takes: this machine's knurlogic strategy (Settings ->
// Knurlogic); a base model's own Launch preset still beats it
window.LOADTUNE='balanced';
getJSON('/strategy.json').then(d=>{ if(d&&d.preset) window.LOADTUNE=d.preset });

// --- which machines, and MTP -------------------------------------------------
const localName=()=>((LASTNODES.find(isLocal)||{}).node)||'this Mac';
function whereLine(){
  const ns=selNodes();
  $('lmwhere').textContent=ns.length ? 'on: '+ns.map(n=>n.node).join(' + ')
    : 'on: none picked (click machines in Memory)';
}
// MTP for this launch: shown only for a model that ships a draft head. The
// default is its saved launch settings (Settings -> Models), else what the
// preset resolves to (the preview), else on; a click here is for this launch.
const LMTP={for:null, mtp:null, dyn:null};
function mtpDefault(name){
  const saved=SETS[name]; if(saved) return String(saved);
  const k=((PREVIEW&&SEL&&PREVIEW.for===SEL.path&&PREVIEW.knobs)||[]).find(k=>k.name===name);
  const v=k && (k.value??k.would_be);
  return v==='off'||v===false||v==='0' ? 'off' : 'on';
}
function mtpState(){
  const m=SEL, box=$('mtpopts');
  if(!m || !m.mtp){ box.hidden=true; return }
  if(LMTP.for!==m.path){ LMTP.for=m.path; LMTP.mtp=LMTP.dyn=null }
  const mtp=LMTP.mtp||mtpDefault('KNURLOGIC_MTP');
  const dyn=LMTP.dyn||mtpDefault('KNURLOGIC_MTP_DYNAMIC');
  box.hidden=false;
  const set=(k,v)=>box.querySelectorAll(`[data-k=${k}] button`).forEach(b=>
    b.setAttribute('aria-pressed', b.dataset.v===v));
  set('mtp',mtp); set('dyn',dyn);
  $('mtpdynrow').hidden=mtp==='off';
}
$('mtpopts').querySelectorAll('.seg').forEach(g=>
  g.querySelectorAll('button').forEach(b=>b.onclick=()=>{
    LMTP[g.dataset.k]=b.dataset.v; mtpState();
  }));
// The launch's settings: the saved ones, with this panel's MTP choice
function launchMTP(m){
  const sets={...SETS};
  if(m.mtp){
    sets.KNURLOGIC_MTP=LMTP.mtp||mtpDefault('KNURLOGIC_MTP');
    if(sets.KNURLOGIC_MTP==='on')
      sets.KNURLOGIC_MTP_DYNAMIC=LMTP.dyn||mtpDefault('KNURLOGIC_MTP_DYNAMIC');
    else delete sets.KNURLOGIC_MTP_DYNAMIC;
  }
  return sets;
}

// --- a launch, followed until it is ready or has failed ----------------------
// Read from the residency the page already polls (/loaded.json?peers=1):
// each machine's `loads` (its servers started lately: phase, bytes held of
// the artifact's size, the last log line, why one exited) and `jobs` (a
// cluster job's phase per machine, and its stop reason).
const LAUNCHES=[];
let LSEQ=0;
function trackLaunch(m, ns, pn, j, t0){
  const L={id:++LSEQ, name:m.name, t0, port:j.port||0, job:j.job||'',
    machines:ns.length?ns.map(n=>n.node):[localName()],
    node:pn?pn.id:'', cluster:ns.length>1, phase:'starting', samples:[]};
  if(j.error||j.refused){ L.phase='failed';
    L.why=j.error||('not loaded: '+j.refused+(j.note?' -- '+j.note:'')) }
  LAUNCHES.unshift(L); if(LAUNCHES.length>4) LAUNCHES.length=4;
  renderLaunches();
}
// every machine's document, named: this page's and each peer's
function machinesOf(d){
  return [{name:localName(), id:'', doc:d}].concat((d.peers||[]).map(p=>
    ({name:p.machine, id:p.id, doc:p})));
}
function followLaunch(L, d){
  if(L.phase==='ready'||L.phase==='failed') return;
  const ms=machinesOf(d), secs=(Date.now()-L.t0)/1000;
  const mine=ms.filter(x=>L.cluster ? true : L.node ? x.id===L.node : x.id==='');
  const nm=L.name.split('/').pop();
  const ld=mine.flatMap(x=>(x.doc.loads||[]).filter(e=>
    (!L.port||e.port===L.port) && e.name===nm));
  const e=ld.sort((a,b)=>a.seconds-b.seconds)[0];
  if(e){ L.bytes=e.bytes; L.total=e.total_bytes; L.last=e.last_log_line;
    L.samples.push([Date.now(), e.bytes]); if(L.samples.length>6) L.samples.shift() }
  if(L.cluster && L.job){
    L.per=[];
    for(const x of ms) for(const jb of (x.doc.jobs||[])) if(jb.job===L.job){
      L.per.push({machine:x.name, phase:jb.phase});
      if(jb.phase==='stopped'||jb.phase==='stopping'){
        L.phase='failed'; L.why=jb.reason||'the cluster job stopped'; return }
    }
    const lead=ms.flatMap(x=>x.doc.resident||[]).find(r=>r.cluster&&r.cluster.job===L.job);
    const ph=L.per.map(p=>p.phase);
    if(lead && lead.state==='loaded' && ph.length && ph.every(p=>p==='ready')
       && (!e || e.phase==='ready')) L.phase='ready';
    else L.phase=ph.includes('joining')?'joining ring':ph.includes('loading')?'loading weights'
      :e&&e.phase==='warming'?'warming':'starting';
    return;
  }
  if(e){
    if(e.phase==='exited'){ L.phase='failed';
      L.why=e.refused||(e.log_tail||[]).slice(-2).join(' / ')||'the server exited'; return }
    L.phase={loading:'loading weights', stalled:'stalled', warming:'warming',
             ready:'ready'}[e.phase]||'starting';
    if(e.phase==='stalled') L.why=`no log output for a while; last: ${e.last_log_line||'--'}`;
    return;
  }
  // a machine that does not report `loads` (older knurlogic): its row says
  const r=mine.flatMap(x=>x.doc.resident||[]).find(r=>r.runtime==='knurlogic'
    && r.name===nm && (!L.port||(r.where||'').endsWith(':'+L.port)));
  if(r) L.phase=r.state==='loaded'?'ready':'loading weights';
  else if(secs>60) L.phase='starting';
}
function renderLaunches(){
  const el=$('lmprog'); if(!el) return;
  el.innerHTML=LAUNCHES.map(L=>{
    const secs=Math.round((Date.now()-L.t0)/1000);
    const el2=secs<90?secs+'s':Math.floor(secs/60)+'m '+(secs%60)+'s';
    const pct=L.total?Math.min(100,Math.round(100*L.bytes/L.total)):null;
    let rate='';
    if(L.samples.length>1){ const [a,b]=[L.samples[0],L.samples[L.samples.length-1]];
      const r=(b[1]-a[1])/((b[0]-a[0])/1000); if(r>0) rate=` · ${(r/1e9).toFixed(2)} GB/s` }
    const busy=L.phase!=='ready'&&L.phase!=='failed';
    return `<div class="lp ${L.phase==='ready'?'ready':L.phase==='failed'?'failed':''}">
      <div class="lph"><span class="ph">${esc(L.phase)}</span><b title="${esc(L.name)}">${esc(String(L.name).split("--").pop())}</b>
        <button class="mini x" data-lx="${L.id}" title="dismiss">✕</button></div>
      <div>on ${esc(L.machines.join(' + '))} · ${el2}${busy&&L.total?` · ${gb(L.bytes)} of ${gb(L.total)}`:''}${busy?rate:''}</div>
      ${busy&&pct!=null?`<div class="bar"><i style="width:${pct}%"></i></div>`:''}
      ${L.per&&L.per.length&&busy?`<div>${L.per.map(p=>esc(p.machine+': '+p.phase)).join(' · ')}</div>`:''}
      ${L.phase==='ready'?'<div>ready: pick it in the chat bar\'s Model:</div>':''}
      ${L.why?`<div class="why">${esc(L.why)}</div>`:''}
    </div>`}).join('');
  el.querySelectorAll('[data-lx]').forEach(b=>b.onclick=()=>{
    const i=LAUNCHES.findIndex(L=>L.id===+b.dataset.lx);
    if(i>=0) LAUNCHES.splice(i,1); renderLaunches();
  });
}
function followLaunches(d){
  if(!LAUNCHES.length) return;
  for(const L of LAUNCHES) followLaunch(L, d);
  renderLaunches();
}

export {BASEKEY, SEL, baseKey, baseOf, followLaunches, loadModels,
        nodeSelChanged, published, setSets};
