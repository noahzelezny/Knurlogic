import {OVL} from '../ui/overlay.js';
import {$, GIB, esc, gb} from '../format.js';
import {LASTNODES, fitWS, fitWhere, isLocal, selNodes} from '../nodes.js';
import {launchSets, migrateLaunchSets} from './settings/knobs.js';
import {getJSON, httpWhy, peekURL} from '../api.js';
import {tick} from './home.js';
import {act, loadResident} from './memory.js';

// --- what else is here ----------------------------------------------------
// Shown, not offered: a loaded model is loaded, and a page that presents a
// list of models as if any of them could be clicked into service would be
// promising something the runtime cannot do. What it can hand over is the
// command, which is why the row opens one.
// Runs AFTER the first status, not beside it: "fits this box" needs the
// working set, and two fetches racing means the list is sorted against zero
// about half the time.
// --- load model -----------------------------------------------------------
// One picker, its options, and a launch. A list with a button on each
// model is a catalogue, not a control: the answer to fifty rungs is a
// picker.
let MODELS=[], GROUPS=[], FAM='All', SEL=null, QUERY='';
// a launch that went ahead clears the pick; loadModels must not re-pick
let SELCLEARED=false;
// Favorites/recents: a set of starred paths and a 20-entry most-recent list,
// both in localStorage (the picker's own bookkeeping).
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
// A running model's name as its runtime gives it: org/name is the
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
  // favorites and recents are families of their own, first after All
  const favMs=MODELS.filter(m=>isFav(m.path));
  const recMs=loadRecents().map(r=>MODELS.find(m=>m.path===r.path)).filter(Boolean);
  const mine={Favorites:favMs, Recent:recMs};
  if(mine[FAM] && !mine[FAM].length) FAM='All';
  const n=f=>f==='All'?GROUPS.length:mine[f]?mine[f].length:famCount[f];
  $('pfams').innerHTML=`<div class="fam hub" data-f="${HUB}" aria-current="${FAM===HUB}">${HUB}</div>`
    +['All',...Object.keys(mine).filter(f=>mine[f].length),
    ...Object.keys(famCount).sort()].map(f=>
    `<div class="fam" data-f="${esc(f)}" aria-current="${f===FAM}">${esc(f)}
      <i>${n(f)}</i></div>`).join('');
  $('pfams').querySelectorAll('.fam').forEach(e=>e.onclick=()=>{
    FAM=e.dataset.f; renderPicker() });
  $('psearch').placeholder=FAM===HUB?'Search Hugging Face (MLX models)':'Search models';
  $('psearch').value=FAM===HUB?HF.q:QUERY;
  if(FAM===HUB){ $('pbudget').innerHTML=ws?`<b>${gb(ws)}</b> to fill`:'';
    return renderHub() }

  const shown=mine[FAM]?[]:GROUPS.filter(g=>(FAM==='All'||g.fam===FAM) &&
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
          <span class="star${isFav(m.path)?' on':''}" data-fav="${esc(m.path)}"
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
  const oneRow=m=>`<div class="var" data-p="${esc(m.path)}"
      aria-current="${SEL&&SEL.path===m.path}">
      <span class="star${isFav(m.path)?' on':''}" data-fav="${esc(m.path)}">${isFav(m.path)?'★':'☆'}</span>
      <span class="vn">${esc(m.name)}</span>
      ${isVision(m)?'<span class="tag">VISION</span>':''}
      <span class="vs">${gb(m.size_bytes)}</span></div>`;
  const favHTML=mine[FAM]?`<div class="vars flat">${mine[FAM].filter(m=>
    !q||m.name.toLowerCase().includes(q)).map(oneRow).join('')}</div>`:'';
  $('prows').innerHTML=favHTML+a.map(grp).join('')
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
                               tune:window.LOADTUNE||'default'});
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
  for(const n of ns){
    const r=(n.cluster||{}).rdma;
    if(!r) return `${n.node} does not report RDMA (update knurlogic there)`;
    if(!r.available) return `RDMA on ${n.node}: ${r.reason}`;
  }
  // a full mesh: every pair needs its own Thunderbolt 5 cable
  for(let i=0;i<ns.length;i++) for(let j=i+1;j<ns.length;j++){
    const w=tb5Why(ns[i].cluster, ns[j].cluster);
    if(w) return ns.length>2 ? `${ns[i].node} and ${ns[j].node}: ${w}` : w;
  }
  return '';
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
$('psearch').oninput=e=>{
  if(FAM===HUB){ HF.q=e.target.value; clearTimeout(HF.timer);
    HF.timer=setTimeout(hubSearch, 350) }
  else{ QUERY=e.target.value; renderPicker() } };

$('launch').onclick=async()=>{
  const m=SEL; if(!m || launchBlock(selNodes())) return;
  const b=$('launch'); b.disabled=true; b.textContent='Launching…';
  const pn=launchPeer(), ns=selNodes();
  const tune=window.LOADTUNE||'default';
  const sets=launchMTP(m);
  const t0=Date.now();
  const L=trackLaunch(m, ns, pn, t0);
  const j=await act(ns.length>1
    ? {action:'load', identity:m.identity, nodes:ns.map(n=>n.id),
       split:MULTI.shard, link:MULTI.link, tune, sets}
    : pn ? {action:'load', node:pn.id, identity:m.identity, tune, sets}
    : {action:'load', target:m.path, tune, sets});
  b.textContent='Launch'; b.disabled=false;
  settleLaunch(L, j);
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
window.LOADTUNE='default';
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

// --- Hugging Face ---------------------------------------------------------
// MLX-format models on the Hub, searched from the picker. A result opens to
// its size, architecture and whether this machine can run it (/hub/repo.json,
// fetched when the row opens); Download puts it in the standard cache on THIS
// machine, and once there it is a local model like any other.
const HUB='Hugging Face';
const HF={q:'', res:null, err:'', open:'', detail:{}, timer:0, seq:0};
let DLS=[];
const dlOf=id=>DLS.find(d=>d.id===id);
async function hubSearch(){
  const n=++HF.seq;
  const d=await getJSON('/hub/search.json?'+new URLSearchParams({q:HF.q}));
  if(n!==HF.seq) return;
  HF.res=d.results||[]; HF.err=d.error||'';
  if(FAM===HUB) renderHub();
}
async function hubOpen(id){
  HF.open=id; renderHub();
  if(!HF.detail[id]||HF.detail[id].error){
    HF.detail[id]={loading:true};
    HF.detail[id]=await getJSON('/hub/repo.json?'+new URLSearchParams({id}));
  }
  if(FAM===HUB) renderHub();
}
function hubDialog(){
  const box=$('picker').querySelector('.box');
  box.querySelector('.hfdlg')?.remove();
  removeEventListener('keydown',HF.esc,true);
  const id=HF.open;
  if(!id) return;
  const d=HF.detail[id], dl=dlOf(id), ws=fitWS();
  const local=MODELS.some(m=>m.name===id)
    ||(window.ALLMODELS||[]).some(m=>m.name===id);
  const busy=dl&&dl.state==='downloading';
  const ready=d&&!d.loading&&!d.error;
  const line=!d||d.loading?'':d.error?d.error
    :local?'already on this Mac':busy?'downloading…'
    :!d.supported?d.why:!d.access?'needs access — run hf auth login'
    :ws&&d.size_bytes>ws?"doesn't fit the picked machines":'';
  const can=ready&&d.supported&&d.access&&!local&&!busy;
  const el=document.createElement('div');
  el.className='hfdlg';
  el.innerHTML=`<div class="hfbox"><div class="hfttl">${esc(id)}</div>
    <div class="hfsz">${ready?gb(d.size_bytes):'…'}</div>
    <div class="vs hfnote">${esc(line)}</div>
    <div class="hfbtns"><button class="mini" data-c>Cancel</button>
      <button class="mini pri" data-d${can?'':' disabled'}>Download</button></div></div>`;
  const esc_=e=>{ if(e.key==='Escape'){ e.stopPropagation(); close() } };
  const close=()=>{
    removeEventListener('keydown',esc_,true);
    HF.open=''; el.remove(); renderHub() };
  HF.esc=esc_; addEventListener('keydown',esc_,true);
  el.onmousedown=e=>{ if(e.target===el) close() };
  el.querySelector('[data-c]').onclick=close;
  el.querySelector('[data-d]').onclick=async()=>{
    close();
    await hubAct('download', id);
    renderHub();
  };
  box.appendChild(el);
}
async function hubAct(action, id){
  const r=await fetch('/hub/download.json',{method:'POST',
    headers:{'Content-Type':'application/json'}, body:JSON.stringify({action,id})});
  await loadDownloads();
  if(!r.ok) alert(await httpWhy(r));
  loadResident();
}
// what the page's server has downloading, done or failed; the Downloads
// overlay lists them, and a finished one refreshes the model list once
const DONESEEN=new Set();
async function loadDownloads(){
  const d=await getJSON('/hub/downloads.json');
  if(d.error) return;
  DLS=d.downloads||[];
  const fresh=DLS.filter(x=>x.state==='done'&&!DONESEEN.has(x.id));
  fresh.forEach(x=>DONESEEN.add(x.id));
  if(fresh.length) await loadModels();
  if(!$('picker').hidden && FAM===HUB) renderHub();
}
const allDownloads=()=>DLS;
function renderHub(){
  const ws=fitWS();
  const row=r=>{
    const d=HF.detail[r.id], open=HF.open===r.id;
    const no=d&&!d.loading&&!d.error&&(!d.supported||!d.access);
    return `<div class="hfrow${no?' no':''}"><div class="var" data-hf="${esc(r.id)}"
        aria-current="${open}">
        <span class="vn">${esc(r.id)}</span>
        ${r.gated?'<span class="tag">GATED</span>':''}
        <span class="vs">${(r.downloads||0).toLocaleString()} downloads</span></div>
      </div>`;
  };
  $('prows').innerHTML=HF.err?`<div class="sect no">${esc(HF.err)}</div>`
    :HF.res===null?'<div class="sect no">searching…</div>'
    :HF.res.length?HF.res.map(row).join(''):'<div class="sect no">nothing matches</div>';
  $('prows').querySelectorAll('[data-hf]').forEach(v=>v.onclick=()=>hubOpen(v.dataset.hf));
  hubDialog();
  if(HF.res===null) hubSearch();
}

// --- a launch, followed until it is ready or has failed ----------------------
// Read from the residency the page already polls (/loaded.json?peers=1):
// each machine's `loads` (its servers started lately: phase, bytes held of
// the artifact's size, the last log line, why one exited) and `jobs` (a
// cluster job's phase per machine, and its stop reason).
const LAUNCHES=[];
let LSEQ=0;
// The card exists from the click, 'preparing' while the server checks and
// prepares every rank; the answer to the launch settles it.
function trackLaunch(m, ns, pn, t0){
  const L={id:++LSEQ, name:m.name, t0, port:0, job:'',
    machines:ns.length?ns.map(n=>n.node):[localName()],
    node:pn?pn.id:'', cluster:ns.length>1, phase:'preparing', samples:[]};
  LAUNCHES.unshift(L); if(LAUNCHES.length>4) LAUNCHES.length=4;
  loadResident();
  return L;
}
function settleLaunch(L, j){
  L.port=j.port||0; L.job=j.job||'';
  if(j.error||j.refused){ L.phase='failed';
    L.why=j.error||('not loaded: '+j.refused+(j.note?' -- '+j.note:'')) }
  else L.phase='starting';
  // said once and kept: a local copy that differs from the shared one
  if(j.alerts&&j.alerts.length) L.alert=j.alerts.join(' · ');
  loadResident();
}
// every machine's document, named: this page's and each peer's
function machinesOf(d){
  return [{name:localName(), id:'', doc:d}].concat((d.peers||[]).map(p=>
    ({name:p.machine, id:p.id, doc:p})));
}
function followLaunch(L, d){
  if(L.phase==='ready'||L.phase==='failed'||L.phase==='preparing') return;
  const ms=machinesOf(d), secs=(Date.now()-L.t0)/1000;
  const mine=ms.filter(x=>L.cluster ? true : L.node ? x.id===L.node : x.id==='');
  const nm=L.name.split('/').pop();
  const ld=mine.flatMap(x=>(x.doc.loads||[]).filter(e=>
    (!L.port||e.port===L.port) && e.name===nm));
  const e=ld.sort((a,b)=>a.seconds-b.seconds)[0];
  if(L.cluster){
    const per=mine.map(x=>(x.doc.loads||[]).filter(k=>k.name===nm)
      .sort((a,b)=>a.seconds-b.seconds)[0]).filter(Boolean);
    if(per.length){ L.bytes=per.reduce((s,k)=>s+k.bytes,0); L.total=per[0].total_bytes }
  } else if(e){ L.bytes=e.bytes; L.total=e.total_bytes; L.last=e.last_log_line;
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
    if(lead && lead.state==='loaded' && ph.length && ph.every(p=>p==='ready')) L.phase='ready';
    else L.phase=ph.includes('joining')?'joining ring':ph.includes('loading')?'loading weights'
      :e&&e.phase==='warming'?'warming':'starting';
    return;
  }
  const row=mine.flatMap(x=>x.doc.resident||[]).find(r=>r.runtime==='knurlogic'
    && r.name===nm && (!L.port||(r.where||'').replace(/\/$/,'').endsWith(':'+L.port)));
  if(row && row.state==='loaded' && !(e&&e.phase==='exited')){ L.phase='ready'; return }
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
  for(let i=LAUNCHES.length-1;i>=0;i--)
    if(LAUNCHES[i].phase==='ready') LAUNCHES.splice(i,1);
}
// the launches that failed, kept as INSTANCES cards until dismissed
function failedLaunches(){ return LAUNCHES.filter(L=>L.phase==='failed') }
function dismissLaunch(id){
  const i=LAUNCHES.findIndex(L=>L.id===id);
  if(i>=0) LAUNCHES.splice(i,1);
}
// the launches still loading, as rows for the INSTANCES card
function loadingLaunches(){
  return LAUNCHES.filter(L=>L.phase!=='ready'&&L.phase!=='failed');
}
function followLaunches(d){
  if(!LAUNCHES.length) return;
  for(const L of LAUNCHES) followLaunch(L, d);
  renderLaunches();
}

export {allDownloads, BASEKEY, SEL, baseKey, baseOf, dismissLaunch,
        failedLaunches, famOf, followLaunches, hubAct,
        loadDownloads, loadModels, loadingLaunches,
        nodeSelChanged, published, setSets};
