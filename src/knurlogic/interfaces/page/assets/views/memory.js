import {$, esc, gb} from '../format.js';
import {dismissLaunch, downloadingNow, failedDownloads, failedLaunches,
  followLaunches, hubAct, loadDownloads, loadingLaunches} from './picker.js';

// --- what is actually in memory, whoever put it there ---------------------
// exo's models, ollama's models and ours in one list. A machine has one pool
// of memory and every runtime is spending from it, so opening another tool's
// page to see what it is holding is a thing this should have made
// unnecessary.
// A runtime's own account of what it holds and the OS's account of what its
// processes cost are different numbers, and the gap is the thing that is
// hard to chase down by hand. Both are shown; neither is derived from the
// other.
let RAM=null;   // the memory map, shared with the topology's gauges
const RAMCOL={knurlogic:'var(--acc)', exo:'var(--warn)',
              ollama:'var(--ok)', 'mlx-lm':'#7aa2f7', 'mlx-vlm':'#bb9af7',
              vqlab:'#e0af68'};
function ramWent(m){
  const el=$('ramwent');
  RAM=m && m.installed_bytes ? m : null;
  if(!RAM){ el.innerHTML=''; return }
  const inst=m.installed_bytes, by=Object.entries(m.by_runtime||{})
    .sort((a,b)=>b[1]-a[1]);
  // DERIVED FROM ONE BASE, never carried alongside. Taking `other` from
  // other_bytes (used - named) and `free` from installed - seen_bytes (the
  // sum of process FOOTPRINTS) mixes two bases: the rows can sum to
  // 118.7 GiB on a 96 GiB machine and disagree with the gauge beside them.
  // Everything here comes from
  // `installed` and `used` in the same object, so the rows add up to the
  // machine by construction.
  const used=Number.isFinite(m.used_bytes)?m.used_bytes:(m.seen_bytes||0);
  const named=by.reduce((x,[,v])=>x+v,0);
  const other=Math.max(used-named,0);
  const free=Math.max(inst-used,0);
  const row=(k,b,c)=>`<div class="ramrow">${
    c?`<em style="background:${c}"></em>`
     :'<em style="border:1px solid var(--line)"></em>'}<span>${k}</span>
    <b>${gb(b)}</b></div>`;
  // No bar. The machines in the topology are already gauges and now fill
  // in these colours, so a second full-width chart of the same split would
  // be the free-space bar's mistake again. This is the key to that picture,
  // and the key carries the numbers.
  el.innerHTML=`<div class="ram">
    ${by.map(([k,b])=>row(k,b,RAMCOL[k]||'var(--dim)')+
      (k==='knurlogic'&&m.knurlogic_cache>0
        ? `<div class="ramrow sub"><em class="hatch" style="--c:${RAMCOL.knurlogic}"></em>
            <span>of which cache</span><b>${gb(m.knurlogic_cache)}</b></div>`:'')).join('')}
    ${row('everything else',other,'var(--faint)')}
    ${row('unused',free,'')}
    ${m.swap_bytes>0?row('swap',m.swap_bytes,'var(--bad)'):''}</div>`;
}
async function act(payload){
  // A server that is down or answers with something other than JSON is an
  // answer too: said, and the buttons that awaited this get theirs back.
  let j;
  try{
    const r=await fetch('/loaded.json',{method:'POST',
      headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)});
    j=await r.json();
  }catch(e){ j={error:`the server did not answer: ${e.message||e}`} }
  // A refusal is an answer, and the page has to say it the way the MCP does
  // -- a Launch button that quietly resets is the page's version of an agent
  // waiting on silence.
  if(j.error) alert(j.error);
  else if(j.refused) alert(refusalText(j));
  await loadResident();
  return j;
}
function refusalText(j){
  const L=['Not loaded: '+j.refused];
  const bl=[...(j.blockers||[]),...(j.detail&&Array.isArray(j.detail)?j.detail:[])];
  bl.forEach(b=>L.push('  - '+b.what+(b.detail&&b.detail.phase?' ('+b.detail.phase+')':'')));
  const d=(j.detail&&!Array.isArray(j.detail))?j.detail:{};
  if(d.short_by_gib) L.push(`  needs ${d.model_gib} GiB, ${d.available_gib_total} free: `+
                            `${d.short_by_gib} GiB short`);
  (j.nodes||[]).filter(n=>!n.fits).forEach(n=>
    L.push(`  - ${n.node}: shard ${n.gib} GiB, ${n.available_gib} free`));
  if((j.holding_memory||[]).length){ L.push('','Holding memory now:');
    j.holding_memory.forEach(h=>L.push(`  - ${h.what} (${h.where}, ${h.phase})`)) }
  if(j.note) L.push('',j.note);
  return L.join('\n');
}
async function loadResident(){
  // ?peers=1: the page process also asks every answering peer what it is
  // holding, so a model on the other machine is on this page too, the way
  // exo's topology shows it. The server bounds the wait; a dead peer comes
  // back as an error line, never as a missing answer.
  let d; try{ d=await (await fetch('/loaded.json?peers=1')).json() }catch(e){ return }
  const peers=d.peers||[];
  window.RESIDENCY=d;       // Settings' machine tabs and Connect read this
  followLaunches(d);
  await loadDownloads();
  // One list, local first; a peer's rows carry `machine`. A peer's model is
  // shown and chatted with, never unloaded from here: this page drives only
  // what its own machine started.
  // A peer's knurlogic model is unloaded through that peer (forwarded
  // by port; the peer stops only what it started).
  const rows=(d.resident||[]).concat(...peers.map(m=>(m.resident||[])
      .map(r=>({...r, node:m.id}))))
    .map(r=>r.machine?{...r, can_unload:r.runtime==='knurlogic' && !!r.node
      && /:\d+\/?$/.test(r.where||'')}:r);
  const el=$('resident');
  // One card per instance, not per machine: a cluster job is one instance,
  // named once with the machines it runs on. The whole card opens a chat.
  const on=r=>r.cluster?(r.cluster.machines||[]).join(' + ')
    :(r.machine||((window.NODES||[]).find(n=>n.role==='local')||{}).node||'this machine');
  const card=(r,i)=>`<div class="card ${esc(r.state)}"${r.runtime==='exo'?
      ' title="exo\'s model: shown, not driven by knurlogic"'
      :` title="click to chat with it" data-chatn="${i}" style="cursor:pointer"`}>
      <div class="cardhd">
        <span class="dot"></span>
        <span class="rt ${esc(r.runtime)}"${r.runtime==='knurlogic'?` title="knurlogic"`:''}>${
          r.runtime!=='knurlogic' ? esc(r.runtime)
          : r.recovery&&r.recovery.state==='recovering' ? 'recovering'
          : r.state!=='loaded' ? esc(r.state)
          : r.cluster&&r.cluster.phase&&r.cluster.phase!=='ready' ? esc(r.cluster.phase)
          : r.requests&&(r.requests.in_flight||r.requests.pending) ? 'running' : 'ready'}</span>
        <span class="grow"></span>
        ${r.can_unload?`<button class="mini danger" data-i="${i}"
          >Unload</button>`:''}
      </div>
      <div class="n">${esc(r.name)}</div>
      <div class="s">${[
        r.bytes_resident?gb(r.bytes_resident)
          :(r.state==='offered'?'':'size not reported'),
        r.cluster?`${r.cluster.split} over ${
          (r.cluster.link==='rdma'||r.cluster.link==='jaccl')?'RDMA':'TCP/IP'}${
          r.cluster.phase&&r.cluster.phase!=='ready'?' · '+r.cluster.phase:''}`:'',
        r.detail, r.state==='loaded'?'':r.state,
        r.instance?`id ${r.instance.slice(0,6)}`:'']
        .filter(Boolean).filter((v,i,x)=>x.indexOf(v)===i)
        .map(esc).join(' · ')}</div>
      <div class="s">on ${esc(on(r))}</div>
    </div>`;
  const none='<div class="card offered"><div class="cardhd">'+
    '<span class="dot"></span><span class="n">nothing loaded</span></div></div>';
  // the same instance reported by more than one page (a cluster job,
  // named by more than one rank's page) is still one card
  const seen=new Set();
  const shown=rows.map((r,i)=>[r,i]).filter(([r])=>{
    const inst=r.instance||(r.cluster&&r.cluster.job);
    if(!inst) return true;
    if(seen.has(inst)) return false;
    seen.add(inst); return true });
  const busy=loadingLaunches();
  const mine=([r])=>busy.some(L=>r.runtime==='knurlogic'
    && (r.name===String(L.name).split('/').pop())
    && (r.cluster ? (L.job ? r.cluster.job===L.job : L.cluster)
        : r.state!=='loaded' && !L.cluster && L.port
          && (r.where||'').replace(/\/$/,'').endsWith(':'+L.port)));
  const cards=shown.filter(x=>!mine(x)).map(([r,i])=>card(r,i));
  const loading=busy.map(L=>{
    const pct=L.total?Math.min(100,Math.round(100*L.bytes/L.total)):null;
    const say=L.phase==='warming'?'warming up':pct!=null&&L.phase==='loading weights'
      ?`loading ${pct}%`:L.phase;
    return `<div class="card loading"><div class="cardhd"><span class="dot"></span>
        <span class="rt knurlogic">${esc(say)}</span></div>
      <div class="n">${esc(String(L.name).split('/').pop())}</div>
      ${pct!=null||L.phase==='preparing'?`<div class="lbar"><i style="width:${pct||0}%"></i></div>`:''}
      <div class="s">on ${esc(L.machines.join(' + '))}${L.per&&L.per.length?' · '
        +esc(L.per.map(p=>p.machine+': '+p.phase).join(' · ')):''}</div></div>`});
  const failed=failedLaunches().map(L=>
    `<div class="card failed"><div class="cardhd"><span class="dot"></span>
        <span class="rt">FAILED</span>
        <button class="mini x" data-lx="${L.id}" title="dismiss">×</button></div>
      <div class="n">${esc(String(L.name).split('/').pop())}</div>
      <div class="s">on ${esc(L.machines.join(' + '))}</div>
      ${L.alert?`<div class="why">${esc(L.alert)}</div>`:''}
      ${L.why?`<div class="why">${esc(L.why)}</div>`:''}</div>`);
  const dls=downloadingNow().map(D=>{
    const pct=D.total_bytes?Math.min(100,Math.round(100*D.bytes/D.total_bytes)):0;
    return `<div class="card loading"><div class="cardhd"><span class="dot"></span>
        <span class="rt knurlogic">${D.total_bytes?`downloading ${pct}%`:'preparing'}</span>
        <button class="mini x" data-dc="${esc(D.id)}" title="cancel">×</button></div>
      <div class="n">${esc(D.id)}</div>
      <div class="lbar"><i style="width:${pct}%"></i></div>
      <div class="s">${D.total_bytes?gb(D.bytes)+' of '+gb(D.total_bytes)+' · ':''}to this Mac</div></div>`});
  const dlFailed=failedDownloads().map(D=>
    `<div class="card failed"><div class="cardhd"><span class="dot"></span>
        <span class="rt">FAILED</span>
        <button class="mini x" data-dx="${esc(D.id)}" title="dismiss">×</button></div>
      <div class="n">${esc(D.id)}</div>
      <div class="s">download to this Mac</div>
      <div class="why">${esc(D.why)}</div></div>`);
  failed.unshift(...dlFailed); loading.unshift(...dls);
  let h=cards.length||loading.length||failed.length
    ?failed.concat(loading,cards).join(''):none;
  for(const m of peers) if(m.error)
    h+=`<div class="s ro peererr" title="${esc(m.error)}">${
      esc(m.machine)} not answering</div>`;
  el.innerHTML=h;
  // What can be chatted with right now, for the chat bar's label: a chat
  // whose model has gone away no longer has a selection to name.
  window.RUNNING=new Set(rows.filter(r=>r.where).map(r=>r.where));
  // what the chat bar's Model: offers -- the rows a click on the name chats with
  window.CHATTABLE=rows.filter(r=>r.where && r.runtime!=='exo');
  if(window.chatGate) window.chatGate();
  el.querySelectorAll('[data-chatn]').forEach(x=>x.onclick=e=>{
    if(e.target.closest('button')) return;
    const r=rows[+x.dataset.chatn];
    if(window.switchTab) window.switchTab('chat');
    if(window.useModel) window.useModel(r);
  });
  // Only when status has told us nothing: in a cluster the key is the sum
  // of what the machines are drawing, and this document is one box.
  // NOT a second writer for the key. `tick` renders it from the node maps
  // the machines are drawn from; this document is a different sample of the
  // same box, and letting both write produced a key that disagreed with the
  // gauge beside it. Only used when status supplied nothing at all.
  if(!RAM) ramWent(d.memory||{});
  el.querySelectorAll('[data-dc]').forEach(x=>x.onclick=()=>hubAct('cancel', x.dataset.dc));
  el.querySelectorAll('[data-dx]').forEach(x=>x.onclick=()=>hubAct('dismiss', x.dataset.dx));
  el.querySelectorAll('[data-lx]').forEach(x=>x.onclick=()=>{
    dismissLaunch(+x.dataset.lx); loadResident();
  });
  el.querySelectorAll('.cardhd button[data-i]').forEach(x=>x.onclick=async()=>{
    const r=rows[+x.dataset.i];
    const how={knurlogic:'unload', ollama:'ollama-unload'};
    x.textContent='…'; x.disabled=true;
    await act(r.machine
      ? {action:'unload', node:r.node, port:+r.where.replace(/\/$/,'').split(':').pop()}
      : {action:how[r.runtime], target:r.ident, where:r.where});
  });
}

export {RAM, RAMCOL, act, loadResident, ramWent};
