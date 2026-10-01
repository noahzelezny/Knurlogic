import {$, GIB, esc, gb, gb0} from '../format.js';
import {NODESEL, isLocal, nodeWS, saveNodeSel, selNodes, setLastNodes,
        setLastWS} from '../nodes.js';
import {nodeSelChanged} from './picker.js';
import {RAM, RAMCOL, ramWent, SWAP_FLOOR} from './memory.js';

// --- status ---------------------------------------------------------------
function setState(ok, text){
  const el=$('state');
  el.classList.toggle('off', !ok);
  el.lastElementChild.textContent=text;
}
async function tick(){
  let d;
  try{ d=await (await fetch('/status.json')).json() }
  catch(e){ setState(false,'no answer'); return }
  const a=d.artifact, m=d.memory, C=d.cluster;
  // In a cluster the honest reading is how many nodes ANSWERED, not that
  // the page it is drawn on is up.
  const ns=d.nodes||[], up=ns.filter(n=>n.reachable!==false).length;
  window.NODES=ns;          // Settings' Machine tab reads chip and memory here
  setState(up===ns.length, ns.length>1 ? `${up}/${ns.length} nodes`
    : (up ? 'serving' : 'no answer'));
  // Nothing loaded is a STATE, not a page still loading, and the bench has
  // nothing to send a prompt to -- so it goes rather than sitting there with
  // a button that would fail.
  // The chat talks to a server `load` started
  // through this page's /chat proxy, so it stays even when this page serves
  // nothing; the bench speaks to this page's own endpoint and goes.
  $('try').style.display='';
  // The Chat/Bench strip only when there is a Bench to switch to: alone,
  // a teal "Chat" tab over the chat is a label for what is already plain.
  $('ptabs').parentElement.style.display=a?'':'none';
  if(!a && window.ACTIVETAB==='bench' && window.switchTab) window.switchTab('chat');
  $('mem').style.display=a?'':'none';
  $('served').style.display=a?'':'none';
  if(!a){ $('title').textContent='no model loaded';
    $('sub').textContent='';
    $('badges').innerHTML=''; }
  if(a){ $('title').textContent=a.name;
    $('sub').textContent=`${a.model_type} · ${gb(a.size_bytes)}`;
    const b=[];
    if(a.is_vq) b.push(['VQ',1]);
    if(d.vision && d.vision.served) b.push(['VISION',1]);
    if(a.bundled_runtime) b.push([a.bundled_runtime,1]);
    (d.architectures||[]).forEach(r=>b.push([r.module+' '+r.state,
      r.state==='OK'||r.state==='UNPINNED']));
    $('badges').innerHTML=b.map(([t,on])=>
      `<span class="badge${on?' on':''}">${esc(t)}</span>`).join('');
  }
  window.PAGEMODEL=a;
  if(window.chatServedChanged) window.chatServedChanged(a);
  // Each node is a card: its name overhead, what the hardware IS in words
  // (a chip and a size say more than a drawing of a laptop), the lines that
  // explain slow tokens on the left, and on the right a bar of where its
  // memory went. The old device pictures filled with one colour and could
  // not say whether the box was hot, swapping or busy; these can.
  //
  // A missing reading is stepped over: the line joins the readings either
  // side of it and never drops to zero (a probe that failed is not an idle
  // GPU). Drawn as gaps it read as a broken line on a busy machine, where
  // one probe in a few runs late.
  function spark(vals, col){
    const W=200, H=30, n=vals.length;
    const pts=vals.map((v,i)=>[i,v]).filter(([,v])=>v!=null);
    if(!pts.length)
      return `<svg class="sp" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none"></svg>`;
    // What history there is spans the width; a new page is not a dash in
    // the corner for three minutes. The scale is 0-100 (or the peak, if a
    // reading goes past it) with a band of headroom above: a peak drawn on
    // the top edge lost half its stroke to the svg's edge.
    const top=Math.max(100, ...pts.map(([,v])=>v)), PAD=4;
    const x=i=>n<2?W:i/(n-1)*W,
          y=v=>H-1-Math.max(0,v)/top*(H-1-PAD);
    const xy=([i,v])=>`${x(i).toFixed(1)} ${y(v).toFixed(1)}`;
    const line='M'+pts.map(xy).join('L');
    const area=`M${x(pts[0][0]).toFixed(1)} ${H}L`+pts.map(xy).join('L')+
      `L${x(pts[pts.length-1][0]).toFixed(1)} ${H}Z`;
    return `<svg class="sp" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">
      <path d="${area}" fill="${col}" opacity=".13"/>
      <path d="${line}" fill="none" stroke="${col}" stroke-width="1.4"
        vector-effect="non-scaling-stroke"/></svg>`;
  }
  const THERMCOL={nominal:'var(--ok)', fair:'var(--warn)',
                  serious:'var(--bad)', critical:'var(--bad)'};
  function lines(mx, swap){
    const h=(mx&&mx.history)||[], now=(mx&&mx.now)||{};
    const row=(lab,val,cls,vals,col)=>`<div class="mrow">
      <div class="mhd"><span>${lab}</span><b class="${cls||''}">${val}</b></div>
      ${vals?spark(vals,col):''}</div>`;
    const pc=v=>v==null?'--':Math.round(v)+'%';
    const th=now.thermal;
    return row('gpu', pc(now.gpu_pct)+(now.gpu_in_use_bytes!=null
          ?` · ${gb(now.gpu_in_use_bytes)}`:''), '',
        h.map(s=>s.gpu_pct), 'var(--acc)')+
      row('cpu', pc(now.cpu_pct), '', h.map(s=>s.cpu_pct), 'var(--warn)')+
      row('memory', pc(now.memory_pct), '',
        h.map(s=>s.memory_pct), '#bb9af7')+
      // °C on a 0-100 scale, ambient near the floor,
      // a hot die near the top.
      row('thermal', (now.temp_c!=null?Math.round(now.temp_c)+'°C · ':'')+(th||'--'),
          'th', h.some(s=>s.temp_c!=null)?h.map(s=>s.temp_c):null, '#fb923c')
        .replace('class="th"', `class="th" style="color:${THERMCOL[th]||'var(--faint)'}"`);
  }
  // TALK (a node beside a conversation): the GPU line and a memory bar
  // laid flat under it, the temperature at the right of the name. HOME
  // keeps the full card; the page switches between them with body.talk.
  // The red swap band: a sensible minimum visible size even when swap is a
  // sliver of installed memory, so it never disappears to nothing.
  function swapFrac(swap, inst){
    return swap==null||!inst ? 0 : Math.max(1.5, Math.min(100, 100*swap/inst));
  }
  function swapTip(swap){ return `${gb0(swap)} swapped`; }
  function hbar(n, mm, off, swap){
    if(off) return '<div class="hstk off"></div>';
    const nm=n.memory||{};
    const swapEl=inst=>swap==null?'':`<div class="swapbar"
      style="width:${swapFrac(swap,inst).toFixed(2)}%" title="${esc(swapTip(swap))}"></div>`;
    if(!mm||!mm.installed_bytes){
      const ws=nm.working_set_bytes||1, f=100*(nm.active_bytes||0)/ws;
      return `<div class="hstk"><i style="width:${f.toFixed(1)}%;--c:var(--dim)"></i>${
        swapEl(ws)}</div>`;
    }
    const inst=mm.installed_bytes;
    const ss=segs(n, mm, inst).filter(s=>s[1]/inst>0.003);
    const wl=isLocal(n) ? nodeWS(n) : nm.working_set_bytes;
    const lim=wl && wl<inst*.99
      ? `<u style="left:${(100*wl/inst).toFixed(1)}%"
          title="limit ${gb(wl)}"></u>` : '';
    return `<div class="hstk">${ss.map(([k,b,c,cls])=>
      `<i class="${cls}" style="width:${(100*b/inst).toFixed(2)}%;--c:${c}"
        title="${esc(k)} ${gb(b)}"></i>`).join('')}${lim}${swapEl(inst)}</div>`;
  }
  function compact(n, mm, off, used, inst, pct, swap){
    const h=(n.metrics&&n.metrics.history)||[], now=(n.metrics&&n.metrics.now)||{};
    const pc=v=>v==null?'--':Math.round(v)+'%';
    return `<div class="ucard ucompact">${off?`<div class="noans">${
        n.problem?esc(n.problem):'no answer'}</div>`:`
      <div class="mrow"><div class="mhd"><span>gpu</span><b>${pc(now.gpu_pct)}${
        now.gpu_in_use_bytes!=null?' · '+gb(now.gpu_in_use_bytes):''}</b></div>
        ${h.length?spark(h.map(s=>s.gpu_pct),'var(--acc)'):''}</div>
      <div class="mrow"><div class="mhd"><span>memory</span><b>${
        Math.round(used/GIB)} / ${Math.round(inst/GIB)} GiB</b></div>${hbar(n, mm, off, swap)}</div>`}</div>`;
  }
  // Swap is the OS's own figure (sysctl vm.swapusage), shown whenever it is
  // above a small floor: macOS keeps swapped pages long after pressure
  // passes, but they are still not in RAM, and a model that was swapped out
  // reads slowly until they come back. Hiding the band once pressure eased
  // dropped it while a Mac still held 5.5 GB there.
  function swapToShow(n){
    const now=((n.metrics||{}).now)||{}, s=now.swap_bytes;
    return s!=null && (s>=SWAP_FLOOR || (now.swapping && s>0)) ? s : null;
  }
  function ttemp(n, off){
    const now=((n.metrics||{}).now)||{};
    if(off || now.temp_c==null) return '';
    return `<span class="utemp" title="thermal: ${esc(now.thermal||'--')}"
      >${Math.round(now.temp_c)}°C</span>`;
  }
  // Bottom up: knurlogic's weights, knurlogic's cache (hatched, because it
  // is reclaimable), each other runtime in its own colour, everything else
  // on the box, then free. Only knurlogic can split weights from cache --
  // it asks its own allocator; every other runtime is one footprint, drawn
  // solid rather than guessed at.
  function segs(n, mm, ws){
    const nm=n.memory||{}, by=Object.entries(mm.by_runtime||{})
      .sort((a,b)=>(a[0]==='knurlogic'?-1:b[0]==='knurlogic'?1:b[1]-a[1]));
    const out=[];
    for(const [k,v] of by){
      if(k==='knurlogic' && nm.scope==='process' && nm.available){
        const w=Math.min(nm.active_bytes||0, v),
              c=Math.min(nm.cache_bytes||0, Math.max(v-w,0));
        out.push(['knurlogic weights', w, RAMCOL.knurlogic, '']);
        out.push(['knurlogic cache', c, RAMCOL.knurlogic, 'hatch']);
        if(v-w-c>0) out.push(['knurlogic', v-w-c, RAMCOL.knurlogic, '']);
      } else out.push([k, v, RAMCOL[k]||'var(--dim)', '']);
    }
    const used=Number.isFinite(mm.used_bytes)?mm.used_bytes:(mm.seen_bytes||0);
    const named=by.reduce((x,[,v])=>x+v,0);
    out.push(['other', Math.max(used-named,0), 'var(--faint)', '']);
    const tot=out.reduce((x,s)=>x+s[1],0);
    // Footprints can overrun the OS's own used figure; scale to fit rather
    // than draw past the top of the machine.
    return tot>ws ? out.map(s=>[s[0],s[1]*ws/tot,s[2],s[3]]) : out;
  }
  function stack(n, mm, off, swap){
    if(off) return '<div class="stk off"></div>';
    const nm=n.memory||{};
    const swapEl=inst=>swap==null?'':`<div class="swapbar"
      style="height:${swapFrac(swap,inst).toFixed(2)}%" title="${esc(swapTip(swap))}"></div>`;
    if(!mm||!mm.installed_bytes){
      // No map from this node: one band of what it reported, nothing split.
      const ws=nm.working_set_bytes||1, f=100*(nm.active_bytes||0)/ws;
      return `<div class="stk"><i style="height:${f.toFixed(1)}%;
        background:var(--dim)" title="in use ${gb(nm.active_bytes||0)}"></i>${
        swapEl(ws)}</div>`;
    }
    const inst=mm.installed_bytes;
    const ss=segs(n, mm, inst).filter(s=>s[1]/inst>0.003);
    // The GPU cannot wire all of installed memory; a model has to fit under
    // this line, so it is drawn where a person would look for it.
    // this machine's limit is its knurlogic allowance (the box reports all
    // of itself); a peer's is its GPU working set
    const wl=isLocal(n) ? nodeWS(n) : nm.working_set_bytes;
    const lim=wl && wl<inst*.99
      ? `<u style="bottom:${(100*wl/inst).toFixed(1)}%"
          title="limit ${gb(wl)}"></u>` : '';
    return `<div class="stk">${ss.map(([k,b,c,cls])=>
      `<i class="${cls}" style="height:${(100*b/inst).toFixed(2)}%;--c:${c}"
        title="${esc(k)} ${gb(b)}"></i>`).join('')}${lim}${swapEl(inst)}</div>`;
  }
  // A quiet line glyph of the kind of Mac, from machine.kind (the server
  // reads it from system_profiler's Model Name, hw.model as a fallback).
  const MACS={
    laptop:'<rect x="5" y="5" width="14" height="10" rx="1"/><path d="M3 18h18"/>',
    // a Mac Studio is a tall square box -- the front is nearly square and
    // carries two USB-C ports, the SD slot and the power light -- where the
    // mini beside it is a flat slab
    studio:'<rect x="3.5" y="4.5" width="17" height="15" rx="3"/>'+
      '<path d="M7 14.5v2M9.5 14.5v2"/>'+
      '<circle cx="17" cy="16" r=".6" fill="currentColor" stroke="none"/>',
    mini:'<rect x="4" y="11" width="16" height="5" rx="1.5"/>',
    imac:'<rect x="3" y="4" width="18" height="12" rx="1"/><path d="M10 16l-1 4h6l-1-4"/>',
    pro:'<rect x="6" y="3" width="12" height="18" rx="1.5"/><path d="M10 7v10M14 7v10"/>',
    desktop:'<rect x="3" y="4" width="18" height="12" rx="1"/><path d="M8 20h8M12 16v4"/>'};
  const macIcon=k=>`<svg class="mk${k==='studio'?' big':''}" viewBox="0 0 24 24" fill="none" stroke-width="1.6"
    stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${
    MACS[k]||MACS.desktop}</svg>`;
  function unit(n){
    const nm=n.memory||{}, hw=n.machine||{};
    const mm=n.memory_map||(n.role==='local'||n.role==='server'?RAM:null);
    const mapped=!!(mm&&mm.installed_bytes);
    const inst=(mapped?mm.installed_bytes:nm.working_set_bytes)||0;
    const used=mapped?(mm.used_bytes||mm.seen_bytes||0):(nm.active_bytes||0);
    const off=n.reachable===false||n.state==='version_mismatch'||(!nm.available&&!mapped);
    const pct=inst?Math.max(0,Math.min(100,100*used/inst)):0;
    const swap=swapToShow(n);
    const what=[hw.chip||hw.model||'', inst?gb(inst).replace(/\.\d+ /,' '):'']
      .filter(Boolean).join(' · ');
    return `<div class="unit${off?' off':''}${NODESEL.has(n.node)?' sel':''}"
      data-node="${esc(n.node)}" title="click to pick this machine for Load model">
      <div class="uhead">${macIcon(hw.kind)}<div class="ut">
        <div class="nm">${esc(n.node)}</div>
        <div class="hw">${esc(what)||'&nbsp;'}</div></div>${ttemp(n, off)}</div>
      ${compact(n, mapped?mm:null, off, used, inst, pct, swap)}
      <div class="ucard ufull">
        <div class="mlines">${off?`<div class="noans">${
            n.problem?esc(n.problem):'no answer'}</div>`
          : (n.metrics?.history||[]).length?lines(n.metrics, swap)
          : '<div class="noans">no metrics from this node</div>'}</div>
        ${stack(n, mapped?mm:null, off, swap)}
      </div>
      <div class="stat"><span class="pct">${off?'--':pct.toFixed(0)+'%'}</span>
        ${off?'':gb(used)+' / '+gb(inst)}</div>
    </div>`;
  }
  // The link between nodes animates along its length. That is the only
  // motion on the page that carries information -- it says the link is
  // live -- so it is also the only one, and it stops under reduced-motion.
  //
  // Two nodes: a connector between the cards, the truth. Three or more:
  // one small ring glyph under the machines (a dot per machine, a line to
  // the next), because the cards wrap into rows and a line chained between
  // them would dangle off every row's end.
  const ns_=d.nodes||[];
  // The key in the rail is a key to THESE machines, so it is the sum of the
  // maps they reported. A key whose numbers are one box's while the picture
  // is four is worse than no key.
  const maps=ns_.map(n=>n.memory_map).filter(Boolean);
  if(maps.length){
    const merged={installed_bytes:0, seen_bytes:0, other_bytes:0,
                  free_bytes:0, used_bytes:0, by_runtime:{}, nodes:maps.length};
    for(const m of maps){
      merged.installed_bytes+=m.installed_bytes||0;
      merged.seen_bytes+=m.seen_bytes||0;
      merged.other_bytes+=m.other_bytes||0;
      merged.free_bytes+=m.free_bytes||0;
      merged.used_bytes+=m.used_bytes||0;
      for(const [k,v] of Object.entries(m.by_runtime||{}))
        merged.by_runtime[k]=(merged.by_runtime[k]||0)+v;
    }
    // knurlogic's reclaimable share, keyed beside its runtime row: the only
    // runtime whose cache can be told apart from its weights.
    merged.knurlogic_cache=ns_.reduce((x,n)=>{const q=n.memory||{};
      return x+(q.scope==='process'&&q.available&&(n.memory_map||{})
        .by_runtime?.knurlogic ? (q.cache_bytes||0) : 0)},0);
    // The swap band's key row, only while some machine draws the band.
    merged.swap_bytes=ns_.reduce((x,n)=>x+(swapToShow(n)||0),0);
    merged.swapped_by_runtime={};
    for(const m of maps) for(const [k,v] of Object.entries(m.swapped_by_runtime||{}))
      merged.swapped_by_runtime[k]=(merged.swapped_by_runtime[k]||0)+v;
    ramWent(merged);
  }
  // Up to sixteen machines (the most a cluster launch joins). Up to six
  // keep the full cards; past six every card takes the compact form the
  // page uses beside an open chat, so sixteen fit. The rest are counted.
  const MAXU=16, DENSE=6, shown=ns_.slice(0,MAXU), more=ns_.length-shown.length;
  const dense=shown.length>DENSE;
  $('memory').querySelector('.topo').className=
    'topo n'+Math.min(Math.max(shown.length,1),3)+(dense?' dense':'');
  const ring=k=>{ const c=30, r=22, pts=Array.from({length:k},(_,i)=>{
      const t=2*Math.PI*i/k-Math.PI/2;
      return [(c+r*Math.cos(t)).toFixed(1),(c+r*Math.sin(t)).toFixed(1)] });
    return '<svg class="topolinks" viewBox="0 0 60 60" aria-hidden="true">'+
      '<polygon points="'+pts.map(q=>q.join(',')).join(' ')+'"/>'+
      pts.map(q=>`<circle cx="${q[0]}" cy="${q[1]}" r="2.4"/>`).join('')+
      '</svg>' };
  $('topo').innerHTML=shown.map(unit).join(shown.length===2
    ? '<svg class="flow" viewBox="0 0 56 20" aria-hidden="true">'+
      '<line x1="2" y1="10" x2="54" y2="10"/></svg>'
    : '')+(shown.length>2?ring(shown.length):'')+(more>0?`<div class="topomore ro">+${more} more machine${
      more===1?'':'s'} -- this page shows ${MAXU}</div>`:'');
  fit();
  setLastNodes(ns_);
  $('multiopts').hidden=selNodes().length<2;
  $('topo').querySelectorAll('.unit[data-node]').forEach(u=>u.onclick=()=>{
    const k=u.dataset.node;
    NODESEL.has(k)?NODESEL.delete(k):NODESEL.add(k); saveNodeSel();
    u.classList.toggle('sel', NODESEL.has(k));
    nodeSelChanged();
  });
  // One line under the machines. The gauges carry the split; this carries
  // the totals, and only the figures that are not already drawn above.
  if(m&&m.available){
    setLastWS((d.nodes||[]).reduce((x,n)=>Math.max(
      x,(n.memory||{}).working_set_bytes||0), 0)||m.working_set_bytes||0);
    const box=m.scope==='box', bit=(k,v)=>`<span>${k} <b>${v}</b></span>`;
    $('mem').innerHTML=
      bit(box?'in use':'weights + live', gb(m.active_bytes))+
      bit('cache', gb(m.cache_bytes))+
      bit('free', gb(m.headroom_bytes))+
      (C&&C.nodes_total>1
        ? bit('nodes', `${C.nodes_reachable}/${C.nodes_total}`)
        : Number.isFinite(m.peak_bytes) ? bit('peak', gb(m.peak_bytes)) : '');
  }
  // Measured by the peers: this machine cannot see the connections its
  // own firewall drops, so their view is the only evidence there is.
  if(d.me&&d.me.problem)
    $('mem').innerHTML+=`<div class="msg">${esc(d.me.problem)}</div>`;
  const w=d.wired;
  if(w&&w.action==='raise'&&w.command)
    $('mem').innerHTML+=`<div class="msg">The GPU may wire only
      <b>${gb(w.limit_bytes)}</b> of ${gb(w.total_bytes)} installed.
      <code style="user-select:all">${esc(w.command)}</code></div>`;
}

export {tick};

// A full card needs about FULL px; the window's width says nothing about
// what the memory panel has (zoom, side panels), so measure the panel: the
// key goes below the machines when beside them the cards would be short of
// FULL, and the cards go compact when even that is not enough.
const FULL=250;
function fit(){
  const t=document.querySelector('#memory .topo'); if(!t) return;
  const n=t.querySelectorAll('.unit').length; if(!n||n>2) {
    t.classList.remove('keybelow','tight'); return }
  const w=t.clientWidth, key=($('ramwent')||{}).offsetWidth||0;
  const flow=n===2?56:0, gap=12*(n===2?3:1);
  const beside=(w-flow-key-gap)/n, below=(w-(n===2?40:0)-12*(n-1))/n;
  t.classList.toggle('keybelow', beside<FULL);
  t.classList.toggle('tight', beside<FULL && below<FULL);
}
try{ new ResizeObserver(()=>fit()).observe(document.getElementById('memory')) }catch(e){}
