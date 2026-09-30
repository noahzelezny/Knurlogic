import {$, esc} from '../../format.js';
import {fitWS} from '../../nodes.js';
import {isGlobal, knobHTML, launchSets} from './knobs.js';
import {stagedVal} from './apply.js';
import {SBASE, SEQ, SRUN, allModels, keepSet, nextSeq, setSBase, shortName} from './index.js';
import {docURL, getJSON} from '../../api.js';
import {BASEKEY, baseKey, baseOf, published} from '../picker.js';

// One base model's settings: its header, where its variants run (or which
// rung the resolver was asked about), the tunes to preview, a row per knob
// a runtime reads, and the rest folded away. `r` is the running variant
// whose live values are shown; none, and every change is a launch setting.
function modelHTML(b, r, doc, runs){
  const hd=`<div class="shd">${esc(b.name)}<span class="ro">${b.ms.length||'no'}
    variant${b.ms.length===1?'':'s'} on this disk</span>`;
  if(doc.error) return `<div class="sgrp">${hd}</div>${runs}
    <div class="msg">could not read its settings: ${esc(doc.error)}</div></div>`;
  const all=doc.knobs||[];
  // knobs these runtimes never read are not shown at all
  const ks=all.filter(k=>k.reach!=='no-effect' && !isGlobal(k.name))
    .sort((a,b)=>(a.tier==='reach'?0:1)-(b.tier==='reach'?0:1));
  const asked=doc.asked||{}, un=doc.unmanaged||[];
  const tunes=r?(doc.tunes||[]).map(t=>`<button data-tune="${esc(t.name)}"
    title="${esc(t.why)}" aria-pressed="${t.name===asked.tune}">${esc(t.name)}</button>`).join(''):'';
  // the confirm groups by g and lists by label
  const c=r ? {g:r.where, where:r.where, model:r.name, base:b.name, label:b.name}
    : {g:'launch|'+b.name, model:(doc.artifact||{}).name||'', base:b.name,
       launch:true, label:b.name};
  // dynamic MTP means something only while MTP is on: its row is hidden
  // while the launch (staged, saved, else running) says off
  const mk=ks.find(k=>k.name==='KNURLOGIC_MTP');
  const mtp=mk ? (stagedVal(c.g,mk.name)??launchSets(b.name)[mk.name]??
    mk.running??mk.would_be) : 'off';
  c.hide=n=>n==='KNURLOGIC_MTP_DYNAMIC' && String(mtp)==='off';
  return `<div class="sgrp"${r?` data-where="${esc(r.where)}"`:''}>
    ${hd}${tunes?`<span class="seg">${tunes}</span>`:''}</div>${runs}
    ${ks.filter(k=>!isVQ(k)).map(k=>knobHTML(k,c)).join('')}
    ${ks.some(isVQ)?`<div class="ksub">VQ</div>${
      ks.filter(isVQ).map(k=>knobHTML(k,c)).join('')}`:''}
    ${ks.length?'':'<div class="msg">no knobs</div>'}
    ${un.length?`<details class="more"><summary>${un.length} other variables this
      runtime reads; knurlogic sets none of them</summary><div class="unl">${
      esc(un.map(u=>u.name).join('  '))}</div></details>`:''}
  </div>`;
}
// the VQ runtime's own knobs, under their own header after the general ones
// (VQ_ only, but not VQ_CACHE_LIMIT_GB: that is the engine's buffer cache)
const isVQ=k=>/^VQ_/.test(k.name) && k.name!=='VQ_CACHE_LIMIT_GB';
const TUNES={};
// Every base model: the picker's groups over what is on this disk, plus any
// running on a peer that this disk does not have; the running ones first.
function settingBases(){
  const by={}, add=k=>by[k]=by[k]||{name:k, ms:[], runs:[]};
  (window.ALLMODELS||[]).filter(published).forEach(m=>
    add(BASEKEY[m.name]||baseOf(m.name)).ms.push(m));
  allModels().forEach(r=>add(baseKey(r.name)).runs.push(r));
  const big=b=>b.ms.length?Math.max(...b.ms.map(m=>m.size_bytes||0)):0;
  return Object.values(by).map(b=>{ b.ms.sort((x,y)=>y.size_bytes-x.size_bytes); return b })
    .sort((x,y)=>(y.runs.length?1:0)-(x.runs.length?1:0) || big(y)-big(x));
}
const runsOn=b=>[...new Set(b.runs.map(r=>r.mach.name))].join(', ');
function showModels(){
  const bs=settingBases(), L=$('setlist');
  if(!bs.some(b=>b.name===SBASE)) setSBase(bs.length?bs[0].name:'');
  const row=b=>`<div class="fam" data-b="${esc(b.name)}" aria-current="${b.name===SBASE}"
      title="${esc(b.ms.map(m=>m.name).join('\n'))}">${esc(b.name)}
      <small${b.runs.length?' class="on"':''}>${b.runs.length?'on '+esc(runsOn(b)):
        b.ms.length+' variant'+(b.ms.length===1?'':'s')}</small></div>`;
  const on=bs.filter(b=>b.runs.length), off=bs.filter(b=>!b.runs.length);
  L.innerHTML=`${on.length?`<div class="sect">running</div>${on.map(row).join('')}`:''}${
    off.length?`<div class="sect">on this disk</div>${off.map(row).join('')}`:''}${
    bs.length?'':'<div class="msg" style="padding:0 12px">no models</div>'}`;
  L.querySelectorAll('[data-b]').forEach(v=>v.onclick=()=>{
    if(v.dataset.b===SBASE) return;
    setSBase(v.dataset.b); keepSet();
    L.querySelectorAll('[data-b]').forEach(x=>x.setAttribute('aria-current', x===v));
    showBase(bs.find(b=>b.name===SBASE));
  });
  keepSet(); showBase(bs.find(b=>b.name===SBASE));
}
// A base model's settings. Running: the running variant's own settings
// (live ones apply to it; the rest wait for a launch), and a line for each
// place a variant runs, with what it runs at. Not running: what the
// resolver would give the rung that fits (the preview Load model asks for),
// every change saved for its next launch. Reads only; Apply on close sends.
async function showBase(b){
  const el=$('machbody'), seq=nextSeq();
  if(!b){ el.innerHTML='<div class="msg">No models on this disk, and nothing running.</div>'; return }
  el.innerHTML='<div class="msg">reading…</div>';
  // what the resolver would give the rung that fits, as launch settings
  const preview=async lead=>{
    const ws=fitWS(), m=b.ms.find(x=>!ws||x.size_bytes<=ws)||b.ms[0];
    // resolved at the base model's saved launch preset
    const pre=launchSets(b.name).KNURLOGIC_PRESET||window.LOADTUNE||'balanced';
    const doc=await getJSON('/settings.json?'+new URLSearchParams({artifact:m.path, tune:pre}));
    if(seq!==SEQ) return;
    el.innerHTML=modelHTML(b, null, doc, `${lead}<div class="msg" style="margin-top:0">${
      b.runs.length?'Its server does not answer, so nothing can change live.':'Not running.'}
      The resolver's values for ${esc(m.name)} (${esc(pre)} preset); a change is kept
      for whichever variant launches next.</div>`);
  };
  if(!b.runs.length) return preview('');
  const r=b.runs.find(x=>x.where===SRUN[b.name])||b.runs[0];
  const docs=await Promise.all(b.runs.map(x=>x===r
    ? getJSON(docURL(x.where, TUNES[x.where]?{tune:TUNES[x.where]}:{}))
    : getJSON(docURL(x.where))));
  if(seq!==SEQ) return;
  const say=(x,d)=>`${esc(x.mach.name)} · ${esc(shortName(x.name))} · ${
    d.error?'not answering':esc((d.live||{}).tune||'?')+' tune'} · ${
    esc((x.where||location.host).replace(/^https?:\/\//,''))}`;
  // more than one: each line picks whose live values the rows show
  const runs=`<div class="runs">${b.runs.map((x,i)=>b.runs.length>1
    ? `<button class="mini" data-run="${esc(x.where)}" aria-pressed="${x===r}">${say(x,docs[i])}</button>`
    : `<span class="ro">running on ${say(x,docs[i])}</span>`).join('')}</div>`;
  const doc=docs[b.runs.indexOf(r)];
  // a server that does not answer still leaves the base model's launch
  // settings to be made, when a variant is on this disk to ask about
  if(doc.error && b.ms.length) return preview(runs);
  el.innerHTML=modelHTML(b, r, doc, runs);
  el.querySelectorAll('[data-tune]').forEach(x=>x.onclick=()=>{
    TUNES[r.where]=x.dataset.tune; showBase(b) });
  el.querySelectorAll('[data-run]').forEach(x=>x.onclick=()=>{
    SRUN[b.name]=x.dataset.run; showBase(b) });
}

export {showModels};
