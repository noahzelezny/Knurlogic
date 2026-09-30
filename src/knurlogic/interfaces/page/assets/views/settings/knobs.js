import {esc} from '../../format.js';
import {stagedVal} from './apply.js';
import {BASEKEY} from '../picker.js';

// --- settings -------------------------------------------------------------
// Three things kept apart: what is RUNNING, what a tune WOULD give, and how
// far a change can reach. Reach is per knob and was measured by reading a
// bundled runtime: some apply live, the rest are read when the model loads,
// and a few are not read by these runtimes at all (those are not listed).
function infoHTML(text, cls){
  return text ? `<i class="info ${cls||''}" tabindex="0">i<span class="bub">${text}</span></i>` : '';
}
// What the two reach words mean, for their hover.
const REACH={live:'applies to the running model without reloading it -- on every machine a split model runs on',
  restart:'read when the model loads: to take effect it needs the running model unloaded and launched again from this page'};
// "4.0" running is the step 4: compared as numbers where both are numbers
const sameVal=(a,b)=>a!=null&&b!=null&&(String(a)===String(b)||
  (String(a).trim()!==''&&String(b).trim()!==''&&+a===+b));

// --- launch settings, per base model ----------------------------------------
// A knob read only at load cannot change a running model; what it CAN be is
// what that model launches with next time. Kept per BASE model in this
// browser -- the picker's group, every quantisation of one model -- so a
// setting chosen for one rung holds for whichever rung launches next; handed
// to Launch as its `sets` (--set KEY=VALUE), which is also where they show,
// pre-filled, under advanced options.
const LSKEY='kn.launchsets';
function launchAll(){
  try{ return JSON.parse(localStorage.getItem(LSKEY)||'{}')||{} }catch(e){ return {} }
}
function launchSets(base){ return {...(launchAll()[base]||{})} }
function saveLaunchSets(base, sets){
  const all=launchAll();
  if(Object.keys(sets).length) all[base]=sets; else delete all[base];
  try{ localStorage.setItem(LSKEY, JSON.stringify(all)) }catch(e){}
}
// Entries saved per artifact name (before settings were the base model's)
// move to their base model's key; where two rungs disagree the one already
// under the base key wins, and then the first moved.
function migrateLaunchSets(){
  const all=launchAll(); let moved=0;
  Object.keys(all).forEach(k=>{ const b=BASEKEY[k];
    if(b && b!==k){ all[b]={...all[k], ...(all[b]||{})}; delete all[k]; moved++ } });
  // the prompt chunk's legacy name, saved before it was shown as
  // KNURLOGIC_PREFILL_CHUNK (the server takes either; this one shows)
  Object.values(all).forEach(v=>{ if(v && 'VQLAB_PREFILL_CHUNK' in v){
    if(!('KNURLOGIC_PREFILL_CHUNK' in v)) v.KNURLOGIC_PREFILL_CHUNK=v.VQLAB_PREFILL_CHUNK;
    delete v.VQLAB_PREFILL_CHUNK; moved++ } });
  // knurlogic-wide now (Settings -> Knurlogic): a model's saved copy would
  // beat the one choice at its launch, so it is dropped
  Object.values(all).forEach(v=>{ if(v) Object.keys(v).forEach(k=>{
    if(isGlobal(k)){ delete v[k]; moved++ } }) });
  // long context is the context length's now (past the native window is
  // YaRN): a saved switch would be one no row shows
  Object.values(all).forEach(v=>{ if(v && 'KNURLOGIC_LONG_CONTEXT' in v){
    delete v.KNURLOGIC_LONG_CONTEXT; moved++ } });
  if(moved) try{ localStorage.setItem(LSKEY, JSON.stringify(all)) }catch(e){}
}
// set once for every model (machine/preferences), never per model
const isGlobal=n=>n==='KNURLOGIC_CROSS_CHIP'||/^KNURLOGIC_COMPACT_/.test(n);

// One knob, one line: NAME (i) live|needs reload, and its control on the
// right. `c` says whose it is: g (the staging group), where (the model
// server, '' this page's own), model (its
// name), base (its base model, whose launch settings these are), launch
// (nothing of it runs: every change is a launch setting), label (how the confirm lists it) and url
// (a setting of this page's own, POSTed there as {gib}: the allowance).
// a launch setting's plain name, where its variable is not one
const KNOB_TITLE={KNURLOGIC_CROSS_CHIP:'Per-chip rounding', KNURLOGIC_PRESET:'Preset',
  KNURLOGIC_CONTEXT_LENGTH:'Context length', KNURLOGIC_KV_BITS:'KV cache',
  KNURLOGIC_PREFILL_CHUNK:'Prompt chunk', KNURLOGIC_CACHE_LIMIT_GB:'Cache reserve',
  KNURLOGIC_MTP:'MTP', KNURLOGIC_MTP_DYNAMIC:'MTP dynamic', KNURLOGIC_KV_KERNEL:'KV kernel'};
// the preset may be left unset: the Knurlogic tab's preset then applies
const UNSET='(unset)';
function knobHTML(k, c){
  // a model not running follows the Knurlogic presets until a value is set
  // here: every knob of it offers "default", the preset's value
  const unsettable=k.name==='KNURLOGIC_PRESET' || !!c.launch;
  const cur=unsettable ? UNSET : k.running??k.would_be??k.value;
  // with no variant running, every knob waits for a launch, live or not
  const reach=c.launch ? 'restart' : k.reach;
  const next=reach==='restart' ? launchSets(c.base)[k.name] : undefined;
  // a live knob's saved launch setting still goes to the next launch
  const saved=reach==='live'&&c.base ? launchSets(c.base)[k.name] : undefined;
  const from=next??cur, st=stagedVal(c.g,k.name), sel=st??from;
  // a tag only where it tells a running model's owner something: live
  // (applies now) or next launch (needs an unload and a relaunch). With
  // nothing of it running there is no tag -- every change simply applies
  // when it launches, so a tag on every row was noise.
  const tag=c.launch ? ''
    : reach==='live' ? `<span class="kw live" title="${esc(REACH.live)}">live</span>`
    : `<span class="kw rst" title="${esc(REACH.restart)}">next launch</span>`;
  const about=(k.help ? esc(k.help).replace(/\n/g,'<br>')
    : (KNOB_TITLE[k.name]?[k.what]:[k.what,k.why]).filter(Boolean).map(esc).join(' — '))+
    (k.max_why?' '+esc(k.max_why):'');
  // the measured steps, stopping where the headroom does; else a field
  const vals=k.values||[], ci=vals.findIndex(x=>sameVal(x,k.cap));
  const cap=k.cap==null||ci<0?vals.length-1:ci;
  let ctl;
  if(unsettable){
    const dflt='default';
    // the preset's default is the unset row itself; the rest by title
    const all=(vals.length?vals:[...new Set([k.would_be??k.value].filter(v=>v!=null&&v!==''))])
      .filter(x=>!(k.name==='KNURLOGIC_PRESET'&&x==='balanced'));
    const title=x=>x===UNSET?dflt:x;
    ctl=`<select aria-label="${esc(k.name)}">${[UNSET,...all].map(x=>
      `<option value="${esc(x)}"${sameVal(x,sel)?' selected':''}>${esc(title(x))}</option>`).join('')}</select>`;
  }else if(vals.length){
    const opts=vals.filter((x,i)=>i<=cap||sameVal(x,cur)||sameVal(x,sel));
    [sel,cur].forEach(v=>{ if(v!=null && !opts.some(x=>sameVal(x,v))) opts.unshift(v) });
    ctl=`<select aria-label="${esc(k.name)}">${opts.map(x=>
      `<option${sameVal(x,sel)?' selected':''}>${esc(x)}</option>`).join('')}</select>`;
  }else{
    const num=String(cur??'').trim()!=='' && !isNaN(+cur);
    ctl=`<input type="${num?'number':'text'}" aria-label="${esc(k.name)}" value="${esc(sel)}"${
      num&&!c.url?' min="1" step="1"':''}${k.max?` max="${esc(k.max)}"`:''}>`;
  }
  const tuneNote=k.changed&&st==null ? `the tune gives ${esc(k.would_be)}` : '';
  return `<div class="knob${st!=null?' staged':''}" data-name="${esc(k.name)}"${
    c.hide&&c.hide(k.name)?' hidden':''}>
    <div class="hd"><span class="k">${esc(KNOB_TITLE[k.name]||k.name)}${infoHTML(about,'down')}${tag}</span>
      <span class="kedit" data-knob="${esc(k.name)}" data-g="${esc(c.g)}"
        ${c.where!=null?`data-where="${esc(c.where)}"`:''}${c.url?` data-url="${esc(c.url)}"`:''}${c.ukey?` data-ukey="${esc(c.ukey)}"`:''} data-model="${esc(c.model)}"
        data-base="${esc(c.base||'')}" data-label="${esc(c.label)}" data-reach="${reach}" data-cur="${esc(cur)}"
        data-from="${esc(from)}"${c.launch?' data-launch="1"':''}${next!=null?` data-next="${esc(next)}"`:''}${
        k.max?` data-max="${esc(k.max)}"`:''}${saved!=null?` data-saved="${esc(saved)}"`:''}>${ctl}${
        k.unit?`<span class="ro">${esc(k.unit)}</span>`:''}</span></div>
    <div class="kstage">${stageNote(reach,cur,st,next,saved,c.launch)||tuneNote}</div></div>`;
}
// The line under a row: what is staged, else a launch setting already saved.
// Only for a model that runs: with none, what the row shows is what launches.
function stageNote(reach, cur, to, next, saved, launch){
  if(launch) return '';
  if(to!=null) return reach==='live'
    ? `changed · ${esc(cur)} → ${esc(to)}, applied on close`
    : sameVal(to,cur) ? `back to ${esc(cur)} when it launches, saved on close`
    : `${esc(to)} when it launches, saved on close`;
  if(next!=null && !sameVal(next,cur)) return `${esc(next)} when it launches`;
  if(saved!=null && !sameVal(saved,cur)) return `saved launch setting ${esc(saved)} · used on its next launch`;
  return '';
}
// A value this knob cannot take, said before it is staged: a number field
// that is not a positive whole number, or past the model's maximum (the
// server refuses both; this only says so sooner).
function knobRefusal(d, el){
  const v=el.value, s=String(v).trim(); if(s==='') return '';
  if(el.type==='number' && (d.url ? !(+s>=0) : !/^\d+$/.test(s) || +s<1))
    return `${esc(s)} is not ${d.url?'a number of GiB':'a positive whole number'}`;
  if(d.max && +v>+d.max) return `${esc(v)} is past this model's maximum of ${(+d.max).toLocaleString()} tokens`;
  return '';
}
// (i) opens on a click too (it takes focus), without the click reaching a
// label's field.
document.addEventListener('click', e=>{
  const i=e.target.closest('.info');
  if(i && !e.target.closest('.bub')){ e.preventDefault(); i.focus() }
});

export {isGlobal, knobHTML, knobRefusal, launchAll, launchSets,
        migrateLaunchSets, sameVal, saveLaunchSets, stageNote};
