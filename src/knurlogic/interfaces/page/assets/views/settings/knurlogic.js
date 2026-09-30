import {$, esc} from '../../format.js';
import {STAGED, stage, stagedVal} from './apply.js';
import {SEQ, machines, nextSeq} from './index.js';
import {getJSON, peekURL} from '../../api.js';
import {SKEEP} from './keep.js';

// --- knurlogic: presets and compaction ---------------------------------------
// Knurlogic settings, saved to every machine: the preset every model
// launches with (and per-chip rounding beside it), and how a server
// compacts a long conversation. Memory (the allowance, the wired limit) is
// each machine's, under Cluster. A base model's own Preset (Models) still
// beats this one.
let SKN=SKEEP.kn||'strategy';
function keepKn(){ try{ const k=JSON.parse(sessionStorage.getItem('kl.settings')||'{}');
  k.kn=SKN; sessionStorage.setItem('kl.settings', JSON.stringify(k)) }catch(e){} }
function showKnurlogic(){
  const L=$('setlist'), ms=machines();
  if(!['strategy','compaction'].includes(SKN)) SKN='strategy';
  const it=(id,name,sub)=>`<div class="fam" data-k="${id}" aria-current="${id===SKN}">${esc(name)}<small>${esc(sub)}</small></div>`;
  L.innerHTML=it('strategy','Presets','')+
    it('compaction','Compaction','');
  L.querySelectorAll('[data-k]').forEach(v=>v.onclick=()=>{
    if(v.dataset.k===SKN) return;
    SKN=v.dataset.k; keepKn();
    L.querySelectorAll('[data-k]').forEach(x=>x.setAttribute('aria-current', x===v));
    showKnOf(ms);
  });
  keepKn(); showKnOf(ms);
}
function showKnOf(ms){ return SKN==='compaction' ? showCompaction(ms) : showStrategy(ms) }
// Each machine's own documents: /strategy.json and /knurlogic.json here, a
// peer's through its /settings.json (/peek reads nothing else).
async function knDocs(ms){
  return Promise.all(ms.map(async m=>{
    if(m.id==='local'){
      const [st,kn]=await Promise.all([getJSON('/strategy.json'), getJSON('/knurlogic.json')]);
      return {strategy:st, knurlogic:kn};
    }
    const d=await getJSON(peekURL(m.page,'/settings.json'));
    return d.error ? {strategy:d, knurlogic:d}
      : {strategy:d.strategy||{error:'its knurlogic predates the strategy'},
         knurlogic:d.knurlogic||{error:'its knurlogic predates knurlogic-wide settings'}};
  }));
}
// Sent to this machine and every answering peer, each through its own page
// (/machine.json -> that machine's /peer/machine.json); one answer per machine.
async function applyEvery(ms, body){
  return Promise.all(ms.map(async m=>{
    try{ const r=await fetch('/machine.json?'+new URLSearchParams({where:m.id==='local'?'':m.page}),
        {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
      let j; try{ j=await r.json() }catch(e){ j={} }
      if(!r.ok && !j.error) j.error='HTTP '+r.status;
      return {name:m.name, error:j.error||''} }
    catch(e){ return {name:m.name, error:String(e.message||e)} } }));
}
// The knurlogic-wide choices are staged like every other Settings change
// (group 'knurlogic', every:true) and go out with the close prompt's Apply:
// to this machine and every answering peer. A machine that refused keeps
// the change staged, to try again or discard.
const KNG='knurlogic';
function stageKn(name, from, to, extra){
  stage(KNG, {label:'Knurlogic', where:'every machine', every:true}, name,
        {from:String(from??''), to:String(to??''), reach:'live', cur:String(from??''), ...(extra||{})});
}
async function applyEveryGroup(key, g){
  const body={}, settings={};
  Object.entries(g.knobs).forEach(([k,v])=>{
    if(k==='strategy') body.strategy=v.to;
    else settings[k]=v.to==='(unset)'||(v.dflt!=null&&String(v.to)===String(v.dflt))?'':v.to });
  if(Object.keys(settings).length) body.settings=settings;
  const res=await applyEvery(machines(), body);
  const bad=res.filter(r=>r.error);
  if(!bad.length){ STAGED.delete(key); if(body.strategy) window.LOADTUNE=body.strategy }
  return `<div class="sg"><span class="ro">Knurlogic · every machine</span>${
    Object.entries(g.knobs).map(([k,v])=>`<div><b>${esc(k)}</b> ${esc(v.to)}</div>`).join('')}${
    res.map(r=>`<div>${esc(r.name)}: ${r.error?`<span style="color:var(--warn)">${esc(r.error)}</span>`:'saved'}</div>`).join('')}</div>`;
}
// One knurlogic-wide choice as one row: its name, an (i) with the
// explanation and trade-off, a native select.
function gsel(name, title, what, why, values, cur, dflt){
  const st=stagedVal(KNG,name), v0=st==='(unset)'?'':st??cur;
  return `<label class="gk${st!=null?' staged':''}"><b>${esc(title)}<i class="info" tabindex="0">i<span
    class="bub">${esc(what)}${why?' '+esc(why):''}</span></i></b>
    <select name="${esc(name)}" data-from="${esc(cur??'')}" data-dflt="${esc(dflt??'')}"
      aria-label="${esc(title)}">${values.map(v=>
      `<option value="${esc(v.v)}"${String(v.v)===String(v0)?' selected':''}>${esc(v.t)}${
        dflt!=null&&String(v.v)===String(dflt)?' (default)':''}</option>`).join('')}</select></label>`;
}
function wireGks(el){
  el.querySelectorAll('.gks:not(.prow) select').forEach(x=>x.onchange=()=>{
    const d=x.dataset;
    // an unset knob reads '' -- choosing a value equal to it clears the stage
    if(x.value===d.from || (d.from==='' && d.dflt!=='' && x.value===d.dflt))
      stage(KNG,{},x.name,{from:'',to:''});
    // '' (the preset's own) is a value here; a stage never holds ''
    else stageKn(x.name, d.from, x.value===''?'(unset)':x.value, d.dflt!==''?{dflt:d.dflt}:{});
    x.closest('.gk').classList.toggle('staged', stagedVal(KNG,x.name)!=null);
  });
}
// The presets, every setting a preset sets as its own row, and per-chip
// rounding last (it is no part of a preset): on unless KNURLOGIC_CROSS_CHIP
// is on (the same rounding on every chip); a saved auto or off reads as on.
// A preset button sets the rows; a row changed by hand leaves no preset
// pressed, unless the rows come to equal one. Saved: the preset when the
// rows equal it, otherwise balanced plus each row that differs from it.
async function showStrategy(ms){
  const el=$('machbody'), seq=nextSeq();
  el.innerHTML='<div class="msg">reading…</div>';
  const docs=await knDocs(ms);
  if(seq!==SEQ) return;
  const me=docs[0].strategy||{}, ps=me.presets||[], rows=me.rows||[], cur=me.preset||'balanced';
  if(!ps.length||!rows.length){ el.innerHTML=`<div class="msg">${esc(me.error||'this page predates the strategy')}</div>`; return }
  const kn=docs[0].knurlogic||{}, cc=kn.cross_chip, saved=kn.saved||{};
  const names=rows.map(r=>r.name), pv=n=>ps.find(p=>p.name===n).values;
  const ov=n=>{ const t=stagedVal(KNG,n); return t==null?saved[n]||'':t==='(unset)'?'':t };
  const V0={...pv(stagedVal(KNG,'strategy')??cur)};
  names.forEach(n=>{ if(ov(n)!=='') V0[n]=ov(n) });
  const V={...V0};
  const same=(a,b)=>names.every(n=>a[n]===b[n]);
  const match=()=>ps.find(p=>same(pv(p.name),V));
  el.innerHTML=`<div class="sgrp"><div class="shd">Presets</div>
    <div class="seg strat" role="group" aria-label="preset">${ps.map(p=>`<button type="button"
      data-p="${esc(p.name)}">${esc(p.title)}${
      p.name===me.default?'<small>default</small>':''}</button>`).join('')}</div>
    <div class="gks prow">${rows.map(r=>`<label class="gk"><b>${esc(r.title)}<i class="info" tabindex="0">i<span
      class="bub">${esc(r.help)}</span></i></b><select name="${esc(r.name)}"
      aria-label="${esc(r.title)}">${r.options.map(o=>
      `<option value="${esc(o.v)}">${esc(o.t)}</option>`).join('')}</select></label>`).join('')}</div>
    ${cc?`<div class="gks">${gsel(cc.name,'Per-chip rounding',cc.help||cc.what,'',
      [{v:'',t:'on'},{v:'on',t:'off'}], cc.value==='on'?'on':'',null)}</div>`:''}
  </div>`;
  const show=()=>{
    const m=match();
    el.querySelectorAll('.strat button').forEach(b=>b.setAttribute('aria-pressed', !!m&&b.dataset.p===m.name));
    el.querySelectorAll('.prow select').forEach(x=>{
      x.value=V[x.name]; x.closest('.gk').classList.toggle('staged', V[x.name]!==V0[x.name]) });
  };
  const put=()=>{
    const m=match(), base=pv(me.default), to=m?{strategy:m.name, ov:{}}
      : {strategy:me.default, ov:Object.fromEntries(names.map(n=>[n, V[n]!==base[n]?V[n]:'']))};
    const one=(n,from,t)=>from===t?stage(KNG,{},n,{from:'',to:''}):stageKn(n, from, t||'(unset)');
    one('strategy', cur, to.strategy);
    names.forEach(n=>one(n, saved[n]||'', to.ov[n]||''));
    show();
  };
  el.querySelectorAll('.strat button').forEach(b=>b.onclick=()=>{
    Object.assign(V, pv(b.dataset.p)); put() });
  el.querySelectorAll('.prow select').forEach(x=>x.onchange=()=>{ V[x.name]=x.value; put() });
  wireGks(el); show();
}
// Compaction: one set for every model on every machine. Every model server
// reads it per request, so a change applies to the next request of every
// running model, and to every model launched later.
async function showCompaction(ms){
  const el=$('machbody'), seq=nextSeq();
  el.innerHTML='<div class="msg">reading…</div>';
  const docs=await knDocs(ms);
  if(seq!==SEQ) return;
  const kn=docs[0].knurlogic||{}, cp=kn.compaction;
  if(!cp){ el.innerHTML=`<div class="msg">${esc(kn.error||'this page predates knurlogic-wide compaction')}</div>`; return }
  const ks=cp.knobs||[];
  const TITLE={KNURLOGIC_COMPACT_AUTO:'Compact unasked', KNURLOGIC_COMPACT_TRIGGER:'Auto compact',
    KNURLOGIC_COMPACT_KEEP_TURNS:'Keep recent', KNURLOGIC_COMPACT_TOOL_RESULTS:'Dropped tool results'};
  el.innerHTML=`<div class="sgrp"><div class="shd">Compaction</div>
    <div class="gks">${ks.map(k=>gsel(k.name, TITLE[k.name]||k.name, k.help||k.what, k.help?'':k.why,
      k.values.map(v=>({v, t:v+(k.unit?' '+k.unit:'')})), k.value, k.default)).join('')}</div>
  </div>`;
  wireGks(el);
}

export {applyEveryGroup, showKnurlogic};
