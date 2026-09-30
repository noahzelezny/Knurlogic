import {OVL} from '../../ui/overlay.js';
import {esc} from '../../format.js';
import {knobRefusal, launchSets, sameVal, saveLaunchSets, stageNote} from './knobs.js';
import {ASK, loadSettings} from './cluster.js';
import {SSEC, showSetTab} from './index.js';
import {applyEveryGroup} from './knurlogic.js';
import {SEL, baseKey, loadModels, setSets} from '../picker.js';

// --- staged knob changes ----------------------------------------------------
// A change in Settings is STAGED, never sent as it is made: a select that
// fires on every arrow key would otherwise change a running model by
// accident. Closing the overlay with something staged asks once -- Apply,
// Discard, Keep editing -- and Apply is the only thing that sends. Live
// knobs go to their server; needs-reload ones become that model's launch
// settings and nothing is restarted.
const STAGED=new Map();       // g -> {label, where, model, knobs:{name:{from,to,reach,cur}}}
function stagedVal(g,name){ const x=STAGED.get(g); return x&&x.knobs[name]?x.knobs[name].to:undefined }
function stagedCount(){ let n=0; STAGED.forEach(g=>n+=Object.keys(g.knobs).length); return n }
function stage(key, grp, name, knob){
  let g=STAGED.get(key);
  if(sameVal(knob.from,knob.to)||String(knob.to).trim()===''){
    if(g){ delete g.knobs[name]; if(!Object.keys(g.knobs).length) STAGED.delete(key) }
    return;
  }
  if(!g) STAGED.set(key, g={...grp, knobs:{}});
  g.knobs[name]=knob;
}
document.addEventListener('change', e=>{
  const box=e.target.closest('.kedit'); if(!box) return;
  const d=box.dataset;
  const no=knobRefusal(d, e.target);
  if(no){ box.closest('.knob').querySelector('.kstage').innerHTML=
      `<span style="color:var(--warn)">${no} -- not staged</span>`;
    e.target.value=stagedVal(d.g,d.knob)??d.from; return }
  stage(d.g, {label:d.label, where:d.where, model:d.model, base:d.base, url:d.url, ukey:d.ukey}, d.knob,
        {from:d.from, to:e.target.value.trim(), reach:d.reach, cur:d.cur});
  const to=stagedVal(d.g,d.knob), row=box.closest('.knob');
  row.classList.toggle('staged', to!=null);
  if(d.knob==='KNURLOGIC_MTP'){
    const dyn=row.parentElement.querySelector('.knob[data-name="KNURLOGIC_MTP_DYNAMIC"]');
    if(dyn) dyn.hidden=e.target.value.trim()==='off';
  }
  row.querySelector('.kstage').innerHTML=stageNote(d.reach,d.cur,to,d.next,d.saved,!!d.launch);
});
// The pop-up: what applies NOW (live, one POST per server -- /apply to a
// model server, or this page's own /settings.json) apart from what waits for
// the model's next launch, then each server's own report where the list was.
const STAGEPOP=(()=>{
  const pop=document.createElement('div');
  pop.id='stagepop'; pop.hidden=true;
  pop.setAttribute('role','dialog'); pop.setAttribute('aria-label','staged changes');
  document.querySelector('#sheet .box').appendChild(pop);
  // a launch setting's staged group: nothing of that model runs, so its g is
  // 'launch|' + the base model (its rows have no running server to reload) --
  // distinct from a running model's g (its server address, '' this page's own)
  const isLaunch=key=>String(key).startsWith('launch|');
  const part=(reach, check=()=>true)=>[...STAGED].map(([key,g])=>{
    if(!check(key,g)) return '';
    const ks=Object.entries(g.knobs).filter(([,v])=>v.reach===reach);
    return ks.length ? `<div class="sg"><span class="ro">${esc(g.label)}${
      g.where?' · '+esc(g.where):''}</span>${ks.map(([k,v])=>
      `<div><b>${esc(k)}</b> <span class="was">${esc(v.from)}</span><span class="chg">${
      esc(v.to)}</span></div>`).join('')}</div>` : '' }).join('');
  function show(){
    const n=stagedCount(), now=part('live'),
      later=part('restart', key=>!isLaunch(key)),
      next=part('restart', isLaunch);
    pop.innerHTML=`<b>${n} change${n===1?'':'s'}</b>
      ${now?`<div class="sh">apply now</div>${now}`:''}
      ${later?`<div class="sh">needs a reload</div>${later}
        <div class="msg">Saved as that base model's launch settings, for
        whichever of its variants is launched next from this page; what is
        running keeps running as it is -- this does not restart it.</div>`:''}
      ${next?`<div class="sh">applies on next launch</div>${next}
        <div class="msg">Nothing of this model is running, so the change
        cannot be applied to it; it is saved as the model's launch setting
        and takes effect when it is launched next from this page.</div>`:''}
      <div class="ctl"><button class="mini" data-s="keep">Keep editing</button>
      <button class="mini" data-s="discard">Discard</button>
      <button class="go" data-s="apply">${now?'Apply':'Save'}</button></div>`;
    pop.hidden=false; pop.querySelector('[data-s=apply]').focus();
  }
  async function apply(){
    pop.querySelectorAll('button').forEach(b=>b.disabled=true);
    pop.querySelector('b').textContent='applying…';
    const out=await Promise.all([...STAGED].map(async ([key,g])=>{
      if(g.every) return applyEveryGroup(key,g);
      const live=Object.entries(g.knobs).filter(([,v])=>v.reach==='live');
      const later=Object.entries(g.knobs).filter(([,v])=>v.reach!=='live');
      const lines=[];
      if(later.length){
        const sets=launchSets(g.base);
        later.forEach(([k,v])=>{ if(sameVal(v.to,v.cur)||v.to==='(unset)') delete sets[k]; else sets[k]=v.to;
          delete g.knobs[k] });
        saveLaunchSets(g.base, sets);
        lines.push(...later.map(([k,v])=>`<div><b>${esc(k)}</b> ${esc(v.to)} on next launch</div>`));
      }
      if(live.length){
        const url=g.url || (g.where ? '/apply?'+new URLSearchParams({where:g.where}) : '/settings.json');
        const body=g.url ? {[g.ukey||'gib']:+live[0][1].to} : Object.fromEntries(live.map(([k,v])=>[k,v.to]));
        let r;
        try{
          const x=await fetch(url,{method:'POST', headers:{'Content-Type':'application/json'},
            body:JSON.stringify(body)});
          r=await x.json();
          if(!x.ok && !r.error) r={error:'HTTP '+x.status};
        }catch(err){ r={error:String(err.message||err)} }
        // what did not go stays staged, to try again or discard next close
        if(!r.error) live.forEach(([k])=>delete g.knobs[k]);
        const done=Object.entries(r.applied||{});
        lines.unshift(r.error ? `<span style="color:var(--warn)">${esc(r.error)}</span>`
          : done.length ? done.map(([k,v])=>`<div><b>${esc(k)}</b> ${esc(v)}</div>`).join('')
          : esc(r.note||'nothing changed'));
      }
      if(!Object.keys(g.knobs).length) STAGED.delete(key);
      return `<div class="sg"><span class="ro">${esc(g.label)}${g.where?' · '+esc(g.where):''}</span>${
        lines.join('')}</div>`;
    }));
    pop.innerHTML=`<b>Done</b>${out.join('')}
      <div class="ctl"><button class="go" data-s="done">Close</button></div>`;
    pop.querySelector('[data-s=done]').focus();
    if(SEL) setSets(launchSets(baseKey(SEL.name)));
    // the allowance moves what fits and the room each model leaves
    if(out.length){ loadSettings(ASK); loadModels();
      if(SSEC==='cluster' || SSEC==='knurlogic') showSetTab() }
  }
  pop.addEventListener('click', e=>{
    const s=e.target.closest('[data-s]'); if(!s) return;
    if(s.dataset.s==='keep') hide();
    else if(s.dataset.s==='discard'){ STAGED.clear(); hide(); OVL.close(true) }
    else if(s.dataset.s==='apply') apply();
    else if(s.dataset.s==='done'){ hide(); OVL.close(true) }
  });
  function hide(){ pop.hidden=true }
  // asked by the overlay before Settings closes: false keeps it open
  function beforeClose(){
    if(!pop.hidden){ if(!pop.querySelector('[data-s=done]')) hide(); return false }
    if(!stagedCount()) return true;
    show(); return false;
  }
  return {beforeClose, hide};
})();

export {STAGED, STAGEPOP, stage, stagedCount, stagedVal};
