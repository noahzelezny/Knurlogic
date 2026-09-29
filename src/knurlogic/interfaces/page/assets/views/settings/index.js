import {$} from '../../format.js';
import {DOC, showCluster} from './cluster.js';
import {SKEEP} from './keep.js';
import {showModels} from './models.js';
import {showKnurlogic} from './knurlogic.js';

// --- settings, per machine --------------------------------------------------
// Why the control page showed one setting: its own /settings.json is the
// MACHINE's (the wired limit) -- the knobs belong to a model's server. So
// each machine's tab reads that machine's page and every model it is
// running, through /peek (GET only, the addresses the page already knows).
// This machine and every peer are drawn by the SAME code; only where the
// data comes from differs. A change goes out only by the Apply asked for on
// close (STAGED), through /apply (a known model server's /settings.json).
// SSEC is the side's section; SBASE the base model picked under Models,
// SRUN which running variant's live values a base model shows (by address)
// when it runs in more than one place. SEQ drops an answer that arrives
// after the view moved on.
let SSEC='cluster', SBASE='', SEQ=0;
const SRUN={};
// Every machine alike: this one first, then each answering peer; its name
// and hardware from /status.json's nodes, its models from the residency.
function machines(){
  const d=window.RESIDENCY||{}, ns=window.NODES||[];
  const ours=r=>r.runtime==='knurlogic' && r.where;
  const me=ns.find(n=>n.role==='local')||{};
  return [{id:'local', name:me.node||'this machine', page:'', node:me,
      models:(d.resident||[]).filter(ours)}]
    .concat((d.peers||[]).map(p=>({id:p.address, name:p.machine,
      page:'http://'+p.address, error:p.error, node:ns.find(n=>n.node===p.machine)||{},
      models:(p.resident||[]).filter(ours)})));
}
// "TheDrainFlorist--GLM-5.3-Flash" and "org/name" read as the name alone
const shortName=n=>String(n||'').replace(/^.*\//,'').replace(/^.*?--/,'');
// A machine's models: a page that is itself a model's server has that
// model as its own.
function machModels(m){
  if(m.id!=='local' || !(DOC.knobs||[]).length) return m.models;
  return [{name:(DOC.artifact||{}).name||'this server', where:''}]
    .concat(m.models.filter(r=>r.where.replace(/\/$/,'')!==location.origin));
}
// Every running model, whichever machine runs it, labelled with the machine.
function allModels(){
  return machines().flatMap(m=>machModels(m).map(r=>({...r, mach:m})));
}
if(SKEEP.sec) SSEC=SKEEP.sec; if(SKEEP.base) SBASE=SKEEP.base;
let SMACH=SKEEP.mach||'local';
function keepSet(){ try{ sessionStorage.setItem('kl.settings',
  JSON.stringify({sec:SSEC, base:SBASE, mach:SMACH})) }catch(e){} }
// The Cluster, Models and Knurlogic modules move these through here: an
// imported binding is read-only, so the module that owns it assigns it.
function nextSeq(){ return ++SEQ }
function setSBase(v){ SBASE=v }
function setSMach(v){ SMACH=v }
function renderSetTabs(){
  if(SSEC==='compaction') SSEC='knurlogic';     // folded into Knurlogic
  const secs=[{id:'knurlogic', name:'Knurlogic'}, {id:'cluster', name:'Cluster'},
    {id:'models', name:'Models'}];
  $('settabs').innerHTML=secs.map(t=>`<button data-s="${t.id}"
    class="${t.id===SSEC?'on':''}">${t.name}</button>`).join('');
  $('settabs').querySelectorAll('button').forEach(f=>f.onclick=()=>{
    if(f.dataset.s===SSEC) return;
    SSEC=f.dataset.s; keepSet(); renderSetTabs(); showSetTab() });
}
function showSetTab(){ return SSEC==='models' ? showModels()
  : SSEC==='knurlogic' ? showKnurlogic() : showCluster() }

export {SBASE, SEQ, SMACH, SRUN, SSEC, allModels, keepSet, machines, nextSeq,
        renderSetTabs, setSBase, setSMach, shortName, showSetTab};
