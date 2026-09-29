// The machines the status reports, and which of them are picked.

let LASTWS=0;   // this box's working set, learned from the first status
// The machines picked in the MEMORY grid, by node name. None = automatic.
// Kept in this browser as a convenience; the page state is the Set.
let LASTNODES=[];
// tick() learns both from each status (an imported binding is read-only there)
function setLastNodes(v){ LASTNODES=v }
function setLastWS(v){ LASTWS=v }
const NODESEL=(()=>{ try{ return new Set(JSON.parse(
  localStorage.getItem('kn.nodesel')||'[]')) }catch(e){ return new Set() } })();
function saveNodeSel(){
  try{ localStorage.setItem('kn.nodesel', JSON.stringify([...NODESEL])) }catch(e){} }
// the picked machines that are still in the cluster
const selNodes=()=>LASTNODES.filter(n=>NODESEL.has(n.node));
const isLocal=n=>n.role==='local'||n.role==='server';
// What a model has to fit in: the picked machines' working sets (each under
// its allowance) added up; with none picked, the whole cluster's -- nothing
// is assumed to be meant for this machine.
const clusterNodes=()=>LASTNODES.filter(n=>isLocal(n)||n.state==='answering');
function fitWS(){
  const ns=selNodes().length ? selNodes() : clusterNodes();
  return ns.length ? ns.reduce((x,n)=>x+nodeWS(n),0) : (LASTWS||0);
}
// A machine's room under its knurlogic allowance. For this one that is the
// server's own answer (every model's `room` is worked out against it); with
// nothing loaded the status reports the whole box, which is more than the
// allowance lets knurlogic take. A peer's status carries its GPU working
// set, the most its own page would allow.
function nodeWS(n){
  if(isLocal(n)){
    const r=(window.ALLMODELS||[]).find(m=>m.room&&m.room.working_set_bytes);
    if(r) return r.room.working_set_bytes;
  }
  return (n.memory||{}).working_set_bytes||0;
}
const fitWhere=()=>{ const ns=selNodes();
  return !ns.length ? (clusterNodes().length>1 ? "the cluster's memory" : "this machine's memory")
    : ns.length===1 ? `${ns[0].node}'s memory` : `the ${ns.length} picked machines' memory` };

export {LASTNODES, NODESEL, fitWS, fitWhere, isLocal, nodeWS, saveNodeSel,
        selNodes, setLastNodes, setLastWS};
