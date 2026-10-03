// A newer knurlogic on PyPI: the page asks /release.json, which the server
// fills from one PyPI request at page start (page/updates.py), so it is
// asked again a little later for the answer to arrive. Click copies the
// upgrade command.
import {getJSON} from '../api.js';

async function checkRelease(){
  const d=await getJSON('/release.json'), a=document.getElementById('relnote');
  if(!a || d.error || !d.update) return !!d.latest;
  a.hidden=false;
  a.textContent=`Update ${d.latest}`;
  a.title=`knurlogic ${d.latest} is out (this is ${d.current}). `
    +`Click to copy: ${d.command} -- then restart knurlogic ui`;
  a.onclick=async()=>{
    try{ await navigator.clipboard.writeText(d.command); a.textContent='Copied' }
    catch(e){ a.textContent=d.command }
    setTimeout(()=>{ a.textContent=`Update ${d.latest}` },2000);
  };
  return true;
}

// the server's one PyPI request may not have answered yet
(async()=>{ for(const s of [3,30,120]){
  await new Promise(r=>setTimeout(r,s*1000));
  if(await checkRelease()) return } })();
