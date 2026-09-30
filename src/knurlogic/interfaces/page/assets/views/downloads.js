import {$, esc, gb} from '../format.js';
import {allDownloads, hubAct, loadDownloads} from './picker.js';

// Every download the page's server has: running, finished or failed. Polled
// while the overlay is open; the nav badge counts the running ones.
let timer=0;
function render(){
  const ds=allDownloads();
  const n=ds.filter(d=>d.state==='downloading').length;
  const b=$('dlbadge'); b.hidden=!n; b.textContent=n;
  const el=$('dllist');
  if(el.closest('.panel').classList.contains('on')===false) return;
  el.innerHTML=ds.length?ds.map(D=>{
    const pct=D.total_bytes?Math.min(100,Math.round(100*D.bytes/D.total_bytes)):0;
    const run=D.state==='downloading';
    const st=run?(D.total_bytes?`${pct}% · ${gb(D.bytes)} of ${gb(D.total_bytes)}`:'preparing')
      :D.state==='done'?'done':'failed';
    return `<div class="dlrow ${D.state}"><div class="dlhd"><span class="vn">${esc(D.id)}</span>
        <span class="vs">${esc(st)}</span>
        <button class="mini x" data-d="${esc(D.id)}" title="${run?'cancel':D.state==='done'?'clear':'dismiss'}">×</button></div>
      ${run?`<div class="lbar"><i style="width:${pct}%"></i></div>`:''}
      ${D.state==='failed'&&D.why?`<div class="vs why">${esc(D.why)}</div>`:''}</div>`}).join('')
    :'<div class="sect no">no downloads</div>';
  el.querySelectorAll('[data-d]').forEach(x=>x.onclick=async()=>{
    const D=allDownloads().find(d=>d.id===x.dataset.d);
    await hubAct(D&&D.state==='downloading'?'cancel':'dismiss', x.dataset.d);
    render();
  });
}
async function refresh(){ await loadDownloads(); render() }
function showDownloads(){ render(); refresh(); clearInterval(timer); timer=setInterval(refresh, 2000) }
function stopDownloads(){ clearInterval(timer); timer=0 }
setInterval(render, 2000);
export {showDownloads, stopDownloads};
