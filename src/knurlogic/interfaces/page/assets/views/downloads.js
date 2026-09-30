import {$, esc, gb} from '../format.js';
import {allDownloads, hubAct, loadDownloads} from './picker.js';

// Every download the page's server has: running, stopped, finished or failed. Polled
// while the overlay is open; the nav badge counts the running ones.
let timer=0, CONFIRM='';
const svg=p=>`<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${p}</svg>`;
const ICON={resume:svg('<path d="M21 12a9 9 0 0 1-15.5 6.2L3 16"/><path d="M3 21v-5h5"/><path d="M3 12a9 9 0 0 1 15.5-6.2L21 8"/><path d="M21 3v5h-5"/>'),
  trash:svg('<path d="M3 6h18"/><path d="M8 6V4a1 1 0 0 1 1-1h6a1 1 0 0 1 1 1v2"/><path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/><path d="M10 11v6M14 11v6"/>')};
function render(){
  const ds=allDownloads();
  const n=ds.filter(d=>d.state==='downloading').length;
  const b=$('dlbadge'); b.hidden=!n; b.textContent=n;
  const el=$('dllist');
  if(el.closest('.panel').classList.contains('on')===false) return;
  el.innerHTML=ds.length?ds.map(D=>{
    const pct=D.total_bytes?Math.min(100,Math.round(100*D.bytes/D.total_bytes)):0;
    const run=D.state==='downloading', id=esc(D.id);
    const st=run?(D.total_bytes?`${pct}% · ${gb(D.bytes)} of ${gb(D.total_bytes)}`:'preparing')
      :D.state==='done'?'done'
      :D.state==='stopped'?`stopped · ${gb(D.bytes)}${D.total_bytes?' of '+gb(D.total_bytes):''}`:'failed';
    const btn=(k,t)=>`<button class="mini" data-${k}="${id}" title="${t}" aria-label="${t}">${ICON[k]||'×'}</button>`;
    const acts=run?btn('stop','stop')
      :D.state==='done'?btn('clear','clear')
      :CONFIRM===D.id?`<span class="dlconf">delete ${gb(D.bytes)}?
          <button class="mini danger" data-yes="${id}">delete</button>
          <button class="mini" data-no="${id}">keep</button></span>`
      :btn('resume','resume')+btn('trash','delete files');
    return `<div class="dlrow ${D.state}"><div class="dlhd"><span class="vn">${id}</span>
        <span class="vs">${esc(st)}</span>${acts}</div>
      ${run?`<div class="lbar"><i style="width:${pct}%"></i></div>`:''}
      ${D.state==='failed'&&D.why?`<div class="vs why">${esc(D.why)}</div>`:''}</div>`}).join('')
    :'<div class="sect no">no downloads</div>';
  const on=(k,fn)=>el.querySelectorAll(`[data-${k}]`).forEach(x=>x.onclick=async()=>{
    await fn(x.dataset[k]); render() });
  on('stop',id=>hubAct('cancel',id));
  on('clear',id=>hubAct('dismiss',id));
  on('resume',id=>hubAct('download',id));
  on('trash',async id=>{ CONFIRM=id });
  on('no',async()=>{ CONFIRM='' });
  on('yes',async id=>{ CONFIRM=''; await hubAct('delete',id) });
}
async function refresh(){ await loadDownloads(); render() }
function showDownloads(){ render(); refresh(); clearInterval(timer); timer=setInterval(refresh, 2000) }
function stopDownloads(){ clearInterval(timer); timer=0 }
setInterval(render, 2000);
export {showDownloads, stopDownloads};
