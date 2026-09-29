// The page's entry point, loaded as a native ES module (no build step):
// importing the modules wires each view's controls (nothing reads another
// module's state while loading); then the panel switching below is wired
// and the poll loop starts.

import './views/logo.js';
import './format.js';
import './nodes.js';
import './views/settings/knobs.js';
import './api.js';
import './views/settings/keep.js';
import './views/settings/models.js';
import './views/settings/knurlogic.js';
import './views/bench.js';
import './views/chat.js';
import {OVL} from './ui/overlay.js';
import {loadSettings} from './views/settings/cluster.js';
import {STAGEPOP, stagedCount} from './views/settings/apply.js';
import {renderSetTabs, showSetTab} from './views/settings/index.js';
import {showConnect} from './views/connect.js';
import {tick} from './views/home.js';
import {loadModels} from './views/picker.js';
import {loadResident} from './views/memory.js';

// Settings and Connect: a thing you go and get, shown as an overlay. The
// page always OPENS with it closed -- an earlier version remembered the last
// panel in localStorage, and open once meant open forever after.
(()=>{
  const sheet=document.getElementById('sheet');
  const links=[...document.querySelectorAll('nav a[data-panel]')];
  const mark=name=>links.forEach(a=>
    a.setAttribute('aria-pressed', a.dataset.panel===name));
  function show(name){
    if(!name){ OVL.close(); return }
    // switching panels is a close of Settings too: ask first
    if(OVL.is(sheet) && stagedCount() && name!=='settings'){ OVL.close(); return }
    document.querySelectorAll('.sheetbody .panel').forEach(p=>
      p.classList.toggle('on', p.id===name));
    document.getElementById('sheettitle').textContent=name;
    if(name==='settings'){ renderSetTabs(); showSetTab() }
    if(name==='connect') showConnect();
    OVL.open(sheet, {face:sheet.querySelector('.box'),
                     opener:document.querySelector('header nav'),
                     onclose:()=>{ mark(null); STAGEPOP.hide() },
                     beforeclose:()=>STAGEPOP.beforeClose()});
    mark(name);
    sheet.querySelector('.sheetbody').scrollTop=0;
  }
  links.forEach(a=>a.onclick=()=>show(
    a.getAttribute('aria-pressed')==='true' ? null : a.dataset.panel));
  document.getElementById('sheetclose').onclick=()=>show(null);
  window.knShow=show;
})();

tick().then(loadModels).then(loadResident);
loadSettings({}); setInterval(tick,2000); setInterval(loadResident,5000);
