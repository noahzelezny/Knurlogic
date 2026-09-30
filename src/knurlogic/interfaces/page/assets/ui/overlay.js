// --- overlays -------------------------------------------------------------
// ONE mechanism for everything that floats over the page: the model picker,
// Settings and Connect. One is open at a time; a click
// anywhere outside its face, or Escape, closes it, and the layout underneath
// never changed, so closing it puts you back exactly where you were.
// `face` is the part that counts as inside (a modal's box, not its dimmed
// backdrop); `opener` is the control that toggles it, which must not count as
// "outside" or the click that closes it would open it again.
const OVL=(()=>{
  let cur=null;
  // `force` skips the overlay's own say (Settings asks about what is staged)
  function close(force){
    if(!cur) return;
    if(!force && cur.beforeclose && !cur.beforeclose()) return;
    const c=cur; cur=null;
    c.el.hidden=true;
    if(c.modal) document.body.style.overflow='';
    if(c.onclose) c.onclose();
  }
  function open(el, o={}){
    close(true);
    cur={el, face:o.face||el, opener:o.opener||null,
         modal:el.classList.contains('modal'), onclose:o.onclose,
         beforeclose:o.beforeclose};
    el.hidden=false;
    if(cur.modal) document.body.style.overflow='hidden';
  }
  // mousedown, not click: a drag that starts in a field and ends outside it
  // (selecting text) is not a click outside.
  addEventListener('mousedown',e=>{
    if(!cur || cur.face.contains(e.target)) return;
    if(cur.opener && cur.opener.contains(e.target)) return;
    close();
  }, true);
  addEventListener('keydown',e=>{
    if(e.key==='Escape' && cur && document.getElementById('lightbox').hidden)
      close();
  });
  return {open, close, is:el=>!!cur && cur.el===el};
})();

// (i) bubbles open above their icon; when that would be cut off by the
// scroll container (or the window) they open below instead.
function placeInfo(e){
  const i=e.target.closest && e.target.closest('.info');
  if(!i) return;
  const b=i.querySelector('.bub'); if(!b) return;
  i.classList.remove('down');
  let top=0, p=i.parentElement;
  while(p && p!==document.body){
    const o=getComputedStyle(p).overflowY;
    if(/auto|scroll|hidden/.test(o)){ top=Math.max(top, p.getBoundingClientRect().top); break }
    p=p.parentElement;
  }
  if(b.getBoundingClientRect().top<top+4) i.classList.add('down');
}
addEventListener('mouseover',placeInfo);
addEventListener('focusin',placeInfo);

export {OVL};
