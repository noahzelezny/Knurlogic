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

export {OVL};
