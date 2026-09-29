// Reading the page's JSON: its own documents, a peer's through /peek.

const peekURL=(where,path,q)=>'/peek?'+new URLSearchParams({where,path,...(q||{})});
// a model server's settings: '' is this page's own
const docURL=(where,q)=>where ? peekURL(where,'/settings.json',q)
  : '/settings.json'+(q&&Object.keys(q).length?'?'+new URLSearchParams(q):'');
async function getJSON(url){
  try{ const r=await fetch(url); const j=await r.json();
       return r.ok ? j : {error:j.error||('HTTP '+r.status)} }
  catch(e){ return {error:String(e.message||e)} }
}

// A failed request in the server's own words: its JSON error's message
// (OpenAI {error:{message}}, ours {error:"..."}), else its text -- never a
// bare status.
async function httpWhy(r){
  let t=''; try{ t=await r.text() }catch(e){}
  let j=null; try{ j=JSON.parse(t) }catch(e){}
  const e=j && (j.error ?? j.message ?? j.detail);
  const msg=typeof e==='string' ? e : e && (e.message||JSON.stringify(e));
  return msg || (t && !/^\s*</.test(t) ? t.slice(0,300) : '')
    || `the server answered ${r.status}${r.statusText?' '+r.statusText:''} with no message`;
}

export {docURL, getJSON, httpWhy, peekURL};
