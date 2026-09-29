// the tab and the pick, remembered for this session
const SKEEP=(()=>{ try{ return JSON.parse(sessionStorage.getItem('kl.settings')||'{}') }catch(e){ return {} } })();

export {SKEEP};
