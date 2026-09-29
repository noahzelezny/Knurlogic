// Shared formatting: the element lookup, GiB, escaping.

const GIB=1073741824, $=id=>document.getElementById(id);
const gb=b=>Number.isFinite(b)?(b/GIB).toFixed(1)+' GiB':'--';
// A whole-GiB rounding, for spots (like an inline swap note) that can't
// afford the decimal's width.
const gb0=b=>Number.isFinite(b)?Math.round(b/GIB)+' GiB':'--';
const row=(k,v)=>`<div class="row"><span>${k}</span><span>${v}</span></div>`;
// quotes too: esc() also fills attributes (title="…"), and a model's name
// is a folder name -- one day a download's -- not something this page chose
const esc=s=>String(s??'').replace(/[<>&"']/g,c=>({'<':'&lt;','>':'&gt;',
  '&':'&amp;','"':'&quot;',"'":'&#39;'}[c]));

export {$, GIB, esc, gb, gb0};
