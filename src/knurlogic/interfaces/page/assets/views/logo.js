// The knurled wheel, drawn rather than shipped as an image file. Teeth are trapezoids around the rim -- a knurl has flanks, and a
// ring of plain radial lines reads as a sun, not as something you grip.
//
// The word covers the wheel's middle, so the teeth that matter are the ones
// at the top and bottom of the circle. Those are the only part visible, and
// they are what makes the name sit ON a wheel rather than beside an icon.
(()=>{
  const g=document.getElementById('knurl'); if(!g) return;
  const cx=60, cy=60, r0=40, r1=56, n=12, half=(180/n)*0.52;
  // The band the word passes through. The wheel is MASKED out across it
  // rather than drawn behind it: a rim crossing the letters is clutter, and
  // the point is that the word is the axle the teeth turn around.
  let out=`<defs><mask id="wm" maskUnits="userSpaceOnUse"
      x="-40" y="-40" width="200" height="200">
      <rect x="-40" y="-40" width="200" height="200" fill="#fff"/>
      <rect x="-40" y="${cy-21}" width="200" height="42" fill="#000"/>
    </mask></defs><g mask="url(#wm)">`;
  for(let i=0;i<n;i++){
    const a=(360/n)*i, p=[];
    for(const [rad,off] of [[r0,-half*1.55],[r1,-half],[r1,half],[r0,half*1.55]]){
      const t=(a+off)*Math.PI/180;
      p.push(`${(cx+rad*Math.sin(t)).toFixed(2)},${(cy-rad*Math.cos(t)).toFixed(2)}`);
    }
    out+=`<polygon points="${p.join(' ')}" fill="var(--acc)" opacity=".6"/>`;
  }
  out+=`<circle cx="${cx}" cy="${cy}" r="${r0}" fill="none"
        stroke="var(--acc)" stroke-width="3" opacity=".6"/></g>`;
  g.innerHTML=out;
  // The same wheel, whole and small, for the chat's Waiting/Thinking mark.
  let gear='';
  for(let i=0;i<n;i++){
    const a=(360/n)*i, p=[];
    for(const [rad,off] of [[r0,-half*1.55],[r1,-half],[r1,half],[r0,half*1.55]]){
      const t=(a+off)*Math.PI/180;
      p.push(`${(cx+rad*Math.sin(t)).toFixed(2)},${(cy-rad*Math.cos(t)).toFixed(2)}`);
    }
    gear+=`<polygon points="${p.join(' ')}" fill="var(--acc)"/>`;
  }
  window.GEAR=`<svg viewBox="0 0 120 120" aria-hidden="true">${gear}<circle
    cx="${cx}" cy="${cy}" r="${r0}" fill="none" stroke="var(--acc)" stroke-width="10"/></svg>`;
  // The logo's own wheel for an instance card's mark: the same twelve teeth
  // and the same thin ring (the chat mark's heavy ring blurs into the teeth
  // that small).
  window.GEARLOGO=`<svg viewBox="0 0 120 120" aria-hidden="true">${gear}<circle
    cx="${cx}" cy="${cy}" r="${r0}" fill="none" stroke="var(--acc)" stroke-width="5"/></svg>`;
})();
