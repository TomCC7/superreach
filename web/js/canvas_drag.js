// Controls are created with no DOM element. The order here breaks overlapping hits.
export function arbitrate(canvas, camera, orbit, controls) {
 let active=null, orbiting=false, hoverFrame=null, hoverEvent=null, lastHover=-Infinity;
 const options={capture:true};
 const coords=(e,button=e.button)=>{const r=canvas.getBoundingClientRect();return {x:2*(e.clientX-r.left)/r.width-1,y:1-2*(e.clientY-r.top)/r.height,button};};
 const claim=e=>{e.preventDefault();e.stopImmediatePropagation();};
 function pick(pointer){
  // Recompute even without preceding hover/render (e.g. first touch).
  camera.updateMatrixWorld();
  let winner=null;
  for(const control of controls){
   if(!control.enabled||!control.getHelper().visible){control.axis=null;continue;}
   control.getHelper().updateMatrixWorld(true);
   control.pointerHover(pointer);
   if(!winner&&control.axis!==null)winner=control;else control.axis=null;
  }
  if(winner)winner.getHelper().updateMatrixWorld(true);
  return winner;
 }
 function cancelHover(){if(hoverFrame!==null)cancelAnimationFrame(hoverFrame);hoverFrame=hoverEvent=null;}
 function hover(now){
  if(now-lastHover<1000/30){hoverFrame=requestAnimationFrame(hover);return;}
  const event=hoverEvent;hoverFrame=hoverEvent=null;
  if(!active&&!orbiting&&event){lastHover=now;pick(coords(event));}
 }
 orbit.addEventListener('start',()=>{cancelHover();orbiting=true;for(const c of controls)c.axis=null;});
 orbit.addEventListener('end',()=>{orbiting=false;});
 canvas.addEventListener('pointerdown',e=>{
  if(active){claim(e);return;}
  if(orbiting||e.button!==0||e.shiftKey||e.ctrlKey||e.metaKey)return;
  cancelHover();const pointer=coords(e),control=pick(pointer);
  if(!control)return;
  canvas.setPointerCapture(e.pointerId);
  active={id:e.pointerId,control,x:e.clientX,y:e.clientY};
  orbit.enabled=false;
  control.pointerDown(pointer);
  claim(e);
 },options);
 function move(e){
  if(e.clientX===active.x&&e.clientY===active.y)return;
  active.x=e.clientX;active.y=e.clientY;
  active.control.pointerMove(coords(e,-1));
 }
 canvas.addEventListener('pointermove',e=>{
  if(active){if(active.id===e.pointerId)move(e);claim(e);}
  else if(!orbiting&&e.buttons===0&&e.pointerType!=='touch'){hoverEvent=e;if(hoverFrame===null)hoverFrame=requestAnimationFrame(hover);}
 },options);
 function finish(e){
  if(!active||active.id!==e.pointerId)return;
  if(e.type==='pointerup')move(e);
  active.control.pointerUp(null); // Flush via dragging-changed before relinquishing ownership.
  active=null;orbit.enabled=true;
  if(canvas.hasPointerCapture(e.pointerId))canvas.releasePointerCapture(e.pointerId);
  claim(e);
 }
 for(const type of ['pointerup','pointercancel','lostpointercapture'])canvas.addEventListener(type,finish,options);
 canvas.addEventListener('pointerleave',()=>{cancelHover();if(!active)for(const c of controls)c.axis=null;},options);
 canvas.addEventListener('wheel',e=>{if(active)claim(e);},{capture:true,passive:false});
}
