import {arbitrate} from './canvas_drag.js?v=macos-opt-v1';
import {interactiveView} from './interactive_view.js?v=macos-opt-v1';
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {TransformControls} from 'three/addons/controls/TransformControls.js';
import {ErrorPlot} from './error_plot.js?v=macos-opt-v1';
const $=id=>document.getElementById(id),host=$('viewport');
const plot=new ErrorPlot($('error-chart'));
const scene=new THREE.Scene();scene.background=new THREE.Color('#e8ede8');
const camera=new THREE.PerspectiveCamera(39,1,.005,30);camera.up.set(0,0,1);camera.position.set(.95,-1.1,.75);
// Avoid multisample resolve/preserved-buffer copies on every interactive frame.
// In particular, software WebGL otherwise stalls message delivery at ~30 fps.
const renderer=new THREE.WebGLRenderer({antialias:false,powerPreference:'high-performance'});const renderDpr=()=>Math.min(window.devicePixelRatio||1,1.5);renderer.setPixelRatio(renderDpr());renderer.outputColorSpace=THREE.SRGBColorSpace;host.append(renderer.domElement);
scene.add(new THREE.HemisphereLight(0xffffff,0x8b998c,2.7));const key=new THREE.DirectionalLight(0xfff5e6,3.5);key.position.set(1,-1,2);scene.add(key);const fill=new THREE.DirectionalLight(0xffffff,1.5);fill.position.set(-1,1,1);scene.add(fill);
const grid=new THREE.GridHelper(2,40,0xb4c3b6,0xd1dcd2);grid.rotation.x=Math.PI/2;grid.position.z=-.055;scene.add(grid);
const target=new THREE.Mesh(new THREE.SphereGeometry(.009,20,16),new THREE.MeshStandardMaterial({color:0x2b9e69,transparent:true,opacity:.7}));scene.add(target);target.add(new THREE.AxesHelper(.055));
const requested=new THREE.Mesh(new THREE.SphereGeometry(.006,16,12),new THREE.MeshBasicMaterial({color:0xdfaa45}));requested.visible=false;scene.add(requested);
const gizmo=new TransformControls(camera);gizmo.setMode('translate');gizmo.setSize(.72);gizmo.setSpace('world');gizmo.attach(target);scene.add(gizmo.getHelper());gizmo.enabled=false;
const obstacleHandle=new TransformControls(camera);obstacleHandle.setMode('translate');obstacleHandle.setSize(.59);obstacleHandle.setSpace('world');scene.add(obstacleHandle.getHelper());
const orbit=new OrbitControls(camera,renderer.domElement);orbit.target.set(.13,0,.17);orbit.enableDamping=true;orbit.minDistance=.3;orbit.maxDistance=3;orbit.update();

let geoms=[],lastState=null,ready=false,frames=0,lastFps=performance.now(),renderFrames=0,errors=[],trace=[],targetPending=false,stateDirty=false,targetFrame=null,targetSequence=0,lastTargetId=null;
const hud={hz:$('worker-hz'),backend:$('webgl-renderer')};
const timingRows=[['infer_ms','infer-ms'],['phys_ms','phys-ms'],['pack_ms','pack-ms'],['timer_late_ms','late-ms']].map(([key,id])=>[key,$(id)]);
let hudDirty=false,lastHud=-Infinity;
let renderDirty=true,displayError=0;
const invalidate=()=>{renderDirty=true;};
function updateBackend(){
 const gl=renderer.getContext(),debug=gl.getExtension('WEBGL_debug_renderer_info');
 hud.backend.textContent=(debug&&gl.getParameter(debug.UNMASKED_RENDERER_WEBGL))||'Unmasked renderer unavailable';
 invalidate();
}
updateBackend();renderer.domElement.addEventListener('webglcontextrestored',updateBackend);
orbit.addEventListener('change',invalidate);gizmo.addEventListener('change',invalidate);obstacleHandle.addEventListener('change',invalidate);
const interactive=interactiveView(scene,obstacleHandle,send,invalidate);
arbitrate(renderer.domElement,camera,orbit,[gizmo,obstacleHandle]);
const setText=(element,value)=>{if(element.textContent!==value)element.textContent=value;};
const timing=(value,unit)=>Number.isFinite(value)?`${value.toFixed(2)} ${unit}`:'—';
const worker=new Worker(new URL('./sim_worker.js?v=macos-opt-v1',import.meta.url),{type:'module'}),waiters=new Map();let seq=0;
const selector=$('checkpoint'), checkpointStatus=$('checkpoint-status');
const controlMode=$('control-mode'),tool=$('eef-tool'),jointPanel=$('joint-controls');
let sliders=[],jointDraft=null,jointPending=0,jointSeq=0;
const deg=180/Math.PI;
function makeSliders(limits){
 if(sliders.length)return;
 sliders=limits.map(([lo,hi],i)=>{
  const label=document.createElement('label'),name=document.createElement('span'),input=document.createElement('input'),value=document.createElement('output');
  name.textContent=`J${i+1}`;input.type='range';input.id=`joint-${i+1}`;input.setAttribute('aria-label',`Joint ${i+1}`);input.min=lo*deg;input.max=hi*deg;input.step='0.1';
  value.htmlFor=input.id;label.append(name,input,value);$('joint-sliders').append(label);
  input.addEventListener('input',()=>{const q=(jointDraft||lastState.qref).slice();q[i]=input.valueAsNumber/deg;sendJoints(q);});
  return {input,value};
 });
}
function renderJoints(q){sliders.forEach(({input,value},i)=>{input.value=q[i]*deg;const text=`${(q[i]*deg).toFixed(1)}°`;setText(value,text);input.setAttribute('aria-valuetext',`${(q[i]*deg).toFixed(1)} degrees`);});}
function sendJoints(q){if(lastState?.mode!==2)return;jointDraft=q;renderJoints(q);jointPending=++jointSeq;send({type:'jointTarget',id:jointPending,q,checkpoint:lastState.checkpoint});}
$('joint-home').onclick=()=>sendJoints([0,1.047,1.047,0,0,0]);
controlMode.onchange=()=>{controlMode.disabled=true;requested.visible=false;send({type:'mode',mode:Number(controlMode.value)});};
tool.onchange=()=>{gizmo.axis=null;gizmo.setMode(tool.value);invalidate();};
const requestedCheckpoint=new URL(location.href).searchParams.get('checkpoint');
function send(message){worker.postMessage(message);}
function request(type,fields={}){return new Promise((resolve,reject)=>{const id=++seq;waiters.set(id,{resolve,reject});send({type,id,...fields});});}
function fail(message){errors.push(message);$('status').textContent='Error: '+message;console.error(message);for(const w of waiters.values())w.reject(Error(message));waiters.clear();}
worker.onerror=e=>fail(e.message);
function makeGeoms(items){for(const x of geoms){scene.remove(x.mesh);x.ghostMesh?.removeFromParent();x.mesh.geometry.dispose();x.mesh.material.dispose();}geoms=[];
 for(const item of items){let g;if(item.type===7){g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.Float32BufferAttribute(item.vertices,3));g.setIndex(item.faces);g.computeVertexNormals();}
 else if(item.type===6)g=new THREE.BoxGeometry(...item.size.map(x=>2*x));else if(item.type===2)g=new THREE.SphereGeometry(item.size[0],24,16);else if(item.type===5){g=new THREE.CylinderGeometry(item.size[0],item.size[0],item.size[1]*2,24);g.rotateX(Math.PI/2);}else continue;
 const color=new THREE.Color().setRGB(...item.rgba.slice(0,3)),mesh=new THREE.Mesh(g,new THREE.MeshStandardMaterial({color,metalness:.15,roughness:.5,side:THREE.DoubleSide}));mesh.matrixAutoUpdate=false;scene.add(mesh);const ghostMesh=item.ghost?new THREE.Mesh(g,interactive.ghostMaterial):null;if(ghostMesh){ghostMesh.matrixAutoUpdate=false;interactive.ghost.add(ghostMesh);}geoms.push({id:item.id,mesh,ghostMesh});}
}
worker.onmessage=({data:msg})=>{
 if(msg.id&&waiters.has(msg.id)){waiters.get(msg.id).resolve(msg);waiters.delete(msg.id);}
 if(msg.type==='error')fail(msg.message);
 if(msg.type==='progress')$('status').textContent=msg.message;
 if(msg.type==='ready'){makeGeoms(msg.geoms);makeSliders(msg.jointLimits);ready=true;interactive.enable();$('run').disabled=false;$('reset').disabled=false;$('status').textContent='Ready · Drag a handle to take control';$('engine').textContent=`MuJoCo ${msg.version} · ORT Web 1.22 · WASM`;invalidate();}
 if(msg.type==='target'&&msg.id===lastTargetId){setText($('target-note'),msg.residual_mm===undefined?'Green: Cartesian target':`IK projection: ${msg.residual_mm.toFixed(2)} mm · Green: reachable reference`);targetPending=targetFrame!==null;if(lastState)stateDirty=true;}
 if(msg.type==='state'){
 lastState=msg;stateDirty=hudDirty=true;
 displayError=Math.hypot(msg.tcp[0]-msg.goal[0],msg.tcp[1]-msg.goal[1],msg.tcp[2]-msg.goal[2])*1000;plot.push(msg.time,displayError);
 }
 if(msg.type==='testDone'||msg.type==='trace')trace=msg.trace;
};
const dragStats={translate:0};
gizmo.addEventListener('dragging-changed',e=>{if(e.value){dragStats.translate++;requested.visible=!lastState?.tri;}else sendTarget(true);});
gizmo.addEventListener('objectChange',queueTarget);
function queueTarget(){if(!ready)return;targetPending=true;if(targetFrame===null)targetFrame=requestAnimationFrame(()=>{targetFrame=null;sendTarget();});}
function sendTarget(final=false){if(!ready||!lastState||lastState.mode===2||controlMode.disabled)return;if(targetFrame!==null){cancelAnimationFrame(targetFrame);targetFrame=null;}targetPending=true;requested.position.copy(target.position);invalidate();lastTargetId=`target-${++targetSequence}`;send({type:'target',id:lastTargetId,gesture:dragStats.translate,final,mode:lastState.mode,checkpoint:lastState.checkpoint,pose:lastState.tri&&lastState.mode===0?[...target.position.toArray(),target.quaternion.w,target.quaternion.x,target.quaternion.y,target.quaternion.z]:target.position.toArray()});$('preset').value='hold';return lastTargetId;}
function applyState(msg){
 function transform(id,mesh,x,v){const p=3*id,r=9*id;mesh.matrix.set(v[r],v[r+1],v[r+2],x[p],v[r+3],v[r+4],v[r+5],x[p+1],v[r+6],v[r+7],v[r+8],x[p+2],0,0,0,1);mesh.matrixWorldNeedsUpdate=true;}
 for(const {id,mesh,ghostMesh} of geoms){transform(id,mesh,msg.positions,msg.rotations);if(ghostMesh&&msg.ghost)transform(id,ghostMesh,msg.ghost.positions,msg.ghost.rotations);}
 interactive.apply(msg);
 const joint=msg.tri&&msg.mode===2, eef=msg.tri&&msg.mode===0;
 $('control-mode-card').hidden=!msg.tri;$('eef-tools').hidden=!eef;jointPanel.hidden=!joint;
 if(controlMode.disabled&&controlMode.value===String(msg.mode))controlMode.disabled=false;
 if(!controlMode.disabled)controlMode.value=String(msg.mode);
 gizmo.enabled=!joint;gizmo.getHelper().visible=!joint;
 if(!gizmo.dragging)gizmo.setMode(eef?tool.value:'translate');
 setText($('mode-label'),msg.tri?['EEF 6D','XYZ-only','Joint 6D'][msg.mode]:'XYZ handle · EEF policy');
 setText($('control-hint'),joint?'Set a joint target with the sliders':eef&&tool.value==='rotate'?'Drag a ring to rotate the target':'Drag target or obstacle handles');
 if(!joint||msg.jointAck===jointPending){jointPending=0;jointDraft=null;}
 if(joint)renderJoints(jointDraft||msg.qref);
 if(joint)setText($('target-note'),'Green: TCP pose of the joint target');
 if(!gizmo.dragging&&!targetPending){target.position.fromArray(msg.goal);target.quaternion.set(msg.goal[4],msg.goal[5],msg.goal[6],msg.goal[3]);}
 if($('preset').value!==msg.preset)$('preset').value=msg.preset;
 invalidate();
}
function updateHud(msg){
 setText($('active-checkpoint'),msg.checkpoint);
 if(selector.disabled&&!checkpointStatus.textContent){selector.value=msg.checkpoint;selector.disabled=false;}
 interactive.updateHud(msg);
 setText($('time'),msg.time.toFixed(2)+' s');setText($('error'),displayError.toFixed(2));
 setText($('velocity'),Math.sqrt(msg.dq.reduce((s,x)=>s+x*x,0)/6).toFixed(5)+' rad/s');
 setText($('run'),msg.running?'Pause':'Resume');setText($('status'),msg.running?`● ${msg.checkpoint} · 50 Hz policy`:`Paused · ${msg.checkpoint}`);
 setText($('engine'),`MuJoCo WASM · ${msg.backend}`);setText(hud.hz,timing(msg.tick_hz,'Hz'));
 timingRows.forEach(([key,node],i)=>setText(node,`${timing(msg.timing_ema[i],'ms')} / ${timing(msg[key],'ms')}`));
}
$('run').onclick=()=>lastState?.running?send({type:'pause'}):send({type:'run',preset:$('preset').value});
function reset(){plot.clear();requested.visible=false;invalidate();$('target-note').textContent='Green target: reachable policy reference.';send({type:'reset',payload:Number($('payload').value),preset:$('preset').value});}
$('reset').onclick=reset;$('payload').onchange=()=>send({type:'payload',payload:Number($('payload').value)});
$('preset').onchange=()=>{reset();};
new ResizeObserver(([entry])=>{const {width,height}=entry.contentRect;if(!width||!height)return;
 renderer.setSize(width,height,false);camera.aspect=width/height;camera.updateProjectionMatrix();invalidate();
}).observe(host);
function render(now){requestAnimationFrame(render);const dpr=renderDpr();if(renderer.getPixelRatio()!==dpr){renderer.setPixelRatio(dpr);invalidate();}if(hudDirty&&now-lastHud>=80){hudDirty=false;lastHud=now;updateHud(lastState);}if(stateDirty){stateDirty=false;applyState(lastState);}plot.renderIfNeeded(now);if(orbit.enabled)orbit.update();if(renderDirty){renderDirty=false;renderer.render(scene,camera);frames++;renderFrames++;}if(now-lastFps>1000){setText($('fps'),(frames*1000/(now-lastFps)).toFixed(0)+' fps');frames=0;lastFps=now;}}requestAnimationFrame(render);
// Explicit, read-only diagnostics and deterministic test hooks for real-browser verification.
window.demo={get ready(){return ready;},get state(){return lastState;},get errors(){return errors;},get renderFrames(){return renderFrames;},get trace(){return trace;},test:fields=>request('test',fields),infer:obs=>request('infer',{obs}),getTrace:()=>request('trace'),setTarget:xyz=>{target.position.fromArray(xyz);requested.visible=!lastState?.tri;return sendTarget(true);},get plotSamples(){return plot.samples.slice();},get dragStats(){return {...dragStats,obstacle:interactive.drags};},interactive,obstacleHandle,geoms:()=>geoms,scene,target,orbit,camera,gizmo,renderer,worker};
selector.onchange=async()=>{
 const checkpoint=selector.value;
 selector.disabled=true;
 setText(checkpointStatus,`Loading ${checkpoint}…`);
 try{
  const result=await request('checkpoint',{checkpoint});
  if(result.message){selector.value=result.checkpoint;setText(checkpointStatus,`Could not load ${checkpoint}. Still using ${result.checkpoint}.`);return;}
  plot.clear();requested.visible=false;invalidate();
  setText(checkpointStatus,'');
  const url=new URL(location.href);url.searchParams.set('checkpoint',result.checkpoint);history.replaceState(null,'',url);
 }finally{selector.disabled=false;}
};
send({type:'init',checkpoint:requestedCheckpoint});
