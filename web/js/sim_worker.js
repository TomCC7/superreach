import {Interactions,obstacleXml} from './interactive_physics.js?v=macos-opt-v1';
import {TraceRing,nextDeadline,geomSnapshot,copyGeoms} from './runtime_timing.mjs?v=macos-opt-v1';
let interactions;
import loadMujoco from '../vendor/mujoco/mujoco.js';
import {Float32Engine} from './mlp_infer.mjs?v=js-f32-v1';
import {Policy,KP,KD,HOME,DT} from './policy.js?v=js-f32-v1';
import {periodicReference} from './trajectories.js?v=scheme1-14k-v1';
let mj,m,d,ref,policy,tcp,xml,ready=false,running=false,busy=false,timer,trace,geomsFrame,qref=HOME.slice(),preset='reach',payload=0;
const SUBSTEPS=10, PHYSICS_DT=0.002;
let checkpoints,activeCheckpoint,mode=0,manualPose=null,jointAck=0;
const limits=()=>Array.from(ikLower,(lo,i)=>[lo,ikUpper[i]]);
const boundQ=(q)=>Array.from(q,(x,i)=>Math.max(ikLower[i],Math.min(ikUpper[i],x)));
function setMode(next){
 if(!policy.tri||![0,1,2].includes(next)||mode===next)return;
 // Hold the measured state in the incoming modality, without resetting plant or history.
 mode=next;preset='hold';qref=boundQ(d.qpos.slice(0,6));manualPose=pose(d);state();
}
let inferMs=null,stepMs=null,ikMs=null,physMs=null,packMs=null,lateMs=null;
let tickHz=null,tickWindowStart=null,tickIntervals=0;
const averages=[null,null,null,null];
const ema=(i,x)=>averages[i]===null?x:averages[i]+.15*(x-averages[i]);
function sampleTick(now){
 if(tickWindowStart===null){tickWindowStart=now;return;}
 tickIntervals++;
 const elapsed=now-tickWindowStart;
 if(elapsed>=1000){tickHz=1000*tickIntervals/elapsed;tickWindowStart=now;tickIntervals=0;}
}
const send=(type,fields={})=>postMessage({type,...fields});
const arr=(a)=>Array.from(a);
async function bytes(path){const r=await fetch(new URL('../'+path+'?v=js-f32-v1',import.meta.url));if(!r.ok)throw Error(`${path}: HTTP ${r.status}`);return new Uint8Array(await r.arrayBuffer());}
async function checked(path,expected){const b=await bytes(path);if(expected&&globalThis.crypto?.subtle?.digest){const hash=arr(new Uint8Array(await globalThis.crypto.subtle.digest('SHA-256',b))).map(x=>x.toString(16).padStart(2,'0')).join('');if(hash!==expected)throw Error('Hash mismatch '+path);}return b;}
function pose(data=ref){return [...data.xpos.slice(3*tcp,3*tcp+3),...data.xquat.slice(4*tcp,4*tcp+4)];}
function fk(q){ref.qpos.set(q,0);mj.mj_kinematics(m,ref);return pose();}
function geometry(){const out=[];for(let g=0;g<m.ngeom;g++){
 if(m.geom_group[g]===3)continue;
 const item={id:g,ghost:m.geom_bodyid[g]>1,type:m.geom_type[g],size:arr(m.geom_size.slice(3*g,3*g+3)),rgba:arr(m.geom_rgba.slice(4*g,4*g+4))};
 const mat=m.geom_matid[g];if(mat>=0)item.rgba=arr(m.mat_rgba.slice(4*mat,4*mat+4));
 if(item.type===7){const id=m.geom_dataid[g],v=m.mesh_vertadr[id],n=m.mesh_vertnum[id],f=m.mesh_faceadr[id],nf=m.mesh_facenum[id];item.vertices=arr(m.mesh_vert.slice(3*v,3*(v+n)));item.faces=arr(m.mesh_face.slice(3*f,3*(f+nf)));}
 out.push(item);
 }return out;}
// Policy input uses tick t; UI/scoring use the current post-physics tick.
function currentGoal(){if(preset!=='hold'&&preset!=='manual'){qref=periodicReference(preset,policy.tick*DT,HOME).q;return fk(qref);}return policy.tri&&mode!==2&&manualPose?manualPose:fk(qref);}
function state(goal=currentGoal(),packStart=null,tcpPose=pose(d)){
 const qDes=policy.tick?arr(d.ctrl.subarray(0,6)):null;
 const frame={type:'state',...interactions.state(ref,qDes),...copyGeoms(d,geomsFrame),tick:policy.tick,time:d.time,
  q:arr(d.qpos.subarray(0,6)),dq:arr(d.qvel.subarray(0,6)),q_des:qDes,tcp:tcpPose,goal,qref,mode,tri:policy.tri,jointAck,
  running,payload,preset,checkpoint:activeCheckpoint,backend:policy.engine.backend,infer_ms:inferMs,step_ms:stepMs,
  phys_ms:physMs,ik_ms:ikMs,timer_late_ms:lateMs,tick_hz:running?tickHz:0};
 if(packStart!==null){
  packMs=performance.now()-packStart;
  averages[0]=ema(0,inferMs);averages[1]=ema(1,physMs);averages[2]=ema(2,packMs);
  if(lateMs!==null)averages[3]=ema(3,lateMs);
 }
 frame.pack_ms=packMs;frame.timing_ema=averages;
 // Structured clone snapshots the compact backing buffers; never detach WASM or reuse detached buffers.
 postMessage(frame);
}
function build(mass=0,preserve=false){
 if(!Number.isFinite(mass)||mass<0)throw Error('Invalid payload mass');
 // Recompile inertial constants instead of relying on optional mj_setConst bindings.
 // Preserve measured state AND intended reference/history when only load changes.
 const saved=preserve?{qpos:arr(d.qpos),qvel:arr(d.qvel),ctrl:arr(d.ctrl),warmstart:arr(d.qacc_warmstart),time:d.time}:null;
 if(d)d.delete();if(ref)ref.delete();if(m)m.delete();
 // Verify original bytes first; strip obsolete task props only in memory.
 let source=xml.replace(/meshdir="[^"]*"/,'meshdir="/openyam/meshes"')
 .replace(/<geom\b[^>]*name="(?:table|target_zone)[^"]*"[^>]*\/>/g,'')
 .replace(/<body\b[^>]*name="(?:object|pick_object)"[^>]*>[\s\S]*?<\/body>/g,'');
 source=source.replace('</worldbody>',obstacleXml+'</worldbody>');
 // Training payload: full 40 mm cube at TCP, I/m=0.0002666667 m².
 // A real welded payload contributes mass/inertia; zero keeps the verified scene unchanged.
 if(mass>0)source=source.replace('<site name="arm_tcp_site"',`<body name="demo_payload"><inertial pos="0 0 0" mass="${mass}" diaginertia="${mass*.0002666667} ${mass*.0002666667} ${mass*.0002666667}"/><geom type="box" size=".02 .02 .02" contype="0" conaffinity="0" rgba=".1 .6 .8 1"/></body><site name="arm_tcp_site"`);
 mj.FS.writeFile('/openyam/scene.xml',source);m=mj.MjModel.from_xml_path('/openyam/scene.xml');d=new mj.MjData(m);ref=new mj.MjData(m);
 tcp=mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY.value,'arm_tcp');if(tcp<0)throw Error('arm_tcp missing');
 for(let i=0;i<6;i++)if(Math.abs(m.actuator_gainprm[10*i]-KP[i])>1e-6||Math.abs(m.actuator_biasprm[10*i+2]+KD[i])>1e-6)throw Error('PD gain mismatch');
 if(Math.abs(m.opt.timestep-PHYSICS_DT)>1e-12)throw Error('Timestep mismatch');
 const ranges=m.jnt_range;
 for(let i=0;i<6;i++){ikLower[i]=ranges[2*i];ikUpper[i]=ranges[2*i+1];}
 geomsFrame=geomSnapshot(m.ngeom);interactions.bind(m,d,tcp);payload=mass;if(saved){d.qpos.set(saved.qpos);d.qvel.set(saved.qvel);d.ctrl.set(saved.ctrl);d.qacc_warmstart.set(saved.warmstart);d.time=saved.time;mj.mj_forward(m,d);}else reset();send('ready',{geoms:geometry(),timestep:m.opt.timestep,kp:KP,kd:KD,payload,version:mj.mj_versionString(),control_dt:DT,substeps:SUBSTEPS,jointLimits:limits()});state();
}
function reset(){mj.mj_resetData(m,d);interactions.reset();d.qpos.set(HOME);d.qpos[6]=.04;d.qpos[7]=.04;d.ctrl.set(HOME);d.ctrl[6]=.04;mj.mj_forward(m,d);policy.reset(arr(d.qpos.slice(0,6)),arr(d.qvel.slice(0,6)));qref=HOME.slice();manualPose=null;trace=new TraceRing(policy.tri?318:311);inferMs=stepMs=ikMs=physMs=packMs=lateMs=null;averages.fill(null);running=false;}
// XYZ only. The six leading arm joints are hinges upstream of arm_tcp.
const ikJ=new Float64Array(18),ikStep=new Float64Array(6),ikQ=new Float64Array(6),ikTrial=new Float64Array(6);
const ikLower=new Float64Array(6),ikUpper=new Float64Array(6),ikNominal=new Float64Array(6),ikFree=new Uint8Array(6);
const ikTolerance=1e-5,ikDamping=.003,ikMaxStep=.18;
function ikPosition(q){ref.qpos.set(q,0);mj.mj_kinematics(m,ref);return ref.xpos;}
function ik(target,id){
 const start=performance.now(),q=ikQ,p=3*tcp;
 q.set(qref);let iterations=0,xyz=ikPosition(q);
 // Prefer the previous branch, with a weak bias toward HOME to limit wrist drift.
 for(let i=0;i<6;i++)ikNominal[i]=.8*q[i]+.2*HOME[i];
 for(;iterations<45;iterations++){
  const ex=target[0]-xyz[p],ey=target[1]-xyz[p+1],ez=target[2]-xyz[p+2];
  const error2=ex*ex+ey*ey+ez*ez;
  if(error2<=ikTolerance*ikTolerance)break;
  const axis=ref.xaxis,anchor=ref.xanchor;
  for(let i=0;i<6;i++){
   const j=3*i,x=xyz[p]-anchor[j],y=xyz[p+1]-anchor[j+1],z=xyz[p+2]-anchor[j+2];
   // Jp column = hinge axis × (TCP - hinge anchor), in world coordinates.
   ikJ[j]=axis[j+1]*z-axis[j+2]*y;
   ikJ[j+1]=axis[j+2]*x-axis[j]*z;
   ikJ[j+2]=axis[j]*y-axis[j+1]*x;
  }
  ikFree.fill(1);
  let maxStep;
  for(;;){
   const lambda2=ikDamping*ikDamping;
   let a=lambda2,b=0,c=0,d=lambda2,e=0,f=lambda2,rx=ex,ry=ey,rz=ez;
   for(let i=0;i<6;i++)if(ikFree[i]){
    const j=3*i,x=ikJ[j],y=ikJ[j+1],z=ikJ[j+2],h=.06*(ikNominal[i]-q[i]);
    a+=x*x;b+=x*y;c+=x*z;d+=y*y;e+=y*z;f+=z*z;
    rx-=x*h;ry-=y*h;rz-=z*h;
   }
   // Cholesky on JJᵀ + λ²I. dq = h + Jᵀ (JJᵀ + λ²I)⁻¹ (error - Jh),
   // i.e. damped position correction plus damped null-space posture correction.
   const l00=Math.sqrt(a),l10=b/l00,l20=c/l00,l11=Math.sqrt(d-l10*l10);
   const l21=(e-l20*l10)/l11,l22=Math.sqrt(f-l20*l20-l21*l21);
   const y0=rx/l00,y1=(ry-l10*y0)/l11,y2=(rz-l20*y0-l21*y1)/l22;
   const uz=y2/l22,uy=(y1-l21*uz)/l11,ux=(y0-l10*uy-l20*uz)/l00;
   let blocked=false;maxStep=0;
   for(let i=0;i<6;i++){
    const j=3*i;
    let v=ikFree[i]?ikJ[j]*ux+ikJ[j+1]*uy+ikJ[j+2]*uz+.06*(ikNominal[i]-q[i]):0;
    if((q[i]<=ikLower[i]&&v<0)||(q[i]>=ikUpper[i]&&v>0)){ikFree[i]=0;v=0;blocked=true;}
    ikStep[i]=v;maxStep=Math.max(maxStep,Math.abs(v));
   }
   if(!blocked)break;
  }
  if(maxStep<1e-8)break;
  // Keep steps bounded and only accept FK improvements, even at a joint stop.
  let accepted=false;
  for(let scale=Math.min(1,ikMaxStep/maxStep),tries=0;tries<8;scale*=.5,tries++){
   for(let i=0;i<6;i++)ikTrial[i]=Math.max(ikLower[i],Math.min(ikUpper[i],q[i]+scale*ikStep[i]));
   xyz=ikPosition(ikTrial);
   const x=target[0]-xyz[p],y=target[1]-xyz[p+1],z=target[2]-xyz[p+2];
   if(x*x+y*y+z*z<error2-1e-15){q.set(ikTrial);accepted=true;break;}
  }
  if(!accepted)break;
 }
 qref=arr(q);
 // Both the projected XYZ and unconstrained quaternion come from this exact qref.
 const actual=fk(qref);
 const solve_ms=performance.now()-start;
 ikMs=solve_ms;
 send('target',{id,requested:target.slice(0,3),reachable:actual,iterations,solve_ms,
  residual_mm:Math.hypot(target[0]-actual[0],target[1]-actual[1],target[2]-actual[2])*1000});
}
function step(){
 const goal=currentGoal(),goal_qref=qref,obs=policy.observation(goal,goal_qref,mode),start=performance.now();
 const q_des=policy.infer(obs);
 if(policy.engine.async)return q_des.then(out=>advance(goal,goal_qref,obs,start,out));
 advance(goal,goal_qref,obs,start,q_des);
}
function advance(goal,goal_qref,obs,start,q_des){
 const physicsStart=performance.now();
 // The six MuJoCo position actuators apply Kp*(q_des-q) - Kd*dq.
 inferMs=physicsStart-start;d.ctrl.set(q_des);d.ctrl[6]=.04;
 for(let i=0;i<SUBSTEPS;i++){mj.mj_step(m,d);interactions.afterSubstep();}
 // implicitfast integrates qpos/qvel after computing derived fields. Refresh final-pose
 // geom/TCP and solved contact/actuator forces (including after a poke expires).
 mj.mj_forward(m,d);
 const packStart=performance.now();physMs=stepMs=packStart-physicsStart;
 policy.observe(d.qpos,d.qvel,q_des);
 if(!d.qpos.every(Number.isFinite))throw Error('Physics divergence');
 // Keep causal targets separate from the post-step display reference.
 const scored_goal=currentGoal(),tcpPose=pose(d);
 trace.push(policy.tick,mode,obs,q_des,d.qpos,d.qvel,tcpPose,goal,goal_qref,scored_goal,inferMs,stepMs);
 state(scored_goal,packStart,tcpPose);
}
// Coalesce previews within a gesture only; preserve other gestures' final poses.
let pendingDrags=[],targetTimer=null,idleWaiters=[];
function flushTarget(){clearTimeout(targetTimer);targetTimer=null;if(!ready||busy||!pendingDrags.length)return;
 for(const msg of pendingDrags.splice(0)){
  if(msg.type==='target'){
   if(mode===2||(policy.tri&&(msg.mode!==mode||msg.checkpoint!==activeCheckpoint)))continue;
   preset='hold';
   if(policy.tri){manualPose=mode===0?msg.pose:[...msg.pose,...(manualPose||pose(d)).slice(3)];send('target',{id:msg.id});}
   else ik(msg.pose,msg.id);
  }else if(msg.type==='jointTarget'){
   if(!policy.tri||mode!==2||msg.checkpoint!==activeCheckpoint)continue;
   preset='hold';qref=boundQ(msg.q);jointAck=msg.id;
  }
  else {interactions.move(msg);mj.mj_forward(m,d);}
 }
 state();
}
function releaseBusy(){busy=false;flushTarget();for(const resolve of idleWaiters.splice(0))resolve();}
function whenIdle(){return busy?new Promise(resolve=>idleWaiters.push(resolve)):Promise.resolve();}
function schedule(){
 clearTimeout(timer);tickHz=tickWindowStart=lateMs=averages[3]=null;tickIntervals=0;if(!running)return;
 const period=1000*DT;
 let deadline=performance.now();
 const arm=()=>{timer=setTimeout(tick,Math.max(0,Math.ceil(deadline-performance.now())));};
 const advanceDeadline=()=>{deadline=nextDeadline(deadline,performance.now(),period);arm();};
 const fail=e=>{running=false;send('error',{message:String(e.stack||e)});};
 const finish=()=>{try{releaseBusy();}catch(e){fail(e);}if(running)advanceDeadline();};
 function tick(){
  if(!running)return;
  const start=performance.now();
  if(start<deadline){arm();return;}
  if(busy){advanceDeadline();return;}
  sampleTick(start);lateMs=Math.max(0,start-deadline);busy=true;
  try{const pending=step();if(pending){pending.then(finish,e=>{fail(e);finish();});return;}}
  catch(e){fail(e);}
  finish();
 }
 arm();
}
async function loadPolicy(id){
 if(!Object.hasOwn(checkpoints,id))throw Error('Unknown checkpoint');
 const {sha256,abi,js:descriptor}=checkpoints[id];
 const tri=abi==='tri-mode-v2',width=tri?318:311;
 let engine;
 if(descriptor){
  if(!/^models\/js\/policy-[0-9a-f]{20}\.graph\.json$/.test(descriptor.path))throw Error('Invalid JS checkpoint path');
  const artifact=JSON.parse(new TextDecoder().decode(await checked(descriptor.path,descriptor.sha256)));
  if(artifact.onnx_sha256!==sha256||artifact.input!==width||
     !/^models\/js\/graph-[0-9a-f]{20}\.mjs$/.test(artifact.graph)||
     !/^models\/js\/weights-[0-9a-f]{20}\.bin$/.test(artifact.weights))throw Error('Invalid JS checkpoint manifest');
  const [{forward},weights]=await Promise.all([
   import(new URL('../'+artifact.graph,import.meta.url).href),checked(artifact.weights,artifact.sha256)
  ]);
  engine=new Float32Engine(artifact,forward,weights);engine.backend='JS Float32';engine.async=false;
 }else{throw Error('Missing compiled JS policy descriptor');}
 const next=new Policy(engine,tri);next.reset(HOME,[0,0,0,0,0,0]);
 try{await next.infer(next.observation([.3,0,.35,1,0,0,0],HOME,0));return next;}
 catch(e){await engine.release?.();throw e;}
}
async function selectCheckpoint(id,requestId){
 if(id!==activeCheckpoint){
  let next;
  try{next=await loadPolicy(id);}
  catch(e){send('checkpointDone',{id:requestId,checkpoint:activeCheckpoint,message:String(e)});return;}
  const wasRunning=running,previous=policy,goal=qref,oldPose=manualPose;
  clearTimeout(timer);
  policy=next;activeCheckpoint=id;mode=0;reset();qref=goal;
  if(preset==='hold'&&oldPose){if(next.tri)manualPose=oldPose;else ik(oldPose,null);}
  running=wasRunning;state();schedule();
  await previous.engine.release?.();
 }
 send('checkpointDone',{id:requestId,checkpoint:activeCheckpoint});
}
async function init(requested){send('progress',{message:'Loading MuJoCo WASM and policy…'});mj=await loadMujoco();interactions=new Interactions(mj);
 const bundle=await (await fetch(new URL('../assets/BUNDLE.json?v=js-f32-v1',import.meta.url))).json();mj.FS.mkdirTree('/openyam/meshes');
 const entries=Object.entries(bundle.files);await Promise.all(entries.filter(([n])=>n.startsWith('meshes/')).map(async([name,hash])=>mj.FS.writeFile('/openyam/'+name,await checked('assets/'+name,hash))));
 xml=new TextDecoder().decode(await checked('assets/scene.xml',bundle.files['scene.xml']));
 checkpoints=bundle.checkpoints;
 activeCheckpoint=Object.hasOwn(checkpoints,requested)?requested:bundle.default_checkpoint;
 policy=await loadPolicy(activeCheckpoint);build(0);ready=true;running=true;schedule();
}
let chain=Promise.resolve();onmessage=({data:msg})=>{
 if(msg.type==='target'||msg.type==='jointTarget'||msg.type==='obstacleMove'){
  if(!ready){send('error',{message:'Not ready'});return;}
  const target=msg.type==='target',joint=msg.type==='jointTarget';
  const values=target?msg.pose:joint?msg.q:msg.position;
  const size=joint?6:target&&policy.tri?(msg.mode===0?7:3):3;
  if(!Array.isArray(values)||values.length!==size||!values.every(Number.isFinite))return;
  if(size===7&&Math.abs(Math.hypot(...values.slice(3))-1)>1e-4)return;
  const prev=pendingDrags.at(-1);
  if(prev?.type===msg.type&&prev.gesture===msg.gesture&&!prev.final)pendingDrags[pendingDrags.length-1]=msg;else pendingDrags.push(msg);
  // One task boundary batches bursts already in the worker message queue.
  if(!busy&&targetTimer===null)targetTimer=setTimeout(()=>{try{flushTarget();}catch(e){send('error',{message:String(e.stack||e)});}},0);
  return;
 }
 chain=chain.then(async()=>{
 if(msg.type==='init')return init(msg.checkpoint);if(!ready)throw Error('Not ready');
 // Legacy inference and checkpoint loads can yield to message events.
 await whenIdle();flushTarget();busy=true;
 try{
 if(msg.type==='run'){preset=msg.preset||preset||'hold';running=true;schedule();}
 else if(msg.type==='pause'){running=false;clearTimeout(timer);state();}
 else if(msg.type==='reset'){const wasRunning=running;running=false;clearTimeout(timer);if(msg.preset)preset=msg.preset;build(Number(msg.payload||0));if(wasRunning){running=true;schedule();}}
 else if(msg.type==='mode')setMode(msg.mode);
 else if(msg.type==='checkpoint')await selectCheckpoint(msg.checkpoint,msg.id);
 else if(msg.type==='obstacleConfig'){interactions.configure(msg);mj.mj_forward(m,d);state();}
 else if(msg.type==='ghost'){interactions.ghost=msg.on===true;state();}
 else if(msg.type==='poke'){if(running)interactions.startPoke(msg.target,msg.magnitude);state();}
 else if(msg.type==='payload'){build(Number(msg.payload),true);}
 else if(msg.type==='trace')send('trace',{id:msg.id,trace:trace.snapshot(activeCheckpoint)});
 else if(msg.type==='infer'){send('inference',{id:msg.id,q_des:arr(await policy.infer(Float32Array.from(msg.obs)))});}
 else if(msg.type==='test'){running=false;clearTimeout(timer);build(Number(msg.payload||0));preset=msg.preset||'hold';const ticks=msg.ticks||180;for(let i=0;i<ticks;i++){const pending=step();if(pending)await pending;if(pendingDrags.length){busy=false;try{flushTarget();}finally{busy=true;}}}send('testDone',{id:msg.id,trace:trace.snapshot(activeCheckpoint)});}
 }finally{releaseBusy();}
 }).catch(e=>send('error',{message:String(e.stack||e)}));};
