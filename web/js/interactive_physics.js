import {shapes,materials,obstacleHome} from './interaction_config.js?v=interact-v1';
import {geomSnapshot,copyGeoms} from './runtime_timing.mjs?v=macos-opt-v1';
const PARK=[0,0,-5], POKE_STEPS=75;
export const obstacleXml=`<body name="demo_obstacle" mocap="true" pos="${PARK.join(' ')}">${Object.entries(shapes).map(([name,{type,size}])=>`<geom name="demo_${name}" type="${type}" size="${size.join(' ')}" group="3" contype="2" conaffinity="1" priority="1" friction=".5 .005 .0001"/>`).join('')}</body>`;
export class Interactions {
 constructor(mj){this.mj=mj;this.force=new mj.DoubleBuffer(6);this.on=false;this.shape='wall';this.material='rigid';this.pos=obstacleHome.slice();this.moveId=null;this.ghost=false;this.builds=0;this.pokeId=0;this.poke=null;}
 bind(m,d,tcp){
  this.m=m;this.d=d;this.ghostFrame=null;this.builds++;
  const name=(type,n)=>this.mj.mj_name2id(m,this.mj.mjtObj[type].value,n);
  this.ids=Object.fromEntries(Object.keys(shapes).map(s=>[s,name('mjOBJ_GEOM',`demo_${s}`)]));
  this.mocap=3*m.body_mocapid[name('mjOBJ_BODY','demo_obstacle')];
  this.bodies={tcp,elbow:name('mjOBJ_BODY','arm_link_3')};
  if([...Object.values(this.ids),this.mocap,this.bodies.elbow].some(i=>i<0))throw Error('Missing interaction model element');
  this.configure({});
  // Preserve an in-flight poke across payload-only recompilation.
  if(this.poke?.remaining)this.writePoke();
 }
 configure({on,shape,material}){
  if(typeof on==='boolean')this.on=on;
  if(Object.hasOwn(shapes,shape))this.shape=shape;
  if(Object.hasOwn(materials,material))this.material=material;
  const m=this.m,{solref,solimp}=materials[this.material];
  this.activeGeom=this.on?this.ids[this.shape]:-1;
  for(const id of Object.values(this.ids)){
   m.geom_contype[id]=id===this.activeGeom?2:0;m.geom_conaffinity[id]=id===this.activeGeom?1:0;
   m.geom_solref.set(solref,2*id);m.geom_solimp.set(solimp,5*id);
  }
  this.updatePosition();
 }
 move({position,id}){
  if(!Array.isArray(position)||position.length!==3||!position.every(Number.isFinite))return;
  this.pos=position.map(x=>Math.max(-2,Math.min(2,x)));this.moveId=id;this.updatePosition();
 }
 updatePosition(){this.d.mocap_pos.set(this.on?this.pos:PARK,this.mocap);}
 reset(){this.poke=null;this.updatePosition();}
 startPoke(target,magnitude){
  if(this.poke?.remaining||!Object.hasOwn(this.bodies,target)||![15,30,50].includes(magnitude))return;
  this.poke={id:++this.pokeId,target,force:[0,magnitude,0],remaining:POKE_STEPS};this.writePoke();
 }
 writePoke(){this.d.xfrc_applied.set(this.poke.force,6*this.bodies[this.poke.target]);}
 afterSubstep(){
  if(this.poke?.remaining&&--this.poke.remaining===0)this.d.xfrc_applied.fill(0,6*this.bodies[this.poke.target],6*this.bodies[this.poke.target]+6);
 }
 contacts(){
  const {m,d,mj,activeGeom}=this,points=[];let peak=0,penetration=0;
  if(activeGeom<0||!d.ncon)return {points,peak,penetration};
  const ncon=d.ncon,contacts=d.contact;
  for(let i=0;i<ncon;i++){
   const c=contacts.get(i);
   if(c.efc_address<0||(c.geom1!==activeGeom&&c.geom2!==activeGeom))continue;
   mj.mj_contactForce(m,d,i,this.force);
   const f=this.force.GetView(),r=c.frame,sign=c.geom1===activeGeom?1:-1;
   // Columns of frame^T map local normal/tangents to world. Show force on the arm.
   const vector=[0,1,2].map(j=>sign*(r[j]*f[0]+r[j+3]*f[1]+r[j+6]*f[2]));
   const magnitude=Math.hypot(...vector);
   peak=Math.max(peak,magnitude);penetration=Math.max(penetration,-c.dist);
   points.push({pos:Array.from(c.pos),vector,magnitude,dist:c.dist,solref:Array.from(c.solref)});
  }
  return {points,peak,penetration};
 }
 state(ref,qDes){
  const {m,d,poke}=this;
  let ghost=null,torqueSat=0;
  if(this.ghost&&qDes){
   ref.qpos.set(d.qpos);ref.qpos.set(qDes);this.mj.mj_kinematics(m,ref);
   ghost=copyGeoms(ref,this.ghostFrame??=geomSnapshot(m.ngeom));
  }
  for(let i=0;i<6;i++){
   const force=d.actuator_force[i],cap=m.actuator_forcerange[2*i+(force<0?0:1)];
   torqueSat=Math.max(torqueSat,100*Math.abs(force/cap));
  }
  const body=poke?this.bodies[poke.target]:0;
  return {ghost,torqueSat,contacts:this.contacts(),
   obstacle:{on:this.on,shape:this.shape,material:this.material,position:this.pos,physicalPosition:Array.from(d.mocap_pos.slice(this.mocap,this.mocap+3)),moveId:this.moveId,builds:this.builds},
   poke:poke?{...poke,steps:POKE_STEPS-poke.remaining,position:Array.from(d.xipos.slice(3*body,3*body+3)),applied:Array.from(d.xfrc_applied.slice(6*body,6*body+6))}:null};
 }
}
