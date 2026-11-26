import * as THREE from 'three';
import {shapes,materials,obstacleHome} from './interaction_config.js?v=interact-v1';
const $=id=>document.getElementById(id);
export function interactiveView(scene,control,send,invalidate){
 const obstacle=new THREE.Group();obstacle.name='Obstacle';obstacle.position.fromArray(obstacleHome);scene.add(obstacle);
 const obstacleMaterial=new THREE.MeshStandardMaterial({transparent:true,depthWrite:false});
 const {wall,pillar,sphere}=shapes;
 const cylinder=new THREE.CylinderGeometry(pillar.size[0],pillar.size[0],2*pillar.size[1],32);cylinder.rotateX(Math.PI/2);
 const meshes=Object.fromEntries(Object.entries({wall:new THREE.BoxGeometry(...wall.size.map(x=>2*x)),pillar:cylinder,sphere:new THREE.SphereGeometry(sphere.size[0],32,20)}).map(([name,g])=>{
  const mesh=new THREE.Mesh(g,obstacleMaterial);mesh.name=name;obstacle.add(mesh);return [name,mesh];
 }));
 const ghost=new THREE.Group();ghost.name='FK(q_des)';scene.add(ghost);ghost.visible=false;
 const ghostMaterial=new THREE.MeshBasicMaterial({color:0x37c8ff,transparent:true,opacity:.28,depthWrite:false});
 const arrows=new THREE.Group();arrows.name='Obstacle contact forces';scene.add(arrows);
 const up=new THREE.Vector3(0,1,0),direction=new THREE.Vector3(),color=new THREE.Color();
 const newArrow=()=>new THREE.ArrowHelper(up,new THREE.Vector3(),.05,0xffaa22,.016,.009);
 const pokeArrow=newArrow();pokeArrow.name='Impulse poke';pokeArrow.visible=false;scene.add(pokeArrow);
 let frame=null,pending=null,seq=0,gesture=0,drags=0;
 function flush(final=false){
  if(frame!==null){cancelAnimationFrame(frame);frame=null;}
  obstacle.position.clampScalar(-2,2);syncCoordinates();
  pending=`obstacle-${++seq}`;
  send({type:'obstacleMove',id:pending,gesture,final,position:obstacle.position.toArray()});invalidate();
 }
 const coords=['obstacle-x','obstacle-y','obstacle-z'].map($);
 function syncCoordinates(){obstacle.position.toArray().forEach((x,i)=>{const value=x.toFixed(3);if(document.activeElement!==coords[i]&&coords[i].value!==value)coords[i].value=value;});}
 coords.forEach((input,i)=>input.onchange=()=>{if(Number.isFinite(input.valueAsNumber)){gesture++;obstacle.position.setComponent(i,input.valueAsNumber);flush(true);}input.value=obstacle.position.getComponent(i).toFixed(3);});
 control.attach(obstacle);
 control.addEventListener('dragging-changed',e=>{if(e.value){gesture++;drags++;}else flush(true);});
 control.addEventListener('objectChange',()=>{if(frame===null)frame=requestAnimationFrame(()=>flush());});
 function visual(on,shape,material){
  obstacle.visible=on;control.enabled=on;control.getHelper().visible=on;
  for(const [name,mesh] of Object.entries(meshes))mesh.visible=name===shape;
  const {color,opacity,roughness}=materials[material];obstacleMaterial.color.setHex(color);Object.assign(obstacleMaterial,{opacity,roughness});invalidate();
 }
 const on=$('obstacle-on'),shape=$('obstacle-shape'),material=$('obstacle-material'),ghostCheck=$('ghost');
 function configure(){
  const config={on:on.checked,shape:shape.value,material:material.value};
  visual(config.on,config.shape,config.material);send({type:'obstacleConfig',...config});
  for(const input of [shape,material,...coords])input.disabled=!config.on;
 }
 on.onchange=shape.onchange=material.onchange=configure;
 ghostCheck.onchange=()=>{if(!ghostCheck.checked){ghost.visible=false;invalidate();}send({type:'ghost',on:ghostCheck.checked});};
 const pokeButton=$('poke');
 pokeButton.onclick=()=>{pokeButton.disabled=true;send({type:'poke',target:$('poke-target').value,magnitude:Number($('poke-force').value)});};
 function updateArrow(arrow,pos,force,magnitude,hex){
  arrow.visible=magnitude>.01;if(!arrow.visible)return;
  const length=Math.min(.22,.027+.024*Math.sqrt(magnitude));
  arrow.position.fromArray(pos);arrow.setDirection(direction.fromArray(force).normalize());
  arrow.setLength(length,Math.min(.028,length*.35),.013);arrow.setColor(hex);
 }
 function apply(state){
  const {obstacle:physical,contacts,poke,running}=state;
  if(physical.moveId===pending)pending=null;
  if(!pending&&!control.dragging&&frame===null&&!physical.position.every((x,i)=>x===obstacle.position.getComponent(i))){obstacle.position.fromArray(physical.position);syncCoordinates();}
  for(let i=arrows.children.length;i<contacts.points.length;i++)arrows.add(newArrow());
  arrows.children.forEach((arrow,i)=>{
   const c=contacts.points[i];if(!c||!on.checked){arrow.visible=false;return;}
   color.setHSL(.15*(1-Math.min(1,c.magnitude/45)),.9,.45);updateArrow(arrow,c.pos,c.vector,c.magnitude,color);
  });
  ghost.visible=ghostCheck.checked&&!!state.ghost;
  pokeArrow.visible=false;if(poke?.remaining)updateArrow(pokeArrow,poke.position,poke.force,Math.hypot(...poke.force),0x037da5);
  pokeButton.disabled=!running||!!poke?.remaining;
  const status=poke?.remaining?(running?'Pushing…':'Push paused'):(running?'Along world +Y · 150 ms':'Resume to poke');
  if($('poke-status').textContent!==status)$('poke-status').textContent=status;
 }
 function updateHud(state){
  const {contacts}=state;
  const force=`${contacts.peak.toFixed(1)} N`,sat=`${state.torqueSat.toFixed(0)}%`;
  if($('contact-force').textContent!==force)$('contact-force').textContent=force;
  if($('torque-sat').textContent!==sat)$('torque-sat').textContent=sat;
 }
 visual(false,shape.value,material.value);syncCoordinates();
 function reset(home){
  if(frame!==null)cancelAnimationFrame(frame);
  frame=null;pending=null;on.checked=ghostCheck.checked=false;
  shape.value='wall';material.value='rigid';$('poke-target').value='tcp';$('poke-force').value='30';
  for(const input of [on,shape,material,ghostCheck,pokeButton,...coords])input.disabled=true;
  obstacle.position.fromArray(home);syncCoordinates();visual(false,'wall','rigid');
  ghost.visible=pokeArrow.visible=false;arrows.children.forEach(a=>a.visible=false);
 }
 return {reset,obstacle,arrows,pokeArrow,ghost,ghostMaterial,apply,updateHud,enable(){on.disabled=ghostCheck.disabled=false;},get drags(){return drags;}};
}
