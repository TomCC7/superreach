// Wall-clock slots are independent of the fixed simulation timestep.
export const MAX_LAG_MS=100;
export function nextDeadline(deadline, now, period){
 const next=deadline+period;
 // Keep the phase after small overruns; skip expired slots instead of replaying them.
 if(now-deadline>MAX_LAG_MS)return now+period;
 return next<now?next+Math.ceil((now-next)/period)*period:next;
}

const CAPACITY=1200;
const FIELDS=[['q_des',6],['q',6],['dq',6],['tcp',3],['goal',7],['goal_qref',6],['scored_goal',7]];
// Keep exact numeric snapshots, but materialize rows/arrays only on an explicit request.
export class TraceRing {
 constructor(width){this.width=width;this.stride=width+FIELDS.reduce((n,[,size])=>n+size,0)+4;this.data=new Float64Array(CAPACITY*this.stride);this.count=0;}
 push(tick,mode,obs,command,q,dq,tcp,goal,qref,scored,infer,step){
  const a=this.data;let i=(this.count++%CAPACITY)*this.stride;
  a[i++]=tick;a[i++]=mode;a[i++]=infer;a[i++]=step;a.set(obs,i);i+=this.width;
  a.set(command,i);i+=6;
  for(let k=0;k<6;k++)a[i++]=q[k];
  for(let k=0;k<6;k++)a[i++]=dq[k];
  for(let k=0;k<3;k++)a[i++]=tcp[k];
  a.set(goal,i);i+=7;a.set(qref,i);i+=6;a.set(scored,i);
 }
 snapshot(checkpoint){
  const a=this.data,out=[];
  for(let row=Math.max(0,this.count-CAPACITY);row<this.count;row++){
   let i=(row%CAPACITY)*this.stride;
   const entry={tick:a[i++],checkpoint,mode:a[i++],infer_ms:a[i++],step_ms:a[i++]};
   entry.obs=Array.from(a.subarray(i,i+=this.width));
   for(const [name,size] of FIELDS)entry[name]=Array.from(a.subarray(i,i+=size));
   out.push(entry);
  }
  return out;
 }
}

export function geomSnapshot(ngeom){
 const packed=new Float32Array(12*ngeom);
 return {positions:packed.subarray(0,3*ngeom),rotations:packed.subarray(3*ngeom)};
}
export function copyGeoms(data,out){
 out.positions.set(data.geom_xpos);out.rotations.set(data.geom_xmat);
 return out;
}
