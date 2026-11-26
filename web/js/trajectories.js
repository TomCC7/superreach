// Closed periodic joint references stay inside the policy's verified envelope.
// FK maps them to dynamically consistent SE(3) goals (not Cartesian IK samples).
export const PERIOD=8;
export function periodicReference(name,t,{home,profiles}){
 const p=profiles[name];if(!p)throw Error('Unknown reference '+name);
 const w=2*Math.PI/PERIOD;
 return {q:home.map((h,i)=>h+p.a[i]*Math.sin(p.n[i]*w*t)+p.b[i]*(1-Math.cos(p.n[i]*w*t))),dq:home.map((_,i)=>p.n[i]*w*(p.a[i]*Math.cos(p.n[i]*w*t)+p.b[i]*Math.sin(p.n[i]*w*t)))};
}
