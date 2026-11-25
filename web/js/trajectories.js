// Closed periodic joint references stay inside the policy's verified envelope.
// FK maps them to dynamically consistent SE(3) goals (not Cartesian IK samples).
export const PERIOD=8;
const profiles={
 reach:{a:[.025,-.13,.17,.035,-.045,.025],b:[0,.025,.015,-.02,.015,0],n:[1,1,1,1,1,1]},
 sweep:{a:[.23,.025,-.035,.04,.065,.04],b:[0,.045,.035,.015,.02,.015],n:[1,2,2,1,2,1]},
 orbit:{a:[.17,-.09,.11,.05,-.035,.04],b:[0,.065,.055,.025,.025,.02],n:[1,1,1,1,1,1]}
};
export function periodicReference(name,t,home){
 const p=profiles[name];if(!p)throw Error('Unknown reference '+name);
 const w=2*Math.PI/PERIOD;
 return {q:home.map((h,i)=>h+p.a[i]*Math.sin(p.n[i]*w*t)+p.b[i]*(1-Math.cos(p.n[i]*w*t))),dq:home.map((_,i)=>p.n[i]*w*(p.a[i]*Math.cos(p.n[i]*w*t)+p.b[i]*Math.sin(p.n[i]*w*t)))};
}
