// Causal 311D contract: 16 × [q6, dq6, previous_u6, valid] + [xyz3, quat_wxyz4].
export const DT = 0.02;
export const KP = [50,60,50,25,15,10], KD = [3,4,3,2,1,.8];
export const HOME = [0,1.047,1.047,0,0,0];
export const RADIUS = [.35,.25,.25,.25,.25,.25];
export const HISTORY_WIDTH = 16*19, OBS_WIDTH = HISTORY_WIDTH+7;
export class Policy {
  constructor(engine,tri=false) {
    this.engine=engine;this.tri=tri;this.width=OBS_WIDTH+(tri?7:0);
    this.obs=new Float32Array(this.width);
    this.history=new Float32Array(HISTORY_WIDTH);
  }
  reset(q,dq) {
    this.tick=0;
    for(let i=0;i<16;i++)this.writeFrame(i*19,q,dq,q,i===15?1:0);
  }
  writeFrame(offset,q,dq,q_des,valid) {
    for(let i=0;i<6;i++) {
      this.history[offset+i]=q[i];this.history[offset+6+i]=dq[i];this.history[offset+12+i]=q_des[i];
    }
    this.history[offset+18]=valid;
  }
  observation(goal,qref,mode) {
    if(goal.length!==7)throw Error('goal must be xyz3+quat_wxyz4');
    const obs=this.obs;obs.set(this.history);obs.set(goal,HISTORY_WIDTH);
    if(this.tri){
      if(qref.length!==6||(mode!==0&&mode!==1&&mode!==2))throw Error('Invalid tri-mode goal');
      if(mode===2)obs.set(qref,OBS_WIDTH);else obs.fill(0,OBS_WIDTH,317);
      obs[317]=mode;
    }
    return obs;
  }
  infer(obs) {return this.engine.infer(obs);}
  observe(q,dq,q_des) {
    this.tick++;this.history.copyWithin(0,19);
    this.writeFrame(HISTORY_WIDTH-19,q,dq,q_des,1);
  }
}
