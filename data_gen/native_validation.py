"""Fail-closed numerical admission; reject bad native candidate, keep producer alive."""
import numpy as np
def validate(t,q,dq,ddq,lo,hi):
 out={}
 if not all(np.isfinite(x).all() for x in [t,q,dq,ddq]):out['nonfinite']=True
 if np.any(np.diff(t)<=0):out['nonmonotonic_clock']=True
 lower=np.maximum(lo-q,0);upper=np.maximum(q-hi,0)
 if np.any(lower>1e-6) or np.any(upper>1e-6):out.update(joint_limit=True,lower_max_rad=float(lower.max()),upper_max_rad=float(upper.max()),lower_indices=np.argwhere(lower>1e-6).tolist(),upper_indices=np.argwhere(upper>1e-6).tolist())
 return out
if __name__=='__main__':
 t=np.array([0.,1.]);q=np.zeros((2,6));lo=np.full(6,-1.);hi=np.ones(6);z=q.copy()
 assert validate(t,q,z,z,lo,hi)=={}
 for bad in ['finite','clock','low','high']:
  bt=t.copy();bq=q.copy()
  if bad=='finite':bq[1,3]=np.nan
  if bad=='clock':bt[1]=0
  if bad=='low':bq[1,2]=-1.00001
  if bad=='high':bq[1,4]=1.00001
  assert validate(bt,bq,z,z,lo,hi),bad
 print('PASS finite/clock/lower/upper rejection, unchanged1e-6 threshold; valid candidate accepted')
