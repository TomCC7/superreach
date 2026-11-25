"""Stationary reference episodes without changing state/replay/WM dimensions.

Normal clips are <2048 ticks. Hold starts occupy the middle of a 4096-tick
bucket, staying exactly representable in float32 throughout the 180-tick clip.
Reference IDs remain unchanged for coverage accounting. Explicit DEV resets never
sample the hold mixture; encoded explicit starts are reserved for hold evaluation.
"""
import torch
HOLD_OFFSET=2048
HOLD_STRIDE=4096
HOLD_PROBABILITY=.20

def encode(ticks):return ticks.long()*HOLD_STRIDE+HOLD_OFFSET

def decode(ticks):
    ticks=ticks.long()
    mask=ticks>=HOLD_OFFSET
    return mask,torch.where(mask,(ticks-HOLD_OFFSET)//HOLD_STRIDE,ticks)

def install(o,recompile):
    original=o.ReferenceBank.lookup
    init=o.ReferenceBank.__init__
    def checked_init(self,device):
        init(self,device)
        assert int(self.lengths.max())<HOLD_OFFSET,'normal bank collides with hold encoding'
        assert (int(self.lengths.max())-181)*HOLD_STRIDE+HOLD_OFFSET+180<2**24,'float32 aux loses ticks'
    def lookup(self,ids,ticks):
        mask,real=decode(ticks)
        out=original(self,ids,real)
        # No mutation of the immutable reference table; normal rows bit-identical.
        return torch.cat((out[...,:6],torch.where(mask[...,None],torch.zeros_like(out[...,6:12]),out[...,6:12]),out[...,12:]),-1)
    o.ReferenceBank.__init__=checked_init;o.ReferenceBank.lookup=lookup
    reset=o.OpenYAMEnv.reset
    namespace=reset.__globals__
    namespace.update(hold_encode=encode,hold_decode=decode,HOLD_PROBABILITY=HOLD_PROBABILITY)
    o.OpenYAMEnv.reset=recompile(reset,[
        (' if not len(ids):return',' if not len(ids):return\n training_reset=ref_ids is None'),
        (' assert torch.all(start_ticks>=0) and torch.all(start_ticks+180<self.bank.lengths[ref_ids])',
         ' _,decoded_ticks=hold_decode(start_ticks)\n assert torch.all(decoded_ticks>=0) and torch.all(decoded_ticks+180<self.bank.lengths[ref_ids])\n if training_reset:\n  hold_mask=torch.as_tensor(self.goal_rng.random(len(ids))<HOLD_PROBABILITY,device=self.device)\n  start_ticks=torch.where(hold_mask,hold_encode(start_ticks),start_ticks)')],namespace,'hold_reset')
    return original
