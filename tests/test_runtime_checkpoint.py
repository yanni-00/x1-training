"""Parity validation must not change a live training policy or CPU RNG."""
import importlib.util
import tempfile
from pathlib import Path
import torch
from torch import nn

path=Path(__file__).resolve().parents[1]/'source/whole_body_tracking/whole_body_tracking/utils/runtime_policy.py'
spec=importlib.util.spec_from_file_location('runtime_policy',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class UpdatingNormalizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('_mean',torch.zeros(1,160))
        self.register_buffer('_std',torch.ones(1,160))
        self.register_buffer('count',torch.tensor(0))
    def forward(self,x):
        if self.training:
            self.count+=len(x)
            self._mean.add_(.1)
        return (x-self._mean)/(self._std+.01)

policy=nn.Module()
policy.actor=nn.Sequential(nn.Linear(160,8),nn.ELU(),nn.Linear(8,29))
policy.actor_obs_normalizer=UpdatingNormalizer()
policy.train()
policy.actor[1].eval()  # Preserve mixed module modes as well as root mode.
before={k:v.clone() for k,v in policy.state_dict().items()}
modes=[m.training for m in policy.modules()]
flags=(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
with tempfile.TemporaryDirectory() as d:
    checkpoint=Path(d)/'model.pt'
    torch.save({'model_state_dict':before,'infos':{'oma_runtime':{
        'format':'beyondmimic_g1_runtime_v1','actor_obs_normalization':True,
        'normalizer_epsilon':.01}}},checkpoint)
    rng=torch.random.get_rng_state().clone()
    result=module.verify_runtime_checkpoint(policy,checkpoint)
    assert torch.equal(rng,torch.random.get_rng_state()),'Parity check consumed training RNG'
assert all(torch.equal(before[k],v) for k,v in policy.state_dict().items()),'Parity check mutated training statistics'
assert modes==[m.training for m in policy.modules()],'Training modes were not restored'
assert flags==(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32),'Training precision was not restored'
print(result)
