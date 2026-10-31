"""Observable actor contracts, legacy compatibility, and fake-native leakage rejection."""
import importlib.util
from pathlib import Path
import tempfile
import types
import torch
from torch import nn
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
UTIL=ROOT/'source/whole_body_tracking/whole_body_tracking/utils'
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
obs=load('actor_observations',UTIL/'actor_observations.py')
runtime=load('runtime_policy',UTIL/'runtime_policy.py')
c=obs.contract(True)
assert c['dimension']==154 and not c['requires_position_estimate'] and not c['requires_linear_velocity_estimate']
cfg=types.SimpleNamespace(motion_anchor_pos_b=None,base_lin_vel=None)
manager=types.SimpleNamespace(active_terms={'policy':c['terms']},group_obs_dim={'policy':(154,),'critic':(286,)})
base=types.SimpleNamespace(oma_proprioceptive_actor=True)
env=types.SimpleNamespace(unwrapped=types.SimpleNamespace(cfg=types.SimpleNamespace(oma_proprioceptive_actor=True,
    observations=types.SimpleNamespace(policy=cfg)),observation_manager=manager))
assert obs.audit_native_observations(env)['passed']
manager.active_terms['policy']=c['terms']+['base_lin_vel']
try:obs.audit_native_observations(env);raise AssertionError('Leaked actor term accepted')
except RuntimeError:pass
manager.active_terms['policy']=c['terms']
rng=np.random.default_rng(17)
qref,dqref,q,dq,prev,default=[rng.normal(size=29) for _ in range(6)]
omega=rng.normal(size=3)
rotation=np.eye(3)
expected=obs.proprioceptive_observation(qref,dqref,rotation,omega,q,dq,prev,default)
# Actual root position and linear velocity are intentionally absent from this API.
import inspect
assert list(inspect.signature(obs.proprioceptive_observation).parameters)==[
    'q_ref','dq_ref','relative_rotation','omega','q','dq','last_action','default_q']
class Norm(nn.Module):
    def __init__(self,n):
        super().__init__();self.register_buffer('_mean',torch.zeros(1,n));self.register_buffer('_std',torch.ones(1,n))
    def forward(self,x):return (x-self._mean)/(self._std+.01)
for n,fmt in [(160,'beyondmimic_g1_velocity_runtime_v1'),(154,obs.PROPRIO_FORMAT)]:
    policy=nn.Module();policy.actor=nn.Sequential(nn.Linear(n,8),nn.ELU(),nn.Linear(8,29))
    policy.actor_obs_normalizer=Norm(n);policy.eval()
    meta={'format':fmt,'actor_obs_normalization':True,'normalizer_epsilon':.01,'actor_observation_contract':obs.contract(n==154)}
    checkpoint={'model_state_dict':policy.state_dict(),'infos':{'oma_runtime':meta}}
    with tempfile.TemporaryDirectory() as directory:
        path=Path(directory)/'model.pt';torch.save(checkpoint,path)
        result=runtime.verify_runtime_checkpoint(policy,path)
        assert result['max_abs_error']<=2e-4
    if n==154:
        checkpoint['infos']['oma_runtime']['actor_observation_contract']['requires_position_estimate']=True
        try:runtime.RuntimeActor(checkpoint);raise AssertionError('Bad154 metadata accepted')
        except ValueError:pass
print('PASS: native positive/negative contract, observable assembler invariance,154 and legacy160 actor parity')
