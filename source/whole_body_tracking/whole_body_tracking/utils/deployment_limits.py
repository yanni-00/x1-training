"""Exact vectorized counterpart of published deployment Core::targets."""
import hashlib,json
from pathlib import Path
import torch

CONTRACT_PATH=Path(__file__).resolve().parents[4]/"configs/deployment_target_contract.json"

def load_contract():
    c=json.loads(CONTRACT_PATH.read_text())
    if c["version"]!="shared_cpp_target_limits_v1":raise ValueError("Unsupported target contract")
    return c

class DeploymentTargetLimiter:
    def __init__(self,contract,num_envs,device="cpu"):
        self.contract=contract;self.num_envs=num_envs
        for key in ["joint_lower","joint_upper","kp","kd","effort_limit","default_joint_pos","action_scale"]:
            value=torch.tensor(contract[key],device=device,dtype=torch.float32)
            if value.shape!=(29,) or not torch.isfinite(value).all():raise ValueError("Invalid target contract "+key)
            setattr(self,key,value)
        if not ((self.joint_lower+contract["joint_margin"]<self.joint_upper-contract["joint_margin"]).all() and
                (self.kp>0).all() and (self.effort_limit>0).all()):raise ValueError("Invalid limit intervals")
        self.last_target=torch.zeros((num_envs,29),device=device)
        self.needs_reset=torch.ones(num_envs,device=device,dtype=torch.bool)
        self.changed_total=torch.zeros((),device=device,dtype=torch.int64)
        self.process_calls=0
    def reset(self,env_ids=None):
        self.needs_reset[slice(None)if env_ids is None else env_ids]=True
    def process(self,raw,q,dq,desired=None):
        if raw.shape!=self.last_target.shape or q.shape!=raw.shape or dq.shape!=raw.shape:
            raise ValueError("Limiter batch shape mismatch")
        if not (torch.isfinite(raw).all() and torch.isfinite(q).all() and torch.isfinite(dq).all()):
            raise ValueError("Non-finite action/sensor state")
        if (raw.abs()>self.contract["max_abs_raw_action"]).any():raise ValueError("Invalid raw policy action")
        self.last_target.copy_(torch.where(self.needs_reset[:,None],q,self.last_target))
        self.needs_reset.fill_(False)
        if desired is None:desired=self.default_joint_pos+self.action_scale*raw
        if not torch.isfinite(desired).all():raise ValueError("Non-finite affine action")
        lo=self.joint_lower+self.contract["joint_margin"];hi=self.joint_upper-self.contract["joint_margin"]
        limited=torch.maximum(torch.minimum(desired,hi),lo)
        delta=self.contract["max_target_speed"]*self.contract["policy_dt"]
        limited=torch.maximum(torch.minimum(limited,self.last_target+delta),self.last_target-delta)
        center=q+self.kd*dq/self.kp;radius=self.effort_limit/self.kp
        limited=torch.maximum(torch.minimum(limited,center+radius),center-radius)
        limited=torch.maximum(torch.minimum(limited,hi),lo)
        self.changed_total+=((desired-limited).abs()>self.contract["changed_tolerance"]).sum()
        self.last_target.copy_(limited);self.process_calls+=1
        return limited
    def execution_snapshot(self):
        return {"process_calls":self.process_calls,"changed_joint_commands":int(self.changed_total.item()),
                "processed_targets_finite":bool(torch.isfinite(self.last_target).all()),
                "reset_pending_envs":int(self.needs_reset.sum().item())}

def audit_deployment_action(env):
    base=env.unwrapped
    if not getattr(base.cfg,"oma_deployment_target_limits",False):return None
    action=base.action_manager.get_term("joint_pos")
    if type(action).__name__!="DeployableJointPositionAction":raise RuntimeError("Wrong native target action")
    contract=action._deployment_limiter.contract
    if abs(base.cfg.sim.dt*base.cfg.decimation-contract["policy_dt"])>1e-8:
        raise RuntimeError("Native/deploy policy clock differs")
    package=Path(__file__).resolve().parents[1]
    files=["utils/deployment_limits.py","tasks/tracking/mdp/deployment_actions.py",
           "tasks/tracking/config/g1/deploy_proprio_env_cfg.py"]
    return {**contract,"contract_sha256":hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest(),
            "implementation_sha256":{name:hashlib.sha256((package/name).read_text().encode()).hexdigest()for name in files},
            "native_action_class":type(action).__name__,"applied_to_native_action":True}
