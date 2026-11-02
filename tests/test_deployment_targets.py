"""Behavioral checks for exact target processing, batching and provenance guards."""
import importlib.util,json,copy
from pathlib import Path
import torch,hashlib
root=Path(__file__).resolve().parents[1];pkg=root/"source/whole_body_tracking/whole_body_tracking"
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
m=load("deployment_limits",pkg/"utils/deployment_limits.py");c=m.load_contract()
q=torch.tensor(c["default_joint_pos"]).repeat(2,1);dq=torch.zeros_like(q);a=torch.ones_like(q)*2
b=m.DeploymentTargetLimiter(c,2);one=[m.DeploymentTargetLimiter(c,1)for _ in range(2)]
for step in range(8):
 if step==4:b.reset(torch.tensor([1]));one[1].reset()
 q=q+.03;dq=dq+.5
 actual=b.process(a,q,dq);expected=torch.cat([one[i].process(a[i:i+1],q[i:i+1],dq[i:i+1])for i in range(2)])
 torch.testing.assert_close(actual,expected,rtol=0,atol=0)
assert b.execution_snapshot()["process_calls"]==8 and b.execution_snapshot()["changed_joint_commands"]>0
for bad in [float("nan"),101.]:
 invalid=a.clone();invalid[0,0]=bad
 try:b.process(invalid,q,dq);raise AssertionError("Accepted invalid policy action")
 except ValueError:pass
bad=copy.deepcopy(c);bad["kp"][0]=0
try:m.DeploymentTargetLimiter(bad,2);raise AssertionError("Accepted bad gains")
except ValueError:pass
bootstrap=load("oma_bootstrap",root/"scripts/oma_bootstrap.py")
# Standalone UNIT fixture only: no archived model dependency or behavior claim.
import tempfile,numpy as np
tmp=tempfile.TemporaryDirectory();reference=Path(tmp.name)/"reference.npz"
names=[f"joint_{i}"for i in range(29)]
np.savez(reference,joint_names=np.array(names),body_names=np.array(["pelvis"]))
meta={"format":"beyondmimic_g1_velocity_proprio_runtime_v1","reference_sha256":hashlib.sha256(reference.read_bytes()).hexdigest(),
      "joint_names":names,"body_names":["pelvis"],"tracked_body_names":["pelvis"],"anchor_body_name":"pelvis",
      "physics_dt":.005,"decimation":4,"policy_dt":c["policy_dt"],"phase_acceleration":2.,
      "normalizer_epsilon":.01,"actor_obs_normalization":True,"reward_parameters":{},
      "command_limits":[[0,1],[0,0],[-.2,.2]],"control":{k:torch.tensor(c[k])for k in ["kp","kd","effort_limit","default_joint_pos","action_scale"]},
      "actor_observation_contract":{"dimension":154,"requires_position_estimate":False,"requires_linear_velocity_estimate":False}}
parent={"iter":3,"infos":{"oma_runtime":meta},"model_state_dict":{"actor.0.weight":torch.zeros(8,154),
    "critic.0.weight":torch.zeros(8,286),"actor_obs_normalizer._mean":torch.zeros(1,154),
    "critic_obs_normalizer._mean":torch.zeros(1,286),"actor_obs_normalizer.count":torch.tensor(64.)},
    "optimizer_state_dict":{"state":{0:{"step":torch.tensor(20.)}}}}
bad_parent=copy.deepcopy(parent);bad_parent["infos"]["oma_runtime"]["reference_sha256"]="0"*64
try:bootstrap.validate_velocity_resume(bad_parent,reference,154);raise AssertionError("Accepted different reference bytes")
except RuntimeError:pass
provenance=bootstrap.validate_velocity_resume(parent,reference,154)
assert provenance["architecture"]==[154,286,29]
try:bootstrap.validate_velocity_resume(parent,reference,160);raise AssertionError("Accepted154 as160")
except RuntimeError:pass
check=load("oma_checkpoint",pkg/"utils/oma_checkpoint.py")
before=copy.deepcopy(parent["infos"]["oma_runtime"]);after=copy.deepcopy(before)
after["deployment_target_contract"]={**c,"applied_to_native_action":True}
try:check.validate_runtime_continuation(before,after);raise AssertionError("Accepted unregistered control delta")
except RuntimeError:pass
check.validate_runtime_continuation(before,after,allowed_target_delta="shared_cpp_target_limits_v1")
after["control"]["kp"][0]+=1
try:check.validate_runtime_continuation(before,after,allowed_target_delta="shared_cpp_target_limits_v1");raise AssertionError("Accepted gains change")
except RuntimeError:pass
print(json.dumps({"batch_and_selective_reset":"passed","nonfinite_action_and_gain_rejection":"passed","explicit154_resume_and160_rejection":"passed","target_delta_and_control_provenance_guards":"passed"}))
