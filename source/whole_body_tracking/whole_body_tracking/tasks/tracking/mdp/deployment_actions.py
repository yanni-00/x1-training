"""Native Isaac Lab action adapter; raw PPO actions/history remain unchanged."""
import torch
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from whole_body_tracking.utils.deployment_limits import DeploymentTargetLimiter,load_contract

class DeployableJointPositionAction(JointPositionAction):
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        contract=load_contract()
        if list(self._joint_names)!=contract["joint_names"]:
            raise RuntimeError("Native/deploy action joint ordering differs")
        self._deployment_limiter=DeploymentTargetLimiter(contract,self.num_envs,self.device)
        data=self._asset.data
        for key,actual in [("kp",data.default_joint_stiffness),("kd",data.default_joint_damping),
                           ("effort_limit",data.joint_effort_limits)]:
            expected=torch.tensor(contract[key],device=self.device,dtype=actual.dtype)
            torch.testing.assert_close(actual[:,self._joint_ids],expected.expand(self.num_envs,-1),rtol=1e-5,atol=1e-5)
        limits=data.joint_pos_limits[:,self._joint_ids]
        for key,actual in [("joint_lower",limits[:,:,0]),("joint_upper",limits[:,:,1])]:
            expected=torch.tensor(contract[key],device=self.device,dtype=actual.dtype)
            torch.testing.assert_close(actual,expected.expand(self.num_envs,-1),rtol=1e-5,atol=1e-5)
    def process_actions(self,actions):
        super().process_actions(actions)
        q=self._asset.data.joint_pos[:,self._joint_ids];dq=self._asset.data.joint_vel[:,self._joint_ids]
        self._processed_actions.copy_(self._deployment_limiter.process(self._raw_actions,q,dq,self._processed_actions))
    def reset(self,env_ids=None):
        super().reset(env_ids)
        if hasattr(self,"_deployment_limiter"):self._deployment_limiter.reset(env_ids)
    def execution_snapshot(self):
        return self._deployment_limiter.execution_snapshot()
