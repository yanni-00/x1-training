"""Bounded adaptation from R41 terminal; accepted arms and body rewards fixed."""
import os
from legged_gym.envs.g1.g1_config import G1Cfg
from legged_gym.envs.g1.g1r41_config import G1R41Cfg,G1R41CfgPPO


def _r41_terminal_penalty_scale():
    scale=G1R41Cfg.rewards.curriculum_init
    for _ in range(G1R41CfgPPO.runner.max_iterations//100):
        scale=pow(scale,G1R41Cfg.rewards.penalize_curriculum_sigma)
    return scale


class G1R42Cfg(G1R41Cfg):
    class control(G1R41Cfg.control):
        stiffness=dict(G1R41Cfg.control.stiffness)
        damping=dict(G1R41Cfg.control.damping)
        for _joint in ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow'):
            stiffness[_joint]=G1Cfg.control.stiffness[_joint]
            damping[_joint]=G1Cfg.control.damping[_joint]
        del _joint

    class rewards(G1R41Cfg.rewards):
        curriculum_init=_r41_terminal_penalty_scale()


class G1R42CfgPPO(G1R41CfgPPO):
    class runner(G1R41CfgPPO.runner):
        experiment_name='g1_r42_fixed_reference_arms_adaptation'
        resume_path=os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..','..',
            'resources','checkpoints','model_12600_g1r41.pt'))
