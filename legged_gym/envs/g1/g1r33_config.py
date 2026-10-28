"""R33: one bounded same-objective continuation of the true R32 terminal."""
import os
from legged_gym.envs.g1.g1r32_config import G1R32Cfg,G1R32CfgPPO


def _r32_terminal_penalty_scale():
    # Archive: R33 curriculum-reconstruction.json. Only this deterministic
    # environment scalar is reconstructed, not terrain/command/per-env states.
    scale=G1R32Cfg.rewards.curriculum_init
    for _ in range(G1R32CfgPPO.runner.max_iterations//100):
        scale=pow(scale,G1R32Cfg.rewards.penalize_curriculum_sigma)
    return scale


class G1R33Cfg(G1R32Cfg):
    class rewards(G1R32Cfg.rewards):
        curriculum_init=_r32_terminal_penalty_scale()


class G1R33CfgPPO(G1R32CfgPPO):
    class runner(G1R32CfgPPO.runner):
        experiment_name='g1_r33_bounded_torso_arm_continuation'
        resume_path=os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..','..',
            'resources','checkpoints','model_12000_g1r32.pt'))
