"""R28: one bounded continuation of the R27 posture stage, same objective."""
import os
from legged_gym.envs.g1.g1r27_config import G1R27Cfg, G1R27CfgPPO


def _r27_terminal_penalty_scale():
    # Source replay and completed update count are archived in R28's
    # curriculum-reconstruction.json. Only this scalar state is restored.
    scale = G1R27Cfg.rewards.curriculum_init
    for _ in range(G1R27CfgPPO.runner.max_iterations // 100):
        scale = pow(scale, G1R27Cfg.rewards.penalize_curriculum_sigma)
    return scale


class G1R28Cfg(G1R27Cfg):
    class rewards(G1R27Cfg.rewards):
        curriculum_init = _r27_terminal_penalty_scale()


class G1R28CfgPPO(G1R27CfgPPO):
    class runner(G1R27CfgPPO.runner):
        experiment_name = 'g1_r28_bounded_posture_continuation'
        resume_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', '..', '..', 'resources',
            'checkpoints', 'model_11400_g1r27.pt'))
        max_iterations = 600
