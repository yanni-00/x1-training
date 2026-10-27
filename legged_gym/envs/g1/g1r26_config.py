"""R26 matched-input probe: restore the deterministic R24 penalty curriculum."""
from legged_gym.envs.g1.g1r24_config import G1R24Cfg, G1R24CfgPPO
from legged_gym.envs.g1.g1r25_config import G1R25Cfg, G1R25CfgPPO


def _r24_terminal_penalty_scale():
    # R24 completed its registered 1200 updates. H1.training_curriculum applies
    # pow(scale, sigma) every 100 local updates, after BaseTask increments once.
    # Exact source replay: exp-20261027-003/curriculum-reconstruction.json.
    scale = G1R24Cfg.rewards.curriculum_init
    for _ in range(G1R24CfgPPO.runner.max_iterations // 100):
        scale = pow(scale, G1R24Cfg.rewards.penalize_curriculum_sigma)
    return scale


class G1R26Cfg(G1R25Cfg):
    class rewards(G1R25Cfg.rewards):
        curriculum_init = _r24_terminal_penalty_scale()


class G1R26CfgPPO(G1R25CfgPPO):
    class runner(G1R25CfgPPO.runner):
        # Same input model_10800_g1r24.pt and 600 updates as R25, not model11400.
        experiment_name = 'g1_r26_restore_penalty_curriculum'
