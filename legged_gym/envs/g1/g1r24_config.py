"""R24: longer optimization of the unchanged R23 hanging-arm stage."""
import os

from legged_gym.envs.g1.g1r23_config import G1R23Cfg, G1R23CfgPPO


class G1R24Cfg(G1R23Cfg):
    pass


class G1R24CfgPPO(G1R23CfgPPO):
    class runner(G1R23CfgPPO.runner):
        experiment_name = 'g1_r24_arm_reference_consolidation'
        resume = True
        resume_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', '..', '..', 'resources',
            'checkpoints', 'model_9600_g1r23.pt'))
        # Preregistered probe length (2 x prior 600), not a convergence estimate.
        max_iterations = 1200
