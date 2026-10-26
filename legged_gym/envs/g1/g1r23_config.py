"""R23: translate the arm-rest reference, without changing reward weights."""
import math
import os

from legged_gym.envs.g1.g1r22_config import G1R22Cfg, G1R22CfgPPO


class G1R23Cfg(G1R22Cfg):
    class rewards(G1R22Cfg.rewards):
        # G1 URDF: elbow -> rubber-hand origin at q_elbow=0 has
        # x=.100+.038+.046+.0415, z=-.010, and elbow axis is +Y.
        # Rotate that vector's sagittal projection toward -Z. At q=0 the
        # forearm points forward; zero is not G1's hanging-arm reference.
        # Reproduction: .oma/experiments/exp-20261026-004/arm-reference.json.
        arm_rest_elbow_rad = math.atan2(0.100 + 0.038 + 0.046 + 0.0415, 0.010)


class G1R23CfgPPO(G1R22CfgPPO):
    class runner(G1R22CfgPPO.runner):
        experiment_name = 'g1_r23_hanging_arm_reference'
        resume = True
        resume_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', '..', '..', 'resources',
            'checkpoints', 'model_9000_g1r22.pt'))
