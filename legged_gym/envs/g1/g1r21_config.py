"""R21: repair G1 translation and train the upright-walk command stage.

All reward functions and coefficients are inherited from R20 except for joint
selection repairs in G1R21Robot. No new reward is introduced by this trial.
"""

import os

from legged_gym.envs.g1.g1r20_config import G1R20Cfg, G1R20CfgPPO


class G1R21Cfg(G1R20Cfg):
    class commands(G1R20Cfg.commands):
        class ranges(G1R20Cfg.commands.ranges):
            # Zero is the already-supported nominal-height/upright command.
            body_height = [0.0, 0.0]
            body_pitch = [0.0, 0.0]
            waist_roll = [0.0, 0.0]  # legacy spelling; command means waist yaw
            # Use the existing range minima validated in the controlled rollout.
            gait_frequency = [G1R20Cfg.commands.ranges.gait_frequency[0]] * 2
            foot_swing_height = [G1R20Cfg.commands.ranges.foot_swing_height[0]] * 2

    class disturb(G1R20Cfg.disturb):
        # Intervention training resumes after nominal upright walking is usable.
        use_disturb = False


class G1R21CfgPPO(G1R20CfgPPO):
    class runner(G1R20CfgPPO.runner):
        experiment_name = 'g1_r21_upright_translation'
        resume = True
        resume_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', '..', '..', 'resources',
            'checkpoints', 'model_7800_g1int.pt'))
        # Pre-registered first probe uses the task card's original 600-iter size;
        # this is a design choice, not a claim about required convergence time.
        max_iterations = 600
