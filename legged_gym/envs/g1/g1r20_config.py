"""R20 recovery trial: resume the first sustained-walking G1 checkpoint."""

import math
import os

from legged_gym.envs.g1.g1interrupt_config import G1InterruptCfg, G1InterruptCfgPPO


class G1R20Cfg(G1InterruptCfg):
    class rewards(G1InterruptCfg.rewards):
        # Gate source: hugwbc_g1 decision_log, R19 pre-registered |waist_pitch| <= 8 deg.
        waist_posture_tolerance_rad = math.radians(8.0)
        reward_curriculum_list = [
            name for name in G1InterruptCfg.rewards.reward_curriculum_list
            if name != "torso_deviation"
        ] + ["waist_posture_excess"]

        class scales(G1InterruptCfg.rewards.scales):
            # Restore the locomotion reward settings used for model_4800.
            tracking_lin_vel = 2.0
            orientation_control = -10.0
            shoulder_deviation = -1.0

            # Replace the always-on 3-joint torso penalty with an excess-only
            # roll/pitch penalty. The -15 scale is the pre-registered R19
            # failure branch for waist error remaining above 15 degrees.
            torso_deviation = 0.0
            waist_posture_excess = -15.0

            # Keep R19's anti-death guard while recovering the walking policy.
            termination = -1500.0

        # Preserve the G1-specific height target correction from the later runs.
        base_height_target = 0.72

    class commands(G1InterruptCfg.commands):
        class ranges(G1InterruptCfg.commands.ranges):
            # A zero-centered body-pitch command makes upright motion trainable.
            body_pitch = [-0.1, 0.1]


class G1R20CfgPPO(G1InterruptCfgPPO):
    class runner(G1InterruptCfgPPO.runner):
        experiment_name = "g1_r20_from4800_waist"
        resume = True
        resume_path = os.path.abspath(os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "..", "..", "resources", "checkpoints", "model_4800_g1int.pt",
        ))
        max_iterations = 3000
