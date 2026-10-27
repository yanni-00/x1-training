"""R29 arm-coordination probe matched to R28's input/duration/curriculum."""
from legged_gym.envs.g1.g1r28_config import G1R28Cfg, G1R28CfgPPO


class G1R29Cfg(G1R28Cfg):
    class rewards(G1R28Cfg.rewards):
        # Exact source: R27 arm-swing-reference-probe.json. The amplitude rule
        # and weight are design choices, not estimates of human norms.
        arm_swing_pitch_amplitude = 0.09178921571001411
        arm_swing_phase_offset = 0.30418359554562746
        arm_swing_speed_reference = 0.5  # recorded fixed upright command
        arm_swing_pitch_weight = 2.0


class G1R29CfgPPO(G1R28CfgPPO):
    class runner(G1R28CfgPPO.runner):
        # Same original R27 checkpoint11400,600 updates and curriculum as R28.
        experiment_name = 'g1_r29_contralateral_arm_swing'
