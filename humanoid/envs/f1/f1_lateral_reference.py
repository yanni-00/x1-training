"""Add the phase-locked lateral hip-roll reference to the stage-one gait reference."""

from .f1_lateral_progress import F1LateralProgressCfg, F1LateralProgressEnv
from .f1_lateral_reference_profile import (
    LEFT_ANKLE_ROLL,
    LEFT_HIP_ROLL,
    RIGHT_ANKLE_ROLL,
    RIGHT_HIP_ROLL,
    lateral_hip_roll_offset,
)

DOF_ORDER = (
    (LEFT_HIP_ROLL, 'left_hip_roll_joint'),
    (LEFT_ANKLE_ROLL, 'left_ankle_roll_joint'),
    (RIGHT_HIP_ROLL, 'right_hip_roll_joint'),
    (RIGHT_ANKLE_ROLL, 'right_ankle_roll_joint'),
)


class F1LateralReferenceCfg(F1LateralProgressCfg):
    class env(F1LateralProgressCfg.env):
        reward_semantics = 'target_speed_ramp_with_lateral_progress_reference_v1'

    class rewards(F1LateralProgressCfg.rewards):
        # One-sided cos amplitude. At the measured steady touchdown phase (~0.07
        # of a cycle after the reference boundary) the stance-portion sweep is
        # 1.905x the amplitude, so 0.126 rad gives 0.240 rad of joint travel and
        # 0.133 m of foot travel (URDF FK at the default pose), matching the
        # offline candidate for 0.4 m/s at the 0.7 s cycle time.
        #
        # Full-reference FK tradeoff: the double-support ankle separation bottoms
        # at 0.180 m (phase 0.5), just under the 0.2 m foot-distance reward
        # reference. The reference weight (2.2) is an order above that soft term
        # (0.2), and the feet stay ~0.10 m apart edge-to-edge, so the geometric
        # amplitude is kept; the sim must confirm the narrow double support.
        lateral_reference_amplitude = 0.126
        lateral_reference_command_scale = 0.4  # lin_vel_y max, f1_config.py:302


class F1LateralReferenceEnv(F1LateralProgressEnv):
    def _init_buffers(self):
        super()._init_buffers()
        got = tuple(self.dof_names[i] for i, _ in DOF_ORDER)
        expected = tuple(name for _, name in DOF_ORDER)
        if got != expected:
            raise ValueError(f'Lateral reference requires the audited dof order, got {got}')

    def compute_ref_state(self):
        super().compute_ref_state()
        offset = lateral_hip_roll_offset(
            self._get_phase(),
            self.commands[:, 0],
            self.commands[:, 1],
            self.cfg.rewards.lateral_reference_amplitude,
            self.cfg.rewards.lateral_reference_command_scale,
            self.cfg.commands.stand_com_threshold,
        )
        # Same offset on both hip rolls: both axes are +x, so the feet move toward
        # the commanded side. Ankle roll cancels the induced foot roll so the sole
        # stays level at touchdown.
        self.ref_dof_pos[:, LEFT_HIP_ROLL] += offset
        self.ref_dof_pos[:, RIGHT_HIP_ROLL] -= offset
        self.ref_dof_pos[:, LEFT_ANKLE_ROLL] -= offset
        self.ref_dof_pos[:, RIGHT_ANKLE_ROLL] += offset
        # Keep the inherited 2x joint-deviation convention for reference actions.
        self.ref_action[:, LEFT_HIP_ROLL] += 2.0 * offset
        self.ref_action[:, RIGHT_HIP_ROLL] -= 2.0 * offset
        self.ref_action[:, LEFT_ANKLE_ROLL] -= 2.0 * offset
        self.ref_action[:, RIGHT_ANKLE_ROLL] += 2.0 * offset