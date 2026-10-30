"""Explicit reference arms with learned leg/waist actions, during train and play."""
import math
import torch
from legged_gym.envs.g1.g1r41 import G1R41Robot


class G1R42Robot(G1R41Robot):
    def _create_envs(self):
        assert not self.cfg.disturb.use_disturb, 'Reference arms must not activate legacy masks'
        super()._create_envs()
        self.reference_arm_indices = [self.dof_names.index(side+'_'+joint+'_joint')
            for side in ('left','right') for joint in ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow')]

    def arm_reference_target(self):
        # [OMA] design-hugwbc-g1-r42: exactly the accepted reference execution.
        # Construction performs one inherited step before creating gait clocks,
        # matching the original diagnostic; resets after initialization use them.
        if not hasattr(self, 'foot_indices'):
            return None
        r = self.cfg.rewards
        phase = self.foot_indices[:, [1,0]]
        swing = -r.arm_swing_pitch_amplitude * torch.cos(2*math.pi*phase-r.arm_swing_phase_offset)
        swing *= (self.commands[:,0]/r.arm_swing_speed_reference).clamp(-1,1).unsqueeze(1)
        swing *= (~self.standing_envs_mask).unsqueeze(1)
        target = self.default_dof_pos[:,self.reference_arm_indices].expand(self.num_envs,-1).clone()
        target[:,0] += r.arm_rest_shoulder_pitch_rad+swing[:,0]
        target[:,4] += r.arm_rest_shoulder_pitch_rad+swing[:,1]
        target[:,1] += r.arm_rest_shoulder_roll_rad
        target[:,5] -= r.arm_rest_shoulder_roll_rad
        target[:,3] += r.arm_rest_elbow_rad
        target[:,7] += r.arm_rest_elbow_rad
        return target

    def calculate_action(self, actions):
        target = self.arm_reference_target()
        if target is None:
            return super().calculate_action(actions)
        hybrid = actions.clone()
        hybrid[:,self.reference_arm_indices] = (
            target-self.default_dof_pos[:,self.reference_arm_indices])/self.cfg.control.action_scale
        # Preserve accepted hybrid history and existing clipping/PD/torque limits.
        # Actor retains 23 outputs for checkpoint compatibility; 8 are overridden.
        return super().calculate_action(hybrid)

    def reference_controller_metadata(self):
        r=self.cfg.rewards
        return {'type':'phase_reference_arms','active':True,'actor_output_dimension':self.num_actions,
            'policy_joint_count':self.num_actions-len(self.reference_arm_indices),'reference_joint_count':len(self.reference_arm_indices),
            'command_interrupt_flag':0,'history':'hybrid_submitted_actions','legacy_disturbances_enabled':bool(self.use_disturb),
            'arm_pd':'g1_asset','reference_pose':{'pitch_bias_rad':r.arm_rest_shoulder_pitch_rad,
                'elbow_rad':r.arm_rest_elbow_rad,'roll_magnitude_rad':r.arm_rest_shoulder_roll_rad},
            'amplitude':r.arm_swing_pitch_amplitude,'phase_offset':r.arm_swing_phase_offset}
