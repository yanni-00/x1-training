"""Select the worse pelvis/torso error in the single existing orientation cost."""
import torch
from isaacgym.torch_utils import quat_from_angle_axis, quat_rotate_inverse
from legged_gym.envs.g1.g1r39 import G1R39Robot


class G1R41Robot(G1R39Robot):
    def _reward_orientation_control(self):
        # [OMA] design-hugwbc-g1-r41: explicit measurement adaptation for G1's
        # articulated waist. This changes torso-only cost semantics; it is not
        # a sum of extra rewards or a scale change. Control/actions stay learned.
        torso_error = super()._reward_orientation_control()
        if not self.cfg.rewards.orientation_control_include_pelvis:
            return torso_error
        target = quat_from_angle_axis(self.commands[:, self.command_body_pitch_dim],
            torch.tensor([0, 1, 0], device=self.device, dtype=torch.float))
        desired = quat_rotate_inverse(target, self.gravity_vec)
        projected = quat_rotate_inverse(self.root_states[:, 3:7], self.gravity_vec)
        pelvis_error = torch.square(projected[:, :2] - desired[:, :2]).sum(dim=1)
        return torch.maximum(torso_error, pelvis_error)
