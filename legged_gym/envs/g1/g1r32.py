"""Map the existing body-pitch cost to G1's actual articulated torso."""
import torch
from isaacgym.torch_utils import quat_from_angle_axis, quat_rotate_inverse
from legged_gym.envs.g1.g1r29 import G1R29Robot


class G1R32Robot(G1R29Robot):
    def _create_envs(self):
        super()._create_envs()
        name=self.cfg.rewards.orientation_control_body
        self.orientation_body_index=self.gym.find_actor_rigid_body_handle(
            self.envs[0],self.actor_handles[0],name)
        if self.orientation_body_index < 0:
            raise RuntimeError('Required orientation-control body not found: '+name)

    def _reward_orientation_control(self):
        # H1's root-based component leaves G1's articulated waist unmeasured.
        # Preserve its pitch target, xy gravity error and external scale; only
        # select the named torso's world quaternion as the measured body.
        torso_quat=self.rigid_body_states[:,self.orientation_body_index,3:7]
        projected=quat_rotate_inverse(torso_quat,self.gravity_vec)
        target=quat_from_angle_axis(self.commands[:,self.command_body_pitch_dim],
            torch.tensor([0,1,0],device=self.device,dtype=torch.float))
        desired=quat_rotate_inverse(target,self.gravity_vec)
        return torch.square(projected[:,:2]-desired[:,:2]).sum(dim=1)
