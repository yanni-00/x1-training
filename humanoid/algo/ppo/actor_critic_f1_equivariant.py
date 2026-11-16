"""F1 mirror-equivariant DH actor, identical during PPO and inference.

The audited F1 observation/URDF mapping is shared by both branches. This is a
trainable policy class, not an evaluation wrapper around a frozen checkpoint.
"""
import torch
from torch.distributions import Normal
from .actor_critic_dh import ActorCriticDH


class ActorCriticF1Equivariant(ActorCriticDH):
    def __init__(self, num_short_obs, num_proprio_obs, num_critic_obs, num_actions,
                 mirror_commands_scale, mirror_stand_threshold, **kwargs):
        if num_proprio_obs != 47 or num_actions != 12:
            raise ValueError('Requires the audited F1 47-component / 12-joint layout')
        super().__init__(num_short_obs, num_proprio_obs, num_critic_obs,
                         num_actions, **kwargs)
        self.mirror_stand_threshold = float(mirror_stand_threshold)
        self.register_buffer('mirror_commands_scale', torch.tensor(mirror_commands_scale, dtype=torch.float32))
        if self.mirror_commands_scale.shape != (3,) or not (self.mirror_commands_scale > 0).all():
            raise ValueError('Need actual positive command normalization scales')
        self.register_buffer('mirror_joint_indices', torch.tensor(list(range(6, 12))+list(range(6))))
        self.register_buffer('mirror_joint_signs', torch.tensor([1., -1., -1., 1., 1., -1.]*2))
        self.register_buffer('mirror_rotation_signs', torch.tensor([-1., 1., -1.]))
        # Persistent keys make a plain DH checkpoint load fail strictly.
        self.register_buffer('f1_equivariant_contract', torch.tensor([num_proprio_obs, num_actions, self.in_channels]))

    def mirror_action(self, actions):
        return actions[..., self.mirror_joint_indices]*self.mirror_joint_signs

    def mirror_observation(self, observations):
        if observations.shape[-1] != self.in_channels*self.num_proprio_obs:
            raise ValueError('Unexpected history width')
        source = observations.reshape(-1, self.in_channels, self.num_proprio_obs)
        result = source.clone()
        moving = torch.norm(source[..., 2:5]/self.mirror_commands_scale, dim=-1) > self.mirror_stand_threshold
        result[..., :2] *= torch.where(moving, -1., 1.).unsqueeze(-1)
        result[..., 3:5] *= -1
        for start in (5, 17, 29):
            result[..., start:start+12] = source[..., start:start+12][..., self.mirror_joint_indices]*self.mirror_joint_signs
        result[..., 41:44] *= self.mirror_rotation_signs
        result[..., 44:47] *= self.mirror_rotation_signs
        return result.reshape(-1, self.in_channels*self.num_proprio_obs)

    def act_inference(self, observations):
        batch = observations.shape[0]
        paired = torch.cat((observations, self.mirror_observation(observations)), dim=0)
        means = super().act_inference(paired)
        return (means[:batch]+self.mirror_action(means[batch:]))/2

    def act(self, observations, **kwargs):
        mean = self.act_inference(observations)
        std = (self.std+self.std[self.mirror_joint_indices])/2
        self.distribution = Normal(mean, mean*0.+std)
        return self.distribution.sample()


def register_policy_class():
    """Register only when the dedicated entry requests this policy."""
    from . import dh_on_policy_runner
    dh_on_policy_runner.ActorCriticF1Equivariant = ActorCriticF1Equivariant
