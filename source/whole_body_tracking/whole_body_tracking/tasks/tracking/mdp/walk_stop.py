"""[OMA] BeyondMimic reference-controlled walk/stop; actor interface is unchanged."""
import numpy as np
import torch
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_inv, quat_mul, sample_uniform, yaw_quat
from .commands import MotionCommand, MotionCommandCfg
from .reference_clock import SupportPhaseClock


class WalkStopCommand(MotionCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        with np.load(cfg.motion_file, allow_pickle=False) as data:
            support = data['support_frames']
        fps = float(np.asarray(self.motion.fps).reshape(-1)[0])
        self.clock = SupportPhaseClock(self.num_envs, fps, support, self.device, cfg.phase_acceleration)
        if abs(fps*env.cfg.sim.dt*env.cfg.decimation-1) > 1e-5:
            raise ValueError('Reference FPS must equal policy rate')
        self.desired_go = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.mode_time = torch.zeros(self.num_envs, device=self.device)
        self.manual_command = False
        for key in ['requested_go','reference_rate','reference_exhausted','actual_planar_speed']:
            self.metrics[key] = torch.zeros(self.num_envs, device=self.device)

    def request(self, go):
        """Evaluation/deployment interface. Switching this never resets robot state."""
        self.manual_command = True
        self.desired_go[:] = torch.as_tensor(go, dtype=torch.bool, device=self.device)

    def _sample(self, data, quaternion=False):
        phase = self.clock.phase.clamp(0, self.motion.time_step_total-1)
        lo = phase.floor().long()
        hi = (lo+1).clamp(max=self.motion.time_step_total-1)
        a = (phase-lo).reshape((self.num_envs,)+(1,)*(data.ndim-1))
        left, right = data[lo], data[hi]
        if quaternion:
            right = torch.where((left*right).sum(-1,keepdim=True)<0,-right,right)
            value = torch.lerp(left,right,a)
            return value/value.norm(dim=-1,keepdim=True).clamp_min(1e-8)
        return torch.lerp(left,right,a)

    @property
    def joint_pos(self):
        return self._sample(self.motion.joint_pos)

    @property
    def joint_vel(self):
        return self._sample(self.motion.joint_vel)*self.clock.rate[:,None]

    @property
    def body_pos_w(self):
        return self._sample(self.motion.body_pos_w)+self._env.scene.env_origins[:,None,:]

    @property
    def body_quat_w(self):
        return self._sample(self.motion.body_quat_w,quaternion=True)

    @property
    def body_lin_vel_w(self):
        return self._sample(self.motion.body_lin_vel_w)*self.clock.rate[:,None,None]

    @property
    def body_ang_vel_w(self):
        return self._sample(self.motion.body_ang_vel_w)*self.clock.rate[:,None,None]

    @property
    def anchor_pos_w(self):
        return self.body_pos_w[:,self.motion_anchor_body_index]

    @property
    def anchor_quat_w(self):
        return self.body_quat_w[:,self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self):
        return self.body_lin_vel_w[:,self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self):
        return self.body_ang_vel_w[:,self.motion_anchor_body_index]

    def _adaptive_sampling(self, env_ids):
        super()._adaptive_sampling(env_ids)
        # Stable-start candidate selection is confined to genuine episode resets.
        self.clock.reset(env_ids,self.time_steps[env_ids].float())
        self.time_steps[env_ids] = self.clock.phase[env_ids].long()
        self.desired_go[env_ids] = False
        self.mode_time[env_ids] = sample_uniform(*self.cfg.stop_duration_range,(len(env_ids),),self.device)

    def _update_command(self):
        dt = self._env.cfg.sim.dt*self._env.cfg.decimation
        if not self.manual_command:
            self.mode_time -= dt
            ids = torch.where(self.mode_time <= 0)[0]
            if len(ids):
                self.desired_go[ids] = ~self.desired_go[ids]
                moving = self.desired_go[ids]
                go_time = sample_uniform(*self.cfg.go_duration_range,(len(ids),),self.device)
                stop_time = sample_uniform(*self.cfg.stop_duration_range,(len(ids),),self.device)
                self.mode_time[ids] = torch.where(moving,go_time,stop_time)
        self.clock.step(self.desired_go,dt)
        self.time_steps[:] = self.clock.phase.long()
        # Native relative-body alignment; no end-of-reference _resample_command.
        anchor_p = self.anchor_pos_w[:,None,:].repeat(1,len(self.cfg.body_names),1)
        anchor_q = self.anchor_quat_w[:,None,:].repeat(1,len(self.cfg.body_names),1)
        robot_p = self.robot_anchor_pos_w[:,None,:].repeat(1,len(self.cfg.body_names),1)
        robot_q = self.robot_anchor_quat_w[:,None,:].repeat(1,len(self.cfg.body_names),1)
        robot_p[...,2] = anchor_p[...,2]
        rotation = yaw_quat(quat_mul(robot_q,quat_inv(anchor_q)))
        self.body_quat_relative_w = quat_mul(rotation,self.body_quat_w)
        self.body_pos_relative_w = robot_p+quat_apply(rotation,self.body_pos_w-anchor_p)
        self.bin_failed_count = self.cfg.adaptive_alpha*self._current_bin_failed+(1-self.cfg.adaptive_alpha)*self.bin_failed_count
        self._current_bin_failed.zero_()

    def _update_metrics(self):
        super()._update_metrics()
        self.metrics['requested_go'] = self.desired_go.float()
        self.metrics['reference_rate'] = self.clock.rate.clone()
        self.metrics['reference_exhausted'] = self.clock.exhausted.float()
        self.metrics['actual_planar_speed'] = self.robot_anchor_lin_vel_w[:,:2].norm(dim=-1)


@configclass
class WalkStopCommandCfg(MotionCommandCfg):
    class_type: type = WalkStopCommand
    # Preregistered control settings; not measured hardware constants.
    phase_acceleration: float = 2.0
    go_duration_range: tuple[float,float] = (2.0,4.0)
    stop_duration_range: tuple[float,float] = (2.0,4.0)
