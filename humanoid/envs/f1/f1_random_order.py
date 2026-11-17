"""Same five native modes and durations, with per-environment order coverage."""
import torch
from .f1_speed_tracking import F1SpeedTrackingCfg, F1SpeedTrackingEnv


class F1RandomOrderCfg(F1SpeedTrackingCfg):
    class env(F1SpeedTrackingCfg.env):
        command_semantics = 'stage1_per_environment_random_order_v1'


class F1RandomOrderEnv(F1SpeedTrackingEnv):
    def _init_buffers(self):
        super()._init_buffers()
        self.gait_mode_order = torch.zeros(
            (self.num_envs, len(self.cfg.commands.gait)),
            dtype=torch.long, device=self.device)

    def generate_gait_time(self, envs):
        if len(envs) == 0:
            return
        # Reuse native weighted sampling and episode scaling. Carry each
        # mode's actual integer duration with that mode when permuting it.
        super().generate_gait_time(envs)
        starts = self.gait_time[envs].clone()
        end = torch.full((len(envs), 1), self.max_episode_length,
                         dtype=starts.dtype, device=self.device)
        duration = torch.diff(torch.cat((starts, end), dim=1), dim=1)
        order = torch.rand(duration.shape, device=self.device).argsort(dim=1)
        ordered_duration = duration.gather(1, order)
        self.gait_mode_order[envs] = order
        self.gait_time[envs] = torch.cat(
            (torch.zeros_like(starts[:, :1]),
             ordered_duration[:, :-1].cumsum(dim=1, dtype=starts.dtype)), dim=1)

    def _resample_commands(self):
        for slot in range(len(self.cfg.commands.gait)):
            boundary = self.episode_length_buf == self.gait_time[:, slot]
            for mode, name in enumerate(self.cfg.commands.gait):
                ids = (boundary & (self.gait_mode_order[:, slot] == mode)).nonzero(
                    as_tuple=False).flatten()
                if len(ids):
                    getattr(self, '_resample_' + name + '_command')(ids)
