"""Stage-one diagnostic variant: explicit stand scheduling emits a zero twist.

The Q3 heading controller remains active for every movement command. No PD,
reward, gait schedule, physics or observation layout is changed.
"""
import torch

from .f1_env import F1WalkingEnv


class F1CleanStopEnv(F1WalkingEnv):
    def _init_buffers(self):
        super()._init_buffers()
        self.explicit_stand_command = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device)

    def _resample_stand_command(self, env_ids):
        super()._resample_stand_command(env_ids)
        self.explicit_stand_command[env_ids] = True
        self.commands[env_ids, :3] = 0

    def _resample_walk_sagittal_command(self, env_ids):
        super()._resample_walk_sagittal_command(env_ids)
        self.explicit_stand_command[env_ids] = False

    def _resample_walk_lateral_command(self, env_ids):
        super()._resample_walk_lateral_command(env_ids)
        self.explicit_stand_command[env_ids] = False

    def _resample_rotate_command(self, env_ids):
        super()._resample_rotate_command(env_ids)
        self.explicit_stand_command[env_ids] = False

    def _resample_walk_omnidirectional_command(self, env_ids):
        super()._resample_walk_omnidirectional_command(env_ids)
        self.explicit_stand_command[env_ids] = False

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        # Clear AFTER heading injection so the scheduled stand cannot be
        # overwritten by a residual heading error. Native stand/phase/reward
        # gates then see the same complete zero command.
        self.commands[self.explicit_stand_command, :3] = 0
