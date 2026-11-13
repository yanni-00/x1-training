"""Stage-one command coverage using existing Q3 modes/ranges; fixed owner PD."""
import json
import torch
from isaacgym.torch_utils import torch_rand_float
from .f1_50hz_config import F1Stage1NoFilter50HzCfg
from .f1_config import F1LowLiftCfg
from .f1_clean_stop_env import F1CleanStopEnv


class F1CommandCoverageCfg(F1Stage1NoFilter50HzCfg):
    class env(F1Stage1NoFilter50HzCfg.env):
        goal_revision = 'F1-WALK-v2'
        command_semantics = 'stage1_explicit_direction_coverage_v1'

    class commands(F1Stage1NoFilter50HzCfg.commands):
        # Stage one keeps its broad ranges, heading controller and rewards.
        # The dedicated modes/timing ranges already exist in the Q3 route.
        gait = ['walk_sagittal', 'walk_lateral', 'rotate', 'stand', 'walk_omnidirectional']
        gait_time_range = dict(F1LowLiftCfg.commands.gait_time_range)


class F1CommandCoverageEnv(F1CleanStopEnv):
    coverage_names = ['scheduled_stand', 'sagittal', 'lateral', 'turn_positive', 'turn_negative', 'omni', 'other']

    def _init_buffers(self):
        super()._init_buffers()
        self.command_coverage_counts = torch.zeros(len(self.coverage_names), dtype=torch.long, device=self.device)
        self.last_coverage_report_step = -1

    def _resample_rotate_command(self, env_ids):
        super()._resample_rotate_command(env_ids)
        # Native atan2(vy,vx) produces zero for a pure turn (both are zero).
        # Use the existing heading target range so the retained heading
        # controller can teach both turn directions, then stop at the target.
        bounds = self.command_ranges['heading']
        self.commands[env_ids, 3] = torch_rand_float(
            bounds[0], bounds[1], (len(env_ids), 1), device=self.device).squeeze(1)

    def compute_observations(self):
        super().compute_observations()
        if not hasattr(self, 'command_coverage_counts'):
            return
        x, y, w = (self.commands[:, i] != 0 for i in range(3))
        stand = self.explicit_stand_command
        masks = [stand, ~stand & x & ~y, ~stand & ~x & y,
                 ~stand & ~x & ~y & (self.commands[:, 2] > 0),
                 ~stand & ~x & ~y & (self.commands[:, 2] < 0), ~stand & x & y]
        masks.append(~(masks[0] | masks[1] | masks[2] | masks[3] | masks[4] | masks[5]))
        self.command_coverage_counts += torch.stack([m.sum() for m in masks])
        # Count actual command values at observation construction, including
        # reset construction. These are not episode counts or training steps.
        step = int(self.common_step_counter)
        if step > 0 and step % self.max_episode_length == 0 and step != self.last_coverage_report_step:
            self.last_coverage_report_step = step
            print('[F1 observation command coverage] '+json.dumps(dict(zip(
                self.coverage_names, self.command_coverage_counts.cpu().tolist()))), flush=True)
