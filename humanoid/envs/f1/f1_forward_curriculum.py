"""r24: pure-forward sampling curriculum on top of the r22 reward configuration.

Only change versus r22: walk_sagittal duration weight raised so the sampler
allocates a larger share of every episode to pure forward walking. The gait
list, the pitch penalty (w2), the swing-direction reward (w60) and everything
else are identical to r22 (exp-20261126-002), giving a paired control.
"""

from .f1_rearward_pitch_gate import F1RearwardPitchGateCfg, F1RearwardPitchGateEnv


class F1ForwardCurriculumCfg(F1RearwardPitchGateCfg):
    class commands(F1RearwardPitchGateCfg.commands):
        # r22 effective draws (means): sagittal 4, lateral 2.5, rotate 2.5,
        # stand 5.5, omni 5 -> sagittal ~20.5% of walk time; measured argo
        # coverage put sagittal at 23.6% of all observation steps and the
        # gate-ON fraction at ~8.5%. Raising the sagittal draw to [8, 12]
        # (mean 10) roughly doubles the pure-forward share; the normalization
        # in generate_gait_time redistributes the rest proportionally.
        gait_time_range = {'walk_sagittal': [8, 12], 'walk_lateral': [2, 3],
                           'rotate': [2, 3], 'stand': [5, 6], 'walk_omnidirectional': [4, 6]}


class F1ForwardCurriculumEnv(F1RearwardPitchGateEnv):
    pass