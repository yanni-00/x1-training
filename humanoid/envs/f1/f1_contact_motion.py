"""Keep linear foot slip and restore the original angular contact cost separately."""
from .f1_linear_slip import F1LinearSlipCfg, F1LinearSlipEnv
from .f1_random_order import F1RandomOrderCfg, F1RandomOrderEnv


class F1ContactMotionCfg(F1LinearSlipCfg):
    class env(F1LinearSlipCfg.env):
        reward_semantics = 'forward_progress_contact_linear_plus_angular_v1'

    class rewards(F1LinearSlipCfg.rewards):
        class scales(F1LinearSlipCfg.rewards.scales):
            # Exact coefficient of the pre-correction angular contact cost.
            foot_contact_angular_velocity = F1RandomOrderCfg.rewards.scales.foot_slip


class F1ContactMotionEnv(F1LinearSlipEnv):
    def _reward_foot_contact_angular_velocity(self):
        # Preserve the original40N gate, angular10:12 norm and foot sum.
        return F1RandomOrderEnv._reward_foot_slip(self)
