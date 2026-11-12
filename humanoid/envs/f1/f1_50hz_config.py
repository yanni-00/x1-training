"""Owner-selected stage-one profile; retain the archived Q3 base configuration."""
import json

from humanoid.f1_contract import CONFIG_ROOT
from .f1_config import F1WalkingCfg, F1WalkingCfgPPO


PROFILE_PATH = CONFIG_ROOT / 'stage1_50hz_no_filter.json'
with PROFILE_PATH.open(encoding='utf-8') as stream:
    PROFILE = json.load(stream)
with (CONFIG_ROOT / 'owner_baseline_model.json').open(encoding='utf-8') as stream:
    OWNER_MODEL = json.load(stream)


class F1Stage1NoFilter50HzCfg(F1WalkingCfg):
    class env(F1WalkingCfg.env):
        physical_model = 'owner_baseline'

    class asset(F1WalkingCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/' + OWNER_MODEL['urdf_relative_path']

    class control(F1WalkingCfg.control):
        stiffness = dict(zip(PROFILE['joint_groups'], PROFILE['kp']))
        damping = dict(zip(PROFILE['joint_groups'], PROFILE['kd']))
        decimation = PROFILE['decimation']

    class domain_rand(F1WalkingCfg.domain_rand):
        randomize_joint_friction = False
        randomize_joint_damping = False
        randomize_joint_armature = False
        randomize_coulomb_friction = False
        enable_delivery = False
        delivery_tau_d = None
        delivery_joint_ids = []
        add_lag = False
        add_dof_lag = False
        add_dof_pos_vel_lag = False
        add_imu_lag = False


class F1Stage1NoFilter50HzCfgPPO(F1WalkingCfgPPO):
    class runner(F1WalkingCfgPPO.runner):
        experiment_name = 'f1_walk_stage1_50hz_no_filter_owner_pd'
        resume = False
        resume_path = None
