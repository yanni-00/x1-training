from legged_gym.envs.g1.g1_config import G1Cfg, G1CfgPPO, PROPRIOCEPTION_DIM, CLOCK_INPUT, PRIVILEGED_DIM, TERRAIN_DIM

# G1 (23-DOF) 干预课程版（design-hugwbc-g1-001 T2 行5/15 + D-d：臂同名直接用，命令/奖励/三课程原样镜像 h1int）
# 维度对照任务卡预注册 obs 75/88/333：
#   75 = 3(ang_vel)+3(proj_gravity)+23×2(dof pos/vel)+23(actions)
#   11 = CMD(3+4+1+2) + INTERRUPT_IN_CMD(1)；88 = 75+11+2(时钟)；333 = 88+24(特权)+221(地形)
INTERRUPT_IN_CMD = True
NOISE_IN_PRIVILEGE = False
EXECUTE_IN_PRIVILEGE = False
CMD_DIM = 3 + 4 + 1 + 2 + INTERRUPT_IN_CMD

DISTURB_DIM = 8

class G1InterruptCfg( G1Cfg ):
    class env( G1Cfg.env ):
        num_observations = PROPRIOCEPTION_DIM + CMD_DIM + CLOCK_INPUT + PRIVILEGED_DIM + TERRAIN_DIM
        num_partial_obs = PROPRIOCEPTION_DIM + CMD_DIM + CLOCK_INPUT

    class rewards( G1Cfg.rewards ):
        reward_curriculum_list = ['action_rate_upper', 'action_rate_lower',
                                  'feet_stumble',
                                  'joint_power_distribution', 'feet_contact_forces',
                                  'dof_acc', 'torques',
                                  'base_height', 'collision', 'stand_still',
                                  'lin_vel_z', 'base_height_min', 'dof_vel_limits',
                                  'ang_vel_xy',
                                  # 'dof_pos_limits',
                                  # Deviation
                                  'shoulder_yaw_deviation', 'shoulder_roll_deviation',
                                  'shoulder_pitch_deviation', 'elbow_deviation',
                                  # 'hip_yaw_deviation', 'hip_roll_deviation',
                                  'torso_deviation',
                                  # Mob
                                  # 'tracking_contacts_shaped_force', 'tracking_contacts_shaped_vel',
                                  # 'feet_clearance_cmd_linear',
                                  # 'feet_clearance_cmd_polynomial',
                                  'hopping_symmetry',
                                  'jump',
                                  'orientation_control',
                                  # 'waist_control',
                                  # Task
                                  # 'tracking_ang_vel'
                                  # Standing
                                  # 'standing',
                                  'standing_air',
                                  'standing_vel',
                                  # 'standing_joint_deviation'
                                  ]
        class scales( G1Cfg.rewards.scales ):
            action_rate = 0
            action_rate_lower = -0.01
            action_rate_upper = -0.01
            base_height = -40.0
            stand_still = -10.0
            standing = 2.0
            orientation_control = -10

            # penalize standing
            standing_air = -2

    class commands( G1Cfg.commands ):
        num_commands = CMD_DIM

    class disturb:
        max_curriculum = 1.0
        use_disturb = True
        disturb_dim = DISTURB_DIM
        disturb_scale = 2
        noise_scale = [
            5.2, # Left Shoulder Pitch -2.6~2.6
            3.3, # Left Shoulder Roll, -0.3~3.0
            5.5, # Left Shoulder Yaw,  -1.2~4.3
            3.7, # Left Shoulder Elbow, -1.2~2.5
            5.2, # Right Shoulder Pitch, -2.6~2.6
            3.3, # Right Shoulder Roll, -3.0~0.3
            5.5, # Right Shoulder Yaw,  -4.3~1.2
            3.7, # Right Shoulder Elbow, -1.2~2.5
        ] # Uniform Distribution Noise for each joint.（H1 档镜像；G1 臂同名直接用——T2 行5，运行时按 G1 URDF dof_pos_limits 截断）
        noise_lowerbound = [
            -2.6,
            -0.3,
            -1.2,
            -1.2,
            -2.6,
            -3.0,
            -4.3,
            -1.2
        ]
        uniform_scale = 1
        uniform_noise = True
        noise_ratio = 1
        interrupt_action_buffer = None
        start_by_curriculum = True
        replace_action = True
        disturb_rad = 0.2
        disturb_rad_curriculum = True
        disturb_curriculum_method = 2

        noise_update_step = 30
        switch_prob = 0.005
        interrupt_in_cmd = INTERRUPT_IN_CMD
        stand_interrupt_only = False
        noise_curriculum_ratio = 0.5
        disturb_in_last_action = False
        obs_target_interrupt_in_privilege = NOISE_IN_PRIVILEGE
        obs_executed_actions_in_privilege = EXECUTE_IN_PRIVILEGE
        disturb_terminate_assets = []


    class curriculum_thresholds( G1Cfg.curriculum_thresholds):
        class disturb:
            tracking_lin_vel = 0.6

class G1InterruptCfgPPO( G1CfgPPO ):
    class runner( G1CfgPPO.runner ):
        experiment_name = "g1_interrupt"
        resume = False
        resume_path = None
        max_iterations = 40000
        save_interval = 2000

    class policy( G1CfgPPO.policy ):
        model_name = "MlpAdaptModel"
        class NetModel:
            class MlpAdaptModel:
                proprioception_dim = PROPRIOCEPTION_DIM
                cmd_dim = CMD_DIM + CLOCK_INPUT
                privileged_dim = PRIVILEGED_DIM
                terrain_dim = TERRAIN_DIM
                latent_dim = 32
                privileged_recon_dim = 3
                max_length = G1InterruptCfg.env.include_history_steps
                actor_hidden_dims = [256, 128, 32]
                mlp_hidden_dims = [256, 128]

        critic_hidden_dims = [512, 256, 128]
        critic_obs_dim = PROPRIOCEPTION_DIM + CMD_DIM + CLOCK_INPUT + PRIVILEGED_DIM + TERRAIN_DIM
