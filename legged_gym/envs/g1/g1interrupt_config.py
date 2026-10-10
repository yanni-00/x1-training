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

            # [lap2 die-fast 机制检验] H1原值0.2×dt=0.004/步 vs 死亡-4：活满20s差价仅+0.8, 无生存梯度
            # (exp-007实测: episode长度塌缩钉死1.0)。提10倍造"多活"梯度, 副作用=站着不动刷alive——
            # 由curriculum内的stand_still(-10)/standing_air(-2)罚随课程推进压制, 预注册签名见decision_log
            alive = 2.0

    class commands( G1Cfg.commands ):
        num_commands = CMD_DIM

    # [lap3 T4变量1] PD 档消融：H1 值折算 ×0.5（H1: hip200/knee300/ankle40/torso300/arm20, damping 5/6/2/6/0.5）
    # 对照 ASAP 档（G1Cfg 继承）。关键差：ankle damping 0.2/0.1→1.0（ASAP 桨状踝是平衡嫌疑），
    # waist 400→150, knee 200→150, arm 90/60/20→10。stiffness+damping 整表换（design T4 预注册的功能单元组）
    class control( G1Cfg.control ):
        stiffness = {'hip_pitch': 100, 'hip_roll': 100, 'hip_yaw': 100,
                     'knee': 150,
                     'ankle_pitch': 20, 'ankle_roll': 20,
                     'waist_yaw': 150, 'waist_roll': 150, 'waist_pitch': 150,
                     'shoulder_pitch': 10, 'shoulder_roll': 10, 'shoulder_yaw': 10, 'elbow': 10,
                     }
        damping = {  'hip_pitch': 2.5, 'hip_roll': 2.5, 'hip_yaw': 2.5,
                    'knee': 3,
                    'ankle_pitch': 1, 'ankle_roll': 1,
                    'waist_yaw': 3, 'waist_roll': 3, 'waist_pitch': 3,
                    'shoulder_pitch': 0.25, 'shoulder_roll': 0.25, 'shoulder_yaw': 0.25, 'elbow': 0.25,
                    }

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
        # lap2 为新奖励经济学下的从头训练（die-fast 权重不可复用），r7 的 resume 配置已复位
        resume = False
        resume_path = None
        max_iterations = 40000
        save_interval = 50  # H1镜像为2000；TASK_051实测90min外部终止无checkpoint可续——降为50(≈49min/个)保断点续训

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
