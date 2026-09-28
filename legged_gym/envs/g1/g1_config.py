from legged_gym.envs.h1.h1_config import H1Cfg, H1CfgPPO

# G1 (23-DOF) 移植（design-hugwbc-g1-001 T2 映射表）
# 腿 6×2: hip_pitch/hip_roll/hip_yaw/knee/ankle_pitch/ankle_roll
# 腰 3:   waist_yaw/waist_roll/waist_pitch
# 臂 4×2: shoulder_pitch/shoulder_roll/shoulder_yaw/elbow（与 H1 同名）
PROPRIOCEPTION_DIM = 69  # 23 dof × (pos+vel) + 12 dof_pos + 23 actions = 46+23+... 由父类布局决定
# 注意：H1 的 63 = 3(ang_vel)+3(proj_gravity)+19*2(dof pos/vel)+19(actions)  → G1 = 3+3+23*2+23 = 75
PROPRIOCEPTION_DIM = 75
CMD_DIM = 3 + 4 + 1 + 2
TERRAIN_DIM = 221
PRIVILEGED_DIM = 3 + 1 + 2 + 1 + 6 + 11
CLOCK_INPUT = 2

class G1Cfg( H1Cfg ):
    class env( H1Cfg.env ):
        num_observations = PROPRIOCEPTION_DIM + CMD_DIM + CLOCK_INPUT + PRIVILEGED_DIM + TERRAIN_DIM
        num_partial_obs = PROPRIOCEPTION_DIM + CMD_DIM + CLOCK_INPUT
        num_privileged_obs = None
        num_actions = 23

    class init_state( H1Cfg.init_state ):
        pos = [0.0, 0.0, 0.8] # x,y,z [m]（G1 体型，ASAP 值）
        default_joint_angles = { # = target angles [rad] when action = 0.0（ASAP G1 默认角）
           'left_hip_pitch_joint' : -0.1,
           'left_hip_roll_joint' : 0.0,
           'left_hip_yaw_joint' : 0.0,
           'left_knee_joint' : 0.3,
           'left_ankle_pitch_joint' : -0.2,
           'left_ankle_roll_joint' : 0.0,
           'right_hip_pitch_joint' : -0.1,
           'right_hip_roll_joint' : 0.0,
           'right_hip_yaw_joint' : 0.0,
           'right_knee_joint' : 0.3,
           'right_ankle_pitch_joint' : -0.2,
           'right_ankle_roll_joint' : 0.0,
           'waist_yaw_joint' : 0.0,
           'waist_roll_joint' : 0.0,
           'waist_pitch_joint' : 0.0,
           'left_shoulder_pitch_joint' : 0.0,
           'left_shoulder_roll_joint' : 0.0,
           'left_shoulder_yaw_joint' : 0.0,
           'left_elbow_joint' : 0.0,
           'right_shoulder_pitch_joint' : 0.0,
           'right_shoulder_roll_joint' : 0.0,
           'right_shoulder_yaw_joint' : 0.0,
           'right_elbow_joint' : 0.0,
        }

    class control( H1Cfg.control ):
        # PD Drive parameters（ASAP G1 训练验证值，[N*m/rad]）
        stiffness = {'hip_pitch': 100,
                     'hip_roll': 100,
                     'hip_yaw': 100,
                     'knee': 200,
                     'ankle_pitch': 20,
                     'ankle_roll': 20,
                     'waist_yaw': 400,
                     'waist_roll': 400,
                     'waist_pitch': 400,
                     'shoulder_pitch': 90,
                     'shoulder_roll': 60,
                     'shoulder_yaw': 20,
                     'elbow': 60,
                     }
        damping = {  'hip_pitch': 2.5,
                    'hip_roll': 2.5,
                    'hip_yaw': 2.5,
                    'knee': 5.0,
                    'ankle_pitch': 0.2,
                    'ankle_roll': 0.1,
                    'waist_yaw': 5.0,
                    'waist_roll': 5.0,
                    'waist_pitch': 5.0,
                    'shoulder_pitch': 2.0,
                    'shoulder_roll': 1.0,
                    'shoulder_yaw': 0.4,
                    'elbow': 1.0,
                    }  # [N*m*s/rad]
        # 空白填默认：ASAP yaml 无 torque_limits → PD 折算保守档（clip_torques 由父类开关）
        torque_limits = {
                     'hip_pitch': 88,
                     'hip_roll': 88,
                     'hip_yaw': 88,
                     'knee': 140,
                     'ankle_pitch': 45,
                     'ankle_roll': 45,
                     'waist_yaw': 88,
                     'waist_roll': 88,
                     'waist_pitch': 88,
                     'shoulder_pitch': 25,
                     'shoulder_roll': 25,
                     'shoulder_yaw': 25,
                     'elbow': 25,
                     }
        action_scale = 0.25
        decimation = 4

    class asset( H1Cfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1/g1_29dof_anneal_23dof.urdf'
        name = "g1"
        base_name = "pelvis"
        foot_name = "ankle_roll"
        auxiliary_foot_link = ["left_ankle_roll_link", "right_ankle_roll_link"]
        penalize_contacts_on = ["elbow", "torso", "hip", "knee", "waist"]
        terminate_after_contacts_on = ["pelvis", "torso"]
        self_collisions = 0

    class domain_rand( H1Cfg.domain_rand ):
        added_mass_range = [-2.3, 6.8]  # G1 35.1kg 等比（H1 -3..9 / 47kg）

class G1CfgPPO( H1CfgPPO ):
    class policy( H1CfgPPO.policy ):
        class NetModel:
            class MlpAdaptModel:
                proprioception_dim = PROPRIOCEPTION_DIM
                cmd_dim = CMD_DIM
                privileged_dim = PRIVILEGED_DIM
                terrain_dim = TERRAIN_DIM
                latent_dim = 32
                privileged_recon_dim = 3
                max_length = G1Cfg.env.include_history_steps
                actor_hidden_dims = [256, 128, 32]
                mlp_hidden_dims = [256, 128]

        critic_obs_dim = PROPRIOCEPTION_DIM + CMD_DIM + PRIVILEGED_DIM + TERRAIN_DIM

    class runner( H1CfgPPO.runner ):
        experiment_name = 'g1_teacher'

    class algorithm( H1CfgPPO.algorithm ):
        robot_type = 'g1'  # PPO 对称性置换表分支（ppo.py 按 robot_type 选 19/23-DOF 镜像表）
