import numpy as np
import os
import sys
# [OMA g1] 同 train.py：按脚本位置回溯两级到 x1-training/（GM gm-run 的 cwd 不在仓库根）
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import isaacgym  # noqa: F401
from legged_gym.envs import *  # noqa: F401
from legged_gym.utils import get_args, task_registry  # noqa: F401
import torch

# 卡 §1 论文口径首轮评估：E_cmd（episodic 平均命令跟踪误差，L1）+ 非终止轨迹比
# checkpoint 路径锚定脚本位置（与 train.py sys.path 同一教训）
CHECKPOINT = os.path.abspath(os.path.join(os.path.dirname(__file__),
                              '..', '..', 'resources', 'checkpoints', 'model_8550_g1int.pt'))
NUM_EVAL_EPISODES = 200
EVAL_NUM_ENVS = 64


def eval_cmd(args):
    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    # play.py 先例：冻结载荷随机化 + 命令重采样窗口拉大（每 episode 固定命令）
    env_cfg.domain_rand.randomize_load = False
    env_cfg.commands.resampling_time = 100
    env_cfg.env.num_envs = EVAL_NUM_ENVS

    env, _ = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    print("load_path:", CHECKPOINT)
    ppo_runner.load(CHECKPOINT)

    obs, critic_obs = env.reset()  # reset 返回 2 元组（step 才是 5 元组）
    actor_critic = ppo_runner.get_inference_policy(device=ppo_runner.device)
    ep_l1_sum = torch.zeros(EVAL_NUM_ENVS, device=env.device)
    ep_steps = torch.zeros(EVAL_NUM_ENVS, device=env.device)
    done_episodes = 0
    terminated = 0
    timeouts = 0
    l1_list = []
    steps_list = []
    cmd_log = []

    while done_episodes < NUM_EVAL_EPISODES:
        with torch.no_grad():
            actions, _ = actor_critic.act_inference(obs, privileged_obs=critic_obs)
        obs, critic_obs, _, dones, infos = env.step(actions)

        # E_cmd: 命令空间前 3 维（vx, vy, ω）对实际基座速度的 L1
        cmd_vel = env.commands[:, :3]
        actual = torch.cat((env.base_lin_vel[:, :2], env.base_ang_vel[:, 2:3]), dim=-1)
        l1 = (cmd_vel - actual).abs().sum(dim=-1)
        ep_l1_sum += l1
        ep_steps += 1

        done_ids = dones.nonzero(as_tuple=False).flatten()
        for i in done_ids.tolist():
            l1_list.append((ep_l1_sum[i] / max(1.0, ep_steps[i].item())).item())
            steps_list.append(int(ep_steps[i].item()))
            cmd_log.append(env.commands[i, :3].cpu().numpy().tolist())
            done_episodes += 1
            if env.time_out_buf[i]:
                timeouts += 1
            else:
                terminated += 1
            ep_l1_sum[i] = 0
            ep_steps[i] = 0

    l1_arr = np.array(l1_list)
    steps_arr = np.array(steps_list)
    cmd_arr = np.array(cmd_log)
    print("=" * 60)
    print(f"EVAL SUMMARY  episodes={done_episodes}  checkpoint={os.path.basename(CHECKPOINT)}")
    print(f"E_cmd (mean L1, cmd vs actual) : {l1_arr.mean():.4f}  (std {l1_arr.std():.4f})")
    print(f"  per-component cmd ranges: vx [{cmd_arr[:,0].min():.2f},{cmd_arr[:,0].max():.2f}]"
          f"  vy [{cmd_arr[:,1].min():.2f},{cmd_arr[:,1].max():.2f}]"
          f"  wyaw [{cmd_arr[:,2].min():.2f},{cmd_arr[:,2].max():.2f}]")
    print(f"non-termination ratio         : {timeouts / done_episodes:.3f}"
          f"  (timeout {timeouts} / terminated {terminated})")
    print(f"episode length (steps, max 1000): mean {steps_arr.mean():.1f}"
          f"  median {np.median(steps_arr):.1f}  max {steps_arr.max()}")
    print("=" * 60)


if __name__ == '__main__':
    args = get_args()
    eval_cmd(args)
