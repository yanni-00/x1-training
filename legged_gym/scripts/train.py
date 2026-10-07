import numpy as np
import os
from datetime import datetime
import sys
# [OMA g1] GM gm-run 的 cwd 不在仓库根——按脚本位置回溯两级（…/x1-training/legged_gym/scripts/ → x1-training/，含 legged_gym 与 rsl_rl 包）
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry, class_to_dict, update_class_from_dict
import torch
import yaml
import shutil
def train(args):

    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

if __name__ == '__main__':
    args = get_args()
    train(args)
