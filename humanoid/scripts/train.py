# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2021 ETH Zurich, Nikita Rudin
# SPDX-FileCopyrightText: Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Copyright (c) 2024, AgiBot Inc. All rights reserved.


import os

from humanoid.envs import *
from humanoid.utils import get_args, task_registry

# GM 云端 clone 后会清理 logs/ 目录（git 跟踪也会被平台删）；resume 时 get_load_path
# 需要 exported_data 目录存在且含至少一个子目录（runs[-1] 取最新 run），故运行时预建占位。
# 仅在 GM 挂载环境（/workspace/x1-training）生效，本地/服务器训练不受影响。
_GM_ROOT = '/workspace/x1-training'
if os.path.isdir(_GM_ROOT):
    os.makedirs(os.path.join(_GM_ROOT, 'logs', 'x1_dh_stand', 'exported_data', 'gm_placeholder'), exist_ok=True)

def train(args):
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg, log_dir = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=False)

if __name__ == '__main__':
    args = get_args()
    train(args)
