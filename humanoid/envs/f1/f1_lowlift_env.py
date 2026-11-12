# SPDX-License-Identifier: BSD-3-Clause
# Stage-two method deltas copied from the selected Q3 v5 copy route.
# Full original copyright and license notice is retained in f1_env.py.
from .f1_env import *


class F1LowLiftEnv(F1WalkingEnv):
    def _get_stand_command(self):
        """ stand 判定用原始用户指令：yaw_hold 会往 commands[:,2] 注入角度纠偏（clip 后可达 0.25），
            若用注入后的 commands 判 stand，零指令站立段会因纠偏量 > 0.05 被误判为 walk，
            gait phase 解冻、reference 切成 walk 轨迹。wz 取 raw_yaw_cmd（注入前的原始指令）。
        """
        if self.cfg.commands.yaw_hold:
            raw_cmd = torch.stack((self.commands[:, 0], self.commands[:, 1], self.raw_yaw_cmd), dim=1)
        else:
            raw_cmd = self.commands[:, :3]
        return torch.norm(raw_cmd, dim=1) <= self.cfg.commands.stand_com_threshold

    def  _get_phase(self):
        cycle_time = self.cfg.rewards.cycle_time
        if self.cfg.commands.sw_switch:
            stand_command = self._get_stand_command()
            self.phase_length_buf[stand_command] = 0 # set this as 0 for which env is standing
            # self.gait_start is rand 0 or 0.5
            phase = (self.phase_length_buf * self.dt / cycle_time + self.gait_start) * (~stand_command)
        else:
            phase = self.episode_length_buf * self.dt / cycle_time + self.gait_start

        # phase continue increase，if want robot stand, set 0
        return phase

    def _resample_commands(self):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        for i in range(len(self.cfg.commands.gait)):
            # if env finish current gait type, resample command for next gait
            env_ids = (self.episode_length_buf == self.gait_time[:,i]).nonzero(as_tuple=False).flatten()
            if len(env_ids) > 0:
                # according to gait type create a name
                name = '_resample_' + self.cfg.commands.gait[i] + '_command'
                # get function from self based on name
                resample_command = getattr(self, name)
                # resample_command stands for _resample_stand_command/_resample_walk_sagittal_command/...
                resample_command(env_ids)
                # v3 yaw_hold：仅在真正 resample 的分支捕获原始 wz（每步全量捕获会吃到注入值造成污染）
                self.raw_yaw_cmd[env_ids] = self.commands[env_ids, 2].clone()

    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        self.phase_length_buf += 1
        self._resample_commands()
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)

        # v3 yaw_hold：近零段注入角度纠偏，借 tracking_ang_vel 闭环（v2 reward 版三层缺陷：强度9:1劣势/
        # 与速率项对撞/24s积分信用分配，全部由"信号走指令通道"消除）。门控 0.15 与 joint_symmetry 对齐。
        if self.cfg.commands.yaw_hold:
            turning = torch.abs(self.raw_yaw_cmd) > 0.15
            # 边沿 recenter：转向→回中首步，锚点重置为当前朝向（手柄语义：转完即新基准，无回拉）
            recenter = (~turning) & self.prev_turning
            forward = quat_apply(self.base_quat, self.forward_vec)
            yaw_now = torch.atan2(forward[:, 1], forward[:, 0])
            if recenter.any():
                self.anchor_yaw[recenter] = yaw_now[recenter]
            self.prev_turning = turning.clone()
            # 静止段注入纠偏指令；转向段透传原始指令
            hold_wz = torch.clip(
                self.cfg.commands.yaw_hold_gain * wrap_to_pi(self.anchor_yaw - yaw_now),
                -self.cfg.commands.yaw_hold_clip, self.cfg.commands.yaw_hold_clip)
            self.commands[:, 2] = torch.where(turning, self.raw_yaw_cmd, hold_wz)

        if self.cfg.terrain.measure_heights:
            # get all robot surrounding height
            self.measured_heights = self._get_heights()

        if self.cfg.domain_rand.push_robots:
            i = int(self.common_step_counter/self.cfg.domain_rand.update_step)
            if i >= len(self.cfg.domain_rand.push_duration):
                i = len(self.cfg.domain_rand.push_duration) - 1
            duration = self.cfg.domain_rand.push_duration[i]/self.dt
            if self.common_step_counter % self.cfg.domain_rand.push_interval <= duration:
                self._push_robots()
            else:
                self.rand_push_force.zero_()
                self.rand_push_torque.zero_()

    def compute_observations(self):

        phase = self._get_phase()
        self.compute_ref_state()

        sin_pos = torch.sin(2 * torch.pi * phase).unsqueeze(1)
        cos_pos = torch.cos(2 * torch.pi * phase).unsqueeze(1)

        stance_mask = self._get_stance_mask()
        contact_mask = self.contact_forces[:, self.feet_indices, 2] > 5.

        self.command_input = torch.cat(
            (sin_pos, cos_pos, self.commands[:, :3] * self.commands_scale), dim=1)

        # critic no lag
        diff = self.dof_pos - self.ref_dof_pos
        # 73
        privileged_obs_buf = torch.cat((
            self.command_input,  # 2 + 3
            (self.dof_pos - self.default_joint_pd_target) * self.obs_scales.dof_pos,  # 12
            self.dof_vel * self.obs_scales.dof_vel,  # 12
            self.actions,  # 12
            diff,  # 12
            self.base_lin_vel * self.obs_scales.lin_vel,  # 3
            self.base_ang_vel * self.obs_scales.ang_vel,  # 3
            self.base_euler_xyz * self.obs_scales.quat,  # 3
            self.rand_push_force[:, :2],  # 2
            self.rand_push_torque,  # 3
            self.env_frictions,  # 1
            self.body_mass / 10.,  # 1 # sum of all fix link mass
            stance_mask,  # 2
            contact_mask,  # 2
        ), dim=-1)

        # random add dof_pos and dof_vel same lag
        if self.cfg.domain_rand.add_dof_lag:
            if self.cfg.domain_rand.randomize_dof_lag_timesteps_perstep:
                self.dof_lag_timestep = torch.randint(self.cfg.domain_rand.dof_lag_timesteps_range[0], 
                                                  self.cfg.domain_rand.dof_lag_timesteps_range[1]+1,(self.num_envs,),device=self.device)
                cond = self.dof_lag_timestep > self.last_dof_lag_timestep + 1
                self.dof_lag_timestep[cond] = self.last_dof_lag_timestep[cond] + 1
                self.last_dof_lag_timestep = self.dof_lag_timestep.clone()
            self.lagged_dof_pos = self.dof_lag_buffer[torch.arange(self.num_envs), :self.num_actions, self.dof_lag_timestep.long()]
            self.lagged_dof_vel = self.dof_lag_buffer[torch.arange(self.num_envs), -self.num_actions:, self.dof_lag_timestep.long()]  
        # random add dof_pos and dof_vel different lag
        elif self.cfg.domain_rand.add_dof_pos_vel_lag:
            if self.cfg.domain_rand.randomize_dof_pos_lag_timesteps_perstep:
                self.dof_pos_lag_timestep = torch.randint(self.cfg.domain_rand.dof_pos_lag_timesteps_range[0], 
                                                  self.cfg.domain_rand.dof_pos_lag_timesteps_range[1]+1,(self.num_envs,),device=self.device)
                cond = self.dof_pos_lag_timestep > self.last_dof_pos_lag_timestep + 1
                self.dof_pos_lag_timestep[cond] = self.last_dof_pos_lag_timestep[cond] + 1
                self.last_dof_pos_lag_timestep = self.dof_pos_lag_timestep.clone()
            self.lagged_dof_pos = self.dof_pos_lag_buffer[torch.arange(self.num_envs), :, self.dof_pos_lag_timestep.long()]

            if self.cfg.domain_rand.randomize_dof_vel_lag_timesteps_perstep:
                self.dof_vel_lag_timestep = torch.randint(self.cfg.domain_rand.dof_vel_lag_timesteps_range[0], 
                                                  self.cfg.domain_rand.dof_vel_lag_timesteps_range[1]+1,(self.num_envs,),device=self.device)
                cond = self.dof_vel_lag_timestep > self.last_dof_vel_lag_timestep + 1
                self.dof_vel_lag_timestep[cond] = self.last_dof_vel_lag_timestep[cond] + 1
                self.last_dof_vel_lag_timestep = self.dof_vel_lag_timestep.clone()
            self.lagged_dof_vel = self.dof_vel_lag_buffer[torch.arange(self.num_envs), :, self.dof_vel_lag_timestep.long()]
        # dof_pos and dof_vel has no lag
        else:
            self.lagged_dof_pos = self.dof_pos
            self.lagged_dof_vel = self.dof_vel

        # imu lag, including rpy and omega
        if self.cfg.domain_rand.add_imu_lag:    
            if self.cfg.domain_rand.randomize_imu_lag_timesteps_perstep:
                self.imu_lag_timestep = torch.randint(self.cfg.domain_rand.imu_lag_timesteps_range[0], 
                                                  self.cfg.domain_rand.imu_lag_timesteps_range[1]+1,(self.num_envs,),device=self.device)
                cond = self.imu_lag_timestep > self.last_imu_lag_timestep + 1
                self.imu_lag_timestep[cond] = self.last_imu_lag_timestep[cond] + 1
                self.last_imu_lag_timestep = self.imu_lag_timestep.clone()
            self.lagged_imu = self.imu_lag_buffer[torch.arange(self.num_envs), :, self.imu_lag_timestep.int()]
            self.lagged_base_ang_vel = self.lagged_imu[:,:3].clone()
            self.lagged_base_euler_xyz = self.lagged_imu[:,-3:].clone()
        # no imu lag
        else:              
            self.lagged_base_ang_vel = self.base_ang_vel[:,:3]
            self.lagged_base_euler_xyz = self.base_euler_xyz[:,-3:]

        # obs q and dq
        q = (self.lagged_dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos
        dq = self.lagged_dof_vel * self.obs_scales.dof_vel  

        # 47
        obs_buf = torch.cat((
            self.command_input,  # 5 = 2D(sin cos) + 3D(vel_x, vel_y, aug_vel_yaw)
            q,    # 12
            dq,  # 12
            self.actions,   # 12
            self.lagged_base_ang_vel * self.obs_scales.ang_vel,  # 3
            self.lagged_base_euler_xyz * self.obs_scales.quat,  # 3
        ), dim=-1)

        if self.cfg.env.num_single_obs == 48:
            stand_command = self._get_stand_command().unsqueeze(1)  # 与 phase 判定同源（原始指令），避免注入纠偏翻转 stand 位
            obs_buf = torch.cat((obs_buf, stand_command),dim=1)

        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            privileged_obs_buf = torch.cat((privileged_obs_buf.clone(), heights), dim=-1)

        if self.add_noise:  
            # add obs noise
            obs_now = obs_buf.clone() + (2 * torch.rand_like(obs_buf) -1) * self.noise_scale_vec * self.cfg.noise.noise_level
        else:
            obs_now = obs_buf.clone()

        self.obs_history.append(obs_now)
        self.critic_history.append(privileged_obs_buf)

        obs_buf_all = torch.stack([self.obs_history[i]
                                   for i in range(self.obs_history.maxlen)], dim=1)  # N,T,K

        self.obs_buf = obs_buf_all.reshape(self.num_envs, -1)  # N, T*K
        self.privileged_obs_buf = torch.cat([self.critic_history[i] for i in range(self.cfg.env.c_frame_stack)], dim=1)

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):
            self.update_command_curriculum(env_ids)

        # reset rand dof_pos and dof_vel=0
        self._reset_dofs(env_ids)

        # reset base position
        self._reset_root_states(env_ids)

        # Randomize joint parameters, like torque gain friction ...
        self.randomize_dof_props(env_ids)
        self._refresh_actor_dof_props(env_ids)
        self.randomize_lag_props(env_ids)

        # reset buffers
        self.last_last_actions[env_ids] = 0.
        self.actions[env_ids] = 0.
        self.last_actions[env_ids] = 0.
        self.last_rigid_state[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.last_root_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.phase_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        if getattr(self.cfg.domain_rand, 'enable_delivery', False):
            self.delivery_tau_lpf[env_ids] = 0.
        # rand 0 or 0.5
        self.gait_start[env_ids] = torch.randint(0, 2, (len(env_ids),)).to(self.device)*0.5

        #resample command
        self.generate_gait_time(env_ids)
        self._resample_commands()

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        if self.cfg.terrain.mesh_type == "trimesh":
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

        # fix reset gravity bug
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.base_quat[env_ids] = self.root_states[env_ids, 3:7]
        self.base_euler_xyz = get_euler_xyz_tensor(self.base_quat)
        # v3 yaw_hold：episode 锚点=初始朝向；prev_turning 清 False（当前 init rot 为 identity → anchor=0，
        # 但按实际朝向取值保持通用）。forward_vec 是全集 (num_envs,3)，必须同步 [env_ids] 取子集（v2 形状 bug 教训）
        fwd = quat_apply(self.base_quat[env_ids], self.forward_vec[env_ids])
        self.anchor_yaw[env_ids] = torch.atan2(fwd[:, 1], fwd[:, 0])
        self.prev_turning[env_ids] = False
        self.projected_gravity[env_ids] = quat_rotate_inverse(self.base_quat[env_ids], self.gravity_vec[env_ids])
        self.base_lin_vel[env_ids] = quat_rotate_inverse(self.base_quat[env_ids], self.root_states[env_ids, 7:10])
        self.base_ang_vel[env_ids] = quat_rotate_inverse(self.base_quat[env_ids], self.root_states[env_ids, 10:13])
        self.feet_quat = self.rigid_state[:, self.feet_indices, 3:7]
        self.feet_euler_xyz = get_euler_xyz_tensor(self.feet_quat)

        # clear obs history buffer and privileged obs buffer
        for i in range(self.obs_history.maxlen):
            self.obs_history[i][env_ids] *= 0
        for i in range(self.critic_history.maxlen):
            self.critic_history[i][env_ids] *= 0

    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        super()._init_buffers()
        self.gait_time = torch.zeros(self.num_envs, len(self.cfg.commands.gait) ,dtype=torch.int, device=self.device, requires_grad=False)
        self.phase_length_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long)
        self.gait_start = torch.randint(0, 2, (self.num_envs,)).to(self.device)*0.5
        # v3 yaw_hold：锚点/原始指令/上步转向态（注入会覆写 commands[:,2]，raw 必须独立 buffer）
        self.anchor_yaw = torch.zeros(self.num_envs, device=self.device)
        self.raw_yaw_cmd = torch.zeros(self.num_envs, device=self.device)
        self.prev_turning = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _reward_ref_joint_pos(self):
        """
        Calculates the reward based on the difference between the current joint positions and the target joint positions.
        """
        joint_pos = self.dof_pos.clone()
        pos_target = self.ref_dof_pos.clone()
        stand_command = self._get_stand_command()  # 原始指令判定：纠偏注入不改变站立位目标
        pos_target[stand_command] = self.default_dof_pos.clone()
        diff = joint_pos - pos_target
        r = torch.exp(-2 * torch.norm(diff, dim=1)) - 0.2 * torch.norm(diff, dim=1).clamp(0, 0.5)
        r[stand_command] = 1.0
        return r

    def _reward_feet_air_time(self):
        """
        Calculates the reward for feet air time, promoting longer steps. This is achieved by
        checking the first contact with the ground after being in the air. The air time is
        limited to a maximum value for reward calculation.
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > 40.  # reward_v1 C3
        stance_mask = self._get_stance_mask().clone()
        stance_mask[self._get_stand_command()] = 1  # air_time：原始指令判定（原 < 0.05 硬编码）
        self.contact_filt = torch.logical_or(torch.logical_or(contact, stance_mask), self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * self.contact_filt
        self.feet_air_time += self.dt
        air_time = self.feet_air_time.clamp(0, 0.5) * first_contact
        self.feet_air_time *= ~self.contact_filt
        return air_time.sum(dim=1)

    def _reward_feet_contact_number(self):
        """
        Calculates a reward based on the number of feet contacts aligning with the gait phase. 
        Rewards or penalizes depending on whether the foot contact matches the expected gait phase.
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > 40.  # reward_v1 C3
        stance_mask = self._get_stance_mask().clone()
        stance_mask[self._get_stand_command()] = 1  # contact_number：原始指令判定
        reward = torch.where(contact == stance_mask, 1, -1.0)  # reward_v1 C2：-0.3→-1.0，节拍失配重罚（唯一验证过的节拍器：ank 4.4→1.9Hz）
        return torch.mean(reward, dim=1)

    def _reward_joint_symmetry(self):
        """
        Penalizes the non-mirror-symmetric component of hip_roll/hip_yaw/ankle_roll.
        For mirror joints, q_left + q_right should stay at the default mirror sum (0).
        A persistent asymmetric hip_yaw/ankle_roll offset acts as a constant yaw torque
        and drives circling (observed: cmd 0.25 m/s, L hy -0.320 / R hy +0.340 rad
        -> yaw rate -2.75 deg/s). Gated off for |ang_vel_yaw cmd| > 0.15, where
        differential hip_yaw is the legitimate turning actuator.
        """
        d = self.dof_pos - self.default_dof_pos
        sym_err = (torch.abs(d[:, 1] + d[:, 7])   # hip_roll L+R
                   + torch.abs(d[:, 2] + d[:, 8])   # hip_yaw L+R
                   + torch.abs(d[:, 5] + d[:, 11])) # ankle_roll L+R
        r = torch.exp(-sym_err * 30.)
        # 转向闸读原始指令：注入的 yaw_hold 纠偏 wz（可达 0.25）不应关闭对称约束，
        # 否则站立纠偏段放行不对称 hip_yaw，回到 circling 根因；仅用户真实转向指令放行
        if self.cfg.commands.yaw_hold:
            turning = torch.abs(self.raw_yaw_cmd) > 0.15
        else:
            turning = torch.abs(self.commands[:, 2]) > 0.15
        r[turning] = 1.
        return r

    def _reward_tracking_lin_vel(self):
        """
        Tracks linear velocity commands along the xy axes. 
        Calculates a reward based on how closely the robot's linear velocity matches the commanded values.
        """
        stand_command = self._get_stand_command()  # tracking_lin：原始指令判定（误差仍对注入后 wz 闭环）
        lin_vel_error_square = torch.sum(torch.square(
            self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        lin_vel_error_abs = torch.sum(torch.abs(
            self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        r_square = torch.exp(-lin_vel_error_square * self.cfg.rewards.tracking_sigma)
        r_abs = torch.exp(-lin_vel_error_abs * self.cfg.rewards.tracking_sigma * 2)
        r = torch.where(stand_command, r_abs, r_square)

        return r

    def _reward_tracking_ang_vel(self):
        """
        Tracks angular velocity commands for yaw rotation.
        Computes a reward based on how closely the robot's angular velocity matches the commanded yaw values.
        """   
        stand_command = self._get_stand_command()  # tracking_ang：原始指令判定（误差仍对注入后 wz 闭环）
        ang_vel_error_square = torch.square(
            self.commands[:, 2] - self.base_ang_vel[:, 2])
        ang_vel_error_abs = torch.abs(
            self.commands[:, 2] - self.base_ang_vel[:, 2])
        r_square = torch.exp(-ang_vel_error_square * self.cfg.rewards.tracking_sigma)
        r_abs = torch.exp(-ang_vel_error_abs * self.cfg.rewards.tracking_sigma * 2)
        r = torch.where(stand_command, r_abs, r_square)

        return r

    def _reward_stand_still(self):
        # penalize motion at zero commands
        stand_command = self._get_stand_command()  # 原始指令判定：纠偏注入不剥夺 stand_still 奖励
        r = torch.exp(-torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=1))
        r = torch.where(stand_command, r.clone(),
                        torch.zeros_like(r))
        return r
