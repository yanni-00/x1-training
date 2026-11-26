# F1 参数接口

当前 `parameter_identification.json` 为 F1 100 Hz 辨识版配置。它绑定原 005 URDF 和用户确认的参数，包含两路随机纯延迟、踝关节力矩低通和名义 Kp/Kd。辨识的三个量固定取配置中的名义值。

## Gradmotion 训练交接

提交前固定本分支的实际 Git 提交，按 OMA 核对任务准入和 GM 账号预算。GM 代码目录为 `x1-training` 时使用以下启动命令。

```shell
gm-run x1-training/humanoid/scripts/train_stage1_identified_100hz.py --task f1_walk_stage1 --headless --num_envs 4096 --seed 5 --max_iterations 6000 --run_name f1_walking_100hz_identified_6000
```

入口使用 r11 最终策略与奖励，从随机初始化训练。启动日志中核对 `[F1 timing]`、`[F1 dynamics]` 和 `[F1 profile]`。保留 `model_3000.pt`、`model_4500.pt` 和 `model_6000.pt` 及各自的 `.f1.json` sidecar。平台若只归档权重，需在同一训练输出中另行保留 sidecar。

在训练仓库根目录运行 `python -m humanoid.scripts.play --task f1_walk_stage1 --checkpoint_path <本版权重>` 可做定性回放。量化验收沿用旧 r11 的协议和工况，对比前进、横移、转向后停步及提前终止。旧基线对应 `TASK_20261125_569` 的 `model_6000`，使用其原始提交和物理配置回放。

## 冻结资产

`robot_contract.json` 记录本次复制 URDF 的 SHA256、每个开放关节的原始轴 / origin / limit / dynamics，以及静态姿态推导。
URDF 和 meshes 保持用户指定版本原字节；任何后续模型改变均使旧辨识、checkpoint 和验证依据失效，应重新核对，不直接改 hash 掩盖差异。
初始姿态保留原路线膝屈曲幅值，按 F1 几何解髋与踝 pitch，令足掌水平、踝原点位于髋 pitch 原点正下方；base 高度取实际碰撞网格最低点。它是工程初始设置，动态效果待验证。

## 辨识 JSON 字段

| 字段 | 要求 |
|---|---|
| `status` | 数据齐备前为 `pending`；经核对后才设为 `ready` |
| `robot` | `F1-12DOF` |
| `urdf_sha256` | 与本次 F1 URDF 精确匹配 |
| `sources` | 实际数据来源数组，每项包含 `file` 和非空 `fields` 数组；字段 / 行应能追溯本次填入数值 |
| `joints` | 与 robot_contract 的开放关节名称严格一一对应 |
| `joints.<name>.armature` | 辨识得到的附加关节等效惯量，单位 kg·m²；必须先核对 URDF 所含惯量的扣除语义，不能直接填总惯量 |
| `joints.<name>.coulomb` | 额外显式库仑摩擦，单位 N·m |
| `joints.<name>.viscous` | 额外显式粘性摩擦，单位 N·m·s/rad |
| `*_range` | 有依据的绝对 DR 区间 `[lower,upper]`，必须非负并包含 nominal；`null` 表示固定 nominal，不自动套旧 X1 随机化幅度 |
| `passive_dynamics.mode` | 本次为 `preserve_urdf`：保留 URDF 被动 damping / friction |
| `passive_dynamics.explicit_friction_semantics` | 额外显式摩擦非零时，必须明确为 `additional_to_urdf`；若辨识的是总摩擦，应先分离语义，不能双计 |
| `delivery.enabled` | 辨识结果明确给出是否启用 LPF，不由旧 X1 设置推断 |
| `delivery.joint_names` | 用关节名映射，禁止依赖未知索引 |
| `delivery.tau_s` | 启用时为用户确认的正时间常数（秒），要求不小于 sim.dt |

注入后程序将 nominal / range 按 F1 关节顺序转换为每关节 armature 和显式摩擦配置。play 侧使用相同 nominal，忽略其 DR 区间；保留原力矩顺序 PD → delivery LPF → 显式摩擦 → 力矩限幅。
代码侧通过 `launch.json` 选择辨识版。正式 GM 提交须核对独立的 OMA 准入和用户授权。

## 两阶段 checkpoint

阶段一从零开始。阶段二要求显式 `--checkpoint_path` 和对应 `.pt.f1.json`；sidecar 含模型 SHA256、F1 URDF / robot_contract / 辨识配置 SHA256、任务身份、已完成更新数与父权重来源。
校验用于防止误用旧 X1 权重、错误阶段或变更物理契约后的权重，不代替实际加载 / 有限值、新权重闭环或 OMA 模型交付验收。
