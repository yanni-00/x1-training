# F1 参数接口

当前 `parameter_identification.json` 是待填模板，`status=pending`。没有 F1 实测数据，禁止以旧 X1 数值、典型值或估算填充。

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
| `delivery.tau_s` | 启用时为实测正时间常数（秒）；本实现要求不小于 sim.dt，否则需审核离散滤波实现 |

注入后程序将 nominal / range 按 F1 关节顺序转换为每关节 armature 和显式摩擦配置。play 侧使用相同 nominal，忽略其 DR 区间；保留原力矩顺序 PD → delivery LPF → 显式摩擦 → 力矩限幅。
参数文件的数据就绪、用户启动授权、OMA 暂停权限和正式 preflight 是不同条件。填写参数不打开 `launch.json`，不创建训练任务，也不解除 `.oma/loop.json` 暂停。

## 两阶段 checkpoint

阶段一从零开始。阶段二要求显式 `--checkpoint_path` 和对应 `.pt.f1.json`；sidecar 含模型 SHA256、F1 URDF / robot_contract / 辨识配置 SHA256、任务身份、已完成更新数与父权重来源。
校验用于防止误用旧 X1 权重、错误阶段或变更物理契约后的权重，不代替实际加载 / 有限值、新权重闭环或 OMA 模型交付验收。
