# ============================================================================
# Isaac Lab 足球机器人强化学习 — 环境配置文件（完整注释版）
# ============================================================================
# 本文件定义了一个 Manager-Based RL 环境，用于训练差速驱动小车踢足球。
#
# 整体架构:
#   场景(Scene) → 动作(Actions) → 观测(Observations) → 事件(Events)
#                                                     → 奖励(Rewards)
#                                                     → 终止(Terminations)
#   最终组合为 SoccerEnvCfg，继承 ManagerBasedRLEnvCfg。
#
# 核心概念:
#   - 每个装饰器 @configclass 定义一个配置"模块"
#   - 每个模块由 Term 组成，Term = 函数指针 + 参数 + 模式/权重
#   - 框架在运行时遍历所有 Term，调用对应的函数，完成仿真闭环
# ============================================================================

import math
import os

# ============================================================================
# Isaac Lab 框架导入
# ============================================================================

# 仿真工具：用于生成物理地面、光照、USD 加载等
import isaaclab.sim as sim_utils

# 资产类型：分别是铰接体(带关节)、静态物体、刚体
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg

# 环境基类：Manager-Based RL 的标准环境配置父类
from isaaclab.envs import ManagerBasedRLEnvCfg

# 管理器 Term 类型：
#   EventTerm     - 事件（重置时随机化、定时扰动等）
#   ObsGroup      - 观测组（一组观测项的容器）
#   ObsTerm       - 单个观测项（一条传感器读数）
#   RewTerm       - 单个奖励项（一个打分规则）
#   DoneTerm      - 单个终止条件（一局何时结束）
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm

# 交互场景基类
from isaaclab.scene import InteractiveSceneCfg

# @configclass 装饰器：Isaac Lab 对 Python @dataclass 的封装
# 作用类似 @dataclass，但增加了框架的配置管理能力
from isaaclab.utils import configclass

# 噪声模型：用于给观测/动作加噪声，模拟真实传感器和电机的不确定性
#   NoiseModelWithAdditiveBiasCfg  = 加性噪声 + 每局重置时的随机偏置
#   GaussianNoiseCfg               = 高斯分布的噪声
from isaaclab.utils.noise import NoiseModelWithAdditiveBiasCfg, GaussianNoiseCfg

# 隐式执行器：不需要自己写电机模型，框架根据关节名自动处理
from isaaclab.actuators import ImplicitActuatorCfg

# 渲染/视觉材质（当前文件中实际未使用，保留为可选扩展）
from isaaclab.sim import SphereCfg, VisualMaterialCfg

# ============================================================================
# 本地模块导入
# ============================================================================

# mdp.py 包含所有自定义的奖励函数、观测函数、终止判断函数、事件函数
# 每个函数的结构为: func(env, **params) -> torch.Tensor
from . import mdp

# ============================================================================
# USD 模型路径
# ============================================================================

# 小车 USD 模型的存放目录
# 小车是一个差速驱动双轮机器人，包含 4 个关节:
#   wheel_left_joint  (驱动轮)
#   wheel_right_joint (驱动轮)
#   wheel_front_joint (被动轮)
#   wheel_back_joint  (被动轮)
ASSETS_DIR = "/home/elong/isaac-sim-wjg/Collected_car"

# 场地 USD 模型的存放目录
# 场地是一个静态场景，包含:
#   - 地面纹理、墙壁/围栏
#   - 已嵌入的球 (golf_ball)
GROUND_DIR = "/home/elong/isaac-sim-wjg/"


# ============================================================================
# 1. 场景定义 (SceneCfg)
# ============================================================================
# 定义仿真场景中有哪些物理实体，以及每个实体的位置、物理属性、加载方式。
# 所有实体在 512 个并行环境中各自拥有一份副本。
#
# 实体类型对比:
#   AssetBaseCfg      - 静态装饰物（地面、场地、灯光）—— 没有物理运动
#   RigidObjectCfg    - 刚体（球）—— 有物理运动，但没有关节
#   ArticulationCfg   - 铰接体（小车）—— 有关节、有执行器、可以主动运动
# ============================================================================

@configclass
class SoccerSceneCfg(InteractiveSceneCfg):
    """足球机器人仿真场景定义"""

    # --------------------------------------------------------------------------
    # 0. 物理地面 (框架内置)
    # --------------------------------------------------------------------------
    # 创建一个无限大的地面平面，带有物理碰撞属性。
    # 不加载外部 USD 文件，直接使用 Isaac Lab 内置的 GroundPlaneCfg 生成。
    # prim_path="/World/ground" 表示这个实体在 USD 层级中属于全局共享
    # （所有并行环境共用同一片地面，而非每环境一份）。
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    # --------------------------------------------------------------------------
    # 1. 场地 — 静态场景 (外部 USD 模型)
    # --------------------------------------------------------------------------
    # 从 elong_ground_copy.usd 文件加载足球场地的 3D 模型。
    # prim_path 中的 {ENV_REGEX_NS} 是 Isaac Lab 的占位符，运行时会自动展开为
    # "/World/envs/env_0", "/World/envs/env_1", ..., "/World/envs/env_511"
    # 这样每个并行环境都能拿到自己独立的场景副本。
    arena = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Arena",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{GROUND_DIR}/elong_ground_copy.usd",
            scale=(1.0, 1.0, 1.0),
        ),
    )

    # --------------------------------------------------------------------------
    # 2. 足球 — 刚体 (引用 Arena 内已有的球)
    # --------------------------------------------------------------------------
    # 球已经在场地 USD 模型中作为 golf_ball 存在，所以 spawn=None，
    # 不必再生成一个。只需要告诉 Isaac Lab "把这个 USD 路径下的物体当作刚体"。
    #
    # 初始 Z=0.072 即球的半径（约 7.2cm），让球贴地放置。
    ball = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Arena/golf_ball",
        spawn=None,  # 不新生成，直接引用 arena 里已有的球
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.072)),
    )

    # --------------------------------------------------------------------------
    # 3. 机器人 (差速驱动小车) — 铰接体 (外部 USD 模型)
    # --------------------------------------------------------------------------
    # 从 car.usd 加载小车模型。
    #
    # 核心物理设置:
    #   fix_root_link=False  ← 关键！基座不固定，小车才能在地面上自由运动
    #   enabled_self_collisions=False ← 防止车轮和车身之间的碰撞导致卡死
    #   disable_gravity=False ← 让小车受重力（自然落在地面上）
    #
    # 执行器配置:
    #   小车有 4 个轮子关节，分成两组:
    #     "wheels" (驱动轮):  left + right  → 通过速度指令主动控制
    #     "passive_wheels" (被动轮): front + back → 只能被动转动，无驱动
    #
    #   差速驱动原理:
    #     左轮快 + 右轮慢 = 右转
    #     右轮快 + 左轮慢 = 左转
    #     两轮等速 = 直行
    #     两轮反向 = 原地旋转
    # ==========================================================================
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ASSETS_DIR}/car.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,  # 受重力影响
            ),
            # +++ 解锁基座，允许小车自由运动 +++
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,  # 防止车身-车轮自碰撞
                fix_root_link=False,            # 核心：绝对不固定基座！
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.1),  # 初始高度 10cm（略高于球），避免生成时与地面穿插
        ),
        # ----------------------------------------------------------------------
        # 执行器字典 — 定义如何控制每个关节
        # ----------------------------------------------------------------------
        actuators={
            # --- 驱动轮组 ---
            "wheels": ImplicitActuatorCfg(
                joint_names_expr=["wheel_left_joint", "wheel_right_joint"],
                effort_limit_sim=200.0,   # 最大力矩 (N·m)，防止电机输出过大
                velocity_limit_sim=120.0,  # 最大转速 (rad/s)
                stiffness=0.0,             # 关节刚度=0 → 纯速度控制模式
                damping=100.0,             # 关节阻尼，模拟传动系统的阻力
            ),
            # --- 被动轮组 ---
            "passive_wheels": ImplicitActuatorCfg(
                joint_names_expr=["wheel_back_joint", "wheel_front_joint"],
                effort_limit_sim=None,   # None = 不施加任何驱动力
                velocity_limit_sim=None,  # None = 不受速度限制
                stiffness=0.0,
                damping=0.1,              # 轻微阻尼，模拟真实轴承摩擦力
            ),
        },
    )

    # --------------------------------------------------------------------------
    # 4. 环境光照 (框架内置)
    # --------------------------------------------------------------------------
    # DomeLight 是半球环境光，模拟均匀的室内/室外光线。
    # intensity=3000 高亮度确保渲染清晰。
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )


# ============================================================================
# 2. 动作空间定义 (ActionsCfg)
# ============================================================================
# 定义策略网络输出什么，以及这些输出如何映射到物理关节。
#
# 当前方案: 关节速度控制 (JointVelocityActionCfg)
#   策略输出 2 个值 → 分别映射为左轮和右轮的目标速度 (rad/s)
#
# scale=10.0 的含义:
#   策略网络通常用 tanh 激活 → 输出范围 [-1, 1]
#   [-1, 1] × 10.0 = [-10, 10] rad/s ← 这才是物理关节收到指令
#   scale 是"无量纲 → 物理单位"的翻译器
# ============================================================================

@configclass
class ActionsCfg:
    """动作空间: 差速驱动的双轮速度控制"""

    drive_velocity = mdp.JointVelocityActionCfg(
        asset_name="robot",                            # 控制场景中名为 "robot" 的物体
        joint_names=["wheel_left_joint", "wheel_right_joint"],  # 只控制两个驱动轮
        scale=10.0,                                    # 缩放: 策略输出[-1,1] → 目标速度[-10,10] rad/s
    )


# ============================================================================
# 3. 观测空间定义 (ObservationsCfg)
# ============================================================================
# 定义策略网络每步"看到"什么信息。
#
# 每个 ObsTerm 的结构:
#   func   = 读取函数 (定义在 mdp.py)，指定"怎么算"
#   params = {"asset_cfg": SceneEntityCfg("xxx")} 指定"从哪个物体读"
#
# 运行时流程:
#   1. SceneEntityCfg("robot") 拿到场景中 robot 物体的引用
#   2. func(env, asset_cfg) 从物理引擎读取刚体状态，返回一个 torch.Tensor
#   3. 所有 ObsTerm 的结果按顺序拼接成一个向量 → 送入策略网络
#
# concatenate_terms = True 表示所有观测项首尾拼接。
# ============================================================================

@configclass
class ObservationsCfg:
    """观测空间: 11 维状态向量"""

    @configclass
    class PolicyCfg(ObsGroup):
        """策略网络的观测组"""

        # ----------------------------------------------------------------------
        # --- 机器人自身状态 ---
        # ----------------------------------------------------------------------

        # 机器人基座线速度 XY 分量 (Vx, Vy)
        # 形状: [num_envs, 2]
        # 只取 XY，忽略 Z（小车在地面上跑，不需要垂直速度）
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel_xy,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        # 机器人基座角速度 Z 分量 (Wz) — 偏航角速度
        # 形状: [num_envs, 1]
        # 只取 Z 轴旋转（水平面内转向），忽略俯仰和滚转
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel_z,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        # ----------------------------------------------------------------------
        # --- 球的运动状态 ---
        # ----------------------------------------------------------------------

        # 球的速度 XY 分量 (Vx, Vy)
        # 形状: [num_envs, 2]
        # 注意: 复用了和机器人线速度一样的函数 base_lin_vel_xy
        # 只是 asset_cfg 指向了 "ball" 而不是 "robot"
        ball_vel = ObsTerm(
            func=mdp.base_lin_vel_xy,
            params={"asset_cfg": SceneEntityCfg("ball")},
        )

        # ----------------------------------------------------------------------
        # --- 方位信息 ---
        # ----------------------------------------------------------------------

        # 机器人朝向 (sin(yaw), cos(yaw))
        # 形状: [num_envs, 2]
        # 使用 sin/cos 而非原始弧度值，原因:
        #   - 弧度 3.14 和弧度 -3.14 其实是一个方向，但神经网络会认为是两个完全不同的值
        #   - sin/cos 把这个"相位环绕"问题消除，连续且唯一
        #   - sin² + cos² = 1 也给网络提供了归一化约束
        robot_ori = ObsTerm(
            func=mdp.robot_yaw_sin_cos,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        # 机器人在场地内的位置 (归一化 XY)
        # 形状: [num_envs, 2]
        # 归一化规则: X / 1.8, Y / 2.2（场地尺寸）
        # 归一化后的值约在 [0, 1] 范围，有利于神经网络训练
        robot_pos = ObsTerm(
            func=mdp.positions_relative_to_arena,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        # 球在场地的位置 (归一化 XY)
        # 形状: [num_envs, 2]
        # 同样的归一化规则，与 robot_pos 使用同一个函数
        ball_pos = ObsTerm(
            func=mdp.positions_relative_to_arena,
            params={"asset_cfg": SceneEntityCfg("ball")},
        )

        # ----------------------------------------------------------------------
        # 观测组后处理
        # ----------------------------------------------------------------------
        def __post_init__(self):
            # 不添加观测噪声破坏（噪声在顶层 SoccerEnvCfg 统一管理）
            self.enable_corruption = False
            # 将上面 6 个 ObsTerm 的结果首尾拼接成一个向量
            # 最终维度: 2 + 1 + 2 + 2 + 2 + 2 = 11 维
            self.concatenate_terms = True

    # policy 是 ObservationsCfg 的唯一成员，类型为上面定义的 PolicyCfg
    policy: PolicyCfg = PolicyCfg()


# ============================================================================
# 4. 事件定义 (EventCfg)
# ============================================================================
# 事件是仿真过程中的"钩子"，在特定时机自动触发。
#
# Isaac Lab 支持三种事件模式:
#   startup  — 整个仿真刚启动时触发（仅一次）。用于加载模型、初始化光照等。
#   reset    — 每一局开始时触发（最常用）。用于随机化初始条件。
#   interval — 按固定时间间隔触发（如每 2 秒）。用于持续性扰动。
#
# 当前代码全部使用 mode="reset"，即每局随机化初始状态，
# 目的是让策略见过足够多的起点 → 提高泛化能力。
# ============================================================================

@configclass
class EventCfg:
    """事件: 每局重置时的随机化规则"""

    # --------------------------------------------------------------------------
    # 1. 随机化机器人初始位置
    # --------------------------------------------------------------------------
    # 每局开始，小车被随机放到场地内的任意位置。
    #
    # pose_range 各字段含义:
    #   x: (0.1, 1.7) — 场地纵向 0.1m ~ 1.7m 之间均匀采样
    #   y: (0.1, 2.0) — 场地横向 0.1m ~ 2.0m 之间均匀采样
    #   z: (0.19, 0.2) — 高度在 19cm~20cm 间微小波动（贴地但有微小随机）
    #
    # velocity_range: {} — 空字典表示初始速度全部归零（静止起步）
    #
    # 为什么这样设计？
    #   - XY 大范围随机 → 策略不能"背答案"，必须学会从任何位置找到球
    #   - Z 微小波动 → 模拟地面不平整，但不是真的飞起来
    #   - 速度归零 → 每局公平起步，不给"偷跑"的机会
    # ==========================================================================
    reset_robot = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {
                "x": (0.1, 1.7),
                "y": (0.1, 2.0),
                "z": (0.19, 0.2),
            },
            "velocity_range": {},  # 初始速度全部为 0
        },
    )

    # --------------------------------------------------------------------------
    # 2. 随机化球的初始位置
    # --------------------------------------------------------------------------
    # 将球放在对方半场附近（X=0.8~0.9, Y=1.8~1.9），接近球门一侧。
    #
    # 设计意图:
    #   - 让球起点偏向对方半场 → 引导小车学会"向前推进"而非后退
    #   - 球不打乱速度（velocity_range={}）→ 球从静止开始，小车主动去碰
    # ==========================================================================
    reset_ball = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("ball"),
            "pose_range": {
                "x": (0.8, 0.9),
                "y": (1.8, 1.9),  # 放在对方半场，靠近球门
            },
            "velocity_range": {},  # 球从静止开始
        },
    )

    # --------------------------------------------------------------------------
    # 3. 随机化摩擦力
    # --------------------------------------------------------------------------
    # 每局开始时随机改变机器人与地面的摩擦系数。
    #
    # 这是"域随机化"的核心手段:
    #   训练时经历了各种摩擦力（从冰面到粗糙地毯）→ 部署时适应真实地面
    #
    # 参数含义:
    #   static_friction_range:  (0.4, 1.2) — 静摩擦系数范围
    #   dynamic_friction_range: (0.3, 1.0) — 动摩擦系数范围
    #   restitution_range:      (0.0, 0.2) — 弹性系数（低弹性防止反弹乱跳）
    #   num_buckets: 64 — 将连续随机值离散化为 64 个桶，加速并行计算
    # ==========================================================================
    randomize_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "static_friction_range": (0.4, 1.2),
            "dynamic_friction_range": (0.3, 1.0),
            "restitution_range": (0.0, 0.2),
            "num_buckets": 64,  # 离散化桶数，提高 GPU 并行效率
        },
    )


# ============================================================================
# 5. 奖励函数定义 (RewardsCfg)
# ============================================================================
# 强化学习的核心——奖励函数就是策略的"价值观"。
# 每个 RewTerm 每步仿真都计算一次，最终总奖励 = Σ(每项分数 × 权重)。
#
# 设计思路: "稠密引导 + 稀疏大奖"
#   稠密引导 (dense reward) — 每步都有信号，帮策略快速找到方向
#   稀疏大奖 (sparse reward) — 只有真正进球才给，确保最终目标不偏离
#
# 奖励层级:
#   找球 → 面向球 → 冲刺撞球 → 推球朝球门 → 进球
#     3      1         2          5          2000
#   └─── 稠密引导 ────────────┘└── 稀疏大奖 ──┘
#
# 同时贯穿全场: is_alive = -1 (时间惩罚，逼迫快速完成)
# ============================================================================

@configclass
class RewardsCfg:
    """奖励函数: 7 项分数 × 各自权重 = 总奖励"""

    # --------------------------------------------------------------------------
    # 1. approach_ball — 靠近球的距离奖励 (权重 +3)
    # --------------------------------------------------------------------------
    # 实现: exp(-2.5 × distance(robot, ball))
    #   距离 0m   → 奖励 ~3.0
    #   距离 0.2m → 奖励 ~1.8
    #   距离 0.5m → 奖励 ~0.9
    #   距离 1.0m → 奖励 ~0.25
    #
    # 使用指数衰减而非线性，因为:
    #   - 远距离时梯度平缓，提供持续的"往球走"的信号
    #   - 近距离时梯度陡峭，精确引导最后的贴球
    # ==========================================================================
    approach_ball = RewTerm(
        func=mdp.reward_approach_ball,
        weight=3,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "ball_cfg": SceneEntityCfg("ball"),
        },
    )

    # --------------------------------------------------------------------------
    # 2. face_ball — 面向球的对齐奖励 (权重 +1)
    # --------------------------------------------------------------------------
    # 实现: |dot(robot_heading, ball_direction)|
    #   正对球 → 约 1.0 分
    #   侧对球 → 约 0 分
    #
    # 差速小车的特殊性:
    #   只有两个驱动轮，前进方向是固定的。如果小车横着对球，
    #   虽然距离近但推不出去。这个奖励强制策略学会"转身面向目标"，
    #   否则小车会横着滑过去——注释里说的"姿态辅助"就是这个意思。
    # ==========================================================================
    face_ball = RewTerm(
        func=mdp.face_target_reward,
        weight=1,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "target_cfg": SceneEntityCfg("ball"),
        },
    )

    # --------------------------------------------------------------------------
    # 3. track_velocity — 朝球冲刺的速度奖励 (权重 +2)
    # --------------------------------------------------------------------------
    # 实现: 机器人当前速度在"指向球方向"上的投影
    #   全速冲向球 → 高奖励
    #   原地打转   → 低奖励
    #   朝反方向跑 → 负奖励
    #
    # 为什么需要这个？
    #   approach_ball 只关心"距离"，不关心"在不在动"。
    #   没有这个奖励，策略可能学会"慢慢爬"——距离在减小，但极慢。
    #   加上速度奖励 → 策略学会"转身 + 加速冲刺"的连贯行为。
    # ==========================================================================
    track_velocity = RewTerm(
        func=mdp.track_ball_velocity_reward,
        weight=2,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "ball_cfg": SceneEntityCfg("ball"),
        },
    )

    # --------------------------------------------------------------------------
    # 4. push_to_goal — 带球向球门推进的引导奖励 (权重 +5)
    # --------------------------------------------------------------------------
    # 实现: 综合考虑三个因素
    #   - 球离球门有多远？(越近分越高)
    #   - 机器人在球和球门的连线上吗？(越对齐分越高)
    #   - 球正在向球门移动吗？(速度方向对的话加分)
    #
    # 这是中后期行为的主驱动。权重 5 是日常奖励中第二高的，
    # 确保策略在碰到球后不会停下来，而是继续往球门方向推。
    # ==========================================================================
    push_to_goal = RewTerm(
        func=mdp.reward_push_ball_to_goal,
        weight=5.0,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "ball_cfg": SceneEntityCfg("ball"),
            "goal_x": 0.9,   # 球门中心 X 坐标
            "goal_y": 2.2,   # 球门线 Y 坐标（对方底线）
        },
    )

    # --------------------------------------------------------------------------
    # 5. goal_bonus — 进球大奖 (权重 +2000)
    # --------------------------------------------------------------------------
    # 实现: 判断球是否越过 Y=2.2 且 X 在 [0.9-0.4, 0.9+0.4] 范围内
    #
    # 这是唯一的"稀疏奖励"——只有真正进球那一刻才触发。
    # 2000 分的权重远大于前面所有奖励之和，确保:
    #   - 策略最终目标是"进球"而非"刷分"
    #   - 前面的稠密奖励只起引导作用，不会喧宾夺主
    #
    # 对比:
    #   一局内所有稠密奖励累加 ≈ 100~300 分
    #   进一个球 = 2000 分
    #   → 策略很清楚: 进球才是王道，其他都是过程
    # ==========================================================================
    goal_bonus = RewTerm(
        func=mdp.check_goal,
        weight=2000.0,
        params={
            "ball_cfg": SceneEntityCfg("ball"),
            "goal_x": 0.9,       # 球门中心 X
            "goal_y": 2.2,       # 球门线 Y
            "goal_width": 0.4,   # 球门宽度的一半
        },
    )

    # --------------------------------------------------------------------------
    # 6. terminal_shoot — 末端射门奖励 (权重 +2)
    # --------------------------------------------------------------------------
    # 当球接近球门附近时，额外奖励球向球门方向的速度分量。
    # 防止策略学会"带球到门口然后停下来"。
    # ==========================================================================
    terminal_shoot = RewTerm(
        func=mdp.reward_terminal_shoot,
        weight=2.0,
        params={
            "ball_cfg": SceneEntityCfg("ball"),
            "goal_y": 2.2,
        },
    )

    # --------------------------------------------------------------------------
    # 7. is_alive — 时间惩罚 (权重 -1)
    # --------------------------------------------------------------------------
    # 实现: 只要本局没结束，每步返回 1
    #
    # 因为权重是负数 (-1)，所以每一步扣 1 分。
    #
    # 这解决了一个关键问题 —— "策略为什么不站在原点不动":
    #   站在原位 → 10 秒超时 → 600 个决策步 × (-1) = -600 分
    #   5 秒进球 → 300 步 × (-1) + 2000 = +1700 分
    #
    # 时间惩罚制造了紧迫感 = 策略必须快速进球来"止损"。
    # 负权重越大策略越激进，越小策略越保守。
    # ==========================================================================
    is_alive = RewTerm(
        func=mdp.is_alive,  # Isaac Lab 内置函数
        weight=-1,          # 负权重 = 惩罚
        params={},
    )

    # ==========================================================================
    # 以下为注释掉的备选奖励项（调试或后续扩展时可用）
    # ==========================================================================

    # --- action_rate: 惩罚动作剧烈变化 ---
    # 防止策略输出在连续两步之间剧烈跳变（避免电机"抽搐"）
    # action_rate = RewTerm(
    #     func=mdp.action_rate_l2,
    #     weight=-0.01,
    #     params={},
    # )

    # --- joint_vel_limit: 惩罚关节速度过快 ---
    # 防止策略让轮子过速转动导致打滑
    # joint_vel_limit = RewTerm(
    #     func=mdp.joint_vel_l2,
    #     weight=-0.001,
    #     params={"asset_cfg": SceneEntityCfg("robot")},
    # )

    # --- touch_bonus: 触球瞬间奖励 ---
    # 当机器人与球距离小于阈值时给予一次性大奖
    # 注意: 这个和 "触球重置" 配合使用 —— 碰到球就算胜利，重新开始
    # touch_bonus = RewTerm(
    #     func=mdp.terminate_on_touch,
    #     weight=500.0,
    #     params={
    #         "robot_cfg": SceneEntityCfg("robot"),
    #         "ball_cfg": SceneEntityCfg("ball"),
    #         "threshold": 0.085,
    #     },
    # )


# ============================================================================
# 6. 终止条件定义 (TerminationsCfg)
# ============================================================================
# 决定一局仿真何时结束。
#
# 终止条件触发的后果:
#   1. 本局立即结束
#   2. 计算最终奖励
#   3. 该环境 reset 到新的初始状态（触发 EventCfg 的 mode="reset"）
#   4. 收集的经验进入训练池，用于更新策略网络
#
# 终止 ≠ 训练停止 —— 每局的结束只是开始下一局，训练持续进行。
# ============================================================================

@configclass
class TerminationsCfg:
    """终止条件: 决定一局何时结束"""

    # --------------------------------------------------------------------------
    # 1. time_out — 超时终止
    # --------------------------------------------------------------------------
    # 当仿真时间达到 episode_length_s (10 秒) 时自动触发。
    # time_out=True 表示这是"正常超时"而非"任务失败"，
    # 框架可能会区别对待（如不额外惩罚）。
    #
    # 为什么设置 10 秒？
    #   - 防止早期随机策略无限拖下去浪费算力
    #   - 配合 is_alive=-1 制造时间紧迫感
    #   - 短对局增加了 reset 频率 → 更多的初始化场景 → 更好的泛化
    #   - 10 秒内完成进球是合理的目标（场地不大）
    # ==========================================================================
    time_out = DoneTerm(
        func=mdp.time_out,
        time_out=True,
    )

    # --------------------------------------------------------------------------
    # 2. goal_scored — 进球终止
    # --------------------------------------------------------------------------
    # 和上面 RewardsCfg 中的 goal_bonus 使用同一个函数 mdp.check_goal。
    #
    # 同一个函数在两个模块中的作用:
    #   在 RewardsCfg 中 → 发放 2000 分大奖（告诉策略"这是好事"）
    #   在 TerminationsCfg 中 → 结束本局（"球已经进了，不需要继续踢了"）
    #
    # 两者独立工作: 奖励系统负责打分，终止系统负责吹哨。
    # ==========================================================================
    goal_scored = DoneTerm(
        func=mdp.check_goal,
        params={
            "ball_cfg": SceneEntityCfg("ball"),
            "goal_x": 0.9,       # 球门中心 X 坐标
            "goal_y": 2.2,       # 球门线 Y 坐标
            "goal_width": 0.4,   # 球门宽度的一半
        },
    )

    # ==========================================================================
    # 以下为注释掉的备选终止条件
    # ==========================================================================

    # --- touch_ball: 触球即重置 ---
    # 当机器人与球距离小于阈值时结束本局。
    # 适合初期的"碰球训练"阶段，降低任务难度:
    #   第一阶段: 只训练"碰到球"就结束 → 学会接近球
    #   第二阶段: 改成"进球"才结束 → 在前者基础上学会推球射门
    # touch_ball = DoneTerm(
    #     func=mdp.terminate_on_touch,
    #     params={
    #         "robot_cfg": SceneEntityCfg("robot"),
    #         "ball_cfg": SceneEntityCfg("ball"),
    #         "threshold": 0.08,
    #     },
    # )


# ============================================================================
# 7. 最终环境配置 (SoccerEnvCfg)
# ============================================================================
# 将上面定义的所有子配置组装成一个完整的环境配置。
# 继承自 ManagerBasedRLEnvCfg，这是 Isaac Lab 的 Manager-Based RL 环境基类。
#
# 顶层参数汇总:
#   num_envs=512      — 512 个并行环境同时训练
#   env_spacing=10.0  — 每个环境的间隔（防止在可视化中重叠）
#   episode_length_s  — 10 秒一局
#   decimation=2       — 每 2 个物理步决策一次 (60 Hz)
#   sim.dt=1/120       — 物理步长 (120 Hz)
#   observation_noise  — 给观测加高斯噪声 (模拟传感器误差)
#   action_noise       — 给动作加高斯噪声 (模拟电机波动)
# ============================================================================

@configclass
class SoccerEnvCfg(ManagerBasedRLEnvCfg):
    """足球机器人 RL 环境 —— 总配置"""

    # ==========================================================================
    # 子模块组装
    # ==========================================================================

    # 场景: 512 个并行环境，间隔 10m
    scene: SoccerSceneCfg = SoccerSceneCfg(num_envs=512, env_spacing=10.0)

    # 观测: 11 维状态向量
    observations: ObservationsCfg = ObservationsCfg()

    # 动作: 双轮差速速度控制
    actions: ActionsCfg = ActionsCfg()

    # 事件: 每局重置时随机化位置和摩擦力
    events: EventCfg = EventCfg()

    # 奖励: 7 项加权奖励函数
    rewards: RewardsCfg = RewardsCfg()

    # 终止: 超时或进球
    terminations: TerminationsCfg = TerminationsCfg()

    # ==========================================================================
    # 噪声模型 (Noise Models)
    # ==========================================================================
    # 噪声是"模拟真实世界"的关键 —— 真实传感器有误差，真实电机有波动。
    # 训练时加入噪声，策略学会"在不确定性中做决策"，
    # 部署到真实硬件时才不会因为传感器噪声而崩溃。
    #
    # 噪声结构: NoiseModelWithAdditiveBiasCfg 包含两层
    #   noise_cfg      — 每步都加的随机噪声（模拟高频波动）
    #   bias_noise_cfg  — 每局重置时采样一次并保持整局（模拟零点漂移/偏置）
    # ==========================================================================

    # 观测噪声 — 模拟传感器误差
    # std=0.002 很小，相当于在 [-0.006, 0.006] 范围内的微小抖动
    # 模拟真实视觉定位/陀螺仪的量化噪声
    observation_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
        noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.002, operation="add"),
        bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.0001, operation="add"),
    )

    # 动作噪声 — 模拟电机波动
    # std=0.05 比观测噪声大，模拟电机无法完美执行目标速度
    # 例如策略输出 5.0 rad/s，实际可能执行 4.9~5.1 rad/s
    action_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
        noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.05, operation="add"),
        bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.01, operation="add"),
    )

    # ==========================================================================
    # 仿真参数 (__post_init__)
    # ==========================================================================
    # __post_init__ 在 @configclass 实例化后自动调用，类似 Python dataclass 的机制。
    # 这里设置底层仿真器的物理和时序参数。
    # ==========================================================================

    def __post_init__(self) -> None:
        """仿真底层参数配置"""

        # --- 决策降采样 ---
        # decimation=2 的含义:
        #   物理引擎每 8.3ms (1/120s) 算一步
        #   策略网络每 16.7ms (2/120s) 决策一次
        #
        # 两个物理步之间保持同一个动作不变，物理引擎只用它来推演运动。
        # 60Hz 的控制频率对差速小车完全足够，同时省下一半的 GPU 推理开销。
        self.decimation = 2

        # --- 每局最大时长 ---
        # 10 秒后如果没进球，time_out 终止条件会结束本局。
        # 10 秒 × 60 决策步/秒 = 600 个决策步/局。
        # 配合 is_alive=-1 的时间惩罚，逼迫策略快速完成任务。
        self.episode_length_s = 10

        # --- 物理仿真步长 ---
        # dt = 1/120 ≈ 0.0083s，即 120Hz 的物理仿真频率。
        # 这个值足够小，保证碰撞检测和摩擦力的精度。
        self.sim.dt = 1 / 120

        # --- 渲染间隔 ---
        # render_interval = decimation = 2，意味着每 2 个物理步渲染一帧。
        # 不对每个物理步都渲染（性能和显存开销太大）。
        # 只在需要可视化时生效，训练时通常关闭渲染。
        self.sim.render_interval = self.decimation
