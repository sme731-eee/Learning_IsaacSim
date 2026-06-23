# ============================================================================
# Isaac Lab 足球机器人强化学习 — MDP 函数定义（完整注释版）
# ============================================================================
# 本文件定义了强化学习环境的所有自定义函数，分为三大类:
#   1. 观测函数 (Observation Functions)  — 从物理引擎读取数据，供策略网络使用
#   2. 奖励函数 (Reward Functions)       — 定义"做什么是对的"，每步计算分数
#   3. 终止函数 (Termination Functions)  — 定义"什么时候一局结束"
#
# 每个函数的签名都是固定的: func(env, **params) -> torch.Tensor
#   - env:       Isaac Lab 环境对象，包含场景、物理数据等
#   - **params:  从 soccer_env_cfg.py 中传入的额外参数
#   - 返回:      torch.Tensor，形状通常为 [num_envs] 或 [num_envs, N]
#
# 数据读取的核心路径:
#   env.scene["物体名"].data.XXX  →  从物理引擎读取实时数据
#   _w = World frame (世界坐标系)     _b = Body frame (机体坐标系)
# ============================================================================

import torch
import torch.nn.functional as F
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import euler_xyz_from_quat

# 导入 Isaac Lab 内置的 MDP 函数（如 is_alive, time_out, reset_root_state_uniform 等）
from isaaclab.envs.mdp import *


# ============================================================================
# 第一部分：观测函数 (Observation Functions)
# ============================================================================
# 观测函数负责从仿真物理引擎中读取实时数据，返回 torch.Tensor。
# 它们在 soccer_env_cfg.py 的 ObservationsCfg 中被 ObsTerm 引用，
# 框架每步调用一次，结果拼接后送入策略网络。
#
# Isaac Lab 内置数据字段 (robot.data.XXX) 的关键后缀:
#   _w = World frame    — 世界坐标系（以仿真世界原点为参照）
#   _b = Body frame     — 机体坐标系（以机器人自身为参照）
#
# 什么时候用局部场地坐标 (- env.scene.env_origins)?
#   - 算两个物体之间的相对关系 → 不需要，原点自动抵消
#   - 算物体离固定点(球门/边线)多远 → 需要，因为球门是场地内的坐标
# ============================================================================


# --------------------------------------------------------------------------
# 观测函数 1: 球相对于机器人的位置（世界坐标系）
# --------------------------------------------------------------------------
# 返回: [num_envs, 3] — (Δx, Δy, Δz) 世界坐标系下的相对位置向量
# 注意: 当前配置中此函数未被使用（被注释掉了），保留作为备用
# --------------------------------------------------------------------------
def object_position_in_robot_frame(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, object_cfg: SceneEntityCfg):
    """计算物体(球)在机器人坐标系下的相对位置"""
    # 从场景字典中按名称取出物理对象
    #   env.scene["robot"] → 小车 (Articulation: 铰接体，有关节)
    #   env.scene["ball"]  → 足球 (RigidObject: 刚体，无关节)
    robot: Articulation = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]

    # 获取世界坐标系下的位置和姿态
    #   root_pos_w  = 物体的基座位置 [num_envs, 3] — (X, Y, Z) 世界坐标
    #   root_quat_w = 物体的四元数姿态 [num_envs, 4] — (x, y, z, w)
    robot_pos = robot.data.root_pos_w
    robot_quat = robot.data.root_quat_w
    object_pos = object.data.root_pos_w

    # 计算相对位置: 球的世界坐标 - 车的世界坐标
    # 原理: 两个世界坐标相减，env_origins 自动抵消
    #   env_0:  球(5,8) - 车(4,8) = (1,0)
    #   env_5:  球(55,8) - 车(54,8) = (1,0)  ← 结果相同！
    target_vec = object_pos - robot_pos

    # 这里为了简化，直接返回世界坐标系的相对距离
    # 如果需要严格的机器人视角（车头方向为X轴），需要用 quat_rotate_inverse 旋转向量
    return target_vec


# --------------------------------------------------------------------------
# 观测函数 2: 机器人偏航角的 sin 和 cos 值
# --------------------------------------------------------------------------
# 返回: [num_envs, 2] — [sin(yaw), cos(yaw)]
#
# 为什么用 sin/cos 而不用原始弧度值？
#   - 弧度在 ±π 处有"相位环绕"问题：+3.14 和 -3.14 是同一个方向
#     但神经网络看到两个完全不同的数字，学习困难
#   - sin/cos 永远在 [-1, 1] 之间连续平滑，没有跳变
#   - sin² + cos² = 1 这个天然约束有助于网络学习
#
# 小车只在水平面运动，绕 Z 轴旋转（偏航），不涉及俯仰和滚转。
# 四元数 → 欧拉角 → 只取 yaw → 转 sin/cos。
# --------------------------------------------------------------------------
def robot_yaw_sin_cos(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]

    # 获取小车在世界坐标系下的四元数 (x, y, z, w)
    # 四元数 = 4个数描述3D朝向，比欧拉角更稳定，不会"万向节锁"
    # 几何意义: 绕旋转轴 (x,y,z) 旋转角度 θ = 2·arccos(w)
    q_world_robot = robot.data.root_quat_w

    # 将四元数转换为欧拉角 (roll, pitch, yaw)
    # roll  ≈ 0  (绕X轴，侧翻)
    # pitch ≈ 0  (绕Y轴，俯仰)
    # yaw   = 有用 (绕Z轴，偏航/面朝方向)
    roll, pitch, yaw_robot = euler_xyz_from_quat(q_world_robot)

    # 计算 sin 和 cos，并调整为列向量形状 [num_envs, 1]
    sin_y = torch.sin(yaw_robot).view(-1, 1)
    cos_y = torch.cos(yaw_robot).view(-1, 1)

    # 拼接返回 [sin, cos]，形状 [num_envs, 2]
    # sin/cos 天然是单位向量: sqrt(sin²+cos²) = 1
    return torch.cat([sin_y, cos_y], dim=-1)


# --------------------------------------------------------------------------
# 观测函数 3: 物体在场地的归一化 XY 坐标
# --------------------------------------------------------------------------
# 返回: [num_envs, 2] — 物体在场地内的 (X/1.8, Y/2.2)，范围约 [0, 1]
#
# 计算步骤:
#   1. 世界坐标 - 环境原点 = 场地内坐标（消除 512 个环境的空间偏移）
#   2. 归一化: X ÷ 1.8, Y ÷ 2.2（场地尺寸）
#
# 当前配置中此函数被调用了两次:
#   - asset_cfg="robot" → 小车在场地内的位置
#   - asset_cfg="ball"  → 球在场地内的位置
# --------------------------------------------------------------------------
def positions_relative_to_arena(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """观测物体相对于场地的 XY 坐标。"""
    obj_asset = env.scene[asset_cfg.name]

    # 一行拆解:
    #   obj_asset.data.root_pos_w[:, :2]  = 世界坐标, 取 XY [num_envs, 2]
    #   env.scene.env_origins[:, :2]       = 每个环境的世界原点 [num_envs, 2]
    #   相减 = 场地局部坐标 [num_envs, 2]
    relative_pos = obj_asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]

    # 归一化到 [0,1] 范围，使神经网络训练更稳定
    relative_pos[:, 0] = relative_pos[:, 0] / 1.8   # X: 场地长边
    relative_pos[:, 1] = relative_pos[:, 1] / 2.2   # Y: 场地短边

    return relative_pos


# --------------------------------------------------------------------------
# 观测函数 4: 物体的机体坐标系 XY 线速度
# --------------------------------------------------------------------------
# 返回: [num_envs, 2] — (Vx_body, Vy_body)
#
# 使用 body frame (_b) 而非 world frame (_w) 的原因:
#   Vx_b = 小车前进方向的速度 → 正数=前进，负数=后退
#   Vy_b = 小车横向的速度     → 非零=打滑（差速车不该横着走）
#
# 用机体坐标系，策略不需要知道车朝向就能判断"我在前进还是打滑"。
# --------------------------------------------------------------------------
def base_lin_vel_xy(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """只观测机器人的 X 和 Y 线速度（机体坐标系）。"""
    # root_lin_vel_b: 机体坐标系下的基座线速度 [num_envs, 3]
    # _b = body frame (机体坐标系，以机器人自身为参照)
    lin_vel = env.scene[asset_cfg.name].data.root_lin_vel_b

    # 只取前两个维度 (X = 前进方向, Y = 横向)，丢弃 Z (垂直速度)
    return lin_vel[:, :2]


# --------------------------------------------------------------------------
# 观测函数 5: 物体的机体坐标系 Z 轴角速度（转向速度）
# --------------------------------------------------------------------------
# 返回: [num_envs, 1] — Wz (偏航角速度)
#
# Wz_b = 小车绕 Z 轴的旋转速度
#   正数 = 向左转
#   负数 = 向右转
# --------------------------------------------------------------------------
def base_ang_vel_z(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """只观测机器人的 Z 轴角速度（转向速度）。"""
    # root_ang_vel_b: 机体坐标系下的基座角速度 [num_envs, 3]
    ang_vel = env.scene[asset_cfg.name].data.root_ang_vel_b

    # 只取最后一个维度 (Wz)，并确保形状是 [num_envs, 1]
    return ang_vel[:, 2:].view(-1, 1)


# --------------------------------------------------------------------------
# 观测函数 6 (备用): 到四面场地围墙的距离
# --------------------------------------------------------------------------
# 返回: [num_envs, 4] — [到左墙, 到右墙, 到下墙, 到上墙] (米)
# 当前配置中此函数未被使用，可作为扩展观测加入
# --------------------------------------------------------------------------
def distance_to_arena_walls(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """观测与四面场地围墙的距离。"""
    # 获取机器人位置并转换为场地局部坐标
    pos = env.scene[asset_cfg.name].data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]

    # 场地边界: X ∈ [0, 1.8], Y ∈ [0, 2.2]
    x_min, x_max = 0.0, 1.8
    y_min, y_max = 0.0, 2.2

    # 计算到四面墙的距离
    dist_left   = pos[:, 0] - x_min      # 到左墙 (X=0)
    dist_right  = x_max - pos[:, 0]       # 到右墙 (X=1.8)
    dist_bottom = pos[:, 1] - y_min      # 到下墙 (Y=0)
    dist_top    = y_max - pos[:, 1]       # 到上墙 (Y=2.2)

    # 打包返回 4 维向量，值越小越危险（→ 0 就撞墙）
    return torch.stack([dist_left, dist_right, dist_bottom, dist_top], dim=-1)


# ============================================================================
# 第二部分：奖励函数 (Reward Functions)
# ============================================================================
# 奖励函数是强化学习的"价值观"，告诉策略网络什么行为是好的。
# 每个函数返回一个标量分数 [num_envs]，由 RewardManager.compute() 加权求和。
#
# 设计思路: "稠密引导 + 稀疏大奖"
#   稠密引导 (dense) — 每步都有信号，帮策略快速找到方向
#   稀疏大奖 (sparse) — 只有真正进球才给 2000 分，确保最终目标不偏离
#
# 奖励层次 (从低到高):
#   靠近球 → 面向球 → 冲刺撞球 → 推球向门 → 禁区爆射 → 进球！
#     3        1         2          5          2       2000
#   └────── 稠密引导 ──────────────┘ └── 稀疏大奖 ──┘
#
# 全程贯穿: is_alive = -1 (时间惩罚，逼迫快速完成)
# ============================================================================


# --------------------------------------------------------------------------
# 奖励函数 1: 靠近球的距离奖励 (权重 +3)
# --------------------------------------------------------------------------
# 公式: exp(-2.5 × 车到球的距离)
#
# 距离与奖励对照表:
#   距离 0m   → 1.0   (贴着球，满分)
#   距离 0.2m → 0.61  (近了)
#   距离 0.5m → 0.29  (半米远)
#   距离 1.0m → 0.08  (一米远)
#
# 使用指数衰减而非线性的原因:
#   - 远距离时梯度平缓，提供持续的"往球走"信号
#   - 近距离时梯度陡峭，精确引导最后的贴球
# --------------------------------------------------------------------------
def reward_approach_ball(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, ball_cfg: SceneEntityCfg):
    """距离奖励：距离越近，奖励越高。"""
    robot: Articulation = env.scene[robot_cfg.name]
    ball: RigidObject = env.scene[ball_cfg.name]

    # 计算车和球在 XY 平面上的欧几里得距离 [num_envs]
    # 注意: 这里不需要减 env_origins，因为两个世界坐标相减原点自动抵消
    dist = torch.norm(ball.data.root_pos_w[:, :2] - robot.data.root_pos_w[:, :2], dim=-1)

    # 负指数函数: k=2.5，小车对距离比较敏感
    return torch.exp(-2.5 * dist)


# --------------------------------------------------------------------------
# 奖励函数 2: 面向球的朝向奖励 (权重 +1)
# --------------------------------------------------------------------------
# 计算车头方向与"车→球"方向的点积绝对值。
# 点积 = 两个单位向量夹角的余弦值:
#   完全同向 (+1.0) → 满分，车头正对球
#   垂直     ( 0.0) → 零分，车侧对球
#   完全反向 (-1.0) → 满分，车尾对球（abs 后）
#
# 为什么取绝对值？
#   差速小车有两个驱动轮，前进和倒车都能推动球。
#   face_ball 只要求"转向"（消除侧对），后续的 track_velocity 再要求正对着冲（消除倒车）。
#   两个奖励函数接力配合。
# --------------------------------------------------------------------------
def face_target_reward(env, robot_cfg: SceneEntityCfg, target_cfg: SceneEntityCfg) -> torch.Tensor:
    """朝向奖励：鼓励机器人的 X 轴（正向或反向）指向球。"""
    robot = env.scene[robot_cfg.name]
    ball_pos = env.scene[target_cfg.name].data.root_pos_w

    # 步骤 1: 获取机器人位置和四元数朝向
    robot_pos = robot.data.root_pos_w
    quat = robot.data.root_quat_w

    # 步骤 2: 计算"机器人 → 球"的方向向量 (XY 平面)，并归一化为单位向量
    to_target_vec = ball_pos[:, :2] - robot_pos[:, :2]
    to_target_dir = torch.nn.functional.normalize(to_target_vec, dim=-1)

    # 步骤 3: 获取"车头方向"的单位向量 (XY 平面)
    #   小车默认正前方 = X 轴 [1, 0, 0]，用四元数旋转得到真实的 3D 前向
    #   取 [:,:2] 得到 XY 平面的分量，再归一化
    from isaaclab.utils.math import quat_apply
    forward_unit_vec = torch.tensor([1.0, 0.0, 0.0], device=env.device).repeat(env.num_envs, 1)
    robot_forward_vec = quat_apply(quat, forward_unit_vec)[:, :2]
    robot_forward_dir = torch.nn.functional.normalize(robot_forward_vec, dim=-1)

    # 步骤 4: 计算两个单位向量的点积（= 夹角的余弦值）
    #   sum(a*b, dim=-1) = dot product
    dot_prod = torch.sum(robot_forward_dir * to_target_dir, dim=-1)

    # 步骤 5: 取绝对值
    #   正对球 (dot=+1) 和 背对球 (dot=-1) 都得满分
    return torch.abs(dot_prod)


# --------------------------------------------------------------------------
# 奖励函数 3: 朝球冲刺的速度奖励 (权重 +2)
# --------------------------------------------------------------------------
# 计算小车当前速度在"指向球方向"上的投影。
#   正数 = 全速冲向球，拿分
#   零   = 原地打转或侧着走
#   负数 = 朝球反方向跑（被 clamp 到 0，不给惩罚）
#
# 为什么用世界坐标系速度 (_w) 而不是机体坐标系 (_b)?
#   因为 to_ball_dir 是在世界坐标系里算的（球和车世界坐标相减），
#   点积要求两个向量在同一个坐标系里，所以速度也必须用世界坐标系。
#
# 为什么 clamp 到 0 而不是给负分惩罚？
#   防止倒车时产生剧烈惩罚导致策略"摆烂"——不学了。
#   不给负分，策略只是拿不到这个奖励，不会因为这项而受苦。
# --------------------------------------------------------------------------
def track_ball_velocity_reward(env, robot_cfg: SceneEntityCfg, ball_cfg: SceneEntityCfg) -> torch.Tensor:
    """速度引导奖励：奖励小车向球移动的速度分量。"""
    robot = env.scene[robot_cfg.name]
    ball = env.scene[ball_cfg.name]

    # 步骤 1: 获取小车和球的世界坐标 XY [num_envs, 2]
    robot_pos = robot.data.root_pos_w[:, :2]
    ball_pos = ball.data.root_pos_w[:, :2]

    # 步骤 2: 计算"小车 → 球"的单位方向向量
    to_ball_vec = ball_pos - robot_pos
    to_ball_dir = torch.nn.functional.normalize(to_ball_vec, dim=-1)

    # 步骤 3: 获取小车当前的世界坐标系线速度 (X, Y) [num_envs, 2]
    robot_vel = robot.data.root_lin_vel_w[:, :2]

    # 步骤 4: 计算速度在"球方向"上的投影（点积）
    velocity_projection = torch.sum(robot_vel * to_ball_dir, dim=-1)

    # 步骤 5: 只奖励正速度（朝向球的），不惩罚负速度（背离球的）
    return torch.clamp(velocity_projection, min=0.0)


# --------------------------------------------------------------------------
# 奖励函数 4: 带球向球门推进的引导奖励 (权重 +5)
# --------------------------------------------------------------------------
# 这是中后期行为的主驱动，由三个子奖励组成:
#
#   奖励 A (ball_to_goal_reward):
#     球离球门越近分越高，权重 1.0
#
#   奖励 B (alignment_reward × center_hit_reward × 3.0):
#     - pos_alignment: 车-球-球门 是否三点一线
#     - heading_alignment: 车头是否正对球
#     - center_hit_reward: 球是否在车身中轴线上
#     - proximity_weight: 车离球越近，上面三项越值钱
#     姿势分 × 距离权重，确保"既要正又要近"
#
#   奖励 C (push_progress_reward × 5.0):
#     球正在向球门方向滚动的速度
#     权重最高 (5.0)，是"终极推球行为"的核心驱动力
# --------------------------------------------------------------------------
def reward_push_ball_to_goal(env, robot_cfg, ball_cfg, goal_x: float, goal_y: float):
    """终极带球奖励：引导策略将球推向球门。"""
    robot = env.scene[robot_cfg.name]
    ball = env.scene[ball_cfg.name]

    # 【核心修复】提取二维位置，并转为场地局部坐标
    # 球门坐标 (0.9, 2.2) 是场地内的固定点，必须用场地坐标系
    robot_pos = robot.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    ball_pos = ball.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]

    robot_quat = robot.data.root_quat_w
    # 创建球门坐标并复制 512 份: shape [512, 2]
    goal_pos = torch.tensor([goal_x, goal_y], device=env.device).repeat(env.num_envs, 1)

    # ================== 奖励 A: 球靠近球门的距离奖 ==================
    # 公式: exp(-1.0 × 球到球门的距离)
    # 距离越小分越高，引导策略将球往球门方向带
    ball_to_goal_dist = torch.norm(goal_pos - ball_pos, dim=-1)
    ball_to_goal_reward = torch.exp(-1.0 * ball_to_goal_dist)

    # ================== 核心计算: 获取各种方向向量 ==================

    # --- 车头朝向 ---
    # 从四元数提取偏航角: tan(θ) = 2(wz+xy) / (1-2(y²+z²))
    w, x, y, z = robot_quat[:, 0], robot_quat[:, 1], robot_quat[:, 2], robot_quat[:, 3]
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    # 车头方向单位向量 [num_envs, 2]
    # (cos θ, sin θ) 天然长度=1
    heading_dir = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=-1)

    # --- 车→球 方向 ---
    robot_to_ball_dir = F.normalize(ball_pos - robot_pos, dim=-1)

    # --- 球→球门 方向 ---
    # 注意: 始终指向球门中心点 (0.9, 2.2)
    # 如果球在边路，这个引导可能不够精确，可考虑 clamp 目标点到球门宽度范围内
    ball_to_goal_dir = F.normalize(goal_pos - ball_pos, dim=-1)

    # ================== 奖励 B: 双重对齐奖 (静态姿态) ==================

    # B1: 位置对齐 — 检查 "车→球" 和 "球→门" 两条线是否共线
    # 点积 ≈ 1.0 → 车在球正后方，推球直飞球门
    # 点积 ≈ 0.0 → 车从侧面推球
    # 点积 ≈ -1.0 → 车在球和球门之间（往反方向推）
    pos_alignment = torch.sum(robot_to_ball_dir * ball_to_goal_dir, dim=-1)

    # B2: 朝向对齐 — 检查车头是否正对球
    # 点积 ≈ 1.0 → 车头直指球
    # 点积 ≈ 0.0 → 车侧对球
    # 点积 ≈ -1.0 → 车尾对球
    heading_alignment = torch.sum(heading_dir * robot_to_ball_dir, dim=-1)

    # B3: 横向偏移 — 球偏离车身中轴线多少米（2D 叉乘）
    # 公式: |x1·y2 - x2·y1| = |heading_X × ball_Y - heading_Y × ball_X|
    # 这是最严格的约束，要求球恰好在中轴线上
    vec_car_to_ball = ball_pos - robot_pos
    lateral_offset = torch.abs(
        heading_dir[:, 0] * vec_car_to_ball[:, 1] - heading_dir[:, 1] * vec_car_to_ball[:, 0]
    )
    # 横向偏差惩罚: exp(-5.0 × 偏移距离)
    #   偏移 0.0m → 1.0
    #   偏移 0.1m → 0.61
    #   偏移 0.3m → 0.22
    center_hit_reward = torch.exp(-5.0 * lateral_offset)

    # 处理对齐分量
    pos_align_clip = torch.clamp(pos_alignment, min=0.0)     # 反向推球不给分
    heading_align_clip = torch.abs(heading_alignment)        # 正对和背对都给满分

    # 姿态综合分 = 位置对齐 × 朝向对齐 (乘积 = 短板效应，一项不行全垮)
    total_alignment = pos_align_clip * heading_align_clip

    # 距离权重: 车离球越近，姿势分越值钱
    robot_to_ball_dist = torch.norm(ball_pos - robot_pos, dim=-1)
    proximity_weight = torch.exp(-3.0 * robot_to_ball_dist)

    # 姿态奖励 = 姿势综合分 × 距离权重
    alignment_reward = total_alignment * proximity_weight

    # ================== 奖励 C: 动态推球奖 ==================
    # 获取球在世界坐标系的线速度 (V_x, V_y)
    ball_vel = ball.data.root_lin_vel_w[:, :2]

    # 计算球速在"指向球门方向"上的投影
    ball_vel_to_goal = torch.sum(ball_vel * ball_to_goal_dir, dim=-1)

    # 只有球往门的方向滚才给分
    push_progress_reward = torch.clamp(ball_vel_to_goal, min=0.0)

    # ================== 综合返回 ==================
    # A (1.0) + B × center_hit × 3.0 + C × 5.0
    # C 的权重最高，是"真正推球射门"的核心驱动力
    return ball_to_goal_reward + (alignment_reward * center_hit_reward * 3.0) + (push_progress_reward * 5.0)


# --------------------------------------------------------------------------
# 奖励函数 5: 进球大奖 (权重 +2000)
# --------------------------------------------------------------------------
# 这是唯一的大规模稀疏奖励——只有真正进球那一刻才触发。
# 2000 分 >> 前面所有稠密奖励的总和（~100-300 分/局），
# 确保策略的最终目标是"进球"而非"刷分"。
#
# 此函数同时用于两个模块:
#   在 RewardsCfg 中 → 发放 2000 分 (告诉策略"这是好事")
#   在 TerminationsCfg 中 → 结束本局 (球已进门，不必继续)
# ==========================================================================
def check_goal(env: ManagerBasedRLEnv, ball_cfg: SceneEntityCfg, goal_x: float, goal_y: float, goal_width: float):
    """进球判定：球越过对方底线且在球门宽度范围内。"""
    ball: RigidObject = env.scene[ball_cfg.name]

    # 转为场地局部坐标（球门坐标是场地内的固定点）
    ball_pos = ball.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]

    # 进球判定:
    #   1. 越过对方底线 (Y > goal_y = 2.2)
    #   2. 在门框范围内 (|X - goal_x| < goal_width/2 = 0.2)
    is_goal = (ball_pos[:, 1] > goal_y) & \
              (torch.abs(ball_pos[:, 0] - goal_x) < (goal_width / 2))

    # 球飞得太远（越过底线 1m 以上），不可能回来了，也触发重置
    # 但注意: is_too_far 让球重置但不给 goal_bonus (is_goal 仍是 False)
    is_too_far = ball_pos[:, 1] > (goal_y + 1.0)

    # 返回布尔值 [num_envs]: True=进或飞远, False=继续
    return is_goal | is_too_far


# --------------------------------------------------------------------------
# 终止函数 (备用): 触球即重置
# --------------------------------------------------------------------------
# 这是第一个训练阶段的"降级"目标——碰到球就算胜利。
# 降低任务难度，让策略先学会"接近球"，再学"推球入球门"。
# 当前代码中被注释掉了，说明训练已经过了碰球阶段，直接训练进球。
# --------------------------------------------------------------------------
def terminate_on_touch(env, robot_cfg: SceneEntityCfg, ball_cfg: SceneEntityCfg, threshold: float = 0.15):
    """当机器人与球的距离小于阈值时，触发环境重置。"""
    # 获取机器人和球的世界位置 [num_envs, 3]
    robot_pos = env.scene[robot_cfg.name].data.root_pos_w
    ball_pos = env.scene[ball_cfg.name].data.root_pos_w

    # 计算三维空间距离
    distance = torch.norm(ball_pos - robot_pos, dim=-1)

    # 判定是否触碰
    is_touched = distance < threshold

    return is_touched


# --------------------------------------------------------------------------
# 奖励函数 6: 末端爆射奖励 (权重 +2)
# --------------------------------------------------------------------------
# 当球进入禁区（Y > 1.7）时，球向球门方向的速度越快，奖励越高。
# 这是"临门一脚"的加速器——前面的奖励解决了"带球到门口"，
# 这个解决"最后一哆嗦"，防止策略推到门口减速。
# --------------------------------------------------------------------------
def reward_terminal_shoot(env, ball_cfg: SceneEntityCfg, goal_y: float):
    """末端爆射奖励：当球进入禁区时，球速越快，奖励越高！"""
    ball = env.scene[ball_cfg.name]

    # 获取球在场地的位置 (需要判断是否在禁区内)
    ball_pos = ball.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    ball_vel = ball.data.root_lin_vel_w[:, :2]

    # 禁区判定: Y > goal_y - 0.5 = 2.2 - 0.5 = 1.7
    # 球在 Y=1.7~2.2 区域内 = 进入射门范围
    in_penalty_area = ball_pos[:, 1] > (goal_y - 0.5)

    # 提取球向球门方向 (Y 轴正向) 的速度分量
    # clamp: 只奖励向前的速度，球往后退不给分
    ball_speed_y = torch.clamp(ball_vel[:, 1], min=0.0)

    # 条件奖励: 禁区内 → 奖励速度；禁区外 → 0
    return torch.where(in_penalty_area, ball_speed_y, 0.0)


# ============================================================================
# 附注: Isaac Lab 内置的 MDP 函数 (from isaaclab.envs.mdp import *)
# ============================================================================
# 以下函数由 Isaac Lab 框架提供，不需要在此文件中定义:
#
#   is_alive()              — 返回全 1 张量，配合负权重 = 时间惩罚
#   time_out()              — 判断是否超过 episode_length_s，超时就终止
#   reset_root_state_uniform() — 按给定范围随机重置物体位置和速度
#   randomize_rigid_body_material() — 随机化刚体的摩擦/弹性系数
#
# 这些在 soccer_env_cfg.py 的 EventsCfg / RewardsCfg / TerminationsCfg 中被引用。


# ============================================================================
# ============================== 总结 ======================================
# ============================================================================

# ┌──────────────────────────────────────────────────────────────────────────┐
# │ 第一部分: 观测函数 — 策略网络每步收到的数据                               │
# ├────┬─────────────────────────────────┬──────────┬────────────────────────┤
# │ #  │ 函数名                           │ 返回维度  │ 数据内容                │
# ├────┼─────────────────────────────────┼──────────┼────────────────────────┤
# │ 1  │ object_position_in_robot_frame  │ [512, 3] │ 球相对车的位置 (备用)   │
# │ 2  │ robot_yaw_sin_cos               │ [512, 2] │ 车朝向 sin/cos          │
# │ 3  │ positions_relative_to_arena     │ [512, 2] │ 车/球在场地内归一化位置  │
# │ 4  │ base_lin_vel_xy                 │ [512, 2] │ 车机体线速度 Vx, Vy     │
# │ 5  │ base_ang_vel_z                  │ [512, 1] │ 车机体角速度 Wz         │
# │ 6  │ distance_to_arena_walls         │ [512, 4] │ 到四面墙的距离 (备用)    │
# ├────┴─────────────────────────────────┼──────────┼────────────────────────┤
# │ 当前配置实际使用 (#2+#3×2+#4+#5)        │ [512, 9] │ 9 维观测向量           │
# ├──────────────────────────────────────┼──────────┼────────────────────────┤
# │ 数据坐标系说明:                                                          │
# │   _w = World frame  — 世界坐标系                                       │
# │   _b = Body frame   — 机体坐标系 (Vx=前进方向, Vy=横向)                 │
# │   局部场地坐标 = 世界坐标 - env_origins (用于和球门/边线比较)            │
# └──────────────────────────────────────────────────────────────────────────┘

# ┌──────────────────────────────────────────────────────────────────────────┐
# │ 第二部分: 奖励函数 — 告诉策略"什么是对的"，每步计算，加权求和后送入 PPO   │
# ├────┬───────────────────────────┬────────┬────────────────────────────────┤
# │ #  │ 函数名                     │ 权重    │ 作用                           │
# ├────┼───────────────────────────┼────────┼────────────────────────────────┤
# │ 1  │ reward_approach_ball      │   3    │ 靠近球: exp(-2.5×距离)          │
# │ 2  │ face_target_reward        │   1    │ 面向球: |车头·球方向|           │
# │ 3  │ track_ball_velocity_reward│   2    │ 冲球: 车速度在球方向上的投影    │
# │ 4  │ reward_push_ball_to_goal  │   5    │ 推球向门: A(球近门) + B(姿势)×3 │
# │    │                           │        │ + C(球向门速度)×5               │
# │ 5  │ check_goal (作为奖励)      │  2000  │ 进球大奖: 稀疏奖励，唯一终点    │
# │ 6  │ reward_terminal_shoot     │   2    │ 禁区爆射: 球在禁区内向门速度    │
# │ 7  │ is_alive (内置)           │  -1    │ 时间惩罚: 每步 -1，逼迫快速完成  │
# ├────┴───────────────────────────┴────────┴────────────────────────────────┤
# │ 奖励体系设计: 稠密引导 → 稀疏大奖                                        │
# │                                                                          │
# │   阶段1 (找球):     approach_ball(3) + face_ball(1) + track_vel(2)       │
# │                     → 引导小车靠近球、面向球、冲刺撞球                    │
# │                                                                          │
# │   阶段2 (推球):     push_to_goal(5)                                      │
# │                     → 姿势对齐 + 推球向门 + 不偏轴                       │
# │                                                                          │
# │   阶段3 (射门):     terminal_shoot(2) + check_goal(2000)                 │
# │                     → 禁区爆射 + 进球大奖                               │
# │                                                                          │
# │   贯穿全程:         is_alive(-1)                                         │
# │                     → 时间压力，防止发呆/绕圈                            │
# ├──────────────────────────────────────────────────────────────────────────┤
# │ 终止条件:                                                                │
# │   time_out      — 10 秒超时自动重置                                     │
# │   check_goal    — 进球或球飞出底线太远 → 立即重置                        │
# │   terminate_on_touch — (备用/第一阶段) 碰球即重置                        │
# └──────────────────────────────────────────────────────────────────────────┘

# ┌──────────────────────────────────────────────────────────────────────────┐
# │ 第三部分: 调用关系 — 这个文件里的函数如何在 soccer_env_cfg.py 中被使用    │
# ├──────────────────────────────────────────────────────────────────────────┤
# │                                                                          │
# │  ObservationsCfg (观测):                                                 │
# │    ObsTerm(func=mdp.robot_yaw_sin_cos,          params=robot)            │
# │    ObsTerm(func=mdp.base_lin_vel_xy,            params=robot)            │
# │    ObsTerm(func=mdp.base_ang_vel_z,             params=robot)            │
# │    ObsTerm(func=mdp.base_lin_vel_xy,            params=ball)             │
# │    ObsTerm(func=mdp.positions_relative_to_arena,params=robot)            │
# │    ObsTerm(func=mdp.positions_relative_to_arena,params=ball)             │
# │                                                                          │
# │  RewardsCfg (奖励):                                                      │
# │    RewTerm(func=mdp.reward_approach_ball,      weight=3)                 │
# │    RewTerm(func=mdp.face_target_reward,        weight=1)                 │
# │    RewTerm(func=mdp.track_ball_velocity_reward,weight=2)                 │
# │    RewTerm(func=mdp.reward_push_ball_to_goal,  weight=5)                 │
# │    RewTerm(func=mdp.check_goal,                weight=2000)              │
# │    RewTerm(func=mdp.reward_terminal_shoot,     weight=2)                 │
# │    RewTerm(func=mdp.is_alive,                  weight=-1)                │
# │                                                                          │
# │  EventsCfg (事件):                                                       │
# │    EventTerm(func=mdp.reset_root_state_uniform, mode="reset")            │
# │    EventTerm(func=mdp.randomize_rigid_body_material, mode="reset")       │
# │                                                                          │
# │  TerminationsCfg (终止):                                                 │
# │    DoneTerm(func=mdp.time_out,   time_out=True)                          │
# │    DoneTerm(func=mdp.check_goal, params={goal_x, goal_y, goal_width})    │
# │                                                                          │
# └──────────────────────────────────────────────────────────────────────────┘
