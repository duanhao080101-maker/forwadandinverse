
# ============================================================
# final_inverse_pipeline.py
# 功能：
# 1. 读取 Fluent 导出的 cell-center + cell-volume 数据
# 2. 清洗流体区域数据
# 3. 采样 PDE 配点和 100 个速度观测点
# 4. 训练特征增强 PINN/FENN 模型
# 5. 在全部 Fluent cell center 点上预测 u, v, p
# 6. 计算误差并生成最终可视化图
#
# 说明：
# 本文件是最终整合版，只保留正式计算流程。
# 已删除环境测试、Stage 03 自动微分测试、特征检查图和中间 npz/csv 采样文件。
# 可视化统一放在训练完成之后执行。
# ============================================================

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

from pathlib import Path
import random
import time
import math
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import FormatStrFormatter


# ============================================================
# 1. 基本设置
# ============================================================

SEED = 42
DTYPE = torch.float32

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# ============================================================
# 2. 物理参数
# ============================================================

R_CYLINDER = 0.5
D_CYLINDER = 1.0

U_INF = 1.0
RHO = 1.0

RE = 40.0
NU = 1.0 / RE
MU = 0.025

P_INF_FEATURE = 0.0


# ============================================================
# 3. 采样参数
# ============================================================

N_PDE_TARGET = 12300
N_DATA = 100

PDE_R_MIN = R_CYLINDER + 1e-4

NEAR_WALL_R_MAX = 2.0

WAKE_X_MIN = 0.5
WAKE_X_MAX = 8.0
WAKE_Y_ABS_MAX = 3.0

DATA_NEAR_WALL_RATIO = 0.35
DATA_WAKE_RATIO = 0.35
DATA_FAR_RATIO = 0.30


# ============================================================
# 4. 训练参数
# ============================================================

ADAM_EPOCHS = 2000
ADAM_LR = 1e-3

LBFGS_OUTER_STEPS = 1500
LBFGS_LR = 0.1
LBFGS_MAX_ITER = 20
LBFGS_HISTORY_SIZE = 50

LAMBDA_DATA = 1.0
LAMBDA_PDE = 100.0
LAMBDA_PRESSURE_ANCHOR = 1e-4

ADAM_EVAL_EVERY = 100
LBFGS_EVAL_EVERY = 10

BATCH_EVAL_SIZE = 20000


# ============================================================
# 5. 路径
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

INPUT_CSV = PROJECT_DIR / "Re40_CellField_WithVolume.csv"

FIG_DIR = PROJECT_DIR / "figures"
RESULT_DIR = PROJECT_DIR / "results"
MODEL_DIR = PROJECT_DIR / "models"

FIG_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

HISTORY_CSV = RESULT_DIR / "training_history.csv"
ERROR_METRICS_CSV = RESULT_DIR / "error_metrics.csv"
PHYSICS_VALIDATION_CSV = RESULT_DIR / "physics_validation.csv"
PREDICTION_CSV = RESULT_DIR / "full_field_prediction.csv"
PREDICTION_NPZ = RESULT_DIR / "full_field_prediction.npz"
PDE_RESIDUAL_NPZ = RESULT_DIR / "pde_residuals.npz"
PLOT_RANGE_JSON = RESULT_DIR / "plot_ranges.json"

BEST_MODEL_PATH = MODEL_DIR / "best_model.pth"
LAST_MODEL_PATH = MODEL_DIR / "last_model.pth"


# ============================================================
# 6. 工具函数
# ============================================================

def print_header(title):
    print(f"\n========== {title} ==========")


def sample_without_replacement(candidate_indices, target_number, random_generator):
    candidate_indices = np.asarray(candidate_indices)
    num_candidates = len(candidate_indices)

    if target_number <= 0:
        sampled_indices = np.array([], dtype=np.int64)
    elif target_number >= num_candidates:
        sampled_indices = candidate_indices.copy()
    else:
        sampled_indices = random_generator.choice(
            candidate_indices,
            size=target_number,
            replace=False
        )

    return np.sort(sampled_indices)


def grad(outputs, inputs):
    return torch.autograd.grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]


def relative_l2(pred, true):
    return torch.linalg.norm(pred - true) / (torch.linalg.norm(true) + 1e-12)


def rmse(pred, true):
    return torch.sqrt(torch.mean((pred - true) ** 2))


def mae(pred, true):
    return torch.mean(torch.abs(pred - true))


def max_abs_error(pred, true):
    return torch.max(torch.abs(pred - true))


# ============================================================
# 7. 读取并清洗 Fluent CSV
# ============================================================

print_header("Final PINN inverse reconstruction")

if torch.cuda.is_available():
    print("device: CUDA GPU -", torch.cuda.get_device_name(0))
else:
    print("device: CPU")

print("project:", PROJECT_DIR)
print("input:", INPUT_CSV)

if not INPUT_CSV.exists():
    raise FileNotFoundError(
        f"没有找到 {INPUT_CSV}。请把 Re40_CellField_WithVolume.csv 和本 py 文件放在同一个文件夹。"
    )

raw_df = pd.read_csv(INPUT_CSV, skipinitialspace=True)
raw_df.columns = [col.strip().lower() for col in raw_df.columns]

required_columns = [
    "cellnumber",
    "x-coordinate",
    "y-coordinate",
    "pressure",
    "x-velocity",
    "y-velocity",
    "cell-volume"
]

missing_columns = [col for col in required_columns if col not in raw_df.columns]

if len(missing_columns) > 0:
    raise ValueError(f"CSV 缺少必要列: {missing_columns}")

df = pd.DataFrame({
    "cell_id": pd.to_numeric(raw_df["cellnumber"], errors="coerce"),
    "x": pd.to_numeric(raw_df["x-coordinate"], errors="coerce"),
    "y": pd.to_numeric(raw_df["y-coordinate"], errors="coerce"),
    "p": pd.to_numeric(raw_df["pressure"], errors="coerce"),
    "u": pd.to_numeric(raw_df["x-velocity"], errors="coerce"),
    "v": pd.to_numeric(raw_df["y-velocity"], errors="coerce"),
    "cell_volume": pd.to_numeric(raw_df["cell-volume"], errors="coerce")
})

finite_mask = np.isfinite(
    df[["cell_id", "x", "y", "p", "u", "v", "cell_volume"]].to_numpy()
).all(axis=1)

positive_volume_mask = df["cell_volume"].to_numpy() > 0.0

df = df.loc[finite_mask & positive_volume_mask].copy().reset_index(drop=True)

df["r"] = np.sqrt(df["x"] ** 2 + df["y"] ** 2)
df = df.loc[df["r"] > R_CYLINDER].copy().reset_index(drop=True)

df["D"] = df["r"] - R_CYLINDER
df["speed"] = np.sqrt(df["u"] ** 2 + df["v"] ** 2)

cell_volume_mean = float(df["cell_volume"].mean())
df["cell_volume_norm"] = df["cell_volume"] / cell_volume_mean

x_np = df["x"].to_numpy(dtype=np.float64)
y_np = df["y"].to_numpy(dtype=np.float64)
p_np = df["p"].to_numpy(dtype=np.float64)
u_np = df["u"].to_numpy(dtype=np.float64)
v_np = df["v"].to_numpy(dtype=np.float64)
r_np = df["r"].to_numpy(dtype=np.float64)
cell_volume_np = df["cell_volume"].to_numpy(dtype=np.float64)
cell_volume_norm_np = df["cell_volume_norm"].to_numpy(dtype=np.float64)
speed_np = df["speed"].to_numpy(dtype=np.float64)

num_all = len(x_np)

if num_all == 0:
    raise ValueError("清洗后没有可用流体 cell，请检查 CSV 数据。")

print(f"fluid cells: {num_all}")
print(f"cell_volume: min={cell_volume_np.min():.4e}, max={cell_volume_np.max():.4e}")


# ============================================================
# 8. PDE 配点与速度观测点采样
# ============================================================

print_header("Sampling")

rng = np.random.default_rng(SEED)

pde_candidate_indices = np.where(r_np > PDE_R_MIN)[0]

if len(pde_candidate_indices) == 0:
    raise ValueError("PDE 候选点数量为 0，请检查数据。")

pde_indices = sample_without_replacement(
    candidate_indices=pde_candidate_indices,
    target_number=N_PDE_TARGET,
    random_generator=rng
)

num_pde = len(pde_indices)

x_pde_np = x_np[pde_indices].astype(np.float32)
y_pde_np = y_np[pde_indices].astype(np.float32)
volume_pde_norm_np = cell_volume_norm_np[pde_indices].astype(np.float32)

x_ref_target = np.max(x_np)
y_ref_target = 0.0
distance_to_ref = (x_np - x_ref_target) ** 2 + (y_np - y_ref_target) ** 2
ref_index = int(np.argmin(distance_to_ref))

x_ref_np = np.array([x_np[ref_index]], dtype=np.float32)
y_ref_np = np.array([y_np[ref_index]], dtype=np.float32)
p_ref_np = np.array([p_np[ref_index]], dtype=np.float32)


def sample_observed_data_indices_stratified(pde_indices, n_data, random_generator):
    pde_indices = np.asarray(pde_indices)

    pde_x = x_np[pde_indices]
    pde_y = y_np[pde_indices]
    pde_r = r_np[pde_indices]

    near_wall_mask = pde_r <= NEAR_WALL_R_MAX

    wake_mask = (
        (pde_r > NEAR_WALL_R_MAX)
        & (pde_x >= WAKE_X_MIN)
        & (pde_x <= WAKE_X_MAX)
        & (np.abs(pde_y) <= WAKE_Y_ABS_MAX)
    )

    far_mask = ~(near_wall_mask | wake_mask)

    near_wall_candidates = pde_indices[near_wall_mask]
    wake_candidates = pde_indices[wake_mask]
    far_candidates = pde_indices[far_mask]

    target_near_wall = int(round(n_data * DATA_NEAR_WALL_RATIO))
    target_wake = int(round(n_data * DATA_WAKE_RATIO))
    target_far = n_data - target_near_wall - target_wake

    sampled_near_wall = sample_without_replacement(
        candidate_indices=near_wall_candidates,
        target_number=target_near_wall,
        random_generator=random_generator
    )

    sampled_wake = sample_without_replacement(
        candidate_indices=wake_candidates,
        target_number=target_wake,
        random_generator=random_generator
    )

    sampled_far = sample_without_replacement(
        candidate_indices=far_candidates,
        target_number=target_far,
        random_generator=random_generator
    )

    sampled_indices = np.concatenate(
        [sampled_near_wall, sampled_wake, sampled_far]
    ).astype(np.int64)

    sampled_indices = np.unique(sampled_indices)

    if len(sampled_indices) < n_data:
        num_need_fill = n_data - len(sampled_indices)

        remaining_candidates = np.setdiff1d(
            pde_indices,
            sampled_indices,
            assume_unique=False
        )

        fill_indices = sample_without_replacement(
            candidate_indices=remaining_candidates,
            target_number=num_need_fill,
            random_generator=random_generator
        )

        sampled_indices = np.concatenate(
            [sampled_indices, fill_indices]
        ).astype(np.int64)

        sampled_indices = np.unique(sampled_indices)

    if len(sampled_indices) > n_data:
        sampled_indices = sample_without_replacement(
            candidate_indices=sampled_indices,
            target_number=n_data,
            random_generator=random_generator
        )

    return np.sort(sampled_indices)


data_indices = sample_observed_data_indices_stratified(
    pde_indices=pde_indices,
    n_data=N_DATA,
    random_generator=rng
)

x_data_np = x_np[data_indices].astype(np.float32)
y_data_np = y_np[data_indices].astype(np.float32)
u_data_np = u_np[data_indices].astype(np.float32)
v_data_np = v_np[data_indices].astype(np.float32)

print(f"PDE points: {num_pde}")
print(f"velocity sensors: {len(data_indices)}")
print(f"pressure anchor: x={x_ref_np[0]:.4f}, y={y_ref_np[0]:.4f}, p={p_ref_np[0]:.4e}")


# ============================================================
# 9. Tensor 数据
# ============================================================

x_pde = torch.tensor(x_pde_np, dtype=DTYPE, device=device).reshape(-1, 1)
y_pde = torch.tensor(y_pde_np, dtype=DTYPE, device=device).reshape(-1, 1)
volume_pde_norm = torch.tensor(volume_pde_norm_np, dtype=DTYPE, device=device).reshape(-1, 1)

x_data = torch.tensor(x_data_np, dtype=DTYPE, device=device).reshape(-1, 1)
y_data = torch.tensor(y_data_np, dtype=DTYPE, device=device).reshape(-1, 1)
u_data = torch.tensor(u_data_np, dtype=DTYPE, device=device).reshape(-1, 1)
v_data = torch.tensor(v_data_np, dtype=DTYPE, device=device).reshape(-1, 1)

x_all = torch.tensor(x_np.astype(np.float32), dtype=DTYPE, device=device).reshape(-1, 1)
y_all = torch.tensor(y_np.astype(np.float32), dtype=DTYPE, device=device).reshape(-1, 1)
u_all = torch.tensor(u_np.astype(np.float32), dtype=DTYPE, device=device).reshape(-1, 1)
v_all = torch.tensor(v_np.astype(np.float32), dtype=DTYPE, device=device).reshape(-1, 1)
p_all = torch.tensor(p_np.astype(np.float32), dtype=DTYPE, device=device).reshape(-1, 1)

x_ref = torch.tensor(x_ref_np, dtype=DTYPE, device=device).reshape(-1, 1)
y_ref = torch.tensor(y_ref_np, dtype=DTYPE, device=device).reshape(-1, 1)
p_ref = torch.tensor(p_ref_np, dtype=DTYPE, device=device).reshape(-1, 1)

u_data_scale = torch.maximum(
    torch.std(u_data).detach(),
    torch.tensor(1e-12, dtype=DTYPE, device=device)
)

v_data_scale = torch.maximum(
    torch.std(v_data).detach(),
    torch.tensor(1e-12, dtype=DTYPE, device=device)
)

print(f"data scales: std(u)={u_data_scale.item():.4e}, std(v)={v_data_scale.item():.4e}")


# ============================================================
# 10. 模型
# ============================================================

class FeatureEnhancedPINN(nn.Module):
    def __init__(self, hidden_layers=5, hidden_neurons=64):
        super().__init__()

        input_dim = 6
        output_dim = 3

        self.x_scale = 20.0
        self.y_scale = 15.0
        self.D_scale = 25.0

        self.upf_scale = 1.0
        self.vpf_scale = 1.0
        self.ppf_scale = 1.0

        layers = []

        layers.append(nn.Linear(input_dim, hidden_neurons))
        layers.append(nn.Tanh())

        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_neurons, hidden_neurons))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(hidden_neurons, output_dim))

        self.net = nn.Sequential(*layers)

        self.initialize_weights()

    def initialize_weights(self):
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def compute_features(self, x, y):
        eps = 1e-12

        r_sq = x ** 2 + y ** 2 + eps
        r = torch.sqrt(r_sq)
        r_fourth = r_sq ** 2

        R_sq = R_CYLINDER ** 2

        D = r - R_CYLINDER

        upf = U_INF * (
            1.0 - R_sq * (x ** 2 - y ** 2) / r_fourth
        )

        vpf = -U_INF * (
            2.0 * R_sq * x * y / r_fourth
        )

        Vpf_sq = upf ** 2 + vpf ** 2

        ppf = P_INF_FEATURE + 0.5 * RHO * (U_INF ** 2 - Vpf_sq)

        return D, upf, vpf, ppf

    def forward(self, x, y):
        D, upf, vpf, ppf = self.compute_features(x, y)

        x_in = x / self.x_scale
        y_in = y / self.y_scale
        D_in = D / self.D_scale

        upf_in = upf / self.upf_scale
        vpf_in = vpf / self.vpf_scale
        ppf_in = ppf / self.ppf_scale

        network_input = torch.cat(
            [x_in, y_in, D_in, upf_in, vpf_in, ppf_in],
            dim=1
        )

        output = self.net(network_input)

        u = output[:, 0:1]
        v = output[:, 1:2]
        p = output[:, 2:3]

        return u, v, p


def navier_stokes_residual(model, x_raw, y_raw):
    x = x_raw.clone().detach().requires_grad_(True)
    y = y_raw.clone().detach().requires_grad_(True)

    u, v, p = model(x, y)

    u_x = grad(u, x)
    u_y = grad(u, y)

    v_x = grad(v, x)
    v_y = grad(v, y)

    p_x = grad(p, x)
    p_y = grad(p, y)

    u_xx = grad(u_x, x)
    u_yy = grad(u_y, y)

    v_xx = grad(v_x, x)
    v_yy = grad(v_y, y)

    f_c = u_x + v_y

    f_u = (
        u * u_x
        + v * u_y
        + p_x
        - NU * (u_xx + u_yy)
    )

    f_v = (
        u * v_x
        + v * v_y
        + p_y
        - NU * (v_xx + v_yy)
    )

    return f_c, f_u, f_v


# ============================================================
# 11. Loss 与评估
# ============================================================

def compute_total_loss(model):
    u_pred_data, v_pred_data, _ = model(x_data, y_data)

    loss_data_u = torch.mean(((u_pred_data - u_data) / u_data_scale) ** 2)
    loss_data_v = torch.mean(((v_pred_data - v_data) / v_data_scale) ** 2)

    loss_data = 0.5 * (loss_data_u + loss_data_v)

    f_c, f_u, f_v = navier_stokes_residual(model, x_pde, y_pde)

    residual_square = f_c ** 2 + f_u ** 2 + f_v ** 2
    volume_square = volume_pde_norm ** 2

    loss_pde = (
        torch.sum(volume_square * residual_square)
        / (torch.sum(volume_square) + 1e-12)
    )

    raw_continuity_rms = torch.sqrt(torch.mean(f_c ** 2))
    raw_momentum_x_rms = torch.sqrt(torch.mean(f_u ** 2))
    raw_momentum_y_rms = torch.sqrt(torch.mean(f_v ** 2))

    _, _, p_pred_ref = model(x_ref, y_ref)
    loss_pressure_anchor = torch.mean((p_pred_ref - p_ref) ** 2)

    total_loss = (
        LAMBDA_DATA * loss_data
        + LAMBDA_PDE * loss_pde
        + LAMBDA_PRESSURE_ANCHOR * loss_pressure_anchor
    )

    loss_dict = {
        "total_loss": total_loss,
        "loss_data": loss_data,
        "loss_pde_weighted": loss_pde,
        "loss_pressure_anchor": loss_pressure_anchor,
        "raw_continuity_rms": raw_continuity_rms,
        "raw_momentum_x_rms": raw_momentum_x_rms,
        "raw_momentum_y_rms": raw_momentum_y_rms
    }

    return loss_dict


@torch.no_grad()
def predict_in_batches(model, x_input, y_input, batch_size=20000):
    model.eval()

    u_list = []
    v_list = []
    p_list = []

    total_points = x_input.shape[0]

    for start in range(0, total_points, batch_size):
        end = min(start + batch_size, total_points)

        u_batch, v_batch, p_batch = model(
            x_input[start:end],
            y_input[start:end]
        )

        u_list.append(u_batch)
        v_list.append(v_batch)
        p_list.append(p_batch)

    u_pred = torch.cat(u_list, dim=0)
    v_pred = torch.cat(v_list, dim=0)
    p_pred = torch.cat(p_list, dim=0)

    return u_pred, v_pred, p_pred


@torch.no_grad()
def evaluate_full_field(model):
    model.eval()

    u_pred_eval, v_pred_eval, p_pred_eval = predict_in_batches(
        model,
        x_all,
        y_all,
        batch_size=BATCH_EVAL_SIZE
    )

    _, _, p_ref_pred = model(x_ref, y_ref)
    pressure_shift_eval = p_ref - p_ref_pred
    p_pred_aligned = p_pred_eval + pressure_shift_eval

    err_u = relative_l2(u_pred_eval, u_all)
    err_v = relative_l2(v_pred_eval, v_all)
    err_p = relative_l2(p_pred_aligned, p_all)

    return err_u.item(), err_v.item(), err_p.item()


# ============================================================
# 12. 训练
# ============================================================

print_header("Training")

model = FeatureEnhancedPINN(
    hidden_layers=5,
    hidden_neurons=64
).to(device)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("trainable parameters:", num_params)

optimizer_adam = torch.optim.Adam(
    model.parameters(),
    lr=ADAM_LR
)

optimizer_lbfgs = torch.optim.LBFGS(
    model.parameters(),
    lr=LBFGS_LR,
    max_iter=LBFGS_MAX_ITER,
    tolerance_grad=1e-7,
    tolerance_change=1e-9,
    history_size=LBFGS_HISTORY_SIZE,
    line_search_fn="strong_wolfe"
)

history = []
best_error_uv = float("inf")
start_time = time.time()


def save_model(path, extra_info=None):
    payload = {
        "model_state_dict": model.state_dict(),
        "RE": RE,
        "NU": NU,
        "R_CYLINDER": R_CYLINDER,
        "D_CYLINDER": D_CYLINDER,
        "U_INF": U_INF,
        "RHO": RHO,
        "LAMBDA_DATA": LAMBDA_DATA,
        "LAMBDA_PDE": LAMBDA_PDE,
        "LAMBDA_PRESSURE_ANCHOR": LAMBDA_PRESSURE_ANCHOR,
        "u_data_scale": u_data_scale.detach().cpu().item(),
        "v_data_scale": v_data_scale.detach().cpu().item(),
        "input_features": ["x", "y", "D", "upf", "vpf", "ppf"],
        "output_fields": ["u", "v", "p"]
    }

    if extra_info is not None:
        payload.update(extra_info)

    torch.save(payload, path)


def save_history_csv():
    pd.DataFrame(history).to_csv(HISTORY_CSV, index=False)


def record_history(stage_name, epoch_id, loss_dict):
    global best_error_uv

    err_u, err_v, err_p = evaluate_full_field(model)

    total_loss_value = loss_dict["total_loss"].detach().cpu().item()
    loss_data_value = loss_dict["loss_data"].detach().cpu().item()
    loss_pde_value = loss_dict["loss_pde_weighted"].detach().cpu().item()
    loss_pressure_anchor_value = loss_dict["loss_pressure_anchor"].detach().cpu().item()

    raw_c_value = loss_dict["raw_continuity_rms"].detach().cpu().item()
    raw_u_value = loss_dict["raw_momentum_x_rms"].detach().cpu().item()
    raw_v_value = loss_dict["raw_momentum_y_rms"].detach().cpu().item()

    elapsed_min = (time.time() - start_time) / 60.0

    history.append({
        "stage": stage_name,
        "epoch": epoch_id,
        "total_loss": total_loss_value,
        "loss_data": loss_data_value,
        "loss_pde_weighted": loss_pde_value,
        "loss_pressure_anchor": loss_pressure_anchor_value,
        "raw_continuity_rms": raw_c_value,
        "raw_momentum_x_rms": raw_u_value,
        "raw_momentum_y_rms": raw_v_value,
        "relative_l2_u": err_u,
        "relative_l2_v": err_v,
        "relative_l2_p": err_p,
        "elapsed_min": elapsed_min
    })

    error_uv = err_u + err_v

    if error_uv < best_error_uv:
        best_error_uv = error_uv
        save_model(
            BEST_MODEL_PATH,
            extra_info={
                "best_error_uv": best_error_uv,
                "relative_l2_u": err_u,
                "relative_l2_v": err_v,
                "relative_l2_p": err_p
            }
        )

    print(
        f"[{stage_name:5s} {epoch_id:5d}] "
        f"Total={total_loss_value:.4e} | "
        f"Data={loss_data_value:.4e} | "
        f"PDEw={loss_pde_value:.4e} | "
        f"Rc={raw_c_value:.4e} | "
        f"Ru={raw_u_value:.4e} | "
        f"Rv={raw_v_value:.4e} | "
        f"L2u={err_u:.4e} | "
        f"L2v={err_v:.4e} | "
        f"L2p={err_p:.4e} | "
        f"time={elapsed_min:.2f} min"
    )


print("Adam start")

for epoch in range(1, ADAM_EPOCHS + 1):
    model.train()

    optimizer_adam.zero_grad()

    loss_dict = compute_total_loss(model)
    loss_dict["total_loss"].backward()

    optimizer_adam.step()

    if epoch == 1 or epoch % ADAM_EVAL_EVERY == 0 or epoch == ADAM_EPOCHS:
        loss_dict_eval = compute_total_loss(model)
        record_history("Adam", epoch, loss_dict_eval)
        save_history_csv()


print("L-BFGS start")

lbfgs_epoch_counter = 0


def closure():
    optimizer_lbfgs.zero_grad()

    loss_dict_closure = compute_total_loss(model)
    loss_dict_closure["total_loss"].backward()

    return loss_dict_closure["total_loss"]


for outer_step in range(1, LBFGS_OUTER_STEPS + 1):
    model.train()

    optimizer_lbfgs.step(closure)

    lbfgs_epoch_counter += 1

    if (
        outer_step == 1
        or outer_step % LBFGS_EVAL_EVERY == 0
        or outer_step == LBFGS_OUTER_STEPS
    ):
        loss_dict_eval = compute_total_loss(model)
        record_history("LBFGS", lbfgs_epoch_counter, loss_dict_eval)
        save_history_csv()


save_model(LAST_MODEL_PATH)
save_history_csv()

total_elapsed_min = (time.time() - start_time) / 60.0

print(f"training finished: {total_elapsed_min:.2f} min")
print("history:", HISTORY_CSV)
print("best model:", BEST_MODEL_PATH)
print("last model:", LAST_MODEL_PATH)


# ============================================================
# 13. 加载 best model 做最终预测
# ============================================================

print_header("Prediction")

model = FeatureEnhancedPINN(
    hidden_layers=5,
    hidden_neurons=64
).to(device)

checkpoint = torch.load(BEST_MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

u_pred, v_pred, p_pred_raw = predict_in_batches(
    model,
    x_all,
    y_all,
    batch_size=BATCH_EVAL_SIZE
)

_, _, p_ref_pred_raw = model(x_ref, y_ref)
pressure_shift = (p_ref - p_ref_pred_raw).detach()
p_pred = p_pred_raw + pressure_shift

speed_true = torch.sqrt(u_all ** 2 + v_all ** 2)
speed_pred = torch.sqrt(u_pred ** 2 + v_pred ** 2)

metrics = {
    "relative_l2_u": relative_l2(u_pred, u_all).item(),
    "relative_l2_v": relative_l2(v_pred, v_all).item(),
    "relative_l2_p": relative_l2(p_pred, p_all).item(),
    "relative_l2_speed": relative_l2(speed_pred, speed_true).item(),

    "rmse_u": rmse(u_pred, u_all).item(),
    "rmse_v": rmse(v_pred, v_all).item(),
    "rmse_p": rmse(p_pred, p_all).item(),
    "rmse_speed": rmse(speed_pred, speed_true).item(),

    "mae_u": mae(u_pred, u_all).item(),
    "mae_v": mae(v_pred, v_all).item(),
    "mae_p": mae(p_pred, p_all).item(),
    "mae_speed": mae(speed_pred, speed_true).item(),

    "max_abs_error_u": max_abs_error(u_pred, u_all).item(),
    "max_abs_error_v": max_abs_error(v_pred, v_all).item(),
    "max_abs_error_p": max_abs_error(p_pred, p_all).item(),
    "max_abs_error_speed": max_abs_error(speed_pred, speed_true).item(),
}

metrics_df = pd.DataFrame([metrics])
metrics_df.to_csv(ERROR_METRICS_CSV, index=False)

print(
    "final L2: "
    f"u={metrics['relative_l2_u']:.4e}, "
    f"v={metrics['relative_l2_v']:.4e}, "
    f"p={metrics['relative_l2_p']:.4e}, "
    f"|V|={metrics['relative_l2_speed']:.4e}"
)

u_pred_np = u_pred.detach().cpu().numpy().reshape(-1)
v_pred_np = v_pred.detach().cpu().numpy().reshape(-1)
p_pred_np = p_pred.detach().cpu().numpy().reshape(-1)
speed_pred_np = speed_pred.detach().cpu().numpy().reshape(-1)

u_true_np = u_np.astype(np.float32)
v_true_np = v_np.astype(np.float32)
p_true_np = p_np.astype(np.float32)
speed_true_np = speed_np.astype(np.float32)

abs_err_u_np = np.abs(u_pred_np - u_true_np)
abs_err_v_np = np.abs(v_pred_np - v_true_np)
abs_err_p_np = np.abs(p_pred_np - p_true_np)
abs_err_speed_np = np.abs(speed_pred_np - speed_true_np)

np.savez(
    PREDICTION_NPZ,
    x=x_np.astype(np.float32),
    y=y_np.astype(np.float32),

    u_true=u_true_np,
    v_true=v_true_np,
    p_true=p_true_np,
    speed_true=speed_true_np,

    u_pred=u_pred_np,
    v_pred=v_pred_np,
    p_pred=p_pred_np,
    speed_pred=speed_pred_np,

    abs_err_u=abs_err_u_np,
    abs_err_v=abs_err_v_np,
    abs_err_p=abs_err_p_np,
    abs_err_speed=abs_err_speed_np,
)

prediction_df = pd.DataFrame({
    "x": x_np.astype(np.float32),
    "y": y_np.astype(np.float32),

    "u_true": u_true_np,
    "v_true": v_true_np,
    "p_true": p_true_np,
    "speed_true": speed_true_np,

    "u_pred": u_pred_np,
    "v_pred": v_pred_np,
    "p_pred": p_pred_np,
    "speed_pred": speed_pred_np,

    "abs_err_u": abs_err_u_np,
    "abs_err_v": abs_err_v_np,
    "abs_err_p": abs_err_p_np,
    "abs_err_speed": abs_err_speed_np,
})

prediction_df.to_csv(PREDICTION_CSV, index=False)


# ============================================================
# 14. 全场 PDE residual
# ============================================================

def compute_residual_in_batches(model, x_input_np, y_input_np, batch_size=4000):
    model.eval()

    res_c_list = []
    res_u_list = []
    res_v_list = []

    total_points = len(x_input_np)

    for start in range(0, total_points, batch_size):
        end = min(start + batch_size, total_points)

        xb = torch.tensor(
            x_input_np[start:end].astype(np.float32),
            dtype=DTYPE,
            device=device
        ).reshape(-1, 1).requires_grad_(True)

        yb = torch.tensor(
            y_input_np[start:end].astype(np.float32),
            dtype=DTYPE,
            device=device
        ).reshape(-1, 1).requires_grad_(True)

        u, v, p = model(xb, yb)

        u_x = grad(u, xb)
        u_y = grad(u, yb)

        v_x = grad(v, xb)
        v_y = grad(v, yb)

        p_x = grad(p, xb)
        p_y = grad(p, yb)

        u_xx = grad(u_x, xb)
        u_yy = grad(u_y, yb)

        v_xx = grad(v_x, xb)
        v_yy = grad(v_y, yb)

        res_c = u_x + v_y

        res_u = (
            u * u_x
            + v * u_y
            + p_x
            - NU * (u_xx + u_yy)
        )

        res_v = (
            u * v_x
            + v * v_y
            + p_y
            - NU * (v_xx + v_yy)
        )

        res_c_list.append(res_c.detach().cpu().numpy().reshape(-1))
        res_u_list.append(res_u.detach().cpu().numpy().reshape(-1))
        res_v_list.append(res_v.detach().cpu().numpy().reshape(-1))

        del xb, yb
        del u, v, p
        del u_x, u_y, v_x, v_y, p_x, p_y
        del u_xx, u_yy, v_xx, v_yy
        del res_c, res_u, res_v

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return (
        np.concatenate(res_c_list),
        np.concatenate(res_u_list),
        np.concatenate(res_v_list)
    )


res_c_np, res_u_np, res_v_np = compute_residual_in_batches(
    model,
    x_np,
    y_np,
    batch_size=4000
)

np.savez(
    PDE_RESIDUAL_NPZ,
    x=x_np.astype(np.float32),
    y=y_np.astype(np.float32),
    residual_continuity=res_c_np,
    residual_momentum_x=res_u_np,
    residual_momentum_y=res_v_np
)


# ============================================================
# 15. 可视化函数
# ============================================================

def add_cylinder(ax):
    circle = plt.Circle(
        (0.0, 0.0),
        R_CYLINDER,
        color="white",
        ec="black",
        linewidth=1.2,
        zorder=20
    )
    ax.add_patch(circle)


def setup_field_axis(ax):
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(np.min(x_np), np.max(x_np))
    ax.set_ylim(np.min(y_np), np.max(y_np))


def build_triangulation(x, y, r_cylinder=0.5):
    triang = mtri.Triangulation(x, y)

    triangles = triang.triangles

    x_tri = x[triangles]
    y_tri = y[triangles]

    x_centroid = x_tri.mean(axis=1)
    y_centroid = y_tri.mean(axis=1)
    r_centroid = np.sqrt(x_centroid ** 2 + y_centroid ** 2)

    edge_01 = np.sqrt((x_tri[:, 0] - x_tri[:, 1]) ** 2 + (y_tri[:, 0] - y_tri[:, 1]) ** 2)
    edge_12 = np.sqrt((x_tri[:, 1] - x_tri[:, 2]) ** 2 + (y_tri[:, 1] - y_tri[:, 2]) ** 2)
    edge_20 = np.sqrt((x_tri[:, 2] - x_tri[:, 0]) ** 2 + (y_tri[:, 2] - y_tri[:, 0]) ** 2)

    max_edge = np.maximum(np.maximum(edge_01, edge_12), edge_20)
    edge_threshold = np.nanpercentile(max_edge, 99.5) * 1.5

    mask_cylinder = r_centroid < (r_cylinder + 0.02)
    mask_large_triangle = max_edge > edge_threshold

    triang.set_mask(mask_cylinder | mask_large_triangle)

    return triang


GLOBAL_TRIANG = build_triangulation(
    x_np.astype(np.float32),
    y_np.astype(np.float32),
    r_cylinder=R_CYLINDER
)


def symmetric_range(*arrays):
    max_abs = 0.0

    for arr in arrays:
        arr = np.asarray(arr)
        local_max = np.nanmax(np.abs(arr))
        max_abs = max(max_abs, local_max)

    if max_abs <= 0.0:
        max_abs = 1.0

    return -max_abs, max_abs


def nice_number_ceil(x):
    if x <= 0.0 or np.isnan(x):
        return 1.0

    exponent = math.floor(math.log10(x))
    base = x / (10.0 ** exponent)

    if base <= 1.0:
        nice_base = 1.0
    elif base <= 2.0:
        nice_base = 2.0
    elif base <= 2.5:
        nice_base = 2.5
    elif base <= 5.0:
        nice_base = 5.0
    else:
        nice_base = 10.0

    return nice_base * (10.0 ** exponent)


def nice_error_ticks(vmax_raw):
    vmax = nice_number_ceil(vmax_raw)

    raw_step = vmax / 6.0
    step = nice_number_ceil(raw_step)

    vmax = math.ceil(vmax / step) * step

    ticks = np.arange(0.0, vmax + 0.5 * step, step)

    return ticks, vmax, step


def positive_error_range_and_ticks(error_array):
    vmax_raw = float(np.nanmax(error_array))

    ticks, vmax, step = nice_error_ticks(vmax_raw)

    return 0.0, vmax, ticks, step


def make_levels(vmin, vmax, n_levels=160):
    if vmax <= vmin:
        vmax = vmin + 1.0

    return np.linspace(vmin, vmax, n_levels)


def save_single_field_figure(
    value,
    title,
    colorbar_label,
    save_name,
    cmap="coolwarm",
    vmin=None,
    vmax=None,
    norm=None,
    n_levels=160,
    error_ticks=None,
    error_step=None
):
    fig, ax = plt.subplots(figsize=(8.5, 6.0))

    if norm is not None:
        levels = make_levels(norm.vmin, norm.vmax, n_levels)

        contour = ax.tricontourf(
            GLOBAL_TRIANG,
            value,
            levels=levels,
            cmap=cmap,
            norm=norm
        )
    else:
        levels = make_levels(vmin, vmax, n_levels)

        contour = ax.tricontourf(
            GLOBAL_TRIANG,
            value,
            levels=levels,
            cmap=cmap
        )

    add_cylinder(ax)
    setup_field_axis(ax)

    ax.set_title(title)

    cbar = fig.colorbar(contour, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label(colorbar_label)

    if error_ticks is not None:
        cbar.set_ticks(error_ticks)

        if error_step is not None and error_step < 0.01:
            cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        else:
            cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    fig.tight_layout()

    save_path = FIG_DIR / save_name
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_field_triplet(
    name,
    true_value,
    pred_value,
    abs_error,
    field_cmap,
    error_cmap,
    signed_field=True
):
    if signed_field:
        vmin, vmax = symmetric_range(true_value, pred_value)
        field_norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    else:
        vmin = float(min(np.nanmin(true_value), np.nanmin(pred_value)))
        vmax = float(max(np.nanmax(true_value), np.nanmax(pred_value)))
        field_norm = None

    err_vmin, err_vmax, err_ticks, err_step = positive_error_range_and_ticks(abs_error)

    save_single_field_figure(
        value=true_value,
        title=f"Fluent Ground Truth {name}",
        colorbar_label=name,
        save_name=f"ground_truth_{name}.png",
        cmap=field_cmap,
        vmin=vmin,
        vmax=vmax,
        norm=field_norm
    )

    save_single_field_figure(
        value=pred_value,
        title=f"PINN Prediction {name}",
        colorbar_label=name,
        save_name=f"pinn_prediction_{name}.png",
        cmap=field_cmap,
        vmin=vmin,
        vmax=vmax,
        norm=field_norm
    )

    save_single_field_figure(
        value=abs_error,
        title=f"Absolute Error {name}",
        colorbar_label=f"|error {name}|",
        save_name=f"absolute_error_{name}.png",
        cmap=error_cmap,
        vmin=err_vmin,
        vmax=err_vmax,
        norm=None,
        error_ticks=err_ticks,
        error_step=err_step
    )


# ============================================================
# 16. 训练历史图
# ============================================================

def build_global_training_axis(history_df):
    if "stage" not in history_df.columns or "epoch" not in history_df.columns:
        history_df["global_step"] = np.arange(len(history_df))
        return history_df, None

    adam_mask = history_df["stage"].astype(str).str.lower().str.contains("adam")

    if np.any(adam_mask):
        adam_end = int(history_df.loc[adam_mask, "epoch"].max())
    else:
        adam_end = 0

    global_steps = []

    for _, row in history_df.iterrows():
        stage_name = str(row["stage"]).lower()
        epoch_value = int(row["epoch"])

        if "adam" in stage_name:
            global_steps.append(epoch_value)
        elif "lbfgs" in stage_name:
            global_steps.append(adam_end + epoch_value)
        else:
            global_steps.append(epoch_value)

    history_df["global_step"] = global_steps

    return history_df, adam_end


def plot_training_history():
    history_df = pd.read_csv(HISTORY_CSV)
    history_df, adam_end = build_global_training_axis(history_df)

    fig, ax = plt.subplots(figsize=(8.5, 5.0))

    ax.semilogy(history_df["global_step"], history_df["total_loss"], label="Total loss")
    ax.semilogy(history_df["global_step"], history_df["loss_data"], label="Data loss")
    ax.semilogy(history_df["global_step"], history_df["loss_pde_weighted"], label="PDE loss")

    if adam_end is not None and adam_end > 0:
        ax.axvline(
            adam_end,
            linestyle="--",
            linewidth=1.0,
            color="black",
            label="Adam / L-BFGS switch"
        )

    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss History")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(FIG_DIR / "history_loss.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5.0))

    ax.semilogy(history_df["global_step"], history_df["raw_continuity_rms"], label="Continuity RMS")
    ax.semilogy(history_df["global_step"], history_df["raw_momentum_x_rms"], label="X-momentum RMS")
    ax.semilogy(history_df["global_step"], history_df["raw_momentum_y_rms"], label="Y-momentum RMS")

    if adam_end is not None and adam_end > 0:
        ax.axvline(
            adam_end,
            linestyle="--",
            linewidth=1.0,
            color="black",
            label="Adam / L-BFGS switch"
        )

    ax.set_xlabel("Training step")
    ax.set_ylabel("Residual RMS")
    ax.set_title("PDE Residual History")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(FIG_DIR / "history_pde_residual.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5.0))

    ax.semilogy(history_df["global_step"], history_df["relative_l2_u"], label="Relative L2 u")
    ax.semilogy(history_df["global_step"], history_df["relative_l2_v"], label="Relative L2 v")
    ax.semilogy(history_df["global_step"], history_df["relative_l2_p"], label="Relative L2 p")

    if adam_end is not None and adam_end > 0:
        ax.axvline(
            adam_end,
            linestyle="--",
            linewidth=1.0,
            color="black",
            label="Adam / L-BFGS switch"
        )

    ax.set_xlabel("Training step")
    ax.set_ylabel("Relative L2 error")
    ax.set_title("Relative L2 Error History")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(FIG_DIR / "history_l2_error.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 17. 物理量评估
# ============================================================

@torch.no_grad()
def predict_line(x_line_np, y_line_np):
    x_line = torch.tensor(
        x_line_np,
        dtype=DTYPE,
        device=device
    ).reshape(-1, 1)

    y_line = torch.tensor(
        y_line_np,
        dtype=DTYPE,
        device=device
    ).reshape(-1, 1)

    u_line, v_line, p_line_raw = model(x_line, y_line)

    p_line = p_line_raw + pressure_shift

    return (
        u_line.detach().cpu().numpy().reshape(-1),
        v_line.detach().cpu().numpy().reshape(-1),
        p_line.detach().cpu().numpy().reshape(-1)
    )


def compute_reattachment_length():
    x_line = np.linspace(0.5005, 8.0, 3000).astype(np.float32)
    y_line = np.zeros_like(x_line, dtype=np.float32)

    u_line, _, _ = predict_line(x_line, y_line)

    x_reattach = np.nan
    Lw = np.nan
    status = "not_found"

    for i in range(len(x_line) - 1):
        if u_line[i] < 0.0 and u_line[i + 1] >= 0.0:
            x1 = x_line[i]
            x2 = x_line[i + 1]
            u1 = u_line[i]
            u2 = u_line[i + 1]

            x_reattach = x1 - u1 * (x2 - x1) / (u2 - u1 + 1e-12)
            Lw = x_reattach - 0.5
            status = "computed"
            break

    fig, ax = plt.subplots(figsize=(8.5, 4.8))

    ax.plot(x_line, u_line, label="PINN u(x, 0)")
    ax.axhline(0.0, linestyle="--", linewidth=1.0, label="u=0")
    ax.axvline(0.5, linestyle=":", linewidth=1.0, label="Cylinder rear edge")

    if status == "computed":
        ax.axvline(
            x_reattach,
            linestyle="--",
            linewidth=1.0,
            label=f"Reattachment x={x_reattach:.3f}"
        )

    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.set_title("Centerline Velocity and Recirculation Length")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(FIG_DIR / "centerline_u_reattachment.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    return float(Lw), float(x_reattach), status


def compute_wall_quantities():
    n_theta = 1440

    theta = np.linspace(
        0.0,
        2.0 * np.pi,
        n_theta + 1,
        endpoint=True
    ).astype(np.float32)

    x_wall_np = (R_CYLINDER * np.cos(theta)).astype(np.float32)
    y_wall_np = (R_CYLINDER * np.sin(theta)).astype(np.float32)

    x_wall = torch.tensor(
        x_wall_np,
        dtype=DTYPE,
        device=device
    ).reshape(-1, 1).requires_grad_(True)

    y_wall = torch.tensor(
        y_wall_np,
        dtype=DTYPE,
        device=device
    ).reshape(-1, 1).requires_grad_(True)

    u, v, p_raw = model(x_wall, y_wall)

    p = p_raw + pressure_shift

    u_x = grad(u, x_wall)
    u_y = grad(u, y_wall)

    v_x = grad(v, x_wall)
    v_y = grad(v, y_wall)

    theta_tensor = torch.tensor(
        theta,
        dtype=DTYPE,
        device=device
    ).reshape(-1, 1)

    nx = torch.cos(theta_tensor)
    ny = torch.sin(theta_tensor)

    tx = -ny
    ty = nx

    sigma_xx = -p + 2.0 * MU * u_x
    sigma_xy = MU * (u_y + v_x)
    sigma_yy = -p + 2.0 * MU * v_y

    traction_x = sigma_xx * nx + sigma_xy * ny
    traction_y = sigma_xy * nx + sigma_yy * ny

    tau_t = traction_x * tx + traction_y * ty

    drag_density = traction_x.reshape(-1)

    drag_force = torch.trapz(
        drag_density,
        theta_tensor.reshape(-1)
    ) * R_CYLINDER

    CD_raw = drag_force / (0.5 * RHO * U_INF ** 2 * D_CYLINDER)
    CD_abs = torch.abs(CD_raw)

    theta_deg = theta * 180.0 / np.pi
    tau_np = tau_t.detach().cpu().numpy().reshape(-1)

    upper_mask = (theta_deg >= 10.0) & (theta_deg <= 110.0)

    theta_upper = theta_deg[upper_mask]
    tau_upper = tau_np[upper_mask]

    theta_sep = np.nan
    theta_status = "not_reliable_no_zero_crossing"

    for i in range(len(theta_upper) - 1):
        t1 = tau_upper[i]
        t2 = tau_upper[i + 1]

        if t1 == 0.0 or t1 * t2 < 0.0:
            th1 = theta_upper[i]
            th2 = theta_upper[i + 1]

            theta_sep = th1 - t1 * (th2 - th1) / (t2 - t1 + 1e-12)
            theta_status = "computed_zero_crossing"
            break

    fig, ax = plt.subplots(figsize=(8.5, 4.8))

    ax.plot(theta_upper, tau_upper, label="PINN wall shear stress")
    ax.axhline(0.0, linestyle="--", linewidth=1.0, label="tau=0")

    if theta_status == "computed_zero_crossing":
        ax.axvline(
            theta_sep,
            linestyle="--",
            linewidth=1.0,
            label=f"theta_s={theta_sep:.2f} deg"
        )
    else:
        ax.text(
            0.02,
            0.95,
            "No reliable zero-crossing found",
            transform=ax.transAxes,
            va="top"
        )

    ax.set_xlabel("theta from rear stagnation point (degree)")
    ax.set_ylabel("wall shear stress")
    ax.set_title("Wall Shear Stress and Separation Angle")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(FIG_DIR / "wall_shear_separation_angle.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    return (
        float(theta_sep),
        theta_status,
        float(CD_abs.detach().cpu().item()),
        float(CD_raw.detach().cpu().item()),
        "diagnostic_wall_gradient_sensitive"
    )


# ============================================================
# 18. 生成最终图与表
# ============================================================

print_header("Visualization")

plot_training_history()

save_field_triplet(
    name="u",
    true_value=u_true_np,
    pred_value=u_pred_np,
    abs_error=abs_err_u_np,
    field_cmap="coolwarm",
    error_cmap="jet",
    signed_field=True
)

save_field_triplet(
    name="v",
    true_value=v_true_np,
    pred_value=v_pred_np,
    abs_error=abs_err_v_np,
    field_cmap="coolwarm",
    error_cmap="jet",
    signed_field=True
)

save_field_triplet(
    name="p",
    true_value=p_true_np,
    pred_value=p_pred_np,
    abs_error=abs_err_p_np,
    field_cmap="coolwarm",
    error_cmap="jet",
    signed_field=True
)

save_field_triplet(
    name="speed",
    true_value=speed_true_np,
    pred_value=speed_pred_np,
    abs_error=abs_err_speed_np,
    field_cmap="jet",
    error_cmap="jet",
    signed_field=False
)

log_res_c = np.log10(np.abs(res_c_np) + 1e-12)
log_res_u = np.log10(np.abs(res_u_np) + 1e-12)
log_res_v = np.log10(np.abs(res_v_np) + 1e-12)

combined_residual = np.concatenate([log_res_c, log_res_u, log_res_v])

res_vmin = np.nanpercentile(combined_residual, 1.0)
res_vmax = np.nanpercentile(combined_residual, 99.0)

save_single_field_figure(
    value=log_res_c,
    title="log10 |Continuity Residual|",
    colorbar_label="log10(|Rc|)",
    save_name="residual_continuity.png",
    cmap="viridis",
    vmin=res_vmin,
    vmax=res_vmax
)

save_single_field_figure(
    value=log_res_u,
    title="log10 |X-Momentum Residual|",
    colorbar_label="log10(|Ru|)",
    save_name="residual_momentum_x.png",
    cmap="viridis",
    vmin=res_vmin,
    vmax=res_vmax
)

save_single_field_figure(
    value=log_res_v,
    title="log10 |Y-Momentum Residual|",
    colorbar_label="log10(|Rv|)",
    save_name="residual_momentum_y.png",
    cmap="viridis",
    vmin=res_vmin,
    vmax=res_vmax
)

Lw_pinn, x_reattach_pinn, Lw_status = compute_reattachment_length()
theta_s_pinn, theta_status, CD_pinn, CD_raw_pinn, CD_status = compute_wall_quantities()

physics_df = pd.DataFrame([
    {
        "quantity": "Drag coefficient CD",
        "PINN": CD_pinn,
        "PINN_raw_signed_value": CD_raw_pinn,
        "Fluent_reference": 1.5695,
        "Literature_reference": "1.52-1.53",
        "status": CD_status,
        "note": "Computed from PINN wall pressure and velocity gradients."
    },
    {
        "quantity": "Recirculation length Lw/D",
        "PINN": Lw_pinn,
        "PINN_raw_signed_value": "",
        "Fluent_reference": 2.25,
        "Literature_reference": "2.2-2.3",
        "status": Lw_status,
        "note": "Computed from centerline u(x,0) zero-crossing behind the cylinder."
    },
    {
        "quantity": "Separation angle theta_s degree",
        "PINN": theta_s_pinn,
        "PINN_raw_signed_value": "",
        "Fluent_reference": 51.7,
        "Literature_reference": "53-54",
        "status": theta_status,
        "note": "Computed from wall shear stress zero-crossing."
    }
])

physics_df.to_csv(PHYSICS_VALIDATION_CSV, index=False)

plot_range_dict = {
    "u_symmetric_range": list(symmetric_range(u_true_np, u_pred_np)),
    "v_symmetric_range": list(symmetric_range(v_true_np, v_pred_np)),
    "p_symmetric_range": list(symmetric_range(p_true_np, p_pred_np)),
    "speed_range": [
        float(min(np.nanmin(speed_true_np), np.nanmin(speed_pred_np))),
        float(max(np.nanmax(speed_true_np), np.nanmax(speed_pred_np)))
    ],
    "error_u_range": [0.0, float(nice_error_ticks(np.nanmax(abs_err_u_np))[1])],
    "error_v_range": [0.0, float(nice_error_ticks(np.nanmax(abs_err_v_np))[1])],
    "error_p_range": [0.0, float(nice_error_ticks(np.nanmax(abs_err_p_np))[1])],
    "error_speed_range": [0.0, float(nice_error_ticks(np.nanmax(abs_err_speed_np))[1])]
}

with open(PLOT_RANGE_JSON, "w", encoding="utf-8") as f:
    json.dump(plot_range_dict, f, indent=4)

print("figures:", FIG_DIR)
print("results:", RESULT_DIR)
print("physics validation:", PHYSICS_VALIDATION_CSV)

print_header("Finished")
print(f"Best L2u={metrics['relative_l2_u']:.4e}, L2v={metrics['relative_l2_v']:.4e}, L2p={metrics['relative_l2_p']:.4e}")
print(f"Lw/D={Lw_pinn:.4f}, theta_s={theta_s_pinn:.4f}, CD={CD_pinn:.4f}")
