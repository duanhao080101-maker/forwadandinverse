import numpy as np
import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt

np.random.seed(42)
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"当前使用的计算设备: {device}")

# ==========================================
# 1. 网格采样
# ==========================================
D_in, D_out = 1.0, 40.0
r_in, r_out = D_in / 2.0, D_out / 2.0
N_bc_wall, N_bc_far = 80, 80

def sample_annulus_interior_structured(n_radial=60, n_angular=80, r_inner=0.5, r_outer=20.0):
    r_lin = np.linspace(r_inner + 0.05, r_outer - 0.05, n_radial)
    theta_lin = np.linspace(0, 2 * np.pi, n_angular, endpoint=False)
    R, Theta = np.meshgrid(r_lin, theta_lin)
    r_flat = R.flatten()
    theta_flat = Theta.flatten()
    x = r_flat * np.cos(theta_flat)
    y = r_flat * np.sin(theta_flat)

    dr = (r_outer - r_inner) / n_radial
    dtheta = (2 * np.pi) / n_angular
    weights = r_flat * dr * dtheta
    return np.stack([x, y], axis=1), weights

def sample_circle_boundary(n_points, radius):
    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    x, y = radius * np.cos(theta), radius * np.sin(theta)
    nx, ny = np.cos(theta), np.sin(theta)
    return np.stack([x, y], axis=1), np.stack([nx, ny], axis=1)

points_interior, volumes_interior = sample_annulus_interior_structured(60, 80, r_in, r_out)
points_wall, normals_wall = sample_circle_boundary(N_bc_wall, r_in)
points_far, _ = sample_circle_boundary(N_bc_far, r_out)

print("正在生成采样网格预览图...")
plt.figure(figsize=(8, 8))
plt.scatter(points_interior[:, 0], points_interior[:, 1], s=1, c='black', alpha=0.5, label='Interior Points')
plt.scatter(points_far[:, 0], points_far[:, 1], s=10, c='black', label='Far-field Boundary')
plt.scatter(points_wall[:, 0], points_wall[:, 1], s=10, c='black', label='Wall Boundary')
plt.title("Stage 1: Structured Grid Sampling", fontsize=14)
plt.xlabel("x")
plt.ylabel("y")
plt.axis('equal')
plt.xlim(-22, 22)
plt.ylim(-22, 22)
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig("Grid_Sampling_Preview.png", dpi=300) # 补上了第一张图的自动保存
plt.show() 

print("网格预览完毕，开始构建模型...")

x_r = torch.tensor(points_interior[:, 0:1], dtype=torch.float32, requires_grad=True).to(device)
y_r = torch.tensor(points_interior[:, 1:2], dtype=torch.float32, requires_grad=True).to(device)
v_weight = torch.tensor(volumes_interior[:, None], dtype=torch.float32).to(device)
x_wall = torch.tensor(points_wall[:, 0:1], dtype=torch.float32).to(device)
y_wall = torch.tensor(points_wall[:, 1:2], dtype=torch.float32).to(device)
nx_wall = torch.tensor(normals_wall[:, 0:1], dtype=torch.float32).to(device)
ny_wall = torch.tensor(normals_wall[:, 1:2], dtype=torch.float32).to(device)
x_far = torch.tensor(points_far[:, 0:1], dtype=torch.float32).to(device)
y_far = torch.tensor(points_far[:, 1:2], dtype=torch.float32).to(device)

# ==========================================
# 2. 代理模型构建
# ==========================================
class FENN_Model(nn.Module):
    def __init__(self, num_layers=5, hidden_dim=64):
        super(FENN_Model, self).__init__()
        self.first_layer = nn.Linear(6, hidden_dim)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 1)])
        self.output_layer = nn.Linear(hidden_dim, 4)
        self.activation = nn.Tanh()
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x, y):
        r_sq = x**2 + y**2 + 1e-12
        r = torch.sqrt(r_sq)
        D = r - 0.5

        R_sq = 0.25
        u_pf = 1.0 - R_sq * (x**2 - y**2) / (r_sq**2)
        v_pf = - R_sq * (2.0 * x * y) / (r_sq**2)
        V_sq = u_pf**2 + v_pf**2
        p_pf = 4.46428 + 0.5 * 1.0 * (1.0 - V_sq)

        inputs = torch.cat([x/20.0, y/20.0, D/20.0, u_pf, v_pf, p_pf], dim=1)

        out = self.activation(self.first_layer(inputs))
        for layer in self.hidden_layers:
            out = self.activation(layer(out))
        outputs = self.output_layer(out)

        return outputs[:, 0:1] + 1.0, outputs[:, 1:2] + 1.0, outputs[:, 2:3], outputs[:, 3:4] + 1.0

fenn_brain = FENN_Model().to(device)

# ==========================================
# 3. 物理残差计算
# ==========================================
def get_grad(target, coord):
    return torch.autograd.grad(target, coord, grad_outputs=torch.ones_like(target), create_graph=True)[0]

def compute_losses(model, x_r, y_r, v_weight, x_wall, y_wall, nx_wall, ny_wall, x_far, y_far):
    gamma, Ma = 1.4, 0.4
    p_inf = 1.0 / (gamma * Ma**2)

    rho, u, v, T = model(x_r, y_r)
    p = (rho * T) / (gamma * Ma**2)

    F_mass, G_mass = rho * u, rho * v
    F_mom_x, G_mom_x = rho * u**2 + p, rho * u * v
    F_mom_y, G_mom_y = rho * u * v, rho * v**2 + p
    H = (gamma / (gamma - 1.0)) * (p / rho) + 0.5 * (u**2 + v**2)
    F_energy, G_energy = rho * u * H, rho * v * H

    eq_mass   = get_grad(F_mass, x_r)   + get_grad(G_mass, y_r)
    eq_mom_x  = get_grad(F_mom_x, x_r)  + get_grad(G_mom_x, y_r)
    eq_mom_y  = get_grad(F_mom_y, x_r)  + get_grad(G_mom_y, y_r)
    eq_energy = get_grad(F_energy, x_r) + get_grad(G_energy, y_r)

    R_sq_total = eq_mass**2 + eq_mom_x**2 + eq_mom_y**2 + eq_energy**2
    loss_pde = torch.sum(R_sq_total * v_weight) / torch.sum(v_weight)

    rho_f, u_f, v_f, T_f = model(x_far, y_far)
    loss_far = torch.mean((rho_f - 1.0)**2 + (u_f - 1.0)**2 + (v_f - 0.0)**2 + (T_f - 1.0)**2)

    _, u_w, v_w, _ = model(x_wall, y_wall)
    loss_wall = torch.mean((u_w * nx_wall + v_w * ny_wall)**2)

    return loss_pde, loss_far, loss_wall

# ==========================================
# 4. 模型训练
# ==========================================
print("开始模型训练...")
start_time = time.time()

history_total, history_pde, history_far, history_wall = [], [], [], []

epochs_adam = 2000
optimizer_adam = torch.optim.Adam(fenn_brain.parameters(), lr=0.001)

for epoch in range(epochs_adam):
    optimizer_adam.zero_grad()
    loss_pde, loss_far, loss_wall = compute_losses(fenn_brain, x_r, y_r, v_weight, x_wall, y_wall, nx_wall, ny_wall, x_far, y_far)
    total_loss = loss_pde + loss_far + loss_wall
    total_loss.backward()
    optimizer_adam.step()

    history_total.append(total_loss.item())
    history_pde.append(loss_pde.item())
    history_far.append(loss_far.item())
    history_wall.append(loss_wall.item())

    if epoch % 500 == 0:
        print(f"Adam Epoch {epoch:4d} | PDE: {loss_pde.item():.10f} | Far: {loss_far.item():.10f} | Wall: {loss_wall.item():.10f}")

epochs_lbfgs = 400
optimizer_lbfgs = torch.optim.LBFGS(fenn_brain.parameters(), lr=0.1, max_iter=20, tolerance_grad=1e-7, tolerance_change=1e-9, history_size=50)

def closure():
    optimizer_lbfgs.zero_grad()
    loss_pde, loss_far, loss_wall = compute_losses(fenn_brain, x_r, y_r, v_weight, x_wall, y_wall, nx_wall, ny_wall, x_far, y_far)
    total_loss = loss_pde + loss_far + loss_wall
    total_loss.backward()
    return total_loss

for epoch in range(epochs_lbfgs):
    optimizer_lbfgs.step(closure)
    loss_pde, loss_far, loss_wall = compute_losses(fenn_brain, x_r, y_r, v_weight, x_wall, y_wall, nx_wall, ny_wall, x_far, y_far)
    total_loss = loss_pde + loss_far + loss_wall

    history_total.append(total_loss.item())
    history_pde.append(loss_pde.item())
    history_far.append(loss_far.item())
    history_wall.append(loss_wall.item())

    if epoch % 50 == 0:
        print(f"L-BFGS Epoch {epoch:3d} | PDE: {loss_pde.item():.10f} | Far: {loss_far.item():.10f} | Wall: {loss_wall.item():.10f}")

print(f"训练结束，耗时: {(time.time() - start_time)/60:.2f} 分钟")

# ==========================================
# 5. 结果可视化与图表保存
# ==========================================
fenn_brain.eval()

# -----------------------------
# 图 1: 残差收敛历史图 (新增)
# -----------------------------
plt.figure(figsize=(10, 5))
iterations = range(len(history_total))

plt.semilogy(iterations, history_pde, label='PDE Residual', color='tab:blue', linewidth=1.5)
plt.semilogy(iterations, history_far, label='Far-field Residual', color='tab:orange', linewidth=1.5)
plt.semilogy(iterations, history_wall, label='Wall Residual', color='tab:green', linewidth=1.5)
plt.semilogy(iterations, history_total, label='Total Residual', color='black', linewidth=2.5)

plt.title("Residual Convergence History")
plt.xlabel("Iteration")
plt.ylabel("Residual (log scale)")
plt.legend()
plt.grid(True, which="both", linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("Residual_Convergence_History.png", dpi=300)
plt.show()

# -----------------------------
# 图 2: 压力云图生成
# -----------------------------
x = np.linspace(-3, 6, 400)
y = np.linspace(-3, 3, 300)
X, Y = np.meshgrid(x, y)

XY = np.stack([X.flatten(), Y.flatten()], axis=1)
x_t = torch.tensor(XY[:, 0:1], dtype=torch.float32).to(device)
y_t = torch.tensor(XY[:, 1:2], dtype=torch.float32).to(device)

with torch.no_grad():
    rho, u, v, T = fenn_brain(x_t, y_t)
    gamma, Ma = 1.4, 0.4
    p = (rho * T) / (gamma * Ma**2)

P = p.cpu().numpy().reshape(X.shape)
R = np.sqrt(X**2 + Y**2)
mask = R <= 0.5
P[mask] = np.nan

plt.figure(figsize=(10, 5))
contour = plt.contourf(X, Y, P, levels=100, cmap='turbo')
cbar = plt.colorbar(contour)
cbar.set_label('Static Pressure [Pa]')

circle = plt.Circle((0, 0), 0.5, color='white', ec='gray', zorder=10)
plt.gca().add_patch(circle)

plt.title("Pressure Field")
plt.xlabel("X")
plt.ylabel("Y")
plt.axis('equal')
plt.xlim(-3, 6)
plt.ylim(-3, 3)
plt.tight_layout()
plt.savefig("Pressure_Field_Result.png", dpi=300) 
plt.show()

# -----------------------------
# 图 3: 表面 Cp 曲线
# -----------------------------
theta = np.linspace(0, 2*np.pi, 200)
x_c = 0.5 * np.cos(theta)
y_c = 0.5 * np.sin(theta)

x_c_t = torch.tensor(x_c[:, None], dtype=torch.float32).to(device)
y_c_t = torch.tensor(y_c[:, None], dtype=torch.float32).to(device)

with torch.no_grad():
    rho_c, u_c, v_c, T_c = fenn_brain(x_c_t, y_c_t)
    p_c = (rho_c * T_c) / (1.4 * 0.4**2)

p_c = p_c.cpu().numpy().flatten()
p_inf = 1.0 / (1.4 * 0.4**2)
Cp = (p_c - p_inf) / (0.5 * 1.0 * 1.0**2)
Ma = 0.4
beta = np.sqrt(1 - Ma**2)
Cp_theory = (1 - 4 * (np.sin(theta))**2) / beta
plt.figure(figsize=(6, 4))
plt.plot(theta * 180/np.pi, Cp, 'r-', label='GU-FENN')
plt.plot(theta * 180/np.pi, Cp_theory, 'k--', label='Theory')
plt.xlabel("Angle (deg)")
plt.ylabel("Pressure Coefficient Cp")
plt.legend()
plt.grid(True)
plt.title("Surface Cp Comparison")
plt.tight_layout()
plt.savefig("Cp_Curve_Result.png", dpi=300)
plt.show()