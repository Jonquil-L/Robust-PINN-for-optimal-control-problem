"""
Robust-PINN for Elliptic Optimal Control Problems
==================================================
Part 1: Unscaled vs. Scaled (Robust) PINN — demonstrates gradient imbalance fix
Part 2: Augmented Lagrangian + DualNet for pointwise state constraints

Manufactured solution on Ω = [0,1]²:
    y_true(x) = sin(2πx₁)sin(2πx₂)
    p_true(x) = sin(2πx₁)sin(2πx₂)

First-order optimality conditions (α-regularized OCP):
    -Δȳ = f + u_d - α⁻¹p̄       (state equation)
    -Δp̄ = ȳ - y_d               (adjoint equation)
    u_d = α⁻¹p̄                  (optimality condition)
"""

import math
import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Device setup: cuda → mps → cpu
# ---------------------------------------------------------------------------
if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

FREQ = 2.0 * math.pi  # 2π frequency

def _ts():
    """Return timestamp string for unique filenames."""
    return time.strftime("%Y%m%d_%H%M%S")

# ===========================================================================
# Manufactured Solution Helpers (2π frequency)
# ===========================================================================
def y_true(x):
    """True state: sin(2πx₁)sin(2πx₂)"""
    return torch.sin(FREQ * x[:, 0:1]) * torch.sin(FREQ * x[:, 1:2])

def p_true(x):
    """True adjoint: sin(2πx₁)sin(2πx₂)"""
    return torch.sin(FREQ * x[:, 0:1]) * torch.sin(FREQ * x[:, 1:2])

def f_source(x, alpha):
    """Source term f: -Δy_true = 2*(2π)² * y_true, so f = 8π² * y_true"""
    return 2.0 * FREQ**2 * y_true(x)

def y_d_target(x):
    """Desired state y_d = y_true - 8π²p_true"""
    return y_true(x) - 2.0 * FREQ**2 * p_true(x)

def u_d_target(x, alpha):
    """Desired control u_d = α⁻¹p_true"""
    return (1.0 / alpha) * p_true(x)


# ===========================================================================
# Laplacian via Automatic Differentiation
# ===========================================================================
def laplacian(u, x):
    grad_u = torch.autograd.grad(
        u, x,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True
    )[0]

    lap = torch.zeros_like(u)
    for i in range(x.shape[1]):
        grad_ui = grad_u[:, i:i+1]
        lap_i = torch.autograd.grad(
            grad_ui, x,
            grad_outputs=torch.ones_like(grad_ui),
            create_graph=True,
            retain_graph=True
        )[0][:, i:i+1]
        lap = lap + lap_i
    return lap


# ===========================================================================
# Fourier Feature Embedding
# ===========================================================================
class FourierEmbed(nn.Module):
    def __init__(self, in_dim=2, n_fourier=64, sigma=4.0):
        super().__init__()
        B = torch.randn(n_fourier, in_dim) * sigma
        self.register_buffer("B", B)

    def forward(self, x):
        proj = x @ self.B.T
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=1)


# ===========================================================================
# Optimistic Adam Optimizer
# ===========================================================================
class OptimisticAdam(torch.optim.Adam):
    """Adam with optimistic gradient correction: g = 2*grad_t - grad_{t-1}"""
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if 'prev_grad' not in state:
                    state['prev_grad'] = p.grad.data.clone()
                else:
                    prev = state['prev_grad']
                    p.grad.data = 2.0 * p.grad.data - prev
                    state['prev_grad'] = p.grad.data.clone()
        return super().step(closure)


# ===========================================================================
# Part 1: PINN with Fourier Features
# ===========================================================================
class PINN(nn.Module):
    def __init__(self, scaled: bool = False, n_fourier=64, sigma=4.0, hidden=128):
        super().__init__()
        self.scaled = scaled
        self.embed = FourierEmbed(2, n_fourier, sigma)
        in_dim = 2 * n_fourier
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x):
        feat = self.embed(x)
        return self.net(feat)

def pinn_loss(model, x_int, x_bnd, alpha):
    x_int = x_int.requires_grad_(True)
    out = model(x_int)
    y_pred = out[:, 0:1]
    p_pred = out[:, 1:2]

    lap_y = laplacian(y_pred, x_int)
    lap_p = laplacian(p_pred, x_int)

    f_val = f_source(x_int, alpha)
    yd_val = y_d_target(x_int)
    ud_val = u_d_target(x_int, alpha)

    if not model.scaled:
        R1 = -lap_y - f_val - ud_val + (1.0 / alpha) * p_pred
        R2 = -lap_p - y_pred + yd_val
    else:
        sqrt_a = alpha ** 0.5
        a34 = alpha ** 0.75
        a14 = alpha ** 0.25
        R1 = -sqrt_a * lap_y + p_pred - a34 * (f_val + ud_val)
        R2 = -sqrt_a * lap_p - y_pred + a14 * yd_val

    loss_pde = (R1**2).mean() + (R2**2).mean()

    out_bnd = model(x_bnd)
    loss_bc = (out_bnd**2).mean()

    return loss_pde, loss_bc, loss_pde + loss_bc

def sample_interior(n, device):
    return torch.rand(n, 2, device=device)

def sample_boundary(n, device):
    t = torch.rand(n, device=device)
    zeros = torch.zeros(n, device=device)
    ones = torch.ones(n, device=device)
    m = n // 4
    edge0 = torch.stack([t[:m], zeros[:m]], dim=1)
    edge1 = torch.stack([t[m:2*m], ones[m:2*m]], dim=1)
    edge2 = torch.stack([zeros[2*m:3*m], t[2*m:3*m]], dim=1)
    edge3 = torch.stack([ones[3*m:], t[3*m:]], dim=1)
    return torch.cat([edge0, edge1, edge2, edge3], dim=0)

def train_part1(alpha=1e-4, n_steps=10000, n_int=2048, n_bnd=256, lr=1e-3):
    print("\n" + "="*60)
    print(f"Part 1: Unscaled vs. Scaled PINN  (α = {alpha})")
    print("="*60)

    model_unscaled = PINN(scaled=False).to(device)
    model_scaled   = PINN(scaled=True).to(device)
    opt_u = torch.optim.Adam(model_unscaled.parameters(), lr=lr)
    opt_s = torch.optim.Adam(model_scaled.parameters(),   lr=lr)

    x_int = sample_interior(n_int, device)
    x_bnd = sample_boundary(n_bnd, device)

    hist_u, hist_s = [], []

    for step in range(1, n_steps + 1):
        opt_u.zero_grad()
        loss_pde_u, loss_bc_u, loss_u = pinn_loss(model_unscaled, x_int, x_bnd, alpha)
        loss_u.backward()
        opt_u.step()

        opt_s.zero_grad()
        loss_pde_s, loss_bc_s, loss_s = pinn_loss(model_scaled, x_int, x_bnd, alpha)
        loss_s.backward()
        opt_s.step()

        hist_u.append(loss_u.item())
        hist_s.append(loss_s.item())

        if step % 500 == 0:
            print(f"  Step {step:5d} | "
                  f"Unscaled Total={loss_u.item():.4e} | "
                  f"Scaled Total={loss_s.item():.4e}")

    return model_unscaled, model_scaled, hist_u, hist_s


# ===========================================================================
# Part 2: PrimalNet with Fourier Features
# ===========================================================================
class PrimalNet(nn.Module):
    def __init__(self, n_fourier=64, sigma=4.0, hidden=128):
        super().__init__()
        self.embed = FourierEmbed(2, n_fourier, sigma)
        in_dim = 2 * n_fourier
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x):
        feat = self.embed(x)
        out = self.net(feat)
        mask = x[:, 0:1] * (1.0 - x[:, 0:1]) * x[:, 1:2] * (1.0 - x[:, 1:2])
        y_sc = mask * out[:, 0:1]
        p_sc = out[:, 1:2]
        return y_sc, p_sc


# ===========================================================================
# Part 2: DualNet (Fourier Features + Softplus)
# ===========================================================================
class DualNet(nn.Module):
    def __init__(self, n_fourier: int = 64, hidden: int = 128):
        super().__init__()
        sigma = 10.0
        B = torch.randn(n_fourier, 2) * sigma
        self.register_buffer("B", B)

        in_dim = 2 * n_fourier
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self.softplus = nn.Softplus()

    def forward(self, x):
        proj = x @ self.B.T
        feat = torch.cat([torch.sin(proj), torch.cos(proj)], dim=1)
        raw = self.mlp(feat)
        return self.softplus(raw)


# ===========================================================================
# Part 2: Augmented Lagrangian Loss
# ===========================================================================

def augmented_lagrangian_loss(primal_net, dual_net, x_int, alpha, y_max, rho):
    x_int = x_int.requires_grad_(True)

    y_sc, p_sc = primal_net(x_int)

    lap_ysc = laplacian(y_sc, x_int)
    lap_psc = laplacian(p_sc, x_int)

    f_val  = f_source(x_int, alpha)
    yd_val = y_d_target(x_int)
    ud_val = u_d_target(x_int, alpha)

    sqrt_a = alpha ** 0.5
    a34    = alpha ** 0.75
    a14    = alpha ** 0.25

    R1 = -sqrt_a * lap_ysc + p_sc - a34 * (f_val + ud_val)
    R2 = -sqrt_a * lap_psc - y_sc + a14 * yd_val

    loss_pde = (R1**2).mean() + (R2**2).mean()

    loss_bc = torch.tensor(0.0, device=x_int.device)

    y_pred = y_sc / a14

    violation = torch.relu(y_pred - y_max)
    mu = dual_net(x_int.detach())
    loss_constraint = (mu.detach() * violation + 0.5 * rho * violation**2).mean()

    loss_total = loss_pde + loss_bc + loss_constraint
    return loss_pde, loss_constraint, loss_total


def dual_loss(primal_net, dual_net, x_int, alpha, y_max, rho):
    with torch.no_grad():
        y_sc, _ = primal_net(x_int)
        y_pred = y_sc / (alpha ** 0.25)
        violation = torch.relu(y_pred - y_max)

    mu = dual_net(x_int)
    # Gradient ascent + entropy regularization
    entropy = 1e-4 * (mu * torch.log(mu + 1e-8)).mean()
    loss_dual = -(mu * violation).mean() + entropy
    return loss_dual


# ===========================================================================
# Part 2 Training: Warm-start + TTUR + Optimistic Adam + Adaptive rho
# ===========================================================================
def train_part2(alpha=1e-4, n_epochs=6000, n_warmup=1500,
                n_int=2048, y_max=0.5,
                rho=1.0,
                K=2, lr_primal=5e-4, lr_dual=5e-4):

    print("\n" + "="*60)
    print(f"Part 2: Augmented Lagrangian (y_max={y_max}, α={alpha})")
    print("="*60)

    primal_net = PrimalNet().to(device)
    dual_net   = DualNet().to(device)

    opt_primal = OptimisticAdam(primal_net.parameters(), lr=lr_primal)
    opt_dual   = OptimisticAdam(dual_net.parameters(),   lr=lr_dual)

    # Cosine annealing for primal after warmup
    scheduler_primal = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_primal, T_max=n_epochs - n_warmup, eta_min=1e-5)

    hist_pde, hist_constraint, hist_total = [], [], []
    hist_max_violation = []

    ema_violation = 0.0
    ema_beta = 0.95

    for epoch in range(1, n_epochs + 1):
        x_int = sample_interior(n_int, device)

        dual_frozen = (epoch <= n_warmup)

        # ---- Step 1: Update PrimalNet (K steps) ----
        for _ in range(K):
            opt_primal.zero_grad()
            loss_pde, loss_c, loss_total = augmented_lagrangian_loss(
                primal_net, dual_net, x_int, alpha, y_max, rho)
            loss_total.backward()
            nn.utils.clip_grad_norm_(primal_net.parameters(), max_norm=1.0)
            opt_primal.step()

        # ---- Step 2: Update DualNet ----
        if not dual_frozen:
            opt_dual.zero_grad()
            loss_d = dual_loss(primal_net, dual_net, x_int, alpha, y_max, rho)
            loss_d.backward()
            nn.utils.clip_grad_norm_(dual_net.parameters(), max_norm=1.0)
            opt_dual.step()

        # Cosine annealing after warmup
        if epoch > n_warmup:
            scheduler_primal.step()

        # Logging
        hist_pde.append(loss_pde.item())
        hist_constraint.append(loss_c.item())
        hist_total.append(loss_total.item())

        with torch.no_grad():
            y_sc_eval, _ = primal_net(x_int)
            y_eval = y_sc_eval / (alpha ** 0.25)
            max_viol = torch.relu(y_eval - y_max).max().item()
        hist_max_violation.append(max_viol)

        # EMA violation tracking + adaptive rho
        ema_violation = ema_beta * ema_violation + (1 - ema_beta) * max_viol
        if not dual_frozen:
            if ema_violation > 0.01:
                rho = min(rho * 1.05, 100.0)
            elif ema_violation < 0.001:
                rho = max(rho * 0.99, 0.1)

        if epoch % 200 == 0:
            status = "WARM-UP" if dual_frozen else f"ρ={rho:.2f}"
            print(f"  Epoch {epoch:4d} [{status:12s}] | "
                  f"PDE={loss_pde.item():.4e}  "
                  f"Constr={loss_c.item():.4e}  "
                  f"MaxViol={max_viol:.4e}  "
                  f"EMA={ema_violation:.4e}")

        # Memory management
        if epoch % 100 == 0:
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()

    return primal_net, dual_net, {
        "pde": hist_pde,
        "constraint": hist_constraint,
        "total": hist_total,
        "max_violation": hist_max_violation,
    }

# ===========================================================================
# Visualization Helpers
# ===========================================================================
def plot_part1(hist_u, hist_s, model_unscaled, model_scaled, alpha):
    fig, ax = plt.subplots(figsize=(8, 5))
    steps = np.arange(1, len(hist_u) + 1)
    ax.semilogy(steps, hist_u, label="Unscaled (Standard PINN)", color="tomato", alpha=0.8)
    ax.semilogy(steps, hist_s, label="Scaled (Robust-PINN)", color="steelblue", alpha=0.8)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Total Loss (log scale)")
    ax.set_title(f"Part 1: Loss Convergence Comparison  (α = {alpha})")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    fname = f"part1_loss_comparison_{_ts()}.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"Saved: {fname}")

    n_grid = 64
    xs = torch.linspace(0, 1, n_grid)
    xg, yg = torch.meshgrid(xs, xs, indexing="ij")
    x_grid = torch.stack([xg.reshape(-1), yg.reshape(-1)], dim=1).to(device)

    with torch.no_grad():
        out_s = model_scaled(x_grid)
        y_sc_pred = out_s[:, 0].reshape(n_grid, n_grid).cpu().numpy()
        y_pred_np = (alpha ** -0.25) * y_sc_pred
        y_ref_np = y_true(x_grid.cpu())[:, 0].reshape(n_grid, n_grid).numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    titles = ["Predicted ȳ (Scaled PINN)", "True y", "Error |ȳ - y_true|"]
    data   = [y_pred_np, y_ref_np, np.abs(y_pred_np - y_ref_np)]
    for ax, d, t in zip(axes, data, titles):
        im = ax.imshow(d, origin="lower", extent=[0,1,0,1], cmap="viridis")
        ax.set_title(t)
        ax.set_xlabel("x₁"); ax.set_ylabel("x₂")
        plt.colorbar(im, ax=ax)
    plt.suptitle(f"Part 1 Solution (α = {alpha})")
    plt.tight_layout()
    fname = f"part1_solution_{_ts()}.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"Saved: {fname}")


def plot_part2(hist, primal_net, alpha, y_max):
    epochs = np.arange(1, len(hist["total"]) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(epochs, hist["pde"],        label="PDE Loss",        alpha=0.8)
    ax.semilogy(epochs, hist["constraint"], label="Constraint Loss",  alpha=0.8)
    ax.semilogy(epochs, hist["total"],      label="Total Loss",       alpha=0.8, lw=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (log scale)")
    ax.set_title("Part 2: Augmented Lagrangian Training Loss")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    fname = f"part2_loss_{_ts()}.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"Saved: {fname}")

    n_grid = 64
    xs = torch.linspace(0, 1, n_grid)
    xg, yg = torch.meshgrid(xs, xs, indexing="ij")
    x_grid = torch.stack([xg.reshape(-1), yg.reshape(-1)], dim=1).to(device)

    with torch.no_grad():
        y_sc_pred, _ = primal_net(x_grid)
        y_pred_np = (y_sc_pred / (alpha ** 0.25))[:, 0].reshape(n_grid, n_grid).cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    im = ax.imshow(y_pred_np, origin="lower", extent=[0,1,0,1],
                   cmap="RdYlBu_r", vmin=0, vmax=max(y_max*1.2, y_pred_np.max()))
    ax.contour(np.linspace(0,1,n_grid), np.linspace(0,1,n_grid),
               y_pred_np.T, levels=[y_max], colors="red", linewidths=2)
    ax.set_title(f"Predicted y(x)  [red line = y_max={y_max}]")
    ax.set_xlabel("x₁"); ax.set_ylabel("x₂")
    plt.colorbar(im, ax=ax)

    ax2 = axes[1]
    ax2.semilogy(epochs, hist["max_violation"], color="crimson", label="Max violation")
    ax2.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("max ReLU(y - y_max)")
    ax2.set_title("Part 2: Constraint Violation Over Training")
    ax2.legend()
    ax2.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    fname = f"part2_constraint_{_ts()}.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"Saved: {fname}")


# ===========================================================================
# Main
# ===========================================================================
def main():
    torch.manual_seed(42)
    np.random.seed(42)

    ALPHA = 1e-4

    # ---- Part 1 ----
    model_u, model_s, hist_u, hist_s = train_part1(alpha=ALPHA, n_steps=10000)
    plot_part1(hist_u, hist_s, model_u, model_s, ALPHA)

    # ---- Part 2 ----
    primal_net, dual_net, hist2 = train_part2(
        alpha=ALPHA,
        n_epochs=6000,
        n_warmup=1500,
        y_max=0.5,
        rho=1.0,
        K=2,
        lr_primal=5e-4,
        lr_dual=5e-4,
    )
    plot_part2(hist2, primal_net, alpha=ALPHA, y_max=0.5)

    print("\nDone. All output images saved with timestamps.")

if __name__ == "__main__":
    main()
