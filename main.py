"""
Robust-PINN for Elliptic Optimal Control Problems
==================================================
Part 1: Unscaled vs. Scaled (Robust) PINN — demonstrates gradient imbalance fix
Part 2: Augmented Lagrangian + DualNet for pointwise state constraints

Manufactured solution on Ω = [0,1]²:
    y_true(x) = sin(πx₁)sin(πx₂)
    p_true(x) = sin(πx₁)sin(πx₂)

First-order optimality conditions (α-regularized OCP):
    -Δȳ = f + u_d - α⁻¹p̄       (state equation)
    -Δp̄ = ȳ - y_d               (adjoint equation)
    u_d = α⁻¹p̄                  (optimality condition)
"""

import math
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


# ===========================================================================
# Manufactured Solution Helpers
# ===========================================================================
# True solution: y_true = sin(πx₁)sin(πx₂), p_true = sin(πx₁)sin(πx₂)
# Laplacian:  Δ(sin(πx₁)sin(πx₂)) = -2π²·sin(πx₁)sin(πx₂)
#
# From optimality conditions:
#   -Δȳ = f + u_d - α⁻¹p̄  →  2π²y_true = f + u_d - α⁻¹p_true
#   -Δp̄ = ȳ - y_d         →  2π²p_true = y_true - y_d
#   u_d = α⁻¹p_true  (so the u_d - α⁻¹p̄ terms cancel at the true solution)
#
# Choose:  u_d = α⁻¹p_true,  y_d = y_true - 2π²p_true,  f = 2π²y_true

def y_true(x):
    """True state: sin(πx₁)sin(πx₂)"""
    return torch.sin(math.pi * x[:, 0:1]) * torch.sin(math.pi * x[:, 1:2])

def p_true(x):
    """True adjoint: sin(πx₁)sin(πx₂)"""
    return torch.sin(math.pi * x[:, 0:1]) * torch.sin(math.pi * x[:, 1:2])

def f_source(x, alpha):
    """Source term f such that optimality conditions are satisfied."""
    # -Δȳ = f  (the u_d - α⁻¹p̄ terms cancel at the true solution)
    # f = 2π²y_true
    return 2.0 * math.pi**2 * y_true(x)

def y_d_target(x):
    """Desired state y_d = y_true - 2π²p_true"""
    return y_true(x) - 2.0 * math.pi**2 * p_true(x)

def u_d_target(x, alpha):
    """Desired control u_d = α⁻¹p_true"""
    return (1.0 / alpha) * p_true(x)


# ===========================================================================
# Laplacian via Automatic Differentiation
# ===========================================================================
def laplacian(u, x):
    """
    Compute Laplacian Δu = Σ_i ∂²u/∂xᵢ²  using torch.autograd.grad.
    Requires x to have requires_grad=True and u computed from x.
    Both create_graph=True calls are needed for higher-order gradients.
    """
    grad_u = torch.autograd.grad(
        u, x,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True
    )[0]  # shape: (N, 2)

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
# Part 1: PINN Network (Unscaled and Scaled modes)
# ===========================================================================
class PINN(nn.Module):
    """
    4-layer MLP (64 neurons, tanh) that outputs (y, p).
    Supports two modes:
      scaled=False → fits unscaled (ȳ, p̄) directly
      scaled=True  → fits scaled variables (y = α^(1/4)ȳ, p = α^(-1/4)p̄)
    """
    def __init__(self, scaled: bool = False):
        super().__init__()
        self.scaled = scaled
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 2),   # outputs (y, p)
        )

    def forward(self, x):
        return self.net(x)


def pinn_loss(model, x_int, x_bnd, alpha):
    """
    Compute PINN loss for both scaled and unscaled modes.

    Unscaled system (Standard PINN):
        R1: -Δȳ - f - u_d + α⁻¹p̄ = 0
        R2: -Δp̄ - ȳ + y_d = 0

    Scaled system (Robust-PINN):
        substitution: y = α^(1/4)ȳ,  p = α^(-1/4)p̄
        R1: -α^(1/2)Δy + p - α^(3/4)(f + u_d) = 0
        R2: -α^(1/2)Δp - y + α^(1/4)y_d = 0
    """
    # ---- Interior residual ----
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
        # Unscaled: ȳ = y_pred, p̄ = p_pred
        R1 = -lap_y - f_val - ud_val + (1.0 / alpha) * p_pred
        R2 = -lap_p - y_pred + yd_val
    else:
        # Scaled: y = α^(1/4)ȳ, p = α^(-1/4)p̄
        sqrt_a = alpha ** 0.5
        a34 = alpha ** 0.75
        a14 = alpha ** 0.25
        R1 = -sqrt_a * lap_y + p_pred - a34 * (f_val + ud_val)
        R2 = -sqrt_a * lap_p - y_pred + a14 * yd_val

    loss_pde = (R1**2).mean() + (R2**2).mean()

    # ---- Boundary condition loss (y = 0, p = 0 on ∂Ω) ----
    out_bnd = model(x_bnd)
    loss_bc = (out_bnd**2).mean()

    return loss_pde, loss_bc, loss_pde + loss_bc


def sample_interior(n, device):
    """Sample n points uniformly in (0,1)²"""
    return torch.rand(n, 2, device=device)

def sample_boundary(n, device):
    """Sample n points on ∂[0,1]² (all 4 edges)"""
    t = torch.rand(n, device=device)
    zeros = torch.zeros(n, device=device)
    ones = torch.ones(n, device=device)
    # Each edge: n//4 points
    m = n // 4
    edge0 = torch.stack([t[:m], zeros[:m]], dim=1)       # bottom
    edge1 = torch.stack([t[m:2*m], ones[m:2*m]], dim=1)  # top
    edge2 = torch.stack([zeros[2*m:3*m], t[2*m:3*m]], dim=1)  # left
    edge3 = torch.stack([ones[3*m:], t[3*m:]], dim=1)    # right
    return torch.cat([edge0, edge1, edge2, edge3], dim=0)


# ===========================================================================
# Part 1 Training
# ===========================================================================
def train_part1(alpha=1e-4, n_steps=10000, n_int=1024, n_bnd=256, lr=1e-3):
    """
    Train both unscaled and scaled PINNs on the same problem.
    Returns loss histories for comparison.
    """
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
        # --- Unscaled update ---
        opt_u.zero_grad()
        loss_pde_u, loss_bc_u, loss_u = pinn_loss(model_unscaled, x_int, x_bnd, alpha)
        loss_u.backward()
        opt_u.step()

        # --- Scaled update ---
        opt_s.zero_grad()
        loss_pde_s, loss_bc_s, loss_s = pinn_loss(model_scaled, x_int, x_bnd, alpha)
        loss_s.backward()
        opt_s.step()

        hist_u.append(loss_u.item())
        hist_s.append(loss_s.item())

        if step % 500 == 0:
            print(f"  Step {step:5d} | "
                  f"Unscaled: PDE={loss_pde_u.item():.4e}  BC={loss_bc_u.item():.4e}  "
                  f"Total={loss_u.item():.4e} | "
                  f"Scaled: PDE={loss_pde_s.item():.4e}  BC={loss_bc_s.item():.4e}  "
                  f"Total={loss_s.item():.4e}")

    return model_unscaled, model_scaled, hist_u, hist_s


# ===========================================================================
# Part 2: PrimalNet
# ===========================================================================
class PrimalNet(nn.Module):
    """
    Approximates (y(x), u(x)) for the state-constrained OCP.
    4-layer MLP with tanh activation.
    Boundary conditions y=0 enforced via multiplication by x₁(1-x₁)x₂(1-x₂).
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 2),  # raw outputs (y_raw, u_raw)
        )

    def forward(self, x):
        out = self.net(x)
        # Enforce y=0 on boundary via smooth masking function
        mask = x[:, 0:1] * (1.0 - x[:, 0:1]) * x[:, 1:2] * (1.0 - x[:, 1:2])
        y = mask * out[:, 0:1]
        u = out[:, 1:2]
        return y, u


# ===========================================================================
# Part 2: DualNet (Fourier Features + Softplus)
# ===========================================================================
class DualNet(nn.Module):
    """
    Approximates Lagrange multiplier μ(x) ≥ 0 for constraint y ≤ y_max.

    Architecture:
      Input x ∈ R²
      → Fourier Feature layer: [sin(Bx), cos(Bx)] ∈ R^(2*n_fourier)
        (overcomes spectral bias, captures sharp transitions at constraint boundary)
      → 3 hidden layers (tanh activation)
      → Softplus output (strictly enforces μ(x) ≥ 0)
    """
    def __init__(self, n_fourier: int = 32, hidden: int = 64):
        super().__init__()
        # Fixed random Fourier frequency matrix B ∈ R^(n_fourier × 2)
        # sampled from N(0, σ²); σ controls frequency bandwidth
        sigma = 5.0
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
        # Fourier feature embedding
        proj = x @ self.B.T       # (N, n_fourier)
        feat = torch.cat([torch.sin(proj), torch.cos(proj)], dim=1)  # (N, 2*n_fourier)
        raw = self.mlp(feat)       # (N, 1)
        return self.softplus(raw)  # μ(x) ≥ 0


# ===========================================================================
# Part 2: Augmented Lagrangian Loss
# ===========================================================================
def augmented_lagrangian_loss(primal_net, dual_net, x_int, alpha, y_max, rho):
    """
    Total loss for the state-constrained OCP using Augmented Lagrangian:

        L = L_PDE + L_BC + L_constraint

    L_PDE enforces the optimality system (scaled mode for stability):
        -α^(1/2)Δy + p - α^(3/4)(f + u_d) = 0
        -α^(1/2)Δp - y + α^(1/4)y_d = 0
    where p ≈ α^(-1/4)·(α·u) = α^(3/4)u (since optimality cond: u = α⁻¹p̄)

    L_constraint = (1/N) Σ [ μ(x)·ReLU(y-y_max) + (ρ/2)·ReLU(y-y_max)² ]

    Note: PrimalNet outputs (y, u) directly; adjoint p̄ = α·u.
    """
    x_int = x_int.requires_grad_(True)
    y_pred, u_pred = primal_net(x_int)

    # Adjoint from optimality: p̄ = αu  (optimality condition)
    p_bar = alpha * u_pred

    lap_y = laplacian(y_pred, x_int)
    lap_p = laplacian(p_bar,  x_int)

    f_val  = f_source(x_int, alpha)
    yd_val = y_d_target(x_int)
    ud_val = u_d_target(x_int, alpha)

    # Scaled residuals
    sqrt_a = alpha ** 0.5
    a34    = alpha ** 0.75
    a14    = alpha ** 0.25

    # Scale primal/dual variables as in Part 1
    y_sc = (alpha ** 0.25) * y_pred   # y = α^(1/4) ȳ
    p_sc = (alpha ** -0.25) * p_bar   # p = α^(-1/4) p̄

    lap_ysc = laplacian(y_sc, x_int)
    lap_psc = laplacian(p_sc, x_int)

    R1 = -sqrt_a * lap_ysc + p_sc - a34 * (f_val + ud_val)
    R2 = -sqrt_a * lap_psc - y_sc + a14 * yd_val

    loss_pde = (R1**2).mean() + (R2**2).mean()

    # Boundary condition on interior points is handled by PrimalNet mask;
    # additional boundary samples for p
    loss_bc = torch.tensor(0.0, device=x_int.device)

    # Constraint loss: L_c = mean[ μ·ReLU(y-y_max) + (ρ/2)·ReLU(y-y_max)² ]
    violation = torch.relu(y_pred - y_max)          # (N,1), ≥0
    mu = dual_net(x_int.detach())                   # (N,1), ≥0  (detach for primal update)
    loss_constraint = (mu.detach() * violation + 0.5 * rho * violation**2).mean()

    loss_total = loss_pde + loss_bc + loss_constraint
    return loss_pde, loss_constraint, loss_total


def dual_loss(primal_net, dual_net, x_int, y_max, rho):
    """
    Dual update: maximize constraint loss w.r.t. DualNet parameters.
    Returns negative of constraint term (for gradient ascent via .backward()).

    L_dual = mean[ μ(x)·ReLU(y-y_max) ]   (ascent: dual increases where violated)
    Note: ρ term does not depend on μ, so only the linear μ·violation term matters.
    """
    with torch.no_grad():
        y_pred, _ = primal_net(x_int)
        violation = torch.relu(y_pred - y_max)

    mu = dual_net(x_int)
    # Maximise: want to ascend, so pass -loss to backward()
    loss_dual = -(mu * violation).mean()
    return loss_dual


# ===========================================================================
# Part 2 Training: Warm-start + TTUR Alternating Optimization
# ===========================================================================
def train_part2(alpha=1e-4, n_epochs=3000, n_warmup=500,
                n_int=1024, y_max=0.5,
                rho_init=0.01, rho_final=10.0,
                K=5, lr_primal=1e-3, lr_dual=1e-4):
    """
    Train PrimalNet + DualNet using Augmented Lagrangian + TTUR.

    Schedule:
      Epochs 0..n_warmup-1:   DualNet frozen, ρ=rho_init  (PDE warm-start)
      Epochs n_warmup..end:   Both nets active, ρ ramps up  (min-max game)
    """
    print("\n" + "="*60)
    print(f"Part 2: Augmented Lagrangian (y_max={y_max}, α={alpha})")
    print("="*60)

    primal_net = PrimalNet().to(device)
    dual_net   = DualNet().to(device)

    opt_primal = torch.optim.Adam(primal_net.parameters(), lr=lr_primal)
    opt_dual   = torch.optim.Adam(dual_net.parameters(),   lr=lr_dual)

    hist_pde, hist_constraint, hist_total = [], [], []
    hist_max_violation = []

    for epoch in range(1, n_epochs + 1):
        # Dynamic resampling: new collocation points each epoch
        x_int = sample_interior(n_int, device)

        # Ramp ρ from rho_init to rho_final after warm-start
        if epoch <= n_warmup:
            rho = rho_init
            dual_frozen = True
        else:
            frac = (epoch - n_warmup) / max(1, n_epochs - n_warmup)
            rho = rho_init + (rho_final - rho_init) * frac
            dual_frozen = False

        # ---- Step 1: Update PrimalNet (K gradient steps) ----
        for _ in range(K):
            opt_primal.zero_grad()
            loss_pde, loss_c, loss_total = augmented_lagrangian_loss(
                primal_net, dual_net, x_int, alpha, y_max, rho)
            loss_total.backward()
            opt_primal.step()

        # ---- Step 2: Update DualNet (1 gradient ascent step) ----
        if not dual_frozen:
            opt_dual.zero_grad()
            loss_d = dual_loss(primal_net, dual_net, x_int, y_max, rho)
            loss_d.backward()
            opt_dual.step()

        # Logging
        hist_pde.append(loss_pde.item())
        hist_constraint.append(loss_c.item())
        hist_total.append(loss_total.item())

        with torch.no_grad():
            y_eval, _ = primal_net(x_int)
            max_viol = torch.relu(y_eval - y_max).max().item()
        hist_max_violation.append(max_viol)

        if epoch % 200 == 0:
            status = "WARM-START" if dual_frozen else f"ρ={rho:.3f}"
            print(f"  Epoch {epoch:4d} [{status:12s}] | "
                  f"PDE={loss_pde.item():.4e}  "
                  f"Constraint={loss_c.item():.4e}  "
                  f"MaxViol={max_viol:.4e}")

        # Memory management: free GPU cache periodically
        if epoch % 100 == 0:
            if torch.cuda.is_available():
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
    """Generate Part 1 plots: loss comparison and solution heatmaps."""

    # --- Loss comparison ---
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
    plt.savefig("part1_loss_comparison.png", dpi=150)
    plt.close()
    print("Saved: part1_loss_comparison.png")

    # --- Solution heatmaps (scaled model) ---
    n_grid = 64
    xs = torch.linspace(0, 1, n_grid)
    xg, yg = torch.meshgrid(xs, xs, indexing="ij")
    x_grid = torch.stack([xg.reshape(-1), yg.reshape(-1)], dim=1).to(device)

    with torch.no_grad():
        out_s = model_scaled(x_grid)
        y_sc_pred = out_s[:, 0].reshape(n_grid, n_grid).cpu().numpy()
        p_sc_pred = out_s[:, 1].reshape(n_grid, n_grid).cpu().numpy()
        # Unscale: ȳ = α^(-1/4) y_sc
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
    plt.savefig("part1_solution.png", dpi=150)
    plt.close()
    print("Saved: part1_solution.png")


def plot_part2(hist, primal_net, y_max):
    """Generate Part 2 plots: loss curve and constraint satisfaction."""

    # --- Training loss ---
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
    plt.savefig("part2_loss.png", dpi=150)
    plt.close()
    print("Saved: part2_loss.png")

    # --- Constraint satisfaction: y distribution ---
    n_grid = 64
    xs = torch.linspace(0, 1, n_grid)
    xg, yg = torch.meshgrid(xs, xs, indexing="ij")
    x_grid = torch.stack([xg.reshape(-1), yg.reshape(-1)], dim=1).to(device)

    with torch.no_grad():
        y_pred, _ = primal_net(x_grid)
        y_pred_np = y_pred[:, 0].reshape(n_grid, n_grid).cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Heatmap of y with y_max contour
    ax = axes[0]
    im = ax.imshow(y_pred_np, origin="lower", extent=[0,1,0,1],
                   cmap="RdYlBu_r", vmin=0, vmax=max(y_max*1.2, y_pred_np.max()))
    ax.contour(np.linspace(0,1,n_grid), np.linspace(0,1,n_grid),
               y_pred_np.T, levels=[y_max], colors="red", linewidths=2)
    ax.set_title(f"Predicted y(x)  [red line = y_max={y_max}]")
    ax.set_xlabel("x₁"); ax.set_ylabel("x₂")
    plt.colorbar(im, ax=ax)

    # Max violation over training
    ax2 = axes[1]
    ax2.semilogy(epochs, hist["max_violation"], color="crimson", label="Max violation")
    ax2.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("max ReLU(y - y_max)")
    ax2.set_title("Part 2: Constraint Violation Over Training")
    ax2.legend()
    ax2.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.savefig("part2_constraint.png", dpi=150)
    plt.close()
    print("Saved: part2_constraint.png")


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
        n_epochs=3000,
        n_warmup=500,
        y_max=0.5,
        rho_init=0.01,
        rho_final=10.0,
        K=5,
    )
    plot_part2(hist2, primal_net, y_max=0.5)

    print("\nDone. Generated files:")
    print("  part1_loss_comparison.png")
    print("  part1_solution.png")
    print("  part2_loss.png")
    print("  part2_constraint.png")


if __name__ == "__main__":
    main()
