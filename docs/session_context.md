# Session Context: Robust-PINN for Optimal Control Problem

## README Analysis

### Problem
Implement a PINN to solve an elliptic optimal control problem (OCP) with PDE constraints.

### Part 1: Unscaled vs. Scaled PINN
- **Domain**: Ω = [0,1]², manufactured solution y_true = sin(πx₁)sin(πx₂)
- **Unscaled (Standard PINN)**: Direct OCP optimality system, suffers gradient imbalance for small α
- **Scaled (Robust-PINN)**: Variable substitution p = α^(-1/4)p̄, y = α^(1/4)ȳ balances the system
- **Key parameter**: α = 1e-4 (small → demonstrates failure of unscaled)

### Part 2 Appendix: DualNet for Pointwise State Constraints
- Constraint: y(x) ≤ y_max
- **PrimalNet**: approximates (y, u)
- **DualNet**: approximates Lagrange multiplier μ(x) ≥ 0
  - Fourier Feature first layer (overcomes spectral bias at constraint boundaries)
  - Softplus final activation (enforces μ ≥ 0)
- **Augmented Lagrangian loss**: L_c = mean[μ·ReLU(y-y_max) + (ρ/2)·ReLU(y-y_max)²]
- **TTUR**: PrimalNet lr=1e-3, DualNet lr=1e-4
- **Alternating**: K=5 primal steps, then 1 dual step (gradient ascent)
- **Warm-start**: First 500 epochs freeze DualNet, ρ=0.01
- **Dynamic resampling**: new collocation points each epoch

## Implementation Plan

### File: main.py (single file)

1. Device detection: cuda → mps → cpu
2. Manufactured solution helpers (y_true, p_true, y_d, u_d, f)
3. `laplacian(u, x)` via torch.autograd.grad twice
4. `PINN(nn.Module)`: 4×64 tanh, output (y,p), scaled/unscaled toggle
5. Part 1 training: α=1e-4, 10000 steps, log every 500
6. `PrimalNet`: output (y,u)
7. `DualNet`: Fourier features + tanh/sine + Softplus
8. Part 2 training: warm-start 500 + TTUR alternating
9. Visualization: 4 PNG files

## Key Implementation Notes

- Linux system (no MPS), use cuda > cpu
- Replace torch.mps.empty_cache() with torch.cuda.empty_cache()
- Math comments for all PDE formulas
- Laplacian must use create_graph=True for second derivatives
