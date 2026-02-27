# 工作日志

## 2026-02-27: 优化 Primal-Dual PINN 以处理高频状态约束

### 问题背景
`main_ver.2.py` 中的 tanh MLP 网络由于频谱偏差(spectral bias)，无法表示高频场 sin(2πx₁)sin(2πx₂)，预测峰值仅约 0.6（真实值为 1.0）。约束 y≤0.5 看似部分满足，实际上是因为网络表达能力不足。

### 实施内容

#### 1. 制造解更新为 2π 频率
- `y_true = sin(2πx₁)sin(2πx₂)`，`p_true` 同理
- `f_source = 8π² * y_true`，`y_d = y_true - 8π²p_true`

#### 2. 傅里叶特征嵌入（关键修复）
- 新增 `FourierEmbed` 模块：随机傅里叶特征 `[sin(Bx), cos(Bx)]`
- 参数：`n_fourier=64, sigma=4.0`，输出维度 128
- 应用于 `PINN`（Part 1）和 `PrimalNet`（Part 2），克服频谱偏差

#### 3. Optimistic Adam 优化器
- 自定义 `OptimisticAdam` 类，继承 `torch.optim.Adam`
- 梯度校正：`g = 2*grad_t - grad_{t-1}`，稳定鞍点动态

#### 4. DualNet 扩展
- `n_fourier=64, sigma=10.0, hidden=128`

#### 5. 自适应惩罚参数 ρ
- 基于 EMA 违反量自动调节：违反>0.01 时 ρ×1.05（上限100），<0.001 时 ρ×0.99（下限0.1）

#### 6. TTUR 重平衡
- `K=2`（原为5），`lr_primal=5e-4, lr_dual=5e-4`

#### 7. 梯度裁剪
- 对两个网络均使用 `clip_grad_norm_(params, max_norm=1.0)`

#### 8. 训练计划调整
- `n_epochs=6000`（原3000），`n_warmup=1500`（原500），`n_int=2048`（原1024）
- 预热后对 primal 使用余弦退火学习率调度

#### 9. EMA 平滑 + 对偶熵正则化
- EMA（beta=0.95）平滑违反量，用于自适应 ρ 和监控
- 对偶损失中增加熵项 `1e-4 * (μ * log(μ+ε)).mean()`

### 验证方法
1. 运行 `python main_ver.2.py`
2. Part 1：缩放 PINN 应达到峰值≈1.0
3. Part 2 热力图：无约束区域 y 应达到 1.0，约束激活区域平坦于 0.5
4. 违反曲线：平滑收敛至 <1e-4
