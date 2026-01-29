# Portfolio Module

## 1. Lý Thuyết

### 1.1. Mean-Variance Optimization (Markowitz)

**Mục tiêu:**
```
max: w'μ - (λ/2) × w'Σw
```

Trong đó:
- `w`: Vector trọng số
- `μ`: Vector kỳ vọng lợi nhuận
- `Σ`: Ma trận hiệp phương sai
- `λ`: Hệ số risk aversion

**Hạn chế:**
- Quá nhạy với ước lượng μ
- Dễ tập trung vào một vài asset

### 1.2. Black-Litterman Model

**Ý tưởng:** Kết hợp:
1. **Prior:** Market equilibrium (CAPM)
2. **Views:** Dự báo của investor (MoE predictions)

**Công thức Posterior:**
```
E[R] = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹ × [(τΣ)⁻¹π + P'Ω⁻¹Q]
```

Trong đó:
- `π`: Equilibrium returns (CAPM)
- `P`: Pick matrix (views)
- `Q`: View returns (MoE predictions)
- `Ω`: View uncertainty
- `τ`: Scalar (uncertainty in prior)

**Ưu điểm:**
- Ít nhạy với estimation error
- Cho phép đưa views chủ quan
- Diversification tự nhiên

### 1.3. Constraints

| Constraint | Công thức | Lý do |
|------------|-----------|-------|
| Long-only | w ≥ 0 | Không short |
| Max position | w ≤ 5% | Diversification |
| Turnover | Σ|w_new - w_old| ≤ 20% | Giảm cost |
| Full investment | Σw = 1 | 100% invested |

---

## 2. Architecture

```
src/portfolio/
├── __init__.py
└── optimizer.py    # BlackLittermanOptimizer
```

---

## 3. Implementation

### 3.1. BlackLittermanOptimizer (`optimizer.py`)

```python
from src.portfolio.optimizer import BlackLittermanOptimizer

optimizer = BlackLittermanOptimizer(
    returns=returns,
    market_caps=market_caps,
    risk_aversion=2.5,
    tau=0.05
)

# Add MoE views
optimizer.set_views(
    view_matrix=P,
    view_returns=Q,
    confidence=Omega
)

# Optimize
weights = optimizer.optimize(
    max_weight=0.05,
    max_turnover=0.20
)
```

---

## 4. Ví Dụ Output

```python
# Optimized weights
{
    'LITE': 0.05,  # Max weight
    'TEAM': 0.05,
    'DOCU': 0.05,
    'SYF':  0.05,
    'INTU': 0.05,
    ...
    'Other stocks': ~0.01 each
}

# Cluster allocation
{
    'Growth': 0.25,
    'Value': 0.28,
    'Cyclical': 0.30,
    'Defensive': 0.16
}
```

---

## 5. References

- Black, F. & Litterman, R. (1992). *Global Portfolio Optimization*
- Markowitz, H. (1952). *Portfolio Selection*
- Idzorek, T. (2005). *A Step-by-Step Guide to Black-Litterman*
