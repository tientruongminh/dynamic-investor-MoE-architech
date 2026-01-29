# Factors Module

## 1. Lý Thuyết

### 1.1. Alpha Factors

**Định nghĩa:** Tín hiệu dự báo lợi nhuận tương lai của cổ phiếu, độc lập với thị trường.

**Công thức Alpha:**
```
α = R_portfolio - β × R_market
```

### 1.2. Năm Nhân Tố Alpha Xác Suất

| Factor | Tên đầy đủ | Công thức | Ý nghĩa |
|--------|------------|-----------|---------|
| **BTP** | Bayesian Trend Persistence | t-stat của hồi quy OLS | Momentum dài hạn |
| **RIE** | Relative Info Entropy | -Σ p(x) log p(x) | Độ "hỗn loạn" |
| **MRD** | Mahalanobis Regime Distance | (r - μ_sector) / σ_sector | So với ngành |
| **KMD** | Kernel Momentum Decay | Σ w_i × r_i với w = Gaussian | Momentum gần đây |
| **RSVS** | Robust Skewness Vol Score | #up / #down moves | Bất đối xứng |

### 1.3. Information Coefficient (IC)

**Định nghĩa:** Tương quan giữa factor score và lợi nhuận tương lai.

**Công thức:**
```
IC = Corr(Factor_t, Return_{t+1})
```

**Ngưỡng đánh giá:**
- IC > 0.05: Strong
- IC 0.02-0.05: Moderate
- IC < 0.02: Weak

### 1.4. Information Ratio (IR)

**Công thức:**
```
IR = Mean(IC) / Std(IC) × √252
```

**Ngưỡng:**
- IR > 0.5: Good
- IR > 1.0: Excellent

---

## 2. Architecture

```
src/factors/
├── __init__.py
├── calculators.py    # Factor calculations
├── validation.py     # IC, IR, Decile analysis
├── orthogonal.py     # Factor orthogonalization
├── preprocessing.py  # Factor preprocessing
├── modeling.py       # Lasso regression
└── eda_driven.py     # EDA-driven factors
```

---

## 3. Implementation

### 3.1. FactorCalculator (`calculators.py`)

```python
from src.factors.calculators import FactorCalculator

calc = FactorCalculator(returns)
factors = calc.calculate_all()
# Returns: DataFrame with BTP, RIE, MRD, KMD, RSVS columns
```

### 3.2. FactorValidator (`validation.py`)

```python
from src.factors.validation import FactorValidator

validator = FactorValidator(factors, forward_returns)
ic_results = validator.compute_ic()
decile_returns = validator.decile_analysis()
```

---

## 4. Kết Quả Validation

| Factor | Mean IC | IR | Đánh giá |
|--------|---------|-----|----------|
| BTP | -0.0027 | -0.26 | Weak (Mean Reversion) |
| RIE | +0.0020 | +0.14 | Moderate |
| **MRD** | **+0.0064** | **+0.68** | **Strong** |
| KMD | -0.0035 | -0.31 | Weak |
| RSVS | -0.0009 | -0.13 | Weak |

**Kết luận:** MRD là factor mạnh nhất (IR = 0.68).

---

## 5. References

- Grinold, R.C. & Kahn, R.N. (1999). *Active Portfolio Management*
- Jegadeesh, N. & Titman, S. (1993). *Returns to Buying Winners and Selling Losers*
