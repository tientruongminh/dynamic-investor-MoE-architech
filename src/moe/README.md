# MoE Module (Mixture of Experts)

## 1. Lý Thuyết

### 1.1. Mixture of Experts (MoE)

**Định nghĩa:** Kiến trúc machine learning chia bài toán thành các phần nhỏ, mỗi phần được xử lý bởi một "expert" chuyên biệt.

**Công thức:**
```
Y = Σ π_k(x) × f_k(x)
```

Trong đó:
- `π_k(x)`: Gating function (trọng số Expert k)
- `f_k(x)`: Expert k prediction
- `Σ π_k = 1`: Ràng buộc

### 1.2. Gemini Clustering

**Ý tưởng:** Sử dụng LLM (Gemini) để phân loại cổ phiếu dựa trên:
- Sector
- Business description
- Financial metrics

**4 Clusters:**
| Cluster | Mô tả | Số mã |
|---------|-------|-------|
| **Growth** | High P/E, revenue growth >20% | 250 |
| **Value** | Low P/E, high dividend | 78 |
| **Cyclical** | Nhạy cảm kinh tế | 101 |
| **Defensive** | Ổn định, recession-resistant | 102 |

### 1.3. Lasso Expert

**Công thức Lasso:**
```
minimize: ||y - Xβ||² + λ Σ|βᵢ|
```

**Ưu điểm:**
- Feature selection tự động (co hệ số về 0)
- Giảm overfitting
- Dễ interpret

### 1.4. Gating Network

**Kiến trúc:** Random Forest Classifier

**Input:** VIX, 10Y Yield, DXY  
**Output:** π = [π_Growth, π_Value, π_Cyclical, π_Defensive]

**Logic:**
- VIX cao → tăng Defensive
- Yield cao → tăng Value
- DXY yếu → tăng Growth

---

## 2. Architecture

```
src/moe/
├── __init__.py
├── clustering.py      # GeminiClusterer (LLM-based)
├── lasso_expert.py    # LassoExpert class
├── gating.py          # Rule-based Gating
├── dynamic_gating.py  # ML-based Gating
├── regime_gating.py   # Regime detection
├── macro.py           # Macro data fetcher
└── supervisor.py      # LLM Supervisor
```

---

## 3. Implementation

### 3.1. GeminiClusterer (`clustering.py`)

```python
from src.moe.clustering import GeminiClusterer

clusterer = GeminiClusterer(api_key='xxx')
clusters = clusterer.classify(tickers, stock_info)
# Returns: {'AAPL': 'Growth', 'KO': 'Value', ...}
```

### 3.2. LassoExpert (`lasso_expert.py`)

```python
from src.moe.lasso_expert import LassoExpert

expert = LassoExpert(cluster='Growth')
expert.fit(X_train, y_train)
predictions = expert.predict(X_test)
```

### 3.3. GatingNetwork (`dynamic_gating.py`)

```python
from src.moe.dynamic_gating import GatingNetwork

gating = GatingNetwork()
gating.fit(macro_features, expert_returns)
weights = gating.predict(current_macro)
# Returns: {'Growth': 0.25, 'Value': 0.28, ...}
```

---

## 4. Feature Importance (Lasso)

| Factor | Growth | Value | Cyclical | Defensive |
|--------|--------|-------|----------|-----------|
| REV_5D | **39%** | **40%** | 19% | **30%** |
| VOL | 34% | 26% | **35%** | 25% |
| DRAWDOWN | 8% | 17% | 5% | 3% |

**Insight:** REV_5D (Mean Reversion) là factor quan trọng nhất.

---

## 5. References

- Jacobs, R. et al. (1991). *Adaptive Mixtures of Local Experts*
- Shazeer, N. et al. (2017). *Outrageously Large Neural Networks*
