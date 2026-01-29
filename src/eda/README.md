# EDA Module

## 1. Lý Thuyết

### 1.1. Fat Tails trong Tài Chính

**Định nghĩa:** Phân phối lợi nhuận không tuân theo phân phối chuẩn, các sự kiện cực đoan xảy ra thường xuyên hơn dự đoán.

**Đo lường:**
- **Kurtosis > 3:** Fat tails (Kurtosis chuẩn = 3)
- Trong dự án: Kurtosis = 15.92

**Ý nghĩa:** Không thể dùng VaR/CVaR dựa trên phân phối chuẩn!

### 1.2. Kiểm Định Tính Dừng (Stationarity)

**Augmented Dickey-Fuller Test:**

H₀: Chuỗi có unit root (không dừng)  
H₁: Chuỗi dừng (stationary)

**Kết quả:**
- Giá (Price): Không dừng (p > 0.05)
- Lợi nhuận (Returns): Dừng (p < 0.05)

**Quyết định:** Sử dụng Returns thay vì Price cho mô hình.

### 1.3. Autocorrelation Function (ACF)

**Công thức:**
```
ACF(k) = Cov(x_t, x_{t-k}) / Var(x_t)
```

**Ý nghĩa:**
- ACF giảm chậm → Chuỗi có trend (không dừng)
- ACF giảm nhanh → Chuỗi dừng

---

## 2. Architecture

```
src/eda/
├── __init__.py
├── analyzer.py     # EDA statistics & plots
└── clustering.py   # Hierarchical clustering
```

---

## 3. Implementation

### 3.1. EDAAnalyzer (`analyzer.py`)

**Chức năng:**
- Missing data heatmap
- Returns distribution
- Q-Q Plot
- ACF/PACF
- Sector distribution

### 3.2. Clustering (`clustering.py`)

**Chức năng:**
- Hierarchical clustering
- Correlation clustermap
- Dendrogram visualization

---

## 4. Ví Dụ

```python
from src.eda.analyzer import EDAAnalyzer

analyzer = EDAAnalyzer(prices, returns)

# Generate all EDA plots
analyzer.plot_missing_heatmap()
analyzer.plot_returns_distribution()
analyzer.plot_qq()
analyzer.run_adf_test()
```

---

## 5. References

- Mandelbrot, B. (1963). *The Variation of Certain Speculative Prices*
- Hamilton, J. (1994). *Time Series Analysis*
