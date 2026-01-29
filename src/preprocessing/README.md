# Preprocessing Module

## 1. Lý Thuyết

### 1.1. Missing Data Imputation

**Vấn đề:** Dữ liệu tài chính thường có giá trị thiếu do:
- Ngày nghỉ lễ
- Lỗi API
- IPO gần đây

**Các phương pháp:**

| Phương pháp | Công thức | Khi nào dùng |
|-------------|-----------|--------------|
| Forward Fill | x_t = x_{t-1} | Gap ≤ 3 ngày |
| Linear Interpolation | x_t = (x_{t-1} + x_{t+1})/2 | Gap 4-10 ngày |
| Cubic Spline | f(x) = ax³ + bx² + cx + d | Gap 11-20 ngày |

### 1.2. Winsorization

**Định nghĩa:** Cắt giá trị cực đoan về ngưỡng percentile.

**Công thức:**
```
x_winsorized = clip(x, percentile_lower, percentile_upper)
```

**Trong dự án:** Sử dụng 0.5% - 99.5%

### 1.3. RobustScaler vs StandardScaler

| Scaler | Công thức | Ưu điểm |
|--------|-----------|---------|
| StandardScaler | (x - μ) / σ | Nhanh |
| **RobustScaler** | (x - median) / IQR | **Robust với outliers** |

**Lý do chọn RobustScaler:** Dữ liệu tài chính có Fat Tails (Kurtosis = 15.92).

---

## 2. Architecture

```
src/preprocessing/
├── __init__.py
└── pipeline.py    # PreprocessingPipeline class
```

### Data Flow

```
Raw Data → Missing Imputation → Winsorization → Scaling → Clean Data
```

---

## 3. Implementation

### 3.1. PreprocessingPipeline (`pipeline.py`)

**Chức năng:**
- Tiered interpolation cho missing values
- Winsorization tại 0.5%-99.5%
- RobustScaler normalization

**API:**
```python
from src.preprocessing.pipeline import PreprocessingPipeline

pipeline = PreprocessingPipeline()
clean_prices = pipeline.fit_transform(raw_prices)
```

---

## 4. Ví Dụ

```python
# Before
raw_prices.isna().sum().sum()  # 52,943 missing values

# After preprocessing
clean_prices = pipeline.fit_transform(raw_prices)
clean_prices.isna().sum().sum()  # 0 missing values
```

---

## 5. References

- Little & Rubin (2019). *Statistical Analysis with Missing Data*
- Taleb, N.N. (2007). *The Black Swan*
