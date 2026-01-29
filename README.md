# MoE Alpha Framework

<p align="center">
  <strong>Dynamic Investor - Mixture of Experts Alpha Architecture</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue.svg">
  <img src="https://img.shields.io/badge/Sharpe-0.98-green.svg">
  <img src="https://img.shields.io/badge/Alpha-+20.94%25-brightgreen.svg">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg">
</p>

---

## 📖 Tổng Quan

**MoE Alpha Framework** là hệ thống đầu tư tự động sử dụng kiến trúc **Mixture of Experts (MoE)** kết hợp với **Gemini AI** để:

1. 🎯 Phân loại cổ phiếu thành 4 clusters (Growth, Value, Cyclical, Defensive)
2. 🧠 Mỗi Expert (Lasso) chuyên dự báo cho cluster của nó
3. 📊 Gating Network quyết định trọng số Expert dựa trên VIX, Yield
4. 💼 Black-Litterman optimization để tối ưu danh mục

---

## 🏆 Kết Quả

| Metric | MoE | S&P 500 | So sánh |
|--------|-----|---------|---------|
| **Sharpe Ratio** | 0.98 | 0.50 | **+96%** |
| **Alpha (CAPM)** | +20.94% | 0% | **Skill-based** |
| **Max Drawdown** | -17.9% | -34% | **+16.1%** |
| **Win Rate** | 54.5% | 52% | **+2.5%** |

---

## 🏗️ Kiến Trúc

```
┌─────────────────────────────────────────────────────────────┐
│                        INPUT DATA                            │
│  Giá 531 mã NASDAQ │ VIX, Yield 10Y, DXY │ Business Info    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    GEMINI CLUSTERING                         │
│         Growth │ Value │ Cyclical │ Defensive               │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
        ┌─────────┐     ┌─────────┐     ┌─────────┐
        │ Expert 1│     │ Expert 2│     │ Expert N│
        │ (Lasso) │     │ (Lasso) │     │ (Lasso) │
        └─────────┘     └─────────┘     └─────────┘
              │               │               │
              └───────────────┼───────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    GATING NETWORK                            │
│             π = [25%, 28%, 30%, 16%]                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  BLACK-LITTERMAN                             │
│            Portfolio Weights (max 5%/stock)                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      OUTPUT                                  │
│           Top 10 Stocks + Weights + Risk Warnings           │
└─────────────────────────────────────────────────────────────┘
```

---

## 📂 Cấu Trúc Dự Án

```
dynamic-investor-MoE-architech/
├── src/
│   ├── data/           # Data fetching          [README](src/data/README.md)
│   ├── preprocessing/  # Data preprocessing     [README](src/preprocessing/README.md)
│   ├── eda/            # Exploratory Analysis   [README](src/eda/README.md)
│   ├── factors/        # Alpha Factors          [README](src/factors/README.md)
│   ├── moe/            # MoE Architecture       [README](src/moe/README.md)
│   ├── backtest/       # Backtesting Engine     [README](src/backtest/README.md)
│   ├── portfolio/      # Portfolio Optimization [README](src/portfolio/README.md)
│   └── reports/        # Report Generation      [README](src/reports/README.md)
├── data/               # Raw data (tickers, prices)
├── outputs/            # MoE outputs & models
├── outputs_clean/      # Midterm outputs
├── dashboard/          # Interactive HTML dashboard
├── main_moe.py         # Main entry point
└── requirements.txt    # Dependencies
```

---

## 🚀 Quick Start

### 1. Cài đặt

```bash
# Clone repo
git clone https://github.com/username/dynamic-investor-MoE-architech.git
cd dynamic-investor-MoE-architech

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Setup API Key

```bash
# Tạo file .env
echo "API_KEY=your_gemini_api_key" > .env
```

### 3. Chạy toàn bộ pipeline

```bash
python main_moe.py
```

---

## 📊 Dashboard

**Xem Dashboard:** `dashboard/index.html`

Hoặc:
```bash
python -m http.server 8080
# Mở http://localhost:8080/dashboard/
```

---

## 📝 Báo Cáo

| Báo cáo | Link |
|---------|------|
| **Giữa Kỳ** | [midterm_report_vietnamese.html](outputs_clean/midterm_report_vietnamese.html) |
| **Cuối Kỳ** | [FINAL_TERM_REPORT.html](outputs/moe/FINAL_TERM_REPORT.html) |
| **Portfolio** | [FULL_PORTFOLIO_REPORT.html](outputs/moe/FULL_PORTFOLIO_REPORT.html) |

---

## 🔮 Hướng Phát Triển

- [ ] **Double-Dynamic MoE:** Clustering động theo thời gian
- [ ] **Factor Return Target:** Thay vì Stock Return
- [ ] **Entropy-based Gating:** Hard khi rõ, Soft khi hỗn loạn
- [ ] **Hysteresis Constraint:** Giảm churning

Chi tiết: [DOUBLE_DYNAMIC_MOE.md](outputs/moe/DOUBLE_DYNAMIC_MOE.md)

---

## 📚 References

1. Jacobs, R. et al. (1991). *Adaptive Mixtures of Local Experts*
2. Black, F. & Litterman, R. (1992). *Global Portfolio Optimization*
3. Grinold, R.C. & Kahn, R.N. (1999). *Active Portfolio Management*

---

## 📄 License

MIT License - Xem [LICENSE](LICENSE) để biết thêm chi tiết.

---

<p align="center">
  <strong>Zhou & Bishop Alpha Framework</strong><br>
  "Probabilistic Thinking for Quantitative Investing"
</p>
