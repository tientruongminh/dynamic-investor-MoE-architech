# Backtest Module

## 1. Lý Thuyết

### 1.1. Walk-Forward Validation

**Vấn đề:** Standard train/test split không phù hợp cho time series vì:
- Data leakage (dùng tương lai để train)
- Non-stationarity

**Giải pháp: Walk-Forward**

```
[--------Train--------][Test]
    [--------Train--------][Test]
        [--------Train--------][Test]
```

**Trong dự án:**
- Training window: 52 tuần (1 năm)
- Test window: 1 tuần
- Rolling: Di chuyển 1 tuần mỗi lần

### 1.2. Transaction Costs

**Thành phần:**
- **Commission:** Phí môi giới (~0.05%)
- **Slippage:** Chênh lệch giá dự kiến vs thực tế (~0.05%)
- **Spread:** Bid-Ask spread (~0.05%)

**Tổng cost trong backtest:** 0.1-0.15% mỗi lần giao dịch

### 1.3. Performance Metrics

| Metric | Công thức | Ý nghĩa |
|--------|-----------|---------|
| **Sharpe Ratio** | (R - Rf) / σ | Risk-adjusted return |
| **Max Drawdown** | max(Peak - Trough) | Worst loss |
| **Win Rate** | #Winning weeks / Total | Tỷ lệ thắng |
| **Calmar Ratio** | Annual Return / Max DD | Return per unit DD |

---

## 2. Architecture

```
src/backtest/
├── __init__.py
└── engine.py    # BacktestEngine class
```

---

## 3. Implementation

### 3.1. BacktestEngine (`engine.py`)

```python
from src.backtest.engine import BacktestEngine

engine = BacktestEngine(
    prices=prices,
    signals=signals,
    initial_capital=1_000_000,
    transaction_cost=0.001
)

results = engine.run()
print(f"Sharpe: {results['sharpe']:.2f}")
print(f"Return: {results['total_return']:.1%}")
```

**Parameters:**
- `prices`: DataFrame giá cổ phiếu
- `signals`: DataFrame tín hiệu mua/bán
- `initial_capital`: Vốn ban đầu
- `transaction_cost`: Chi phí giao dịch (%)
- `max_position`: Giới hạn % mỗi mã

---

## 4. Kết Quả Backtest

| Metric | MoE | S&P 500 | So sánh |
|--------|-----|---------|---------|
| **Sharpe** | 0.98 | 0.50 | +96% |
| **Return (3Y)** | +57.8% | +32% | +25.8% |
| **Max Drawdown** | -17.9% | -34% | +16.1% |
| **Win Rate** | 54.5% | 52% | +2.5% |

---

## 5. Robustness Tests

| Test | Kết quả |
|------|---------|
| Slippage 5x (0.5%) | Sharpe = 0.86 > SPY |
| Turnover limit 10% | Sharpe = 0.92 |
| VIX > 25 periods | Win Rate = 100% |

---

## 6. References

- Bailey, D. et al. (2014). *The Deflated Sharpe Ratio*
- Harvey, C. et al. (2016). *... and the Cross-Section of Expected Returns*
