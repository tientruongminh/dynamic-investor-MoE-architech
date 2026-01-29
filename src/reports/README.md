# Reports Module

## 1. Lý Thuyết

### 1.1. Nguyên Tắc Trình Bày Kết Quả

**Pyramid Principle (Barbara Minto):**
1. **Bắt đầu bằng kết luận** (Sharpe = 0.98)
2. **Hỗ trợ bằng arguments** (Alpha, Beta, Robustness)
3. **Chi tiết cuối cùng** (Tables, Charts)

### 1.2. Metrics Storytelling

| Metric | Ý nghĩa cho Investor |
|--------|---------------------|
| Sharpe | "Với cùng rủi ro, bạn được thêm 96% lợi nhuận" |
| Alpha | "Không phải may mắn, có kỹ năng thật" |
| Max DD | "Tệ nhất bạn mất 17.9%, tốt hơn thị trường" |

---

## 2. Architecture

```
src/reports/
├── __init__.py
└── generator.py    # ReportGenerator class
```

---

## 3. Implementation

### 3.1. ReportGenerator (`generator.py`)

```python
from src.reports.generator import ReportGenerator

gen = ReportGenerator(
    backtest_results=results,
    portfolio_weights=weights,
    llm_analysis=analysis
)

# Generate HTML report
gen.generate_html('FINAL_TERM_REPORT.html')
```

---

## 4. Output Reports

| Report | Nội dung |
|--------|----------|
| `FINAL_TERM_REPORT.html` | Báo cáo cuối kỳ đầy đủ |
| `FULL_PORTFOLIO_REPORT.html` | Danh mục + Risk analysis |
| `midterm_report_vietnamese.html` | Báo cáo giữa kỳ |

---

## 5. References

- Minto, B. (2009). *The Pyramid Principle*
- Tufte, E. (2001). *The Visual Display of Quantitative Information*
