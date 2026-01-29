"""
AI Supervisor Module
====================
Gemini-based trade explanation and weekly reports.
"""

import os
import json
from typing import Dict, List
from datetime import datetime
import google.generativeai as genai
from dotenv import load_dotenv


load_dotenv()
API_KEY = os.getenv("API_KEY")


class AISupervisor:
    """
    AI-powered investment supervisor using Gemini.
    
    Functions:
    1. Explain trade decisions
    2. Generate weekly reports
    3. Risk alerts
    4. Gap analysis
    """
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or API_KEY
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
    
    def explain_trade(
        self,
        ticker: str,
        action: str,  # 'BUY', 'SELL', 'HOLD'
        weight_change: float,
        factor_contributions: Dict[str, float],
        macro_context: Dict[str, float],
        cluster: str
    ) -> str:
        """
        Giải thích tại sao mô hình chọn trade này.
        """
        prompt = f"""Bạn là Chủ tịch Hội đồng Đầu tư. Hãy giải thích quyết định giao dịch này cho ban giám đốc.

QUYẾT ĐỊNH GIAO DỊCH:
- Cổ phiếu: {ticker}
- Hành động: {action}
- Thay đổi tỷ trọng: {weight_change:+.1%}
- Nhóm cổ phiếu: {cluster}

ĐÓNG GÓP CỦA CÁC NHÂN TỐ (từ mô hình Lasso):
{json.dumps(factor_contributions, indent=2, ensure_ascii=False)}

BỐI CẢNH VĨ MÔ:
- VIX (Chỉ số sợ hãi): {macro_context.get('VIX', 'N/A')}
- Lãi suất 10 năm: {macro_context.get('YIELD_10Y', 'N/A')}%
- DXY (Sức mạnh USD): {macro_context.get('DXY', 'N/A')}

Viết giải trình bằng tiếng Việt với 3 đoạn:
1. Nhân tố nào là động lực chính cho quyết định này
2. Bối cảnh vĩ mô hỗ trợ hay thách thức quyết định
3. Rủi ro cần theo dõi

Ngắn gọn, chuyên nghiệp, không quá 200 từ."""

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Lỗi khi tạo giải trình: {e}"
    
    def weekly_report(
        self,
        week_return: float,
        ytd_return: float,
        top_contributors: List[tuple],  # [(ticker, return), ...]
        top_detractors: List[tuple],
        current_weights: Dict[str, float],
        macro_summary: Dict[str, float],
        regime: str
    ) -> str:
        """
        Tạo báo cáo tuần tự động.
        """
        contributors_str = "\n".join([f"  - {t}: {r:+.2%}" for t, r in top_contributors[:5]])
        detractors_str = "\n".join([f"  - {t}: {r:+.2%}" for t, r in top_detractors[:5]])
        
        # Top holdings
        top_holdings = sorted(current_weights.items(), key=lambda x: x[1], reverse=True)[:10]
        holdings_str = "\n".join([f"  - {t}: {w:.1%}" for t, w in top_holdings])
        
        prompt = f"""Bạn là Chief Investment Officer. Viết báo cáo tuần cho quỹ đầu tư.

HIỆU SUẤT TUẦN NÀY:
- Lợi nhuận tuần: {week_return:+.2%}
- Lợi nhuận từ đầu năm (YTD): {ytd_return:+.2%}

ĐÓNG GÓP TÍCH CỰC:
{contributors_str}

ĐÓNG GÓP TIÊU CỰC:
{detractors_str}

TOP 10 VỊ THẾ:
{holdings_str}

BỐI CẢNH VĨ MÔ:
- VIX: {macro_summary.get('VIX', 'N/A')}
- Lãi suất 10Y: {macro_summary.get('YIELD_10Y', 'N/A')}%
- DXY: {macro_summary.get('DXY', 'N/A')}
- Chế độ thị trường: {regime}

Viết báo cáo chuyên nghiệp bằng tiếng Việt với các phần:
1. TỔNG QUAN TUẦN
2. PHÂN TÍCH HIỆU SUẤT
3. TRIỂN VỌNG & ĐIỀU CHỈNH

Khoảng 300 từ, tone chuyên nghiệp."""

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Lỗi khi tạo báo cáo: {e}"
    
    def risk_alert(
        self,
        alert_type: str,  # 'VIX_SPIKE', 'DRAWDOWN', 'CONCENTRATION', etc.
        current_value: float,
        threshold: float,
        affected_positions: List[str]
    ) -> str:
        """
        Tạo cảnh báo rủi ro.
        """
        prompt = f"""Bạn là Risk Manager. Tạo cảnh báo rủi ro khẩn cấp.

LOẠI CẢNH BÁO: {alert_type}
GIÁ TRỊ HIỆN TẠI: {current_value}
NGƯỠNG CẢNH BÁO: {threshold}
VỊ THẾ BỊ ẢNH HƯỞNG: {', '.join(affected_positions[:10])}

Viết cảnh báo ngắn gọn (100 từ) bằng tiếng Việt:
1. Mức độ nghiêm trọng
2. Nguyên nhân có thể
3. Hành động khuyến nghị"""

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Lỗi: {e}"
    
    def gap_analysis(
        self,
        expected_sharpe: float,
        realized_sharpe: float,
        expected_return: float,
        realized_return: float,
        period: str
    ) -> str:
        """
        Phân tích khoảng cách giữa dự báo và thực tế.
        """
        sharpe_gap = realized_sharpe - expected_sharpe
        return_gap = realized_return - expected_return
        
        prompt = f"""Bạn là Quant Analyst. Phân tích khoảng cách hiệu suất.

KỲ VỌNG (EX-ANTE):
- Sharpe Ratio: {expected_sharpe:.2f}
- Expected Return: {expected_return:+.1%}

THỰC TẾ (EX-POST):
- Sharpe Ratio: {realized_sharpe:.2f}
- Realized Return: {realized_return:+.1%}

KHOẢNG CÁCH:
- Sharpe Gap: {sharpe_gap:+.2f}
- Return Gap: {return_gap:+.1%}

GIAI ĐOẠN: {period}

Viết phân tích chuyên sâu (250 từ) bằng tiếng Việt:
1. ĐỊNH LƯỢNG KHOẢNG CÁCH
2. NGUYÊN NHÂN GỐC RỄ (Alpha decay, Regime shifts, Estimation error)
3. BÀI HỌC & CẢI THIỆN"""

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Lỗi: {e}"
    
    def save_report(self, report: str, output_path: str, report_type: str = "weekly"):
        """Lưu báo cáo."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"# {report_type.upper()} REPORT\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
            f.write(report)
        
        print(f"Report saved to {output_path}")


def test_supervisor():
    """Test AI Supervisor."""
    supervisor = AISupervisor()
    
    # Test trade explanation
    print("="*60)
    print("TEST: TRADE EXPLANATION")
    print("="*60)
    
    explanation = supervisor.explain_trade(
        ticker="NVDA",
        action="BUY",
        weight_change=0.02,
        factor_contributions={
            "MOM_12M": 0.5,
            "CONSISTENCY": 0.3,
            "MRD": 0.2
        },
        macro_context={
            "VIX": 15.5,
            "YIELD_10Y": 4.2,
            "DXY": 103
        },
        cluster="Growth"
    )
    
    print(explanation)
    
    return supervisor


if __name__ == "__main__":
    test_supervisor()
