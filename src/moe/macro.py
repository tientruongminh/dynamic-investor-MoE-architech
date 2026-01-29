"""
Macro Data Module
=================
Thu thập dữ liệu vĩ mô: VIX, 10Y Yield, DXY
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Optional
import os


class MacroDataFetcher:
    """
    Thu thập và xử lý dữ liệu vĩ mô cho Gating Network.
    """
    
    TICKERS = {
        'VIX': '^VIX',        # CBOE Volatility Index
        'YIELD_10Y': '^TNX',  # 10-Year Treasury Yield
        'DXY': 'DX-Y.NYB',    # US Dollar Index
        'SPY': 'SPY',         # S&P 500 ETF (market benchmark)
        'QQQ': 'QQQ'          # NASDAQ-100 ETF
    }
    
    def __init__(self):
        self.data = {}
    
    def fetch(self, start: str, end: str) -> pd.DataFrame:
        """
        Lấy dữ liệu vĩ mô.
        
        Args:
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame với columns: VIX, YIELD_10Y, DXY, SPY, QQQ
        """
        print("Fetching macro data...")
        
        for name, ticker in self.TICKERS.items():
            try:
                df = yf.download(ticker, start=start, end=end, progress=False)
                self.data[name] = df['Close'] if 'Close' in df.columns else df['Adj Close']
                print(f"  {name}: {len(self.data[name])} rows")
            except Exception as e:
                print(f"  Warning: Could not fetch {name}: {e}")
                self.data[name] = pd.Series(dtype=float)
        
        # Combine into single DataFrame
        result = pd.DataFrame(self.data)
        
        # Forward fill missing values
        result = result.ffill().bfill()
        
        return result
    
    def calculate_features(self, macro_df: pd.DataFrame) -> pd.DataFrame:
        """
        Tính toán các features cho Gating Network.
        
        Features:
        - VIX level
        - VIX change (5d)
        - 10Y Yield level
        - Yield change (5d)
        - DXY level
        - DXY change (5d)
        - Market momentum (SPY 20d return)
        - Volatility regime (VIX percentile)
        """
        features = pd.DataFrame(index=macro_df.index)
        
        # Levels
        features['VIX'] = macro_df['VIX']
        features['YIELD_10Y'] = macro_df['YIELD_10Y']
        features['DXY'] = macro_df['DXY']
        
        # Changes
        features['VIX_CHG_5D'] = macro_df['VIX'].pct_change(5)
        features['YIELD_CHG_5D'] = macro_df['YIELD_10Y'].diff(5)
        features['DXY_CHG_5D'] = macro_df['DXY'].pct_change(5)
        
        # Market momentum
        if 'SPY' in macro_df.columns:
            features['MKT_MOM_20D'] = macro_df['SPY'].pct_change(20)
        
        # Volatility regime (rolling percentile)
        features['VIX_PERCENTILE'] = macro_df['VIX'].rolling(252).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1]
        )
        
        # Risk regime flags
        features['HIGH_VIX'] = (macro_df['VIX'] > 25).astype(int)
        features['LOW_VIX'] = (macro_df['VIX'] < 15).astype(int)
        features['RISING_RATES'] = (features['YIELD_CHG_5D'] > 0.1).astype(int)
        
        return features.dropna()
    
    def get_regime(self, macro_features: pd.DataFrame) -> pd.Series:
        """
        Xác định regime thị trường.
        
        Regimes:
        - RISK_ON: Low VIX, positive momentum
        - RISK_OFF: High VIX, negative momentum
        - NEUTRAL: Everything else
        """
        regimes = []
        
        for idx in macro_features.index:
            row = macro_features.loc[idx]
            
            if row.get('HIGH_VIX', 0) == 1:
                regime = 'RISK_OFF'
            elif row.get('LOW_VIX', 0) == 1 and row.get('MKT_MOM_20D', 0) > 0:
                regime = 'RISK_ON'
            else:
                regime = 'NEUTRAL'
            
            regimes.append(regime)
        
        return pd.Series(regimes, index=macro_features.index, name='regime')
    
    def save(self, macro_df: pd.DataFrame, path: str):
        """Lưu macro data."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        macro_df.to_csv(path)
        print(f"Saved macro data to {path}")
    
    @staticmethod
    def load(path: str) -> pd.DataFrame:
        """Load macro data."""
        return pd.read_csv(path, index_col=0, parse_dates=True)


def run_macro_fetch(
    start: str = "2016-01-01",
    end: str = "2026-01-20",
    output_path: str = "./outputs/moe/macro_data.csv"
):
    """
    Main function to fetch and process macro data.
    """
    fetcher = MacroDataFetcher()
    
    # Fetch raw data
    macro_df = fetcher.fetch(start, end)
    
    # Calculate features
    features = fetcher.calculate_features(macro_df)
    
    # Get regimes
    regimes = fetcher.get_regime(features)
    features['regime'] = regimes
    
    # Save
    fetcher.save(features, output_path)
    
    # Print summary
    print("\n" + "="*60)
    print("MACRO DATA SUMMARY")
    print("="*60)
    print(f"Date range: {features.index[0]} to {features.index[-1]}")
    print(f"Total rows: {len(features)}")
    print("\nRegime distribution:")
    print(regimes.value_counts())
    print("\nLatest values:")
    print(features.iloc[-1])
    
    return features


if __name__ == "__main__":
    features = run_macro_fetch()
