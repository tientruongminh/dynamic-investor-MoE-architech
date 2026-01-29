"""
Factor Preprocessing Module
============================
Neutralization và Smoothing cho Alpha Factors
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import os


class FactorPreprocessor:
    """
    Xử lý factors trước khi đưa vào mô hình:
    1. Winsorization - Cắt outliers
    2. Neutralization - Loại bỏ bias ngành
    3. Standardization - Z-score
    4. Smoothing - Giảm noise
    """
    
    def __init__(
        self,
        winsor_limits: tuple = (0.01, 0.99),
        smoothing_halflife: int = 5,
        min_stocks_per_sector: int = 5
    ):
        self.winsor_limits = winsor_limits
        self.halflife = smoothing_halflife
        self.min_stocks = min_stocks_per_sector
    
    def winsorize(self, factor: pd.DataFrame) -> pd.DataFrame:
        """
        Winsorization: Cắt outliers tại percentile limits.
        Cross-sectional (mỗi ngày riêng).
        """
        def winsorize_row(row):
            lower = row.quantile(self.winsor_limits[0])
            upper = row.quantile(self.winsor_limits[1])
            return row.clip(lower=lower, upper=upper)
        
        return factor.apply(winsorize_row, axis=1)
    
    def neutralize_sector(
        self,
        factor: pd.DataFrame,
        sector_map: Dict[str, str]
    ) -> pd.DataFrame:
        """
        Sector Neutralization: Loại bỏ bias ngành.
        
        Công thức: factor_neutral[i] = factor[i] - mean(factor trong ngành của i)
        
        Sau neutralization, mean của factor trong mỗi ngành = 0
        """
        result = factor.copy()
        
        for date in factor.index:
            row = factor.loc[date]
            
            # Group by sector
            sector_means = {}
            sector_stds = {}
            
            for sector in set(sector_map.values()):
                sector_stocks = [s for s, sec in sector_map.items() 
                               if sec == sector and s in row.index]
                
                if len(sector_stocks) >= self.min_stocks:
                    sector_vals = row[sector_stocks].dropna()
                    if len(sector_vals) > 0:
                        sector_means[sector] = sector_vals.mean()
                        sector_stds[sector] = sector_vals.std()
            
            # Neutralize each stock
            for stock in row.index:
                sector = sector_map.get(stock, 'Unknown')
                if sector in sector_means:
                    # Demean by sector
                    result.loc[date, stock] = row[stock] - sector_means[sector]
                    
                    # Optional: Also divide by sector std for full standardization
                    if sector_stds.get(sector, 0) > 0:
                        result.loc[date, stock] /= sector_stds[sector]
        
        return result
    
    def neutralize_market(self, factor: pd.DataFrame) -> pd.DataFrame:
        """
        Market Neutralization: Loại bỏ bias toàn thị trường.
        
        Công thức: factor_neutral[i] = factor[i] - mean(all factors)
        
        Sau neutralization, cross-sectional mean = 0
        """
        # Cross-sectional demean
        return factor.sub(factor.mean(axis=1), axis=0)
    
    def standardize(self, factor: pd.DataFrame) -> pd.DataFrame:
        """
        Cross-sectional Z-score standardization.
        
        Công thức: z[i] = (factor[i] - mean) / std
        """
        mean = factor.mean(axis=1)
        std = factor.std(axis=1)
        
        result = factor.sub(mean, axis=0).div(std, axis=0)
        return result
    
    def smooth(self, factor: pd.DataFrame) -> pd.DataFrame:
        """
        Exponential Moving Average smoothing theo thời gian.
        
        Giảm noise và tăng stability của factor signals.
        Halflife = số ngày để weight giảm còn 50%
        """
        return factor.ewm(halflife=self.halflife, min_periods=1).mean()
    
    def process(
        self,
        factor: pd.DataFrame,
        sector_map: Dict[str, str] = None,
        steps: list = ['winsorize', 'neutralize_sector', 'standardize', 'smooth']
    ) -> pd.DataFrame:
        """
        Apply full preprocessing pipeline.
        
        Args:
            factor: Raw factor DataFrame
            sector_map: {ticker: sector} mapping
            steps: List of processing steps to apply
            
        Returns:
            Processed factor DataFrame
        """
        result = factor.copy()
        
        for step in steps:
            if step == 'winsorize':
                result = self.winsorize(result)
            elif step == 'neutralize_sector' and sector_map is not None:
                result = self.neutralize_sector(result, sector_map)
            elif step == 'neutralize_market':
                result = self.neutralize_market(result)
            elif step == 'standardize':
                result = self.standardize(result)
            elif step == 'smooth':
                result = self.smooth(result)
        
        return result
    
    def process_all(
        self,
        factors: Dict[str, pd.DataFrame],
        sector_map: Dict[str, str] = None,
        steps: list = ['winsorize', 'neutralize_sector', 'standardize', 'smooth']
    ) -> Dict[str, pd.DataFrame]:
        """
        Process all factors.
        """
        processed = {}
        
        for name, factor in factors.items():
            print(f"Processing {name}...", end=" ")
            processed[name] = self.process(factor, sector_map, steps)
            print("done")
        
        return processed


def run_factor_preprocessing(
    factors_dir: str = "./outputs/factors_eda",
    info_dir: str = "./data/nasdaq_574/info",
    output_dir: str = "./outputs/factors_preprocessed",
    steps: list = ['winsorize', 'neutralize_sector', 'standardize', 'smooth']
):
    """
    Main function to preprocess all factors.
    """
    import json
    from pathlib import Path
    
    print("="*60)
    print("FACTOR PREPROCESSING PIPELINE")
    print("="*60)
    print(f"Steps: {' → '.join(steps)}")
    
    # Load factors
    factors = {}
    factors_path = Path(factors_dir)
    for fpath in factors_path.glob("*.csv"):
        name = fpath.stem
        factors[name] = pd.read_csv(fpath, index_col=0, parse_dates=True)
        # Handle timezone
        factors[name].index = pd.to_datetime(factors[name].index, utc=True).tz_convert(None)
    
    print(f"\nLoaded {len(factors)} factors")
    
    # Load sector map
    sector_map = {}
    info_path = Path(info_dir)
    for fpath in info_path.glob("*.json"):
        ticker = fpath.stem
        with open(fpath) as f:
            info = json.load(f)
            sector_map[ticker] = info.get('sector', 'Unknown')
    
    print(f"Loaded sector map for {len(sector_map)} stocks")
    
    # Sector distribution
    sector_counts = {}
    for s in sector_map.values():
        sector_counts[s] = sector_counts.get(s, 0) + 1
    print("\nSector distribution:")
    for s, c in sorted(sector_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {s}: {c}")
    
    # Preprocess
    print("\nPreprocessing factors...")
    preprocessor = FactorPreprocessor(
        winsor_limits=(0.01, 0.99),
        smoothing_halflife=5,
        min_stocks_per_sector=5
    )
    
    processed = preprocessor.process_all(factors, sector_map, steps)
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    for name, factor in processed.items():
        factor.to_csv(f"{output_dir}/{name}.csv")
    
    print(f"\nSaved {len(processed)} processed factors to {output_dir}")
    
    # Compare before/after
    print("\n" + "="*60)
    print("BEFORE vs AFTER COMPARISON")
    print("="*60)
    
    for name in list(factors.keys())[:3]:
        raw = factors[name].iloc[-1].dropna()
        proc = processed[name].iloc[-1].dropna()
        
        print(f"\n{name}:")
        print(f"  Raw:  mean={raw.mean():.4f}, std={raw.std():.4f}, range=[{raw.min():.2f}, {raw.max():.2f}]")
        print(f"  Proc: mean={proc.mean():.4f}, std={proc.std():.4f}, range=[{proc.min():.2f}, {proc.max():.2f}]")
    
    return processed


if __name__ == "__main__":
    processed = run_factor_preprocessing()
