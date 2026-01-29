"""
Redesigned Alpha Factors - Low Multicollinearity Version
=========================================================
Problem: Original factors (BTP, MRD, KMD) highly correlated (0.83-0.94)

Solution: New factor set capturing DIFFERENT signals:
1. MRD - Sector-relative performance (KEEP - significant)
2. RIE - Information entropy (KEEP - independent)
3. VOL - Realized volatility (NEW - risk signal)
4. REV - Short-term reversal (NEW - contrarian)
5. LIQ - Liquidity signal (NEW - volume-based)

Theoretical basis:
- MRD: Sector rotation effect
- RIE: Uncertainty premium
- VOL: Low volatility anomaly
- REV: Short-term mean reversion
- LIQ: Liquidity premium
"""

import pandas as pd
import numpy as np
from typing import Dict
import time


class OrthogonalFactorEngine:
    """
    Calculates orthogonal (low-correlation) alpha factors.
    """
    
    def __init__(self, prices: pd.DataFrame, volumes: pd.DataFrame = None, info: Dict = None):
        self.prices = prices
        self.volumes = volumes if volumes is not None else pd.DataFrame()
        self.returns = prices.pct_change()
        self.info = info or {}
        self.factors = {}
    
    def calculate_mrd(self, window: int = 60) -> pd.DataFrame:
        """
        Mahalanobis Regime Distance - Sector-relative performance.
        Z-score of stock return vs sector mean.
        """
        print("  Calculating MRD (Sector Momentum)...", end=" ", flush=True)
        start = time.time()
        
        returns_np = self.returns.values
        n_days, n_stocks = returns_np.shape
        
        # Build sector map
        sector_map = {}
        for idx, ticker in enumerate(self.prices.columns):
            sector = self.info.get(ticker, {}).get('sector', 'Unknown')
            if sector not in sector_map:
                sector_map[sector] = []
            sector_map[sector].append(idx)
        
        mrd = np.full((n_days, n_stocks), np.nan)
        
        for i in range(window, n_days):
            window_returns = returns_np[i-window:i, :]
            
            sector_means = {}
            sector_stds = {}
            for sector, indices in sector_map.items():
                sector_ret = window_returns[:, indices]
                sector_means[sector] = np.nanmean(sector_ret)
                sector_stds[sector] = np.nanstd(sector_ret) + 1e-10
            
            for sector, indices in sector_map.items():
                for idx in indices:
                    stock_ret = np.nanmean(window_returns[:, idx])
                    z_score = (stock_ret - sector_means[sector]) / sector_stds[sector]
                    mrd[i, idx] = z_score
        
        result = pd.DataFrame(mrd, index=self.prices.index, columns=self.prices.columns)
        self.factors['MRD'] = result
        print(f"done ({time.time()-start:.1f}s)")
        return result
    
    def calculate_rie(self, window: int = 20, n_bins: int = 10) -> pd.DataFrame:
        """
        Relative Information Entropy - Return unpredictability.
        Shannon entropy of return distribution.
        """
        print("  Calculating RIE (Entropy)...", end=" ", flush=True)
        start = time.time()
        
        returns_np = self.returns.values
        n_days, n_stocks = returns_np.shape
        bins = np.linspace(-0.15, 0.15, n_bins + 1)
        
        rie = np.full((n_days, n_stocks), np.nan)
        
        for i in range(window, n_days):
            window_returns = returns_np[i-window:i, :]
            
            for j in range(n_stocks):
                r = window_returns[:, j]
                r = r[~np.isnan(r)]
                
                if len(r) < window // 2:
                    continue
                
                hist, _ = np.histogram(r, bins=bins, density=False)
                hist = hist / (hist.sum() + 1e-10)
                hist = hist[hist > 0]
                
                entropy = -np.sum(hist * np.log(hist + 1e-10))
                rie[i, j] = entropy
        
        result = pd.DataFrame(rie, index=self.prices.index, columns=self.prices.columns)
        self.factors['RIE'] = result
        print(f"done ({time.time()-start:.1f}s)")
        return result
    
    def calculate_vol(self, window: int = 20) -> pd.DataFrame:
        """
        Realized Volatility - Risk signal.
        Low volatility stocks tend to outperform (Low Vol Anomaly).
        NEGATIVE of volatility so higher score = lower risk = better.
        """
        print("  Calculating VOL (Volatility)...", end=" ", flush=True)
        start = time.time()
        
        # Rolling std of returns
        vol = self.returns.rolling(window=window).std() * np.sqrt(252)
        
        # Negative so low vol = high score
        vol = -vol
        
        self.factors['VOL'] = vol
        print(f"done ({time.time()-start:.1f}s)")
        return vol
    
    def calculate_rev(self, window: int = 5) -> pd.DataFrame:
        """
        Short-Term Reversal - Contrarian signal.
        Stocks that dropped recently tend to bounce back.
        NEGATIVE of short-term return.
        """
        print("  Calculating REV (Reversal)...", end=" ", flush=True)
        start = time.time()
        
        # Short-term cumulative return
        short_return = self.returns.rolling(window=window).sum()
        
        # Negative for reversal effect
        rev = -short_return
        
        self.factors['REV'] = rev
        print(f"done ({time.time()-start:.1f}s)")
        return rev
    
    def calculate_liq(self, window: int = 20) -> pd.DataFrame:
        """
        Liquidity Signal - Volume-based.
        High turnover = more attention = potential alpha.
        Uses price-volume correlation as proxy.
        """
        print("  Calculating LIQ (Liquidity)...", end=" ", flush=True)
        start = time.time()
        
        if self.volumes.empty:
            # Fallback: use Amihud illiquidity (return/volume proxy)
            # Approximate with return^2 as illiquidity proxy
            liq = -self.returns.abs().rolling(window=window).mean()
        else:
            # Volume change
            vol_change = self.volumes.pct_change().rolling(window=window).mean()
            liq = vol_change
        
        self.factors['LIQ'] = liq
        print(f"done ({time.time()-start:.1f}s)")
        return liq
    
    def calculate_all(self) -> Dict[str, pd.DataFrame]:
        """Calculate all orthogonal factors."""
        print("\n" + "="*60)
        print("ORTHOGONAL FACTOR CALCULATIONS")
        print("="*60)
        
        total_start = time.time()
        
        self.calculate_mrd()
        self.calculate_rie()
        self.calculate_vol()
        self.calculate_rev()
        self.calculate_liq()
        
        print(f"\nTotal: {time.time()-total_start:.1f}s")
        return self.factors
    
    def standardize(self) -> Dict[str, pd.DataFrame]:
        """Cross-sectional standardization."""
        standardized = {}
        for name, df in self.factors.items():
            mean = df.mean(axis=1)
            std = df.std(axis=1)
            standardized[name] = df.sub(mean, axis=0).div(std + 1e-10, axis=0)
        return standardized
    
    def get_correlation_matrix(self) -> pd.DataFrame:
        """Get correlation matrix between factors."""
        # Use last year of data
        latest = pd.DataFrame()
        for name, df in self.factors.items():
            latest[name] = df.iloc[-252:].mean()
        return latest.corr()
    
    def save(self, output_dir: str = "./outputs/factors_v2"):
        """Save factors to disk."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for name, df in self.factors.items():
            df.to_csv(os.path.join(output_dir, f"{name}_raw.csv"))
        
        std_factors = self.standardize()
        for name, df in std_factors.items():
            df.to_csv(os.path.join(output_dir, f"{name}_standardized.csv"))
        
        print(f"Factors saved to {output_dir}")


def run_orthogonal_factors(
    prices_file: str = "./outputs_clean/clean/processed_prices.csv",
    info_dir: str = "./data/nasdaq_574",
    output_dir: str = "./outputs/factors_v2"
):
    """Run orthogonal factor calculation and validation."""
    import json
    import os
    from scipy import stats
    
    print("="*70)
    print("ORTHOGONAL FACTOR ENGINE V2")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
    prices = pd.read_csv(prices_file, index_col=0, parse_dates=True)
    
    # Load info
    info = {}
    info_path = os.path.join(info_dir, "info")
    if os.path.exists(info_path):
        for f in os.listdir(info_path):
            if f.endswith('.json'):
                ticker = f.replace('.json', '')
                with open(os.path.join(info_path, f)) as fp:
                    info[ticker] = json.load(fp)
    
    print(f"Loaded: {prices.shape[1]} stocks, {prices.shape[0]} days")
    
    # Calculate factors
    engine = OrthogonalFactorEngine(prices, info=info)
    factors = engine.calculate_all()
    
    # Save
    engine.save(output_dir)
    
    # Validate
    print("\n" + "="*60)
    print("VALIDATION: CORRELATION MATRIX (TARGET: < 0.5)")
    print("="*60)
    corr = engine.get_correlation_matrix()
    print(corr.round(3).to_string())
    
    # Check significance
    print("\n" + "="*60)
    print("VALIDATION: STATISTICAL SIGNIFICANCE")
    print("="*60)
    
    returns = prices.pct_change()
    forward_returns = returns.shift(-5)
    
    print(f"\n{'Factor':<8} {'IC Mean':>10} {'t-stat':>10} {'p-value':>12} {'Significant':>12}")
    print("-"*60)
    
    for name, factor_df in factors.items():
        common_idx = factor_df.index.intersection(forward_returns.index)
        common_cols = factor_df.columns.intersection(forward_returns.columns)
        
        f_aligned = factor_df.loc[common_idx, common_cols]
        r_aligned = forward_returns.loc[common_idx, common_cols]
        
        f_rank = f_aligned.rank(axis=1)
        r_rank = r_aligned.rank(axis=1)
        ic_series = f_rank.corrwith(r_rank, axis=1).dropna()
        
        ic_mean = ic_series.mean()
        ic_std = ic_series.std()
        n = len(ic_series)
        t_stat = ic_mean / (ic_std / np.sqrt(n))
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-1))
        
        sig = "YES ***" if p_value < 0.05 else "NO"
        print(f"{name:<8} {ic_mean:>+10.5f} {t_stat:>+10.2f} {p_value:>12.6f} {sig:>12}")
    
    return engine


if __name__ == "__main__":
    run_orthogonal_factors()
