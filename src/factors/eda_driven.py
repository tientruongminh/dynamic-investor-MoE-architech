"""
EDA-Driven Factor Design
=========================
Mỗi factor được thiết kế dựa trên một phát hiện cụ thể từ EDA.

EDA Finding → Factor Design:
1. Fat Tails (K=15.92) → SKEW, KURT, TAIL_RISK (robust to extremes)
2. Mean Reversion (IC âm cho momentum) → REV_5D, REV_20D
3. Sector Clustering → MRD, SECTOR_MOM
4. Non-stationarity → Dùng Returns không dùng Prices
5. Outliers (+347%, -85%) → VOL, DRAWDOWN (risk factors)
6. Risk-on Clusters → BETA, CORR_MARKET
7. Entropy insight → RIE (uncertainty premium)

Tổng: 12 factors chia thành 4 nhóm độc lập
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import time
from scipy import stats


class EDADrivenFactorEngine:
    """
    Factor engine designed from EDA insights.
    Each factor addresses a specific EDA finding.
    """
    
    def __init__(self, prices: pd.DataFrame, info: Dict = None):
        self.prices = prices
        self.returns = prices.pct_change()
        self.log_returns = np.log(prices / prices.shift(1))
        self.info = info or {}
        self.factors = {}
        
        # Build sector map
        self.sector_map = {}
        self.stock_sector = {}
        for idx, ticker in enumerate(prices.columns):
            sector = self.info.get(ticker, {}).get('sector', 'Unknown')
            if sector not in self.sector_map:
                self.sector_map[sector] = []
            self.sector_map[sector].append(idx)
            self.stock_sector[ticker] = sector
    
    # ========== GROUP 1: MOMENTUM & REVERSAL ==========
    # EDA Finding: BTP/KMD showed MEAN REVERSION, not momentum
    
    def calc_mom_12m_1m(self, window: int = 252, skip: int = 21) -> pd.DataFrame:
        """
        12-Month Momentum (skip recent month).
        Classic Jegadeesh-Titman momentum factor.
        EDA Insight: Long-term momentum may still work.
        """
        cum_ret = (1 + self.returns).rolling(window=window).apply(lambda x: x.prod() - 1, raw=True)
        recent_ret = (1 + self.returns).rolling(window=skip).apply(lambda x: x.prod() - 1, raw=True)
        return cum_ret - recent_ret
    
    def calc_rev_5d(self) -> pd.DataFrame:
        """
        5-Day Reversal Factor.
        EDA Insight: Short-term momentum factors (BTP, KMD) had NEGATIVE IC.
        → Short-term REVERSAL should have POSITIVE IC.
        """
        return -self.returns.rolling(window=5).sum()
    
    def calc_rev_20d(self) -> pd.DataFrame:
        """
        20-Day Reversal Factor.
        EDA Insight: Medium-term mean reversion.
        """
        return -self.returns.rolling(window=20).sum()
    
    # ========== GROUP 2: RISK FACTORS ==========
    # EDA Finding: Fat Tails (Kurtosis=15.92), Extreme Outliers
    
    def calc_vol(self, window: int = 20) -> pd.DataFrame:
        """
        Realized Volatility (NEGATIVE).
        EDA Insight: Low Volatility Anomaly - low vol stocks outperform.
        """
        vol = self.returns.rolling(window=window).std() * np.sqrt(252)
        return -vol  # Negative so low vol = high score
    
    def calc_skew(self, window: int = 60) -> pd.DataFrame:
        """
        Return Skewness.
        EDA Insight: Fat tails with varying skewness across stocks.
        Positive skew = more upside potential.
        """
        return self.returns.rolling(window=window).skew()
    
    def calc_kurt(self, window: int = 60) -> pd.DataFrame:
        """
        Excess Kurtosis (NEGATIVE).
        EDA Insight: High kurtosis = fat tails = extreme risk.
        Low kurtosis stocks are "safer".
        """
        kurt = self.returns.rolling(window=window).kurt()
        return -kurt  # Negative so low kurtosis = high score
    
    def calc_drawdown(self, window: int = 252) -> pd.DataFrame:
        """
        Maximum Drawdown (NEGATIVE).
        EDA Insight: Outliers like -85% in 1 day indicate drawdown risk.
        """
        rolling_max = self.prices.rolling(window=window).max()
        drawdown = (self.prices - rolling_max) / rolling_max
        max_dd = drawdown.rolling(window=window).min()
        return -max_dd  # Negative DD is bad, so negate to get "lower DD = higher score"
    
    # ========== GROUP 3: SECTOR/RELATIVE FACTORS ==========
    # EDA Finding: Sector imbalance, Tech dominance, Risk-on clusters
    
    def calc_mrd(self, window: int = 60) -> pd.DataFrame:
        """
        Sector-Relative Momentum (MRD).
        EDA Insight: This was the ONLY significant factor (p<0.05)!
        Measures outperformance vs sector peers.
        """
        returns_np = self.returns.values
        n_days, n_stocks = returns_np.shape
        
        mrd = np.full((n_days, n_stocks), np.nan)
        
        for i in range(window, n_days):
            window_returns = returns_np[i-window:i, :]
            
            sector_means = {}
            sector_stds = {}
            for sector, indices in self.sector_map.items():
                sector_ret = window_returns[:, indices]
                sector_means[sector] = np.nanmean(sector_ret)
                sector_stds[sector] = np.nanstd(sector_ret) + 1e-10
            
            for sector, indices in self.sector_map.items():
                for idx in indices:
                    stock_ret = np.nanmean(window_returns[:, idx])
                    z_score = (stock_ret - sector_means[sector]) / sector_stds[sector]
                    mrd[i, idx] = z_score
        
        return pd.DataFrame(mrd, index=self.prices.index, columns=self.prices.columns)
    
    def calc_beta(self, window: int = 252) -> pd.DataFrame:
        """
        Market Beta (NEGATIVE for low-beta anomaly).
        EDA Insight: Risk-on clusters (Semi+Defense) have high beta.
        Low beta stocks tend to outperform risk-adjusted.
        """
        # Use equal-weighted market return
        market_ret = self.returns.mean(axis=1)
        
        betas = pd.DataFrame(index=self.returns.index, columns=self.returns.columns)
        
        for col in self.returns.columns:
            stock_ret = self.returns[col]
            rolling_cov = stock_ret.rolling(window=window).cov(market_ret)
            rolling_var = market_ret.rolling(window=window).var()
            betas[col] = rolling_cov / (rolling_var + 1e-10)
        
        return -betas.astype(float)  # Negative so low beta = high score
    
    # ========== GROUP 4: INFORMATION/ENTROPY FACTORS ==========
    # EDA Finding: RIE was independent (low correlation with others)
    
    def calc_rie(self, window: int = 20, n_bins: int = 10) -> pd.DataFrame:
        """
        Relative Information Entropy.
        EDA Insight: Independent factor measuring uncertainty.
        High entropy = unpredictable = risk premium.
        """
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
        
        return pd.DataFrame(rie, index=self.prices.index, columns=self.prices.columns)
    
    def calc_idiovol(self, window: int = 60) -> pd.DataFrame:
        """
        Idiosyncratic Volatility (NEGATIVE).
        EDA Insight: Residual risk after removing market factor.
        Low idiosyncratic vol = more predictable.
        """
        market_ret = self.returns.mean(axis=1)
        
        idiovol = pd.DataFrame(index=self.returns.index, columns=self.returns.columns)
        
        for col in self.returns.columns:
            stock_ret = self.returns[col]
            # Residual = stock - beta * market
            rolling_cov = stock_ret.rolling(window=window).cov(market_ret)
            rolling_var = market_ret.rolling(window=window).var()
            beta = rolling_cov / (rolling_var + 1e-10)
            residual = stock_ret - beta * market_ret
            idiovol[col] = residual.rolling(window=window).std()
        
        return -idiovol.astype(float)  # Negative so low idiovol = high score
    
    def calc_price_range(self, window: int = 20) -> pd.DataFrame:
        """
        Price Range (High-Low) normalized.
        EDA Insight: Stocks with wide ranges are more volatile.
        """
        rolling_high = self.prices.rolling(window=window).max()
        rolling_low = self.prices.rolling(window=window).min()
        range_pct = (rolling_high - rolling_low) / self.prices
        return -range_pct  # Negative so narrow range = high score
    
    def calc_consistency(self, window: int = 60) -> pd.DataFrame:
        """
        Return Consistency - % of positive days.
        EDA Insight: Stocks with more positive days are "quality" stocks.
        """
        return (self.returns > 0).rolling(window=window).mean()
    
    # ========== MAIN METHODS ==========
    
    def calculate_all(self) -> Dict[str, pd.DataFrame]:
        """Calculate all EDA-driven factors."""
        print("\n" + "="*70)
        print("EDA-DRIVEN FACTOR ENGINE - 12 FACTORS")
        print("="*70)
        
        total_start = time.time()
        
        # Group 1: Momentum & Reversal
        print("\nGroup 1: MOMENTUM & REVERSAL (EDA: Mean reversion detected)")
        start = time.time()
        self.factors['MOM_12M'] = self.calc_mom_12m_1m()
        print(f"  MOM_12M done ({time.time()-start:.1f}s)")
        
        start = time.time()
        self.factors['REV_5D'] = self.calc_rev_5d()
        print(f"  REV_5D done ({time.time()-start:.1f}s)")
        
        start = time.time()
        self.factors['REV_20D'] = self.calc_rev_20d()
        print(f"  REV_20D done ({time.time()-start:.1f}s)")
        
        # Group 2: Risk
        print("\nGroup 2: RISK (EDA: Fat tails, Kurtosis=15.92)")
        start = time.time()
        self.factors['VOL'] = self.calc_vol()
        print(f"  VOL done ({time.time()-start:.1f}s)")
        
        start = time.time()
        self.factors['SKEW'] = self.calc_skew()
        print(f"  SKEW done ({time.time()-start:.1f}s)")
        
        start = time.time()
        self.factors['KURT'] = self.calc_kurt()
        print(f"  KURT done ({time.time()-start:.1f}s)")
        
        start = time.time()
        self.factors['DRAWDOWN'] = self.calc_drawdown()
        print(f"  DRAWDOWN done ({time.time()-start:.1f}s)")
        
        # Group 3: Relative
        print("\nGroup 3: SECTOR/RELATIVE (EDA: Sector clusters)")
        start = time.time()
        self.factors['MRD'] = self.calc_mrd()
        print(f"  MRD done ({time.time()-start:.1f}s)")
        
        start = time.time()
        self.factors['BETA'] = self.calc_beta()
        print(f"  BETA done ({time.time()-start:.1f}s)")
        
        # Group 4: Information
        print("\nGroup 4: INFORMATION (EDA: RIE was independent)")
        start = time.time()
        self.factors['RIE'] = self.calc_rie()
        print(f"  RIE done ({time.time()-start:.1f}s)")
        
        start = time.time()
        self.factors['IDIOVOL'] = self.calc_idiovol()
        print(f"  IDIOVOL done ({time.time()-start:.1f}s)")
        
        start = time.time()
        self.factors['CONSISTENCY'] = self.calc_consistency()
        print(f"  CONSISTENCY done ({time.time()-start:.1f}s)")
        
        print(f"\nTotal: {time.time()-total_start:.1f}s for 12 factors")
        return self.factors
    
    def standardize(self) -> Dict[str, pd.DataFrame]:
        """Cross-sectional Z-score."""
        standardized = {}
        for name, df in self.factors.items():
            mean = df.mean(axis=1)
            std = df.std(axis=1) + 1e-10
            standardized[name] = df.sub(mean, axis=0).div(std, axis=0)
        return standardized
    
    def get_correlation_matrix(self) -> pd.DataFrame:
        """Factor correlation matrix."""
        latest = pd.DataFrame()
        for name, df in self.factors.items():
            series = df.iloc[-252:].mean()
            if not series.isna().all():
                latest[name] = series
        return latest.corr()
    
    def save(self, output_dir: str = "./outputs/factors_eda"):
        """Save all factors."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for name, df in self.factors.items():
            df.to_csv(os.path.join(output_dir, f"{name}.csv"))
        
        std_factors = self.standardize()
        for name, df in std_factors.items():
            df.to_csv(os.path.join(output_dir, f"{name}_std.csv"))
        
        corr = self.get_correlation_matrix()
        corr.to_csv(os.path.join(output_dir, "correlation_matrix.csv"))
        
        print(f"\nFactors saved to {output_dir}")


def run_eda_factors(
    prices_file: str = "./outputs_clean/clean/processed_prices.csv",
    info_dir: str = "./data/nasdaq_574",
    output_dir: str = "./outputs/factors_eda"
):
    """Run EDA-driven factor calculation and validation."""
    import json
    import os
    
    print("="*70)
    print("EDA-DRIVEN FACTOR ENGINE")
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
    
    print(f"Loaded: {prices.shape[1]} stocks, {prices.shape[0]} days, {len(info)} info files")
    
    # Calculate factors
    engine = EDADrivenFactorEngine(prices, info=info)
    factors = engine.calculate_all()
    
    # Save
    engine.save(output_dir)
    
    # Correlation matrix
    print("\n" + "="*70)
    print("FACTOR CORRELATION MATRIX")
    print("="*70)
    corr = engine.get_correlation_matrix()
    print(corr.round(2).to_string())
    
    # Validate significance
    print("\n" + "="*70)
    print("STATISTICAL SIGNIFICANCE TEST (IC ≠ 0)")
    print("="*70)
    
    returns = prices.pct_change()
    forward_returns = returns.shift(-5)
    
    results = []
    for name, factor_df in factors.items():
        common_idx = factor_df.index.intersection(forward_returns.index)
        common_cols = factor_df.columns.intersection(forward_returns.columns)
        
        f_aligned = factor_df.loc[common_idx, common_cols]
        r_aligned = forward_returns.loc[common_idx, common_cols]
        
        f_rank = f_aligned.rank(axis=1)
        r_rank = r_aligned.rank(axis=1)
        ic_series = f_rank.corrwith(r_rank, axis=1).dropna()
        
        if len(ic_series) < 100:
            continue
            
        ic_mean = ic_series.mean()
        ic_std = ic_series.std()
        n = len(ic_series)
        t_stat = ic_mean / (ic_std / np.sqrt(n))
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-1))
        ir = ic_mean / ic_std * np.sqrt(252)
        
        results.append({
            'Factor': name,
            'IC_Mean': ic_mean,
            't_stat': t_stat,
            'p_value': p_value,
            'IR': ir,
            'Significant': p_value < 0.05
        })
    
    results_df = pd.DataFrame(results).sort_values('p_value')
    
    print(f"\n{'Factor':<12} {'IC Mean':>10} {'t-stat':>10} {'p-value':>12} {'IR':>8} {'Sig':>8}")
    print("-"*70)
    for _, r in results_df.iterrows():
        sig = "***" if r['Significant'] else ""
        print(f"{r['Factor']:<12} {r['IC_Mean']:>+10.5f} {r['t_stat']:>+10.2f} {r['p_value']:>12.6f} {r['IR']:>+8.2f} {sig:>8}")
    
    sig_count = results_df['Significant'].sum()
    print(f"\n{sig_count}/{len(results_df)} factors are statistically significant (p < 0.05)")
    
    return engine, results_df


if __name__ == "__main__":
    engine, results = run_eda_factors()
