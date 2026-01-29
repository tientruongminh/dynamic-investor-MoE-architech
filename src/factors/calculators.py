"""
Factors Module - Alpha Factor Calculations
==========================================
Single Responsibility: Calculate alpha factors from price/return data

Classes:
    - FactorCalculator: Abstract base for factor calculators
    - BTPCalculator: Bayesian Trend Persistence
    - RIECalculator: Relative Information Entropy
    - MRDCalculator: Mahalanobis Regime Distance
    - KMDCalculator: Kernel Momentum Decay
    - RSVSCalculator: Robust Skewness Volatility Score
    - FactorEngine: Orchestrates all factor calculations
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass
import time


@dataclass
class FactorConfig:
    """
    Configuration for factor calculations.
    
    Open/Closed: Add new parameters without modifying existing code.
    """
    btp_window: int = 20
    rie_window: int = 20
    rie_bins: int = 10
    mrd_window: int = 60
    kmd_window: int = 60
    kmd_gamma: float = 0.1
    rsvs_window: int = 60


class FactorCalculator(ABC):
    """
    Abstract base for factor calculators.
    
    Interface Segregation: Single method interface.
    """
    
    @abstractmethod
    def calculate(self, prices: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
        """Calculate factor values."""
        pass


class BTPCalculator(FactorCalculator):
    """
    Bayesian Trend Persistence (BTP)
    
    Measures statistical significance of price trends using t-statistic.
    High BTP = strong persistent trend (Momentum).
    
    Theory: Bayesian t-test (Bishop, 2006)
    """
    
    def __init__(self, window: int = 20):
        self.window = window
    
    def calculate(self, prices: pd.DataFrame, returns: pd.DataFrame = None) -> pd.DataFrame:
        log_prices = np.log(prices.values + 1e-10)
        n_days, n_stocks = log_prices.shape
        
        # Pre-compute regression components
        t = np.arange(self.window).reshape(-1, 1)
        t_mean = t.mean()
        t_centered = t - t_mean
        t_var = (t_centered ** 2).sum()
        
        btp = np.full((n_days, n_stocks), np.nan)
        
        for i in range(self.window, n_days):
            y = log_prices[i-self.window:i, :]
            y_mean = y.mean(axis=0)
            y_centered = y - y_mean
            
            cov_ty = (t_centered * y_centered).sum(axis=0)
            slope = cov_ty / t_var
            
            y_pred = y_mean + slope * (t - t_mean).flatten().reshape(-1, 1)
            residuals = y - y_pred
            sse = (residuals ** 2).sum(axis=0)
            mse = sse / (self.window - 2)
            se_slope = np.sqrt(mse / t_var)
            
            with np.errstate(divide='ignore', invalid='ignore'):
                t_stat = slope / (se_slope + 1e-10)
            
            btp[i, :] = t_stat
        
        return pd.DataFrame(btp, index=prices.index, columns=prices.columns)


class RIECalculator(FactorCalculator):
    """
    Relative Information Entropy (RIE)
    
    Shannon Entropy of return distribution.
    High RIE = unpredictable returns (higher risk).
    
    Theory: Shannon Entropy, Maximum Entropy Principle (Bishop Ch.1.6)
    """
    
    def __init__(self, window: int = 20, n_bins: int = 10):
        self.window = window
        self.n_bins = n_bins
    
    def calculate(self, prices: pd.DataFrame, returns: pd.DataFrame = None) -> pd.DataFrame:
        if returns is None:
            returns = prices.pct_change()
        
        returns_np = returns.values
        n_days, n_stocks = returns_np.shape
        bins = np.linspace(-0.15, 0.15, self.n_bins + 1)
        
        rie = np.full((n_days, n_stocks), np.nan)
        
        for i in range(self.window, n_days):
            window_returns = returns_np[i-self.window:i, :]
            
            for j in range(n_stocks):
                r = window_returns[:, j]
                r = r[~np.isnan(r)]
                
                if len(r) < self.window // 2:
                    continue
                
                hist, _ = np.histogram(r, bins=bins, density=False)
                hist = hist / (hist.sum() + 1e-10)
                hist = hist[hist > 0]
                
                entropy = -np.sum(hist * np.log(hist + 1e-10))
                rie[i, j] = entropy
        
        return pd.DataFrame(rie, index=prices.index, columns=prices.columns)


class MRDCalculator(FactorCalculator):
    """
    Mahalanobis Regime Distance (MRD) - Simplified
    
    Z-score distance from sector mean return.
    High MRD = outperforming sector peers.
    
    Theory: Mahalanobis Distance (simplified for performance)
    """
    
    def __init__(self, window: int = 60, info: Dict = None):
        self.window = window
        self.info = info or {}
    
    def calculate(self, prices: pd.DataFrame, returns: pd.DataFrame = None) -> pd.DataFrame:
        if returns is None:
            returns = prices.pct_change()
        
        returns_np = returns.values
        n_days, n_stocks = returns_np.shape
        
        # Build sector map
        sector_map = {}
        for idx, ticker in enumerate(prices.columns):
            sector = self.info.get(ticker, {}).get('sector', 'Unknown')
            if sector not in sector_map:
                sector_map[sector] = []
            sector_map[sector].append(idx)
        
        mrd = np.full((n_days, n_stocks), np.nan)
        
        for i in range(self.window, n_days):
            window_returns = returns_np[i-self.window:i, :]
            
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
        
        return pd.DataFrame(mrd, index=prices.index, columns=prices.columns)


class KMDCalculator(FactorCalculator):
    """
    Kernel Momentum Decay (KMD)
    
    Gaussian kernel weighted momentum.
    Recent returns weighted more heavily than distant returns.
    
    Theory: Gaussian Kernel Methods (Bishop Ch.6)
    """
    
    def __init__(self, window: int = 60, gamma: float = 0.1):
        self.window = window
        self.gamma = gamma
    
    def calculate(self, prices: pd.DataFrame, returns: pd.DataFrame = None) -> pd.DataFrame:
        if returns is None:
            returns = prices.pct_change()
        
        returns_np = returns.values
        n_days, n_stocks = returns_np.shape
        
        # Pre-compute kernel weights
        t = np.arange(self.window)
        weights = np.exp(-self.gamma * (self.window - 1 - t))
        weights = weights / weights.sum()
        weights = weights.reshape(-1, 1)
        
        kmd = np.full((n_days, n_stocks), np.nan)
        
        for i in range(self.window, n_days):
            window_returns = returns_np[i-self.window:i, :]
            window_returns_filled = np.nan_to_num(window_returns, 0)
            weighted_mom = (weights * window_returns_filled).sum(axis=0)
            kmd[i, :] = weighted_mom
        
        return pd.DataFrame(kmd, index=prices.index, columns=prices.columns)


class RSVSCalculator(FactorCalculator):
    """
    Robust Skewness Volatility Score (RSVS)
    
    Ratio of upside vs downside moves.
    Positive RSVS = more/bigger upside moves.
    
    Theory: Robust Skewness (resistant to fat tails)
    """
    
    def __init__(self, window: int = 60):
        self.window = window
    
    def calculate(self, prices: pd.DataFrame, returns: pd.DataFrame = None) -> pd.DataFrame:
        if returns is None:
            returns = prices.pct_change()
        
        returns_np = returns.values
        n_days, n_stocks = returns_np.shape
        
        rsvs = np.full((n_days, n_stocks), np.nan)
        
        for i in range(self.window, n_days):
            window_returns = returns_np[i-self.window:i, :]
            
            for j in range(n_stocks):
                r = window_returns[:, j]
                r = r[~np.isnan(r)]
                
                if len(r) < self.window // 2:
                    continue
                
                pos = r[r > 0]
                neg = r[r < 0]
                
                if len(pos) > 0 and len(neg) > 0:
                    avg_pos = np.mean(np.abs(pos))
                    avg_neg = np.mean(np.abs(neg))
                    freq_ratio = len(pos) / len(neg)
                    
                    skew = (avg_pos / (avg_neg + 1e-10) - 1) * np.log(freq_ratio + 1)
                    rsvs[i, j] = skew
        
        return pd.DataFrame(rsvs, index=prices.index, columns=prices.columns)


class FactorEngine:
    """
    Orchestrates all factor calculations.
    
    Facade Pattern: Single interface to all factor calculators.
    Dependency Inversion: Depends on FactorCalculator interface.
    """
    
    def __init__(self, config: FactorConfig = None, info: Dict = None):
        self.config = config or FactorConfig()
        self.info = info or {}
        
        # Initialize calculators
        self.calculators = {
            'BTP': BTPCalculator(self.config.btp_window),
            'RIE': RIECalculator(self.config.rie_window, self.config.rie_bins),
            'MRD': MRDCalculator(self.config.mrd_window, self.info),
            'KMD': KMDCalculator(self.config.kmd_window, self.config.kmd_gamma),
            'RSVS': RSVSCalculator(self.config.rsvs_window)
        }
        
        self.factors = {}
        self.standardized = {}
    
    def calculate_all(self, prices: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Calculate all factors."""
        print("\n" + "="*60)
        print("FACTOR CALCULATIONS")
        print("="*60)
        
        returns = prices.pct_change()
        total_start = time.time()
        
        for name, calculator in self.calculators.items():
            start = time.time()
            print(f"  Calculating {name}...", end=" ")
            
            self.factors[name] = calculator.calculate(prices, returns)
            
            elapsed = time.time() - start
            print(f"done ({elapsed:.1f}s)")
        
        total_elapsed = time.time() - total_start
        print(f"\nTotal: {total_elapsed:.1f}s")
        print("="*60)
        
        return self.factors
    
    def standardize(self) -> Dict[str, pd.DataFrame]:
        """Cross-sectional standardization (Z-score by row)."""
        for name, df in self.factors.items():
            mean = df.mean(axis=1)
            std = df.std(axis=1)
            self.standardized[name] = df.sub(mean, axis=0).div(std + 1e-10, axis=0)
        
        return self.standardized
    
    def get_exposures(self) -> pd.DataFrame:
        """Get latest factor exposures for all stocks."""
        if not self.standardized:
            self.standardize()
        
        exposures = pd.DataFrame(index=list(self.factors.values())[0].columns)
        
        for name, df in self.standardized.items():
            exposures[name] = df.iloc[-1]
        
        return exposures
    
    def get_composite_score(self) -> pd.Series:
        """Get composite score (equal-weighted average of all factors)."""
        exposures = self.get_exposures()
        return exposures.mean(axis=1).sort_values(ascending=False)
    
    def save(self, output_dir: str = "./outputs/factors"):
        """Save all factors to disk."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for name, df in self.factors.items():
            df.to_csv(os.path.join(output_dir, f"{name}_raw.csv"))
        
        for name, df in self.standardized.items():
            df.to_csv(os.path.join(output_dir, f"{name}_standardized.csv"))
        
        exposures = self.get_exposures()
        exposures.to_csv(os.path.join(output_dir, "factor_exposures.csv"))
        
        print(f"Factors saved to {output_dir}")
