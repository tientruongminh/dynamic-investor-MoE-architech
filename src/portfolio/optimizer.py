"""
Black-Litterman Portfolio Optimizer
===================================
Kết hợp Market Equilibrium với MoE Investor Views.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from scipy.optimize import minimize
import os


class BlackLittermanOptimizer:
    """
    Black-Litterman Portfolio Optimization.
    
    Combines:
    - Market equilibrium returns (from market cap weights)
    - Investor views (from MoE predictions)
    """
    
    def __init__(
        self,
        risk_aversion: float = 2.5,  # γ (gamma)
        tau: float = 0.05,            # Uncertainty in equilibrium
        risk_free_rate: float = 0.045  # 4.5% annual
    ):
        self.gamma = risk_aversion
        self.tau = tau
        self.rf = risk_free_rate / 252  # Daily
    
    def get_equilibrium_returns(
        self,
        covariance: pd.DataFrame,
        market_caps: pd.Series
    ) -> pd.Series:
        """
        Reverse optimization: Infer equilibrium returns from market weights.
        
        π = γ * Σ * w_mkt
        """
        # Market weights
        w_mkt = market_caps / market_caps.sum()
        
        # Align
        common = covariance.columns.intersection(w_mkt.index)
        w_mkt = w_mkt.loc[common]
        cov = covariance.loc[common, common]
        
        # Equilibrium returns
        pi = self.gamma * cov @ w_mkt
        
        return pi
    
    def incorporate_views(
        self,
        pi: pd.Series,
        covariance: pd.DataFrame,
        views: Dict[str, float],  # {ticker: expected_return}
        view_confidence: float = 0.5  # 0 to 1
    ) -> pd.Series:
        """
        Combine equilibrium returns with investor views.
        
        μ_BL = [(τΣ)^-1 + P'Ω^-1P]^-1 * [(τΣ)^-1π + P'Ω^-1Q]
        """
        tickers = pi.index.tolist()
        n = len(tickers)
        
        # Filter views to only include stocks in our universe
        valid_views = {k: v for k, v in views.items() if k in tickers}
        
        if not valid_views:
            return pi
        
        # Create P matrix (pick matrix) and Q vector (expected returns from views)
        k = len(valid_views)
        P = np.zeros((k, n))
        Q = np.zeros(k)
        
        for i, (ticker, expected_ret) in enumerate(valid_views.items()):
            j = tickers.index(ticker)
            P[i, j] = 1.0
            Q[i] = expected_ret
        
        # Omega (view uncertainty) - diagonal matrix
        # Higher confidence -> lower uncertainty
        omega_diag = (1 - view_confidence) * np.diag(covariance.loc[list(valid_views.keys()), list(valid_views.keys())].values).diagonal()
        omega_diag = np.maximum(omega_diag, 1e-6)
        omega_inv = np.diag(1.0 / omega_diag)
        
        # Covariance
        Sigma = covariance.loc[tickers, tickers].values
        tau_Sigma_inv = np.linalg.inv(self.tau * Sigma)
        
        # Posterior returns
        M = np.linalg.inv(tau_Sigma_inv + P.T @ omega_inv @ P)
        mu_bl = M @ (tau_Sigma_inv @ pi.values + P.T @ omega_inv @ Q)
        
        return pd.Series(mu_bl, index=tickers)
    
    def optimize_weights(
        self,
        expected_returns: pd.Series,
        covariance: pd.DataFrame,
        max_weight: float = 0.10,
        min_weight: float = 0.0
    ) -> pd.Series:
        """
        Mean-Variance Optimization with constraints.
        """
        tickers = expected_returns.index.tolist()
        n = len(tickers)
        
        mu = expected_returns.values
        Sigma = covariance.loc[tickers, tickers].values
        
        # Objective: minimize -μ'w + (γ/2)*w'Σw
        def objective(w):
            return -w @ mu + (self.gamma / 2) * w @ Sigma @ w
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}  # Sum to 1
        ]
        
        # Bounds
        bounds = [(min_weight, max_weight) for _ in range(n)]
        
        # Initial guess
        w0 = np.ones(n) / n
        
        # Optimize
        result = minimize(
            objective,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            weights = pd.Series(result.x, index=tickers)
        else:
            print("Optimization failed, using equal weights")
            weights = pd.Series(np.ones(n) / n, index=tickers)
        
        return weights
    
    def optimize(
        self,
        covariance: pd.DataFrame,
        market_caps: pd.Series,
        views: Dict[str, float],
        view_confidence: float = 0.5,
        max_weight: float = 0.10
    ) -> pd.Series:
        """
        Full Black-Litterman optimization pipeline.
        """
        # Step 1: Equilibrium returns
        pi = self.get_equilibrium_returns(covariance, market_caps)
        
        # Step 2: Incorporate views
        mu_bl = self.incorporate_views(pi, covariance, views, view_confidence)
        
        # Step 3: Optimize
        weights = self.optimize_weights(mu_bl, covariance, max_weight)
        
        return weights


class PortfolioConstraints:
    """
    Ràng buộc thực tế cho danh mục.
    """
    
    def __init__(
        self,
        transaction_cost: float = 0.001,  # 0.1%
        slippage: float = 0.0005,         # 0.05%
        max_turnover: float = 0.30,       # 30% per rebalance
        sector_limit: float = 0.25,       # Max 25% per sector
        min_holding: float = 0.001        # Min 0.1% per stock
    ):
        self.tc = transaction_cost
        self.slip = slippage
        self.max_turn = max_turnover
        self.sector_lim = sector_limit
        self.min_hold = min_holding
    
    def apply_turnover_limit(
        self,
        current_weights: pd.Series,
        target_weights: pd.Series
    ) -> pd.Series:
        """
        Giới hạn turnover.
        """
        # Align
        all_tickers = current_weights.index.union(target_weights.index)
        current = current_weights.reindex(all_tickers, fill_value=0)
        target = target_weights.reindex(all_tickers, fill_value=0)
        
        delta = target - current
        turnover = delta.abs().sum() / 2
        
        if turnover > self.max_turn:
            scale = self.max_turn / turnover
            new_weights = current + delta * scale
        else:
            new_weights = target
        
        # Normalize
        return new_weights / new_weights.sum()
    
    def apply_sector_neutral(
        self,
        weights: pd.Series,
        sector_map: Dict[str, str]
    ) -> pd.Series:
        """
        Giới hạn exposure per sector.
        """
        sectors = {}
        for ticker, weight in weights.items():
            sector = sector_map.get(ticker, 'Unknown')
            if sector not in sectors:
                sectors[sector] = []
            sectors[sector].append((ticker, weight))
        
        new_weights = weights.copy()
        
        for sector, holdings in sectors.items():
            sector_weight = sum(w for _, w in holdings)
            
            if sector_weight > self.sector_lim:
                scale = self.sector_lim / sector_weight
                for ticker, _ in holdings:
                    new_weights[ticker] *= scale
        
        # Normalize
        return new_weights / new_weights.sum()
    
    def apply_min_holding(self, weights: pd.Series) -> pd.Series:
        """
        Loại bỏ positions quá nhỏ.
        """
        # Zero out positions below threshold
        weights = weights.where(weights >= self.min_hold, 0)
        
        # Normalize
        if weights.sum() > 0:
            return weights / weights.sum()
        else:
            return weights
    
    def calculate_costs(
        self,
        current_weights: pd.Series,
        target_weights: pd.Series
    ) -> float:
        """
        Tính tổng chi phí giao dịch.
        """
        # Align
        all_tickers = current_weights.index.union(target_weights.index)
        current = current_weights.reindex(all_tickers, fill_value=0)
        target = target_weights.reindex(all_tickers, fill_value=0)
        
        turnover = (target - current).abs().sum() / 2
        return turnover * (self.tc + self.slip)
    
    def apply_all(
        self,
        current_weights: pd.Series,
        target_weights: pd.Series,
        sector_map: Dict[str, str] = None
    ) -> Tuple[pd.Series, float]:
        """
        Apply all constraints.
        
        Returns:
            (final_weights, transaction_cost)
        """
        weights = target_weights.copy()
        
        # Apply turnover limit
        weights = self.apply_turnover_limit(current_weights, weights)
        
        # Apply sector limit
        if sector_map:
            weights = self.apply_sector_neutral(weights, sector_map)
        
        # Apply min holding
        weights = self.apply_min_holding(weights)
        
        # Calculate costs
        cost = self.calculate_costs(current_weights, weights)
        
        return weights, cost


def test_black_litterman():
    """Test BL optimizer."""
    print("="*60)
    print("BLACK-LITTERMAN TEST")
    print("="*60)
    
    # Create sample data
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    n = len(tickers)
    
    # Random covariance matrix
    np.random.seed(42)
    A = np.random.randn(n, n)
    cov = pd.DataFrame(A @ A.T / 100, index=tickers, columns=tickers)
    
    # Market caps (billions)
    market_caps = pd.Series([3000, 2800, 2000, 1800, 1200], index=tickers)
    
    # Views from MoE
    views = {
        'AAPL': 0.001,   # Slightly bullish (0.1% daily expected)
        'META': 0.002,   # More bullish
        'AMZN': -0.0005  # Slightly bearish
    }
    
    # Optimize
    optimizer = BlackLittermanOptimizer()
    weights = optimizer.optimize(cov, market_caps, views, view_confidence=0.6)
    
    print("\nOptimal Weights:")
    for ticker, weight in weights.items():
        bar = "█" * int(weight * 100)
        print(f"  {ticker}: {weight:.1%} {bar}")
    
    return weights


if __name__ == "__main__":
    test_black_litterman()
