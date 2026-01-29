"""
Modeling Module - Chapter 6
============================
Implements modeling case studies:
- Lasso Expert for Feature Selection
- Anchor Analysis (Growth vs Value)
- Factor Attribution

Theory: Sparse modeling (Lasso) identifies the true drivers of alpha
by eliminating noise in high-dimensional factor space.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Dict, List, Tuple
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

class LassoExpert:
    """
    Applies Lasso Regression to identify significant alpha factors.
    """
    
    def __init__(self, factors: Dict[str, pd.DataFrame], returns: pd.DataFrame):
        self.factors = factors
        self.returns = returns
        
    def fit_model(self, ticker: str, forward_period: int = 5) -> pd.Series:
        """
        Fit Lasso model for a specific ticker to find its drivers.
        
        Args:
            ticker: Stock symbol
            forward_period: Target return horizon
            
        Returns:
            Series of coefficients for each factor
        """
        # Align data
        y = self.returns[ticker].shift(-forward_period).dropna()
        
        X_data = {}
        for name, df in self.factors.items():
            if ticker in df.columns:
                X_data[name] = df[ticker]
        
        X = pd.DataFrame(X_data).dropna()
        
        # Intersection of indices
        common_index = X.index.intersection(y.index)
        if len(common_index) < 100:
            return pd.Series(0, index=self.factors.keys())
            
        X = X.loc[common_index]
        y = y.loc[common_index]
        
        # Fit LassoCV (auto-tune alpha)
        # Normalize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = LassoCV(cv=5, max_iter=2000, random_state=42)
        model.fit(X_scaled, y)
        
        return pd.Series(model.coef_, index=X.columns)

    def run_case_study(self, tickers: List[str]) -> pd.DataFrame:
        """Run Lasso analysis for a list of case study tickers."""
        results = {}
        for t in tickers:
            if t in self.returns.columns:
                coefs = self.fit_model(t)
                results[t] = coefs
        return pd.DataFrame(results)

class AnchorAnalyzer:
    """
    Comparative analysis between different stock profiles (Growth vs Value).
    """
    
    def __init__(self, factors: Dict[str, pd.DataFrame]):
        self.factors = factors
        
    def get_factor_profile(self, ticker: str) -> pd.Series:
        """Get average standardized factor exposure for a ticker."""
        profile = {}
        for name, df in self.factors.items():
            if ticker in df.columns:
                # Use last year average for robust profile
                exp = df[ticker].iloc[-252:].mean()
                profile[name] = exp
            else:
                profile[name] = np.nan
        return pd.Series(profile)

    def compare_anchors(self, growth_ticker: str = 'NVDA', value_ticker: str = 'KO') -> pd.DataFrame:
        """Compare factor profiles of two anchor stocks."""
        g_prof = self.get_factor_profile(growth_ticker)
        v_prof = self.get_factor_profile(value_ticker)
        
        df = pd.DataFrame({
            growth_ticker: g_prof,
            value_ticker: v_prof
        })
        return df

class ModelingPlotter:
    """Generates modeling visualizations."""
    
    def __init__(self, output_dir: str = "./outputs/modeling"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def plot_lasso_coefs(self, coefs_df: pd.DataFrame, filename: str = "lasso_coefs.png"):
        """Plot Lasso coefficients heatmap."""
        plt.figure(figsize=(10, 6))
        sns.heatmap(coefs_df, annot=True, cmap='RdBu_r', center=0, fmt='.3f')
        plt.title('Lasso Feature Importance (Coefficient Magnitude)', fontsize=14, fontweight='bold')
        plt.ylabel('Alpha Factor')
        plt.xlabel('Case Study Ticker')
        plt.tight_layout()
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, dpi=150)
        plt.close()
        return path
        
    def plot_anchor_comparison(self, profile_df: pd.DataFrame, filename: str = "anchor_comparison.png"):
        """Plot radar or bar chart comparison."""
        # Simple bar comparison
        profile_df.plot(kind='bar', figsize=(10, 6), width=0.8)
        plt.title(f'Factor Profile Comparison: {profile_df.columns[0]} vs {profile_df.columns[1]}', fontsize=14, fontweight='bold')
        plt.ylabel('Average Z-Score Exposure (1 Year)')
        plt.axhline(0, color='black', linewidth=0.5)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, dpi=150)
        plt.close()
        return path

def run_modeling_analysis(
    prices_file: str = "./outputs/clean/processed_prices.csv",
    factors_dir: str = "./outputs/factors",
    output_dir: str = "./outputs/modeling",
    case_study_tickers: List[str] = ['NVDA', 'KO', 'TSLA', 'JNJ']
) -> Dict:
    """Run complete modeling analysis."""
    print("\n" + "="*60)
    print("CHAPTER 6: MODELING CASE STUDY")
    print("="*60)
    
    # Load data
    print("Loading data...")
    prices = pd.read_csv(prices_file, index_col=0, parse_dates=True)
    returns = prices.pct_change()
    
    factors = {}
    for name in ['BTP', 'RIE', 'MRD', 'KMD', 'RSVS']:
        path = os.path.join(factors_dir, f"{name}_standardized.csv")
        if os.path.exists(path):
            factors[name] = pd.read_csv(path, index_col=0, parse_dates=True)
            
    # Normalize tickers (ensure they exist)
    valid_tickers = [t for t in case_study_tickers if t in prices.columns]
    print(f"Case Study Tickers: {valid_tickers}")
    
    expert = LassoExpert(factors, returns)
    anchor = AnchorAnalyzer(factors)
    plotter = ModelingPlotter(output_dir)
    
    # 1. Lasso Analysis
    print("1. Running Lasso Feature Selection...")
    lasso_res = expert.run_case_study(valid_tickers)
    print(lasso_res)
    
    # 2. Anchor Comparison (first two valid)
    print("2. Running Anchor Analysis...")
    if len(valid_tickers) >= 2:
        anchor_res = anchor.compare_anchors(valid_tickers[0], valid_tickers[1])
    else:
        anchor_res = pd.DataFrame()
        
    # 3. Visualizations
    print("3. Generating Visualizations...")
    plots = {
        'lasso': plotter.plot_lasso_coefs(lasso_res),
        'anchor': plotter.plot_anchor_comparison(anchor_res)
    }
    
    print(f"Plots saved to {output_dir}")
    
    return {
        'lasso_coefs': lasso_res,
        'anchor_profile': anchor_res,
        'plots': plots
    }

if __name__ == "__main__":
    run_modeling_analysis()
