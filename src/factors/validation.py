"""
Alpha Validation Module - Chapter 4
====================================
Implements alpha factor validation metrics:
- Information Coefficient (IC)
- Information Ratio (IR)
- Decile/Quintile Returns Analysis
- Monotonicity Test
- Factor Decay Analysis

Theory: A valid alpha factor should predict future returns consistently.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os


class AlphaValidator:
    """
    Validates alpha factors using IC, IR, and Monotonicity tests.
    
    Key Metrics:
    - IC (Information Coefficient): Spearman correlation between factor and forward returns
    - IR (Information Ratio): IC_mean / IC_std (signal stability)
    - Monotonicity: Top decile should outperform bottom decile
    """
    
    def __init__(self, prices: pd.DataFrame, factors: Dict[str, pd.DataFrame]):
        """
        Initialize validator.
        
        Args:
            prices: Price data (index=dates, columns=tickers)
            factors: Dict of factor DataFrames {name: DataFrame}
        """
        self.prices = prices
        self.factors = factors
        self.returns = prices.pct_change()
        
        self.ic_results = {}
        self.ir_results = {}
        self.decile_results = {}
        
    def calculate_forward_returns(self, periods: List[int] = [1, 5, 20]) -> Dict[int, pd.DataFrame]:
        """Calculate forward returns at different horizons."""
        forward_returns = {}
        
        for period in periods:
            forward_returns[period] = self.returns.shift(-period)
        
        return forward_returns
    
    def calculate_ic(self, factor_name: str, forward_period: int = 5) -> pd.Series:
        """
        Calculate Information Coefficient (IC) time series using OPTIMIZED VECTORIZATION.
        
        IC = Spearman correlation between factor values and forward returns
        
        Args:
            factor_name: Name of factor to validate
            forward_period: Number of days forward for returns
            
        Returns:
            Series of daily IC values
        """
        factor_df = self.factors[factor_name]
        forward_returns = self.returns.shift(-forward_period)
        
        # Align dataframes
        common_index = factor_df.index.intersection(forward_returns.index)
        common_columns = factor_df.columns.intersection(forward_returns.columns)
        
        f_aligned = factor_df.loc[common_index, common_columns]
        r_aligned = forward_returns.loc[common_index, common_columns]
        
        # Rank cross-sectionally (axis=1)
        # Only rank valid numbers (handle NaNs implicitly)
        f_rank = f_aligned.rank(axis=1, numeric_only=True)
        r_rank = r_aligned.rank(axis=1, numeric_only=True)
        
        # Calculate daily correlation between ranks (Spearman IC)
        # corrwith(axis=1) computes row-wise correlation
        ic_series = f_rank.corrwith(r_rank, axis=1)
        
        # Name and store
        ic_series.name = f"{factor_name}_IC"
        
        # Store primary IC (usually 5-day)
        if forward_period == 5:
            self.ic_results[factor_name] = ic_series
        
        return ic_series
    
    def calculate_all_ic(self, forward_period: int = 5) -> pd.DataFrame:
        """Calculate IC for all factors."""
        all_ic = {}
        
        for factor_name in self.factors.keys():
            # print(f"  Calculating IC for {factor_name}...")
            ic = self.calculate_ic(factor_name, forward_period)
            all_ic[factor_name] = ic
        
        return pd.DataFrame(all_ic)
    
    def calculate_ir(self) -> Dict[str, float]:
        """
        Calculate Information Ratio for all factors.
        
        IR = IC_mean / IC_std
        """
        ir_results = {}
        
        for factor_name, ic_series in self.ic_results.items():
            ic_mean = ic_series.mean()
            ic_std = ic_series.std()
            ir = ic_mean / (ic_std + 1e-10) * np.sqrt(252)  # Annualized
            ir_results[factor_name] = {
                'IC_mean': ic_mean,
                'IC_std': ic_std,
                'IR': ir,
                'IC_positive_pct': (ic_series > 0).mean() * 100
            }
        
        self.ir_results = ir_results
        return ir_results
    
    def calculate_decile_returns(self, factor_name: str, forward_period: int = 20, n_groups: int = 10) -> pd.DataFrame:
        """
        Calculate average returns for each factor decile. Vectorized approach.
        """
        factor_df = self.factors[factor_name]
        forward_returns = self.returns.shift(-forward_period)
        
        # Align
        common_index = factor_df.index.intersection(forward_returns.index)
        common_columns = factor_df.columns.intersection(forward_returns.columns)
        
        f_aligned = factor_df.loc[common_index, common_columns]
        r_aligned = forward_returns.loc[common_index, common_columns]
        
        # Calculate decile ranks (0 to 9)
        # qcut produces categories, codes give integer 0..n-1
        # Apply qcut row-wise
        
        def get_deciles(row):
            try:
                # dropna to handle missing values correctly
                valid = row.dropna()
                if len(valid) < n_groups:
                    return pd.Series(np.nan, index=row.index)
                
                # Rank then cut
                ranks = valid.rank(method='first')
                bins = pd.qcut(ranks, n_groups, labels=False) + 1
                return bins
            except:
                return pd.Series(np.nan, index=row.index)

        # Note: apply per row is slower than pure numpy but much faster than iterating Python loops
        # However, for huge matrices, we can do rank().div(count).mul(n_groups).ceil()
        
        # Fast vectorized binning:
        ranks = f_aligned.rank(axis=1, pct=True)
        deciles = (ranks * n_groups).apply(np.ceil).fillna(0).astype(int)
        
        # Collect returns
        # Using a simple loop over groups is reliable enough given the groupby
        
        decile_stats = []
        
        for g in range(1, n_groups + 1):
            # Mask for current group
            mask = (deciles == g)
            # Returns for this group across all time/stocks
            # We want time-series mean of cross-sectional means? 
            # Or grand mean? Usually Mean Return per Decile = Average of (Average return of stocks in decile d at time t)
            
            # Daily mean return for this decile
            daily_means = r_aligned[mask].mean(axis=1)
            
            # Aggregate stats
            mean_ret = daily_means.mean() * 252 * 100 # Annualized
            std_ret = daily_means.std() * np.sqrt(252) * 100
            
            decile_stats.append({
                'Decile': g,
                'Mean_Return': mean_ret,
                'Std_Return': std_ret,
                'Count': mask.sum().sum() # Total stock-days
            })
            
        result_df = pd.DataFrame(decile_stats)
        self.decile_results[factor_name] = result_df
        
        return result_df
    
    def calculate_factor_decay(self, factor_name: str, horizons: List[int] = [1, 5, 10, 20, 60]) -> pd.DataFrame:
        """
        Calculate IC at different forward horizons.
        """
        decay_results = []
        
        print(f"    Calculating decay for {factor_name}...", end=" ", flush=True)
        for horizon in horizons:
            ic = self.calculate_ic(factor_name, horizon)
            decay_results.append({
                'Horizon': horizon,
                'IC_mean': ic.mean(),
                'IC_std': ic.std(),
                'IC_positive_pct': (ic > 0).mean() * 100
            })
        print("Done")
        
        return pd.DataFrame(decay_results)
    
    def monotonicity_test(self, factor_name: str) -> Dict:
        """
        Test for monotonicity.
        """
        decile_df = self.decile_results.get(factor_name)
        if decile_df is None:
            decile_df = self.calculate_decile_returns(factor_name)
        
        returns = decile_df['Mean_Return'].values
        
        if len(returns) < 2:
             return {
                'monotonicity_corr': 0,
                'p_value': 1.0,
                'long_short_return': 0,
                'increasing_pairs': 0,
                'is_monotonic': False
            }
            
        corr, p_value = stats.spearmanr(range(len(returns)), returns)
        long_short = returns[-1] - returns[0]
        increasing = sum(returns[i] < returns[i+1] for i in range(len(returns)-1))
        
        return {
            'monotonicity_corr': corr,
            'p_value': p_value,
            'long_short_return': long_short,
            'increasing_pairs': increasing,
            'is_monotonic': corr > 0.7 and p_value < 0.05
        }


class ValidationPlotter:
    """Generates validation visualizations."""
    
    def __init__(self, output_dir: str = "./outputs/validation"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_ic_timeseries(self, ic_df: pd.DataFrame, filename: str = "ic_timeseries.png"):
        """Plot IC time series for all factors."""
        if ic_df.empty:
            return None
            
        fig, axes = plt.subplots(len(ic_df.columns), 1, figsize=(14, 3*len(ic_df.columns)), sharex=True)
        
        if len(ic_df.columns) == 1:
            axes = [axes]
        
        for ax, col in zip(axes, ic_df.columns):
            ic_series = ic_df[col].dropna()
            
            # Rolling mean
            rolling_ic = ic_series.rolling(20).mean()
            
            ax.bar(ic_series.index, ic_series, alpha=0.3, color='blue', width=1)
            ax.plot(rolling_ic.index, rolling_ic, color='red', linewidth=2, label='20-day MA')
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax.axhline(y=ic_series.mean(), color='green', linestyle='--', linewidth=1, label=f'Mean: {ic_series.mean():.4f}')
            
            ax.set_ylabel(col)
            ax.set_title(f'{col} Information Coefficient (IC)', fontweight='bold')
            ax.legend(loc='upper right')
            ax.set_ylim(-0.25, 0.25)
        
        plt.xlabel('Date')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()
        
        return os.path.join(self.output_dir, filename)
    
    def plot_decile_returns(self, decile_results: Dict[str, pd.DataFrame], filename: str = "decile_returns.png"):
        """Plot decile returns bar chart for all factors."""
        n_factors = len(decile_results)
        if n_factors == 0:
            return None
            
        fig, axes = plt.subplots(1, n_factors, figsize=(4*n_factors, 5))
        
        if n_factors == 1:
            axes = [axes]
        
        colors = plt.cm.RdYlGn(np.linspace(0.1, 0.9, 10))
        
        for ax, (factor_name, df) in zip(axes, decile_results.items()):
            if df.empty:
                continue
                
            bars = ax.bar(df['Decile'], df['Mean_Return'], color=colors, edgecolor='black')
            
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax.set_xlabel('Decile (1=Low, 10=High)')
            ax.set_ylabel('Annualized Return (%)')
            ax.set_title(f'{factor_name} Decile Returns', fontweight='bold')
            
            # Add value labels
            for bar, val in zip(bars, df['Mean_Return']):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       f'{val:.1f}%', ha='center', va='bottom', fontsize=8)
            
            # Long-short annotation
            long_short = df['Mean_Return'].iloc[-1] - df['Mean_Return'].iloc[0]
            ax.annotate(f'L/S: {long_short:.1f}%', xy=(0.95, 0.95), xycoords='axes fraction',
                       ha='right', va='top', fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()
        
        return os.path.join(self.output_dir, filename)
    
    def plot_ir_summary(self, ir_results: Dict, filename: str = "ir_summary.png"):
        """Plot IR summary bar chart."""
        if not ir_results:
            return None
            
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        factors = list(ir_results.keys())
        ic_means = [ir_results[f]['IC_mean'] for f in factors]
        irs = [ir_results[f]['IR'] for f in factors]
        
        # IC Mean
        colors = ['green' if ic > 0 else 'red' for ic in ic_means]
        axes[0].barh(factors, ic_means, color=colors, edgecolor='black')
        axes[0].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        axes[0].set_xlabel('Mean IC')
        axes[0].set_title('Mean Information Coefficient', fontweight='bold')
        
        for i, (f, ic) in enumerate(zip(factors, ic_means)):
            axes[0].text(ic + 0.001 if ic > 0 else ic - 0.001, i, f'{ic:.4f}',
                        va='center', ha='left' if ic > 0 else 'right')
        
        # IR
        colors = ['green' if ir > 0.5 else 'orange' if ir > 0 else 'red' for ir in irs]
        axes[1].barh(factors, irs, color=colors, edgecolor='black')
        axes[1].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        axes[1].axvline(x=0.5, color='green', linestyle='--', linewidth=1, label='Good (0.5)')
        axes[1].set_xlabel('Information Ratio (Annualized)')
        axes[1].set_title('Information Ratio', fontweight='bold')
        axes[1].legend()
        
        for i, (f, ir) in enumerate(zip(factors, irs)):
            axes[1].text(ir + 0.05, i, f'{ir:.2f}', va='center', ha='left')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()
        
        return os.path.join(self.output_dir, filename)
    
    def plot_factor_decay(self, decay_results: Dict[str, pd.DataFrame], filename: str = "factor_decay.png"):
        """Plot factor decay curves."""
        if not decay_results:
            return None
            
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for factor_name, df in decay_results.items():
            ax.plot(df['Horizon'], df['IC_mean'], marker='o', linewidth=2, label=factor_name)
        
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Forward Horizon (Days)')
        ax.set_ylabel('Mean IC')
        ax.set_title('Factor Decay Analysis', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()
        
        return os.path.join(self.output_dir, filename)


def run_validation(
    prices_file: str = "./outputs/clean/processed_prices.csv",
    factors_dir: str = "./outputs/factors",
    output_dir: str = "./outputs/validation"
) -> Dict:
    """Run complete alpha validation."""
    
    print("\n" + "="*60)
    print("CHAPTER 4: ALPHA VALIDATION")
    print("="*60)
    
    # Load data
    print("\nLoading data...")
    prices = pd.read_csv(prices_file, index_col=0, parse_dates=True)
    
    factors = {}
    for factor_name in ['BTP', 'RIE', 'MRD', 'KMD', 'RSVS']:
        factor_file = os.path.join(factors_dir, f"{factor_name}_standardized.csv")
        if os.path.exists(factor_file):
            factors[factor_name] = pd.read_csv(factor_file, index_col=0, parse_dates=True)
    
    print(f"Loaded: {prices.shape[1]} stocks, {len(factors)} factors")
    
    # Initialize validator
    validator = AlphaValidator(prices, factors)
    plotter = ValidationPlotter(output_dir)
    
    # 1. Calculate IC
    print("\n1. Calculating Information Coefficients...")
    ic_df = validator.calculate_all_ic(forward_period=5)
    
    # 2. Calculate IR
    print("\n2. Calculating Information Ratios...")
    ir_results = validator.calculate_ir()
    
    for factor, metrics in ir_results.items():
        print(f"  {factor}: IC={metrics['IC_mean']:.4f}, IR={metrics['IR']:.2f}, Positive={metrics['IC_positive_pct']:.1f}%")
    
    # 3. Decile Analysis
    print("\n3. Calculating Decile Returns...")
    for factor_name in factors.keys():
        validator.calculate_decile_returns(factor_name, forward_period=20)
        mono = validator.monotonicity_test(factor_name)
        print(f"  {factor_name}: L/S={mono['long_short_return']:.1f}%, Monotonic={mono['is_monotonic']}")
    
    # 4. Factor Decay
    print("\n4. Calculating Factor Decay...")
    decay_results = {}
    for factor_name in factors.keys():
        decay_results[factor_name] = validator.calculate_factor_decay(factor_name)
    
    # Generate plots
    print("\n5. Generating Visualizations...")
    plots = {
        'ic_timeseries': plotter.plot_ic_timeseries(ic_df),
        'decile_returns': plotter.plot_decile_returns(validator.decile_results),
        'ir_summary': plotter.plot_ir_summary(ir_results),
        'factor_decay': plotter.plot_factor_decay(decay_results)
    }
    
    print(f"\nPlots saved to {output_dir}")
    
    return {
        'ic': ic_df,
        'ir': ir_results,
        'decile': validator.decile_results,
        'decay': decay_results,
        'plots': plots
    }


if __name__ == "__main__":
    results = run_validation()
