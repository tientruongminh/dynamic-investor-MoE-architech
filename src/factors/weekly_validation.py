import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Dict, List

class WeeklyAlphaValidator:
    """
    Weekly validation of alpha factors to assess consistency and stability.
    Calculates IC and IR on a weekly basis.
    """
    
    def __init__(self, prices_path: str, factors_dir: str, factor_names: List[str]):
        self.prices = pd.read_csv(prices_path, index_col=0, parse_dates=True)
        self.returns = self.prices.pct_change()
        self.factors = {}
        for name in factor_names:
            path = os.path.join(factors_dir, f"{name}_standardized.csv")
            if os.path.exists(path):
                self.factors[name] = pd.read_csv(path, index_col=0, parse_dates=True)
                
        self.output_dir = "./outputs/validation_weekly"
        os.makedirs(self.output_dir, exist_ok=True)

    def calculate_weekly_ic(self, factor_name: str, forward_period: int = 5) -> pd.Series:
        """Calculate IC by resampling daily data to weekly."""
        factor_df = self.factors[factor_name]
        forward_returns = self.returns.shift(-forward_period)
        
        # Daily IC
        f_rank = factor_df.rank(axis=1)
        r_rank = forward_returns.rank(axis=1)
        daily_ic = f_rank.corrwith(r_rank, axis=1)
        
        # Resample to Weekly (taking the mean IC of that week)
        # This provides a smoother view of factor performance
        weekly_ic = daily_ic.resample('W').mean().dropna()
        return weekly_ic

    def run_analysis(self):
        all_weekly_ic = {}
        for name in self.factors:
            all_weekly_ic[name] = self.calculate_weekly_ic(name)
            
        ic_df = pd.DataFrame(all_weekly_ic)
        
        # Calculate Weekly IR (Rolling 12-week window for stability)
        # Rolling IR = Rolling Mean / Rolling Std
        rolling_mean = ic_df.rolling(12).mean()
        rolling_std = ic_df.rolling(12).std()
        rolling_ir = (rolling_mean / (rolling_std + 1e-10)) * np.sqrt(52) # Weekly IR annualized
        
        self.plot_results(ic_df, rolling_ir)
        return ic_df, rolling_ir

    def plot_results(self, ic_df: pd.DataFrame, rolling_ir: pd.DataFrame):
        # 1. Weekly IC Time Series
        plt.figure(figsize=(14, 8))
        for col in ic_df.columns:
            plt.plot(ic_df.index, ic_df[col], label=f"{col} Weekly IC", alpha=0.7)
        plt.axhline(0, color='black', linestyle='--')
        plt.title("Weekly Information Coefficient (IC) - Factor Performance per Week", fontweight='bold')
        plt.ylabel("Spearman IC")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(self.output_dir, "weekly_ic_timeseries.png"))
        
        # 2. Rolling IR Stability
        plt.figure(figsize=(14, 8))
        for col in rolling_ir.columns:
            plt.plot(rolling_ir.index, rolling_ir[col], label=f"{col} Rolling IR (12-week)", linewidth=2)
        plt.axhline(0.5, color='green', linestyle='--', label='Good Stability (0.5)')
        plt.axhline(0, color='black', linestyle='-')
        plt.title("Rolling Weekly Information Ratio (IR) - Assessing Signal Stability", fontweight='bold')
        plt.ylabel("Annualized Rolling IR")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(self.output_dir, "weekly_rolling_ir_stability.png"))
        
        # 3. Distribution of Weekly IC
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=ic_df)
        plt.title("Distribution of Weekly IC - Consistency Check", fontweight='bold')
        plt.ylabel("IC Value")
        plt.savefig(os.path.join(self.output_dir, "weekly_ic_distribution.png"))

if __name__ == "__main__":
    validator = WeeklyAlphaValidator(
        prices_path="./outputs/clean/processed_prices.csv",
        factors_dir="./outputs/factors",
        factor_names=['BTP', 'RIE', 'MRD', 'KMD', 'RSVS']
    )
    validator.run_analysis()
    print("Weekly validation plots saved to ./outputs/validation_weekly")
