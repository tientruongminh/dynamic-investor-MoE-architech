import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Dict, List
from scipy import stats

class MonthlyRobustnessValidator:
    """
    Monthly robustness validation of alpha factors.
    Calculates IC on a monthly basis to check for regime shifts.
    """
    
    def __init__(self, prices_path: str, factors_dir: str, factor_names: List[str]):
        self.prices = pd.read_csv(prices_path, index_col=0, parse_dates=True)
        self.returns = self.prices.pct_change()
        self.factors = {}
        for name in factor_names:
            path = os.path.join(factors_dir, f"{name}_standardized.csv")
            if os.path.exists(path):
                self.factors[name] = pd.read_csv(path, index_col=0, parse_dates=True)
                
        self.output_dir = "./outputs/validation_robustness"
        os.makedirs(self.output_dir, exist_ok=True)

    def calculate_monthly_ic(self, factor_name: str, forward_period: int = 5) -> pd.Series:
        factor_df = self.factors[factor_name]
        forward_returns = self.returns.shift(-forward_period)
        
        # Daily IC
        f_rank = factor_df.rank(axis=1)
        r_rank = forward_returns.rank(axis=1)
        daily_ic = f_rank.corrwith(r_rank, axis=1)
        
        # Resample to Monthly Mean
        monthly_ic = daily_ic.resample('M').mean().dropna()
        return monthly_ic

    def run_analysis(self):
        all_monthly_ic = {}
        for name in self.factors:
            all_monthly_ic[name] = self.calculate_monthly_ic(name)
            
        ic_df = pd.DataFrame(all_monthly_ic)
        
        # Train/Test Split (7 years / 3 years)
        # Assuming data starts around 2016
        split_point = ic_df.index[int(len(ic_df) * 0.7)]
        train_ic = ic_df[ic_df.index < split_point]
        test_ic = ic_df[ic_df.index >= split_point]
        
        print(f"Split Point: {split_point}")
        print(f"Train Mean IC:\n{train_ic.mean()}")
        print(f"Test Mean IC:\n{test_ic.mean()}")
        
        self.plot_robustness(ic_df, split_point)
        return ic_df

    def plot_robustness(self, ic_df: pd.DataFrame, split_point):
        # 1. Heatmap of Monthly IC (Robustness check)
        # Group by Year and Month
        for col in ic_df.columns:
            temp_df = ic_df[[col]].copy()
            temp_df['Year'] = temp_df.index.year
            temp_df['Month'] = temp_df.index.month
            pivot_ic = temp_df.pivot(index='Year', columns='Month', values=col)
            
            plt.figure(figsize=(12, 6))
            sns.heatmap(pivot_ic, annot=True, fmt=".2f", cmap="RdYlGn", center=0)
            plt.title(f"Monthly IC Robustness Heatmap - {col}", fontweight='bold')
            plt.savefig(os.path.join(self.output_dir, f"robustness_heatmap_{col}.png"))
            plt.close()

        # 2. Cumulative IC (Stability across Train/Test)
        plt.figure(figsize=(14, 8))
        for col in ic_df.columns:
            plt.plot(ic_df.index, ic_df[col].cumsum(), label=f"Cum IC {col}")
        plt.axvline(split_point, color='red', linestyle='--', label='Train/Test Split (7Y/3Y)')
        plt.title("Cumulative Information Coefficient - Stability Across Split", fontweight='bold')
        plt.ylabel("Cumulative IC")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(self.output_dir, "cumulative_ic_stability.png"))
        plt.close()

if __name__ == "__main__":
    validator = MonthlyRobustnessValidator(
        prices_path="./outputs/clean/processed_prices.csv",
        factors_dir="./outputs/factors",
        factor_names=['BTP', 'RIE', 'MRD', 'KMD', 'RSVS']
    )
    validator.run_analysis()
    print("Robustness validation plots saved to ./outputs/validation_robustness")
