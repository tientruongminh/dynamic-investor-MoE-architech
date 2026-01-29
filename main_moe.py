 """
MoE Alpha Pipeline - Main Orchestrator
=======================================
Full pipeline từ Clustering -> Lasso -> Gating -> Black-Litterman -> Backtest
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime

# Import modules
from src.moe.clustering import run_clustering, load_stocks_info
from src.moe.lasso_expert import MoELassoEnsemble, run_lasso_training
from src.moe.macro import run_macro_fetch, MacroDataFetcher
from src.moe.gating import RuleBasedGating, MLPGatingNetwork
from src.moe.supervisor import AISupervisor
from src.portfolio.optimizer import BlackLittermanOptimizer, PortfolioConstraints
from src.backtest.engine import WalkForwardBacktest, SimpleBacktest


class MoEPipeline:
    """
    Full MoE Alpha Pipeline.
    
    Pipeline:
    1. Clustering (Gemini) -> 5 Expert Groups
    2. Lasso Training -> 5 Expert Models
    3. Macro Data -> Regime Detection
    4. Gating Network -> Expert Weights
    5. Black-Litterman -> Portfolio Optimization
    6. Backtest -> Performance Metrics
    7. AI Supervisor -> Explainability
    """
    
    def __init__(
        self,
        data_dir: str = "./outputs_clean/clean",
        factors_dir: str = "./outputs/factors_eda",
        output_dir: str = "./outputs/moe",
        info_dir: str = "./data/nasdaq_574/info"
    ):
        self.data_dir = data_dir
        self.factors_dir = factors_dir
        self.output_dir = output_dir
        self.info_dir = info_dir
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/experts", exist_ok=True)
        os.makedirs(f"{output_dir}/backtest", exist_ok=True)
        os.makedirs(f"{output_dir}/reports", exist_ok=True)
        
        # Components
        self.clusters = None
        self.ensemble = None
        self.gating = None
        self.optimizer = None
        self.constraints = None
        self.supervisor = None
        
        # Data
        self.prices = None
        self.factors = None
        self.macro = None
        self.sector_map = None
    
    def load_data(self):
        """Load all required data."""
        print("="*60)
        print("LOADING DATA")
        print("="*60)
        
        # Prices
        prices_path = f"{self.data_dir}/processed_prices.csv"
        self.prices = pd.read_csv(prices_path, index_col=0, parse_dates=True)
        print(f"Prices: {self.prices.shape}")
        
        # Factors
        self.factors = {}
        for fname in ['MOM_12M', 'CONSISTENCY', 'MRD', 'REV_5D', 'VOL', 
                      'SKEW', 'KURT', 'DRAWDOWN', 'BETA', 'RIE', 'IDIOVOL']:
            fpath = f"{self.factors_dir}/{fname}.csv"
            if os.path.exists(fpath):
                self.factors[fname] = pd.read_csv(fpath, index_col=0, parse_dates=True)
        print(f"Loaded {len(self.factors)} factors")
        
        # Sector map
        self.sector_map = {}
        stocks_info = load_stocks_info(self.info_dir)
        for ticker, info in stocks_info.items():
            self.sector_map[ticker] = info.get('sector', 'Unknown')
        print(f"Sector map: {len(self.sector_map)} stocks")
        
        return self
    
    def run_clustering(self, use_cache: bool = True):
        """Phase 1.1: Gemini Clustering."""
        print("\n" + "="*60)
        print("PHASE 1.1: GEMINI CLUSTERING")
        print("="*60)
        
        clusters_path = f"{self.output_dir}/clusters.json"
        
        self.clusters = run_clustering(
            info_dir=self.info_dir,
            output_path=clusters_path,
            use_cache=use_cache
        )
        
        return self
    
    def run_lasso_training(self, forward_period: int = 5):
        """Phase 1.2: Train Lasso Experts."""
        print("\n" + "="*60)
        print("PHASE 1.2: LASSO EXPERT TRAINING")
        print("="*60)
        
        self.ensemble = run_lasso_training(
            factors_dir=self.factors_dir,
            clusters_path=f"{self.output_dir}/clusters.json",
            prices_path=f"{self.data_dir}/processed_prices.csv",
            output_dir=f"{self.output_dir}/experts",
            forward_period=forward_period
        )
        
        return self
    
    def run_macro_fetch(self, start: str = "2016-01-01", end: str = "2026-01-20"):
        """Phase 2.1: Fetch Macro Data."""
        print("\n" + "="*60)
        print("PHASE 2.1: MACRO DATA FETCH")
        print("="*60)
        
        self.macro = run_macro_fetch(
            start=start,
            end=end,
            output_path=f"{self.output_dir}/macro_data.csv"
        )
        
        return self
    
    def setup_gating(self, gating_type: str = 'rule_based'):
        """Phase 2.2: Setup Gating Network."""
        print("\n" + "="*60)
        print("PHASE 2.2: GATING NETWORK SETUP")
        print("="*60)
        
        if gating_type == 'mlp':
            self.gating = MLPGatingNetwork()
            # Would need training data
            print("MLP Gating initialized (needs training)")
        else:
            self.gating = RuleBasedGating()
            print("Rule-based Gating initialized")
        
        return self
    
    def setup_optimizer(self):
        """Phase 3: Setup Portfolio Optimizer."""
        print("\n" + "="*60)
        print("PHASE 3: PORTFOLIO OPTIMIZER SETUP")
        print("="*60)
        
        self.optimizer = BlackLittermanOptimizer(
            risk_aversion=2.5,
            tau=0.05
        )
        
        self.constraints = PortfolioConstraints(
            transaction_cost=0.001,
            slippage=0.0005,
            max_turnover=0.30,
            sector_limit=0.25
        )
        
        print("Black-Litterman optimizer initialized")
        print("Constraints: TC=0.1%, Slippage=0.05%, MaxTurnover=30%, SectorLimit=25%")
        
        return self
    
    def predict(self, date: pd.Timestamp) -> dict:
        """
        Generate predictions for a given date.
        Combines Lasso predictions with Gating weights.
        """
        if self.ensemble is None or self.clusters is None:
            return {}
        
        cluster_assignments = self.clusters.get('stock_to_cluster', {})
        
        # Get Lasso predictions for all stocks
        raw_predictions = self.ensemble.predict(
            self.factors,
            cluster_assignments,
            date
        )
        
        # Get macro data for gating
        if self.macro is not None and date in self.macro.index:
            macro_row = self.macro.loc[date]
            gating_weights = self.gating.predict_from_df(macro_row)
        else:
            gating_weights = {c: 0.2 for c in self.gating.CLUSTERS}
        
        # Apply gating weights to predictions
        weighted_predictions = {}
        for ticker, pred in raw_predictions.items():
            cluster = cluster_assignments.get(ticker, 'Growth')
            weight = gating_weights.get(cluster, 0.2)
            weighted_predictions[ticker] = pred * weight
        
        return weighted_predictions
    
    def run_backtest(
        self,
        start: str = '2023-01-01',
        end: str = '2026-01-01',
        rebalance_freq: str = 'W'
    ):
        """Phase 4: Run Backtest."""
        print("\n" + "="*60)
        print("PHASE 4: WALK-FORWARD BACKTEST")
        print("="*60)
        
        # Calculate covariance
        returns = self.prices.pct_change().dropna()
        covariance = returns.iloc[-252:].cov()  # 1-year rolling cov
        
        # Market caps (use price * 1B as proxy)
        market_caps = self.prices.iloc[-1] * 1e9
        
        # Create backtest engine
        backtest = WalkForwardBacktest(
            moe_predictor=self,
            optimizer=self.optimizer,
            constraints=self.constraints,
            rebalance_freq=rebalance_freq
        )
        
        # Run
        metrics = backtest.run(
            prices=self.prices,
            covariance=covariance,
            market_caps=market_caps,
            sector_map=self.sector_map,
            start=start,
            end=end
        )
        
        # Save results
        backtest.save_results(f"{self.output_dir}/backtest")
        
        return metrics
    
    def generate_report(self, backtest_metrics: dict):
        """Phase 5: Generate AI Report."""
        print("\n" + "="*60)
        print("PHASE 5: AI SUPERVISOR REPORT")
        print("="*60)
        
        self.supervisor = AISupervisor()
        
        # Gap Analysis
        gap_report = self.supervisor.gap_analysis(
            expected_sharpe=1.1,  # From midterm
            realized_sharpe=backtest_metrics.get('sharpe', 0.5),
            expected_return=0.15,
            realized_return=backtest_metrics.get('total_return', 0.1),
            period="2023-2026"
        )
        
        # Save report
        self.supervisor.save_report(
            gap_report,
            f"{self.output_dir}/reports/gap_analysis.md",
            "gap_analysis"
        )
        
        print("\nGap Analysis Report:")
        print(gap_report[:500] + "...")
        
        return gap_report
    
    def run_full_pipeline(self):
        """Run complete MoE pipeline."""
        print("\n" + "="*70)
        print("MOE ALPHA PIPELINE - FULL EXECUTION")
        print("="*70)
        
        start_time = datetime.now()
        
        # Load data
        self.load_data()
        
        # Phase 1: Expert Layer
        self.run_clustering(use_cache=True)
        self.run_lasso_training()
        
        # Phase 2: Gating
        self.run_macro_fetch()
        self.setup_gating('rule_based')
        
        # Phase 3: Optimizer
        self.setup_optimizer()
        
        # Phase 4: Backtest
        metrics = self.run_backtest()
        
        # Phase 5: Report
        self.generate_report(metrics)
        
        elapsed = (datetime.now() - start_time).seconds
        print(f"\n{'='*70}")
        print(f"PIPELINE COMPLETE - {elapsed}s")
        print(f"{'='*70}")
        
        return metrics


def main():
    """Main entry point."""
    pipeline = MoEPipeline()
    metrics = pipeline.run_full_pipeline()
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Sharpe Ratio: {metrics.get('sharpe', 'N/A'):.3f}")
    print(f"Total Return: {metrics.get('total_return', 'N/A'):.1%}")
    print(f"Max Drawdown: {metrics.get('max_drawdown', 'N/A'):.1%}")
    
    return pipeline, metrics


if __name__ == "__main__":
    pipeline, metrics = main()
