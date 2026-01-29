"""
Walk-Forward Backtest Engine
============================
Backtesting MoE strategy với walk-forward validation (2023-2026).
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import os
import json
from datetime import datetime, timedelta


class WalkForwardBacktest:
    """
    Walk-forward backtesting engine.
    
    Features:
    - Weekly rebalancing
    - Transaction costs
    - Performance metrics (Sharpe, Sortino, Max DD)
    - Benchmark comparison
    """
    
    def __init__(
        self,
        moe_predictor,
        optimizer,
        constraints,
        rebalance_freq: str = 'W',
        initial_capital: float = 1_000_000
    ):
        self.moe = moe_predictor
        self.optimizer = optimizer
        self.constraints = constraints
        self.freq = rebalance_freq
        self.capital = initial_capital
        
        self.results = {
            'dates': [],
            'returns': [],
            'portfolio_values': [],
            'weights': [],
            'costs': [],
            'turnover': []
        }
    
    def run(
        self,
        prices: pd.DataFrame,
        covariance: pd.DataFrame,
        market_caps: pd.Series,
        sector_map: Dict[str, str],
        start: str = '2023-01-01',
        end: str = '2026-01-01'
    ) -> dict:
        """
        Run walk-forward backtest.
        """
        print("="*60)
        print("WALK-FORWARD BACKTEST")
        print("="*60)
        print(f"Period: {start} to {end}")
        print(f"Rebalance: {self.freq}")
        
        # Filter prices to backtest period
        prices = prices.loc[start:end]
        returns = prices.pct_change().fillna(0)
        
        # Get rebalance dates
        rebalance_dates = prices.resample(self.freq).last().index
        
        current_weights = pd.Series(0, index=prices.columns)
        portfolio_value = self.capital
        
        print(f"\nRebalancing {len(rebalance_dates)} times...")
        
        for i, date in enumerate(rebalance_dates[:-1]):
            next_date = rebalance_dates[i + 1]
            
            # 1. Get MoE predictions
            predictions = self.moe.predict(date) if self.moe else {}
            
            # 2. Optimize (use simple momentum if no predictions)
            if predictions:
                target_weights = self.optimizer.optimize(
                    covariance,
                    market_caps,
                    predictions,
                    view_confidence=0.5
                )
            else:
                # Fallback: equal weight
                target_weights = pd.Series(1/len(prices.columns), index=prices.columns)
            
            # 3. Apply constraints
            final_weights, cost = self.constraints.apply_all(
                current_weights,
                target_weights,
                sector_map
            )
            
            # 4. Calculate period return
            period_returns = returns.loc[date:next_date]
            if len(period_returns) > 1:
                period_return = (period_returns.iloc[1:] * final_weights).sum(axis=1).sum()
            else:
                period_return = 0
            
            # 5. Update portfolio value
            portfolio_value = portfolio_value * (1 + period_return - cost)
            
            # 6. Calculate turnover
            turnover = (final_weights - current_weights).abs().sum() / 2
            
            # Store results
            self.results['dates'].append(date)
            self.results['returns'].append(period_return - cost)
            self.results['portfolio_values'].append(portfolio_value)
            self.results['weights'].append(final_weights.to_dict())
            self.results['costs'].append(cost)
            self.results['turnover'].append(turnover)
            
            current_weights = final_weights
            
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(rebalance_dates)} - Value: ${portfolio_value:,.0f}")
        
        # Calculate metrics
        metrics = self._calculate_metrics()
        
        return metrics
    
    def _calculate_metrics(self) -> dict:
        """Calculate performance metrics."""
        returns = pd.Series(self.results['returns'], index=self.results['dates'])
        values = pd.Series(self.results['portfolio_values'], index=self.results['dates'])
        
        # Annualization factor (weekly -> annual)
        ann_factor = 52 if self.freq == 'W' else 12
        
        # Sharpe Ratio
        sharpe = returns.mean() / (returns.std() + 1e-6) * np.sqrt(ann_factor)
        
        # Sortino Ratio (downside risk only)
        downside_returns = returns[returns < 0]
        sortino = returns.mean() / (downside_returns.std() + 1e-6) * np.sqrt(ann_factor)
        
        # Max Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdowns = (cumulative - running_max) / running_max
        max_drawdown = drawdowns.min()
        
        # Win Rate
        win_rate = (returns > 0).mean()
        
        # Total Return
        total_return = values.iloc[-1] / self.capital - 1
        
        # CAGR
        n_years = len(returns) / ann_factor
        cagr = (values.iloc[-1] / self.capital) ** (1/n_years) - 1 if n_years > 0 else 0
        
        # Average turnover
        avg_turnover = np.mean(self.results['turnover'])
        
        # Total costs
        total_costs = sum(self.results['costs'])
        
        metrics = {
            'sharpe': sharpe,
            'sortino': sortino,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_return': total_return,
            'cagr': cagr,
            'avg_turnover': avg_turnover,
            'total_costs': total_costs,
            'final_value': values.iloc[-1],
            'returns_series': returns,
            'equity_curve': values
        }
        
        # Print summary
        print("\n" + "="*60)
        print("BACKTEST RESULTS")
        print("="*60)
        print(f"Sharpe Ratio:    {sharpe:.3f}")
        print(f"Sortino Ratio:   {sortino:.3f}")
        print(f"Max Drawdown:    {max_drawdown:.1%}")
        print(f"Win Rate:        {win_rate:.1%}")
        print(f"Total Return:    {total_return:.1%}")
        print(f"CAGR:            {cagr:.1%}")
        print(f"Avg Turnover:    {avg_turnover:.1%}")
        print(f"Total Costs:     ${total_costs*self.capital:,.0f}")
        print(f"Final Value:     ${values.iloc[-1]:,.0f}")
        
        return metrics
    
    def get_equity_curve(self) -> pd.Series:
        """Get equity curve."""
        return pd.Series(
            self.results['portfolio_values'],
            index=self.results['dates']
        )
    
    def compare_to_benchmark(
        self,
        benchmark_prices: pd.Series,
        start: str = '2023-01-01'
    ) -> pd.DataFrame:
        """Compare strategy to benchmark."""
        strategy = self.get_equity_curve()
        
        # Align benchmark
        benchmark = benchmark_prices.loc[start:]
        benchmark = benchmark / benchmark.iloc[0] * self.capital
        benchmark = benchmark.reindex(strategy.index, method='ffill')
        
        comparison = pd.DataFrame({
            'Strategy': strategy,
            'Benchmark': benchmark
        })
        
        return comparison
    
    def save_results(self, output_dir: str = "./outputs/backtest"):
        """Save backtest results."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save returns
        returns = pd.Series(self.results['returns'], index=self.results['dates'])
        returns.to_csv(os.path.join(output_dir, "returns.csv"))
        
        # Save equity curve
        equity = self.get_equity_curve()
        equity.to_csv(os.path.join(output_dir, "equity_curve.csv"))
        
        # Save summary
        with open(os.path.join(output_dir, "summary.json"), 'w') as f:
            summary = {
                'sharpe': float(returns.mean() / returns.std() * np.sqrt(52)),
                'total_return': float((1 + returns).prod() - 1),
                'max_drawdown': float(((1+returns).cumprod() / (1+returns).cumprod().cummax() - 1).min()),
                'n_periods': len(returns)
            }
            json.dump(summary, f, indent=2)
        
        print(f"Results saved to {output_dir}")


class SimpleBacktest:
    """
    Simple backtest without MoE - for baseline comparison.
    """
    
    def __init__(self, strategy: str = 'equal_weight'):
        self.strategy = strategy
    
    def run(
        self,
        prices: pd.DataFrame,
        start: str = '2023-01-01',
        end: str = '2026-01-01',
        rebalance_freq: str = 'W'
    ) -> dict:
        """Run simple backtest."""
        prices = prices.loc[start:end]
        returns = prices.pct_change().fillna(0)
        
        if self.strategy == 'equal_weight':
            n = len(prices.columns)
            weights = pd.Series(1/n, index=prices.columns)
        elif self.strategy == 'market_cap':
            # Would need market cap data
            weights = pd.Series(1/len(prices.columns), index=prices.columns)
        
        # Portfolio returns
        portfolio_returns = (returns * weights).sum(axis=1)
        
        # Resample to weekly
        weekly_returns = portfolio_returns.resample(rebalance_freq).sum()
        
        # Metrics
        sharpe = weekly_returns.mean() / weekly_returns.std() * np.sqrt(52)
        total_return = (1 + weekly_returns).prod() - 1
        
        return {
            'sharpe': sharpe,
            'total_return': total_return,
            'returns': weekly_returns
        }


if __name__ == "__main__":
    print("Backtest engine ready. Use run() method to execute.")
