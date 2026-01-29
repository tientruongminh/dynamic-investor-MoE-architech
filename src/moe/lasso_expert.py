"""
Lasso Expert Module
===================
Huấn luyện 5 Lasso Regression models cho từng cluster.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
import joblib
import os
import json


class LassoExpert:
    """
    Một Lasso Regression Expert cho một cluster cụ thể.
    """
    
    def __init__(self, cluster_name: str, alpha_range: tuple = (0.0001, 1.0)):
        self.cluster_name = cluster_name
        self.model = LassoCV(
            alphas=np.logspace(np.log10(alpha_range[0]), np.log10(alpha_range[1]), 50),
            cv=5,
            max_iter=10000,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.feature_names = []
        self.feature_importance = {}
        self.is_fitted = False
        self.train_metrics = {}
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'LassoExpert':
        """
        Huấn luyện Lasso Expert.
        
        Args:
            X: DataFrame với 12-13 factors cho stocks trong cluster
            y: Forward returns (5d hoặc 20d)
        """
        # Store feature names
        self.feature_names = list(X.columns)
        
        # Drop NaN
        mask = ~(X.isna().any(axis=1) | y.isna())
        X_clean = X.loc[mask]
        y_clean = y.loc[mask]
        
        if len(X_clean) < 50:
            print(f"Warning: Only {len(X_clean)} samples for {self.cluster_name}")
            self.is_fitted = False
            return self
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_clean)
        
        # Fit Lasso
        self.model.fit(X_scaled, y_clean)
        
        # Store feature importance
        self.feature_importance = dict(zip(self.feature_names, self.model.coef_))
        
        # Calculate training metrics
        y_pred = self.model.predict(X_scaled)
        self.train_metrics = {
            'alpha': self.model.alpha_,
            'r2': self.model.score(X_scaled, y_clean),
            'n_samples': len(y_clean),
            'n_nonzero_coefs': np.sum(self.model.coef_ != 0)
        }
        
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Dự báo expected returns."""
        if not self.is_fitted:
            return np.zeros(len(X))
        
        X_scaled = self.scaler.transform(X[self.feature_names])
        return self.model.predict(X_scaled)
    
    def get_top_factors(self, n: int = 5) -> List[tuple]:
        """Lấy top N factors quan trọng nhất."""
        sorted_factors = sorted(
            self.feature_importance.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        return sorted_factors[:n]
    
    def save(self, path: str):
        """Lưu model."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            'cluster_name': self.cluster_name,
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance,
            'train_metrics': self.train_metrics
        }, path)
    
    @classmethod
    def load(cls, path: str) -> 'LassoExpert':
        """Load model từ file."""
        data = joblib.load(path)
        expert = cls(data['cluster_name'])
        expert.model = data['model']
        expert.scaler = data['scaler']
        expert.feature_names = data['feature_names']
        expert.feature_importance = data['feature_importance']
        expert.train_metrics = data['train_metrics']
        expert.is_fitted = True
        return expert


class MoELassoEnsemble:
    """
    Ensemble của 5 Lasso Experts.
    """
    
    CLUSTERS = ['Growth', 'Value', 'Cyclical', 'Defensive', 'Speculative']
    
    def __init__(self):
        self.experts: Dict[str, LassoExpert] = {}
        for cluster in self.CLUSTERS:
            self.experts[cluster] = LassoExpert(cluster)
    
    def fit_all(
        self,
        factors: Dict[str, pd.DataFrame],  # {factor_name: DataFrame}
        forward_returns: pd.DataFrame,
        cluster_assignments: Dict[str, str],  # {ticker: cluster}
        forward_period: int = 5
    ):
        """
        Huấn luyện tất cả experts.
        
        Args:
            factors: Dictionary of factor DataFrames (same index/columns structure)
            forward_returns: Forward returns DataFrame
            cluster_assignments: {ticker: cluster_name}
        """
        print("\n" + "="*60)
        print("TRAINING LASSO EXPERTS")
        print("="*60)
        
        # Combine all factors into single DataFrame per date
        # Stack factors column-wise for each stock
        factor_names = list(factors.keys())
        
        for cluster in self.CLUSTERS:
            # Get stocks in this cluster
            cluster_stocks = [t for t, c in cluster_assignments.items() if c == cluster]
            print(f"\n{cluster} Expert: {len(cluster_stocks)} stocks")
            
            if len(cluster_stocks) < 10:
                print(f"  Skipping: not enough stocks")
                continue
            
            # Prepare training data
            # Flatten: rows = (date, stock), cols = factors
            X_list = []
            y_list = []
            
            for date in forward_returns.index[:-forward_period]:
                for stock in cluster_stocks:
                    if stock not in forward_returns.columns:
                        continue
                    
                    # Get factor values for this stock at this date
                    factor_values = {}
                    valid = True
                    for fname, fdf in factors.items():
                        if stock in fdf.columns and date in fdf.index:
                            val = fdf.loc[date, stock]
                            if pd.isna(val):
                                valid = False
                                break
                            factor_values[fname] = val
                        else:
                            valid = False
                            break
                    
                    if not valid:
                        continue
                    
                    # Get forward return
                    future_date = forward_returns.index[forward_returns.index.get_loc(date) + forward_period]
                    fwd_ret = forward_returns.loc[future_date, stock] if stock in forward_returns.columns else None
                    
                    if fwd_ret is None or pd.isna(fwd_ret):
                        continue
                    
                    X_list.append(factor_values)
                    y_list.append(fwd_ret)
            
            if len(X_list) < 50:
                print(f"  Skipping: only {len(X_list)} samples")
                continue
            
            X = pd.DataFrame(X_list)
            y = pd.Series(y_list)
            
            # Train expert
            self.experts[cluster].fit(X, y)
            
            if self.experts[cluster].is_fitted:
                metrics = self.experts[cluster].train_metrics
                print(f"  Trained: R² = {metrics['r2']:.4f}, "
                      f"Alpha = {metrics['alpha']:.6f}, "
                      f"Non-zero coefs = {metrics['n_nonzero_coefs']}/{len(factor_names)}")
                
                # Show top factors
                top = self.experts[cluster].get_top_factors(3)
                print(f"  Top factors: {', '.join([f'{f}({c:.4f})' for f, c in top])}")
    
    def predict(
        self,
        factors: Dict[str, pd.DataFrame],
        cluster_assignments: Dict[str, str],
        date: pd.Timestamp
    ) -> Dict[str, float]:
        """
        Dự báo expected returns cho tất cả stocks.
        """
        predictions = {}
        
        for cluster, expert in self.experts.items():
            if not expert.is_fitted:
                continue
            
            # Get stocks in this cluster
            cluster_stocks = [t for t, c in cluster_assignments.items() if c == cluster]
            
            for stock in cluster_stocks:
                # Get factor values
                factor_values = {}
                valid = True
                
                for fname, fdf in factors.items():
                    if stock in fdf.columns and date in fdf.index:
                        val = fdf.loc[date, stock]
                        factor_values[fname] = val
                    else:
                        valid = False
                        break
                
                if valid:
                    X = pd.DataFrame([factor_values])
                    pred = expert.predict(X)
                    predictions[stock] = pred[0]
        
        return predictions
    
    def save_all(self, output_dir: str = "./outputs/moe/experts"):
        """Lưu tất cả experts."""
        os.makedirs(output_dir, exist_ok=True)
        
        for cluster, expert in self.experts.items():
            if expert.is_fitted:
                expert.save(os.path.join(output_dir, f"{cluster}_expert.pkl"))
        
        print(f"Saved experts to {output_dir}")
    
    @classmethod
    def load_all(cls, input_dir: str = "./outputs/moe/experts") -> 'MoELassoEnsemble':
        """Load tất cả experts."""
        ensemble = cls()
        
        for cluster in cls.CLUSTERS:
            path = os.path.join(input_dir, f"{cluster}_expert.pkl")
            if os.path.exists(path):
                ensemble.experts[cluster] = LassoExpert.load(path)
                print(f"Loaded {cluster} expert")
        
        return ensemble
    
    def get_feature_importance_matrix(self) -> pd.DataFrame:
        """Tạo ma trận feature importance cho tất cả experts."""
        importance = {}
        
        for cluster, expert in self.experts.items():
            if expert.is_fitted:
                importance[cluster] = expert.feature_importance
        
        return pd.DataFrame(importance)


def run_lasso_training(
    factors_dir: str = "./outputs/factors_eda",
    clusters_path: str = "./outputs/moe/clusters.json",
    prices_path: str = "./outputs_clean/clean/processed_prices.csv",
    output_dir: str = "./outputs/moe/experts",
    forward_period: int = 5
):
    """
    Main function to train all Lasso experts.
    """
    print("="*60)
    print("LASSO EXPERT TRAINING PIPELINE")
    print("="*60)
    
    # Load clusters
    print("\nLoading cluster assignments...")
    with open(clusters_path) as f:
        cluster_data = json.load(f)
    cluster_assignments = cluster_data['stock_to_cluster']
    print(f"Loaded {len(cluster_assignments)} cluster assignments")
    
    # Load factors
    print("\nLoading factors...")
    factors = {}
    for fname in ['MOM_12M', 'CONSISTENCY', 'MRD', 'REV_5D', 'REV_20D', 
                  'VOL', 'SKEW', 'KURT', 'DRAWDOWN', 'BETA', 'RIE', 'IDIOVOL']:
        fpath = os.path.join(factors_dir, f"{fname}.csv")
        if os.path.exists(fpath):
            factors[fname] = pd.read_csv(fpath, index_col=0, parse_dates=True)
            print(f"  Loaded {fname}: {factors[fname].shape}")
    
    # Load prices and calculate forward returns
    print("\nCalculating forward returns...")
    prices = pd.read_csv(prices_path, index_col=0, parse_dates=True)
    returns = prices.pct_change()
    forward_returns = returns.shift(-forward_period).rolling(forward_period).sum()
    print(f"Forward returns shape: {forward_returns.shape}")
    
    # Create and train ensemble
    ensemble = MoELassoEnsemble()
    ensemble.fit_all(factors, forward_returns, cluster_assignments, forward_period)
    
    # Save
    ensemble.save_all(output_dir)
    
    # Print feature importance matrix
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE MATRIX")
    print("="*60)
    importance = ensemble.get_feature_importance_matrix()
    print(importance.round(4).to_string())
    
    return ensemble


if __name__ == "__main__":
    ensemble = run_lasso_training()
