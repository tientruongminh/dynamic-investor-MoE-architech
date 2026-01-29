"""
Dynamic Gating Network
======================
Neural Network-based Gating để điều phối trọng số cho 5 Lasso Experts.
Trainable dựa trên historical expert performance.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
import os
import joblib
import json

# Try to import PyTorch, fallback to sklearn
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available, using sklearn fallback")


class GatingMLP(nn.Module):
    """
    Lightweight MLP for Gating.
    Input: Macro features (VIX, Yield, PCA, etc.)
    Output: Softmax weights for 5 Experts
    """
    def __init__(self, input_dim: int = 6, hidden_dims: List[int] = [32, 16], n_experts: int = 5):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, n_experts))
        
        self.network = nn.Sequential(*layers)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        logits = self.network(x)
        return self.softmax(logits)


class DynamicGatingNetwork:
    """
    Trainable Dynamic Gating Network.
    
    Features:
    - VIX và VIX change
    - 10Y-2Y Yield Spread
    - Market momentum (PC1)
    - Sector dispersion
    
    Target: Expert weights based on which expert performed best
    """
    
    EXPERTS = ['Growth', 'Value', 'Cyclical', 'Defensive', 'Speculative']
    
    def __init__(
        self,
        input_dim: int = 6,
        hidden_dims: List[int] = [32, 16],
        lr: float = 0.001,
        use_torch: bool = True
    ):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.use_torch = use_torch and TORCH_AVAILABLE
        
        self.scaler = StandardScaler()
        self.trained = False
        
        if self.use_torch:
            self.model = GatingMLP(input_dim, hidden_dims, len(self.EXPERTS))
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
            self.criterion = nn.CrossEntropyLoss()
        else:
            # Fallback: Random Forest
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42
            )
    
    def prepare_features(self, macro_df: pd.DataFrame) -> np.ndarray:
        """
        Extract features from macro data.
        
        Features:
        1. VIX level
        2. VIX change (5d)
        3. VIX percentile (rolling 252d)
        4. Yield 10Y level
        5. Yield change (5d)
        6. Market momentum (20d SPY return)
        """
        features = pd.DataFrame(index=macro_df.index)
        
        # VIX features
        features['vix'] = macro_df['VIX']
        features['vix_change'] = macro_df['VIX'].pct_change(5)
        features['vix_pct'] = macro_df['VIX'].rolling(252).apply(
            lambda x: (x.iloc[-1] > x).mean(), raw=False
        )
        
        # Yield features
        if 'YIELD_10Y' in macro_df.columns:
            features['yield'] = macro_df['YIELD_10Y']
            features['yield_change'] = macro_df['YIELD_10Y'].diff(5)
        else:
            features['yield'] = 4.0
            features['yield_change'] = 0
        
        # Momentum
        if 'MKT_MOM_20D' in macro_df.columns:
            features['momentum'] = macro_df['MKT_MOM_20D']
        else:
            features['momentum'] = 0
        
        return features.fillna(0).values
    
    def create_targets(
        self,
        expert_returns: Dict[str, pd.Series],
        lookback: int = 4
    ) -> np.ndarray:
        """
        Create training targets based on which expert performed best.
        
        Target = argmax(rolling sharpe of each expert)
        """
        # Combine returns
        returns_df = pd.DataFrame(expert_returns)
        
        # Rolling mean return for each expert
        rolling_mean = returns_df.rolling(lookback).mean()
        rolling_std = returns_df.rolling(lookback).std()
        sharpe = rolling_mean / (rolling_std + 1e-6)
        
        # Best expert index
        best_expert = sharpe.idxmax(axis=1)
        
        # Convert to numeric
        expert_to_idx = {e: i for i, e in enumerate(self.EXPERTS)}
        targets = best_expert.map(expert_to_idx).values
        
        return targets
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        verbose: bool = True
    ):
        """
        Train the Gating Network.
        
        Args:
            X: Features (n_samples, n_features)
            y: Target expert indices (n_samples,)
        """
        # Remove NaN
        mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X = X[mask]
        y = y[mask].astype(int)
        
        if len(X) < 50:
            print(f"Not enough samples: {len(X)}")
            return self
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        if self.use_torch:
            self._fit_torch(X_scaled, y, epochs, batch_size, verbose)
        else:
            self._fit_sklearn(X_scaled, y, verbose)
        
        self.trained = True
        return self
    
    def _fit_torch(self, X: np.ndarray, y: np.ndarray, epochs: int, batch_size: int, verbose: bool):
        """Train with PyTorch."""
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for batch_X, batch_y in loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += (predicted == batch_y).sum().item()
                total += batch_y.size(0)
            
            if verbose and (epoch + 1) % 20 == 0:
                acc = correct / total * 100
                print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}, Acc: {acc:.1f}%")
    
    def _fit_sklearn(self, X: np.ndarray, y: np.ndarray, verbose: bool):
        """Train with sklearn Random Forest."""
        self.model.fit(X, y)
        
        if verbose:
            score = self.model.score(X, y)
            print(f"Random Forest Accuracy: {score:.1%}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict expert weights.
        
        Returns:
            weights: (n_samples, n_experts) softmax weights
        """
        if not self.trained:
            # Return equal weights
            return np.ones((len(X), len(self.EXPERTS))) / len(self.EXPERTS)
        
        X_scaled = self.scaler.transform(X)
        
        if self.use_torch:
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_scaled)
                weights = self.model(X_tensor).numpy()
        else:
            # RF gives class probabilities
            weights = self.model.predict_proba(X_scaled)
        
        return weights
    
    def predict_single(self, macro_features: Dict[str, float]) -> Dict[str, float]:
        """Predict weights for a single timestep."""
        # Build feature vector
        feature_names = ['vix', 'vix_change', 'vix_pct', 'yield', 'yield_change', 'momentum']
        X = np.array([[macro_features.get(f, 0) for f in feature_names]])
        
        weights = self.predict(X)[0]
        return dict(zip(self.EXPERTS, weights))
    
    def get_entropy(self, weights: np.ndarray) -> float:
        """
        Calculate entropy of weights.
        Low entropy = confident decision
        High entropy = uncertain
        """
        weights = np.clip(weights, 1e-10, 1)
        entropy = -np.sum(weights * np.log(weights), axis=-1)
        return np.mean(entropy)
    
    def save(self, path: str):
        """Save model."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        if self.use_torch:
            torch.save({
                'model_state': self.model.state_dict(),
                'scaler': self.scaler,
                'input_dim': self.input_dim,
                'hidden_dims': self.hidden_dims,
                'trained': self.trained
            }, path)
        else:
            joblib.dump({
                'model': self.model,
                'scaler': self.scaler,
                'trained': self.trained
            }, path)
        
        print(f"Saved Gating Network to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'DynamicGatingNetwork':
        """Load model."""
        if path.endswith('.pt') or path.endswith('.pth'):
            data = torch.load(path)
            gating = cls(
                input_dim=data['input_dim'],
                hidden_dims=data['hidden_dims'],
                use_torch=True
            )
            gating.model.load_state_dict(data['model_state'])
            gating.scaler = data['scaler']
            gating.trained = data['trained']
        else:
            data = joblib.load(path)
            gating = cls(use_torch=False)
            gating.model = data['model']
            gating.scaler = data['scaler']
            gating.trained = data['trained']
        
        return gating


def train_dynamic_gating(
    macro_path: str = "./outputs/moe/macro_data.csv",
    prices_path: str = "./outputs_clean/clean/processed_prices.csv",
    clusters_path: str = "./outputs/moe/clusters.json",
    output_path: str = "./outputs/moe/gating_network.pkl"
):
    """
    Train Dynamic Gating Network from historical data.
    """
    print("="*60)
    print("TRAINING DYNAMIC GATING NETWORK")
    print("="*60)
    
    # Load macro data
    print("\nLoading macro data...")
    macro = pd.read_csv(macro_path, index_col=0, parse_dates=True)
    print(f"Macro data: {macro.shape}")
    
    # Load prices and calculate returns
    print("\nLoading price data...")
    prices = pd.read_csv(prices_path, index_col=0, parse_dates=True)
    prices.index = pd.to_datetime(prices.index, utc=True).tz_convert(None)
    returns = prices.pct_change()
    
    # Load clusters
    print("\nLoading clusters...")
    with open(clusters_path) as f:
        cluster_data = json.load(f)
    cluster_assignments = cluster_data['stock_to_cluster']
    
    # Calculate expert returns (average return of stocks in each cluster)
    print("\nCalculating expert returns...")
    expert_returns = {}
    
    for expert in DynamicGatingNetwork.EXPERTS:
        expert_stocks = [s for s, c in cluster_assignments.items() if c == expert and s in returns.columns]
        if len(expert_stocks) > 0:
            expert_returns[expert] = returns[expert_stocks].mean(axis=1)
        else:
            expert_returns[expert] = pd.Series(0, index=returns.index)
    
    for expert, rets in expert_returns.items():
        print(f"  {expert}: {len(rets)} days, mean={rets.mean()*100:.3f}%")
    
    # Prepare features and targets
    print("\nPreparing training data...")
    gating = DynamicGatingNetwork(input_dim=6, hidden_dims=[32, 16], use_torch=TORCH_AVAILABLE)
    
    # Align macro with returns
    common_dates = macro.index.intersection(returns.index)
    macro_aligned = macro.loc[common_dates]
    
    X = gating.prepare_features(macro_aligned)
    
    # Create DataFrame of expert returns aligned
    expert_returns_df = pd.DataFrame({
        e: r.reindex(common_dates) for e, r in expert_returns.items()
    })
    
    y = gating.create_targets(expert_returns_df.to_dict('series'), lookback=4)
    
    print(f"Training samples: {len(X)}")
    print(f"Feature dim: {X.shape[1] if len(X) > 0 else 0}")
    
    # Train
    print("\nTraining...")
    gating.fit(X, y, epochs=100, batch_size=32, verbose=True)
    
    # Evaluate
    print("\n" + "="*60)
    print("EVALUATION")
    print("="*60)
    
    # Predict on test set (last 20%)
    n_test = len(X) // 5
    X_test = X[-n_test:]
    y_test = y[-n_test:]
    
    weights = gating.predict(X_test)
    predictions = weights.argmax(axis=1)
    
    # Filter valid
    mask = ~np.isnan(y_test)
    accuracy = (predictions[mask] == y_test[mask]).mean()
    entropy = gating.get_entropy(weights)
    
    print(f"Test Accuracy: {accuracy:.1%}")
    print(f"Average Entropy: {entropy:.3f} (low=confident, high=uncertain)")
    
    # Show weight distribution
    print("\nAverage Expert Weights:")
    avg_weights = weights.mean(axis=0)
    for i, expert in enumerate(gating.EXPERTS):
        bar = "█" * int(avg_weights[i] * 40)
        print(f"  {expert:<12}: {avg_weights[i]:.1%} {bar}")
    
    # Save
    gating.save(output_path)
    
    return gating


if __name__ == "__main__":
    gating = train_dynamic_gating()
