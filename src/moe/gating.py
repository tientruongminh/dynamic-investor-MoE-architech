"""
Gating Network Module
=====================
MLP-based Gating Network để phân bổ trọng số cho 5 Experts.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
import os
import joblib

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not installed. Using numpy fallback.")


class MLPGatingNetwork:
    """
    MLP Gating Network với PyTorch.
    
    Input: Macro features (VIX, Yield, DXY, etc.)
    Output: Softmax weights cho 5 Experts
    """
    
    CLUSTERS = ['Growth', 'Value', 'Cyclical', 'Defensive', 'Speculative']
    
    def __init__(
        self,
        input_dim: int = 6,
        hidden_dims: list = [32, 16],
        n_experts: int = 5,
        lr: float = 0.001
    ):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.n_experts = n_experts
        self.lr = lr
        
        if TORCH_AVAILABLE:
            self._build_model()
        else:
            self.model = None
    
    def _build_model(self):
        """Build PyTorch MLP."""
        layers = []
        prev_dim = self.input_dim
        
        for hidden_dim in self.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, self.n_experts))
        layers.append(nn.Softmax(dim=1))
        
        self.model = nn.Sequential(*layers)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32
    ):
        """
        Huấn luyện Gating Network.
        
        Args:
            X: Macro features (n_samples, input_dim)
            y: Target weights (n_samples, n_experts) - based on historical performance
        """
        if not TORCH_AVAILABLE:
            print("PyTorch not available. Skipping training.")
            return self
        
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        criterion = nn.MSELoss()  # Or KL divergence
        
        print(f"Training Gating Network for {epochs} epochs...")
        
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Dự báo expert weights."""
        if not TORCH_AVAILABLE:
            return self._fallback_predict(X)
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            weights = self.model(X_tensor).numpy()
        
        return weights
    
    def predict_single(self, macro_features: dict) -> Dict[str, float]:
        """Dự báo weights cho single timestep."""
        feature_order = ['VIX', 'YIELD_10Y', 'DXY', 'VIX_CHG_5D', 'YIELD_CHG_5D', 'MKT_MOM_20D']
        X = np.array([[macro_features.get(f, 0) for f in feature_order]])
        
        if TORCH_AVAILABLE and self.model is not None:
            weights = self.predict(X)[0]
        else:
            weights = self._fallback_predict(X)[0]
        
        return dict(zip(self.CLUSTERS, weights))
    
    def _fallback_predict(self, X: np.ndarray) -> np.ndarray:
        """Rule-based fallback when PyTorch not available."""
        n_samples = X.shape[0]
        weights = np.zeros((n_samples, self.n_experts))
        
        for i in range(n_samples):
            vix = X[i, 0] if X.shape[1] > 0 else 20
            
            # Base weights
            w = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
            
            # VIX-based adjustments
            if vix > 30:
                # High fear: increase Defensive, decrease Growth/Speculative
                w[0] -= 0.08  # Growth
                w[3] += 0.12  # Defensive
                w[4] -= 0.04  # Speculative
            elif vix > 25:
                w[0] -= 0.05
                w[3] += 0.08
                w[4] -= 0.03
            elif vix < 15:
                # Low fear: increase Growth/Speculative
                w[0] += 0.08  # Growth
                w[3] -= 0.06  # Defensive
                w[4] += 0.05  # Speculative
            elif vix < 18:
                w[0] += 0.05
                w[4] += 0.03
            
            # Ensure non-negative and normalize
            w = np.clip(w, 0.05, 0.4)
            w = w / w.sum()
            
            weights[i] = w
        
        return weights
    
    def save(self, path: str):
        """Save model."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        if TORCH_AVAILABLE and self.model is not None:
            torch.save({
                'model_state': self.model.state_dict(),
                'input_dim': self.input_dim,
                'hidden_dims': self.hidden_dims,
                'n_experts': self.n_experts
            }, path)
        else:
            joblib.dump({'fallback': True}, path)
        
        print(f"Saved Gating Network to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'MLPGatingNetwork':
        """Load model."""
        if TORCH_AVAILABLE:
            data = torch.load(path)
            if 'fallback' in data:
                return cls()
            
            gating = cls(
                input_dim=data['input_dim'],
                hidden_dims=data['hidden_dims'],
                n_experts=data['n_experts']
            )
            gating.model.load_state_dict(data['model_state'])
            return gating
        else:
            return cls()


class RuleBasedGating:
    """
    Alternative: Rule-based Gating với logic đơn giản.
    Không cần training, deterministic.
    """
    
    CLUSTERS = ['Growth', 'Value', 'Cyclical', 'Defensive', 'Speculative']
    
    def predict(self, vix: float, yield_10y: float, dxy: float) -> Dict[str, float]:
        """
        Rule-based weight allocation.
        """
        weights = {
            'Growth': 0.20,
            'Value': 0.20,
            'Cyclical': 0.20,
            'Defensive': 0.20,
            'Speculative': 0.20
        }
        
        # VIX-based rules
        if vix > 30:
            weights['Defensive'] += 0.15
            weights['Value'] += 0.05
            weights['Growth'] -= 0.10
            weights['Speculative'] -= 0.08
            weights['Cyclical'] -= 0.02
        elif vix > 25:
            weights['Defensive'] += 0.10
            weights['Growth'] -= 0.06
            weights['Speculative'] -= 0.04
        elif vix < 15:
            weights['Growth'] += 0.10
            weights['Speculative'] += 0.06
            weights['Defensive'] -= 0.12
            weights['Cyclical'] += 0.04
        elif vix < 18:
            weights['Growth'] += 0.05
            weights['Speculative'] += 0.03
            weights['Defensive'] -= 0.06
        
        # Yield-based rules
        if yield_10y > 5.0:
            weights['Value'] += 0.08
            weights['Growth'] -= 0.05
            weights['Speculative'] -= 0.03
        elif yield_10y < 3.0:
            weights['Growth'] += 0.05
            weights['Value'] -= 0.03
        
        # DXY-based rules (strong dollar)
        if dxy > 105:
            weights['Defensive'] += 0.03
            weights['Cyclical'] -= 0.03
        elif dxy < 95:
            weights['Cyclical'] += 0.03
            weights['Defensive'] -= 0.02
        
        # Normalize
        total = sum(weights.values())
        weights = {k: max(0.05, v/total) for k, v in weights.items()}  # Min 5% each
        total = sum(weights.values())
        weights = {k: v/total for k, v in weights.items()}
        
        return weights
    
    def predict_from_df(self, macro_row: pd.Series) -> Dict[str, float]:
        """Predict from DataFrame row."""
        vix = macro_row.get('VIX', 20)
        yield_10y = macro_row.get('YIELD_10Y', 4.0)
        dxy = macro_row.get('DXY', 100)
        
        return self.predict(vix, yield_10y, dxy)


def create_training_targets(
    expert_returns: Dict[str, pd.Series],
    lookback: int = 20
) -> pd.DataFrame:
    """
    Create training targets based on historical expert performance.
    The expert with best recent returns gets higher weight.
    """
    # Combine returns
    returns_df = pd.DataFrame(expert_returns)
    
    # Rolling sharpe-like score
    rolling_mean = returns_df.rolling(lookback).mean()
    rolling_std = returns_df.rolling(lookback).std() + 1e-6
    scores = rolling_mean / rolling_std
    
    # Softmax to get weights
    def softmax(x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()
    
    weights = scores.apply(softmax, axis=1)
    
    return weights.dropna()


if __name__ == "__main__":
    # Test rule-based gating
    gating = RuleBasedGating()
    
    test_cases = [
        (35, 4.5, 100, "High VIX (Fear)"),
        (12, 3.5, 98, "Low VIX (Greed)"),
        (20, 5.5, 105, "High Yield + Strong Dollar"),
        (22, 4.0, 100, "Neutral")
    ]
    
    print("="*60)
    print("RULE-BASED GATING TEST")
    print("="*60)
    
    for vix, yield_10y, dxy, scenario in test_cases:
        weights = gating.predict(vix, yield_10y, dxy)
        print(f"\n{scenario} (VIX={vix}, Yield={yield_10y}%, DXY={dxy}):")
        for cluster, weight in weights.items():
            bar = "█" * int(weight * 50)
            print(f"  {cluster:<12}: {weight:.1%} {bar}")
