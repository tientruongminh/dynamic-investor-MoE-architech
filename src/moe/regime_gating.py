"""
Regime-Based Dynamic Gating Network
====================================
Sử dụng Random Forest + Gemini Supervisor để điều phối 5 Lasso Experts.
Không cần train neural network phức tạp.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import os
import json
import joblib

# Gemini integration
try:
    import google.generativeai as genai
    from dotenv import load_dotenv
    load_dotenv()
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


class RegimeGating:
    """
    Regime-Based Gating Network.
    
    Features đầu vào:
    1. VIX Index (Fear)
    2. Yield Spread 10Y-2Y (Economic expectation)
    3. Market Momentum PC1 (Overall market state)
    4. Gemini Macro Sentiment (Optional override)
    
    Output: Softmax weights cho 5 Experts [Growth, Value, Cyclical, Defensive, Speculative]
    """
    
    EXPERTS = ['Growth', 'Value', 'Cyclical', 'Defensive', 'Speculative']
    
    # Regime definitions
    REGIMES = {
        'RISK_ON': {'VIX_max': 15, 'momentum_min': 0.02},
        'NEUTRAL': {'VIX_range': (15, 25)},
        'RISK_OFF': {'VIX_min': 25},
        'RECESSION_FEAR': {'yield_spread_max': 0}  # Inverted yield curve
    }
    
    # Base weights per regime
    REGIME_WEIGHTS = {
        'RISK_ON': {
            'Growth': 0.35, 'Value': 0.15, 'Cyclical': 0.25, 
            'Defensive': 0.10, 'Speculative': 0.15
        },
        'NEUTRAL': {
            'Growth': 0.25, 'Value': 0.20, 'Cyclical': 0.20, 
            'Defensive': 0.25, 'Speculative': 0.10
        },
        'RISK_OFF': {
            'Growth': 0.10, 'Value': 0.25, 'Cyclical': 0.10, 
            'Defensive': 0.45, 'Speculative': 0.10
        },
        'RECESSION_FEAR': {
            'Growth': 0.05, 'Value': 0.30, 'Cyclical': 0.05, 
            'Defensive': 0.55, 'Speculative': 0.05
        }
    }
    
    def __init__(self, use_gemini: bool = True):
        self.use_gemini = use_gemini and GEMINI_AVAILABLE
        self.rf_model = None
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=3)
        self.trained = False
        
        if self.use_gemini:
            api_key = os.getenv("API_KEY")
            if api_key:
                genai.configure(api_key=api_key)
                self.gemini = genai.GenerativeModel('gemini-2.5-flash')
            else:
                self.use_gemini = False
    
    def detect_regime(
        self, 
        vix: float, 
        yield_spread: float = 1.0,
        momentum: float = 0.0
    ) -> str:
        """
        Detect current market regime based on indicators.
        """
        # Check inverted yield curve first (most severe)
        if yield_spread < 0:
            return 'RECESSION_FEAR'
        
        # VIX-based regimes
        if vix > 30:
            return 'RISK_OFF'
        elif vix > 25:
            return 'RISK_OFF' if momentum < 0 else 'NEUTRAL'
        elif vix < 15 and momentum > 0.02:
            return 'RISK_ON'
        else:
            return 'NEUTRAL'
    
    def get_base_weights(self, regime: str) -> Dict[str, float]:
        """Get base weights for a regime."""
        return self.REGIME_WEIGHTS.get(regime, self.REGIME_WEIGHTS['NEUTRAL']).copy()
    
    def calculate_features(
        self,
        vix: float,
        yield_10y: float,
        yield_2y: float = None,
        market_returns: np.ndarray = None
    ) -> Dict[str, float]:
        """
        Calculate all features for gating.
        """
        features = {
            'vix': vix,
            'vix_level': 'high' if vix > 25 else ('low' if vix < 15 else 'normal'),
            'yield_spread': yield_10y - (yield_2y or yield_10y * 0.8),
            'momentum': 0.0
        }
        
        # Calculate PC1 if returns provided
        if market_returns is not None and len(market_returns) > 20:
            # Simple momentum proxy
            features['momentum'] = np.mean(market_returns[-20:])
        
        return features
    
    def gemini_override(
        self,
        proposed_weights: Dict[str, float],
        macro_context: Dict[str, float]
    ) -> Tuple[Dict[str, float], str]:
        """
        Ask Gemini to validate/override gating decision.
        
        Returns:
            (final_weights, explanation)
        """
        if not self.use_gemini:
            return proposed_weights, "Gemini not available"
        
        prompt = f"""You are a risk manager reviewing portfolio allocation decisions.

CURRENT MACRO CONTEXT:
- VIX: {macro_context.get('vix', 'N/A')}
- Yield Spread (10Y-2Y): {macro_context.get('yield_spread', 'N/A')}
- Market Momentum: {macro_context.get('momentum', 'N/A')}

PROPOSED ALLOCATION:
{json.dumps(proposed_weights, indent=2)}

Based on current geopolitical risks (wars, sanctions, elections) and economic indicators:
1. Should we OVERRIDE this allocation?
2. If yes, what adjustment? (e.g., "reduce Speculative by 10%, add to Defensive")

Reply in JSON format:
{{"override": true/false, "adjustment": "description or null", "risk_level": "low/medium/high"}}
"""
        
        try:
            response = self.gemini.generate_content(prompt)
            text = response.text
            
            # Parse JSON from response
            import re
            json_match = re.search(r'\{[^}]+\}', text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                
                if result.get('override', False):
                    # Apply simple adjustment
                    adjusted = proposed_weights.copy()
                    if result.get('risk_level') == 'high':
                        # Shift to defensive
                        adjusted['Defensive'] = min(0.5, adjusted['Defensive'] + 0.15)
                        adjusted['Speculative'] = max(0.05, adjusted['Speculative'] - 0.10)
                        adjusted['Growth'] = max(0.05, adjusted['Growth'] - 0.05)
                        # Normalize
                        total = sum(adjusted.values())
                        adjusted = {k: v/total for k, v in adjusted.items()}
                    
                    return adjusted, f"Gemini override: {result.get('adjustment', 'risk adjustment')}"
            
            return proposed_weights, "Gemini approved allocation"
            
        except Exception as e:
            return proposed_weights, f"Gemini error: {str(e)[:50]}"
    
    def predict_weights(
        self,
        vix: float,
        yield_10y: float,
        yield_2y: float = None,
        momentum: float = 0.0,
        use_gemini_override: bool = False
    ) -> Tuple[Dict[str, float], str, Dict]:
        """
        Predict expert weights based on current macro state.
        
        Returns:
            (weights, regime, metrics)
        """
        # Calculate yield spread
        yield_spread = yield_10y - (yield_2y or yield_10y * 0.8)
        
        # Detect regime
        regime = self.detect_regime(vix, yield_spread, momentum)
        
        # Get base weights
        weights = self.get_base_weights(regime)
        
        # Fine-tune based on VIX level within regime
        if regime == 'NEUTRAL':
            # Interpolate between risk-on and risk-off
            risk_factor = (vix - 15) / 10  # 0 at VIX=15, 1 at VIX=25
            risk_factor = np.clip(risk_factor, 0, 1)
            
            weights['Defensive'] += risk_factor * 0.10
            weights['Growth'] -= risk_factor * 0.10
        
        # Normalize
        total = sum(weights.values())
        weights = {k: v/total for k, v in weights.items()}
        
        # Calculate metrics
        entropy = self._calculate_entropy(list(weights.values()))
        
        # Gemini override (optional)
        explanation = f"Regime: {regime}"
        if use_gemini_override:
            macro_context = {
                'vix': vix,
                'yield_spread': yield_spread,
                'momentum': momentum
            }
            weights, gemini_explanation = self.gemini_override(weights, macro_context)
            explanation += f" | {gemini_explanation}"
        
        metrics = {
            'regime': regime,
            'entropy': entropy,
            'vix': vix,
            'yield_spread': yield_spread
        }
        
        return weights, explanation, metrics
    
    def _calculate_entropy(self, weights: List[float]) -> float:
        """Calculate Shannon entropy of weights."""
        weights = np.array(weights)
        weights = np.clip(weights, 1e-10, 1)
        return -np.sum(weights * np.log(weights))
    
    def train_random_forest(
        self,
        macro_df: pd.DataFrame,
        expert_returns: Dict[str, pd.Series],
        lookback: int = 4
    ):
        """
        Optional: Train Random Forest for regime classification.
        Uses historical data to learn optimal regime for each macro state.
        """
        # Prepare features
        X_list = []
        y_list = []
        
        for date in macro_df.index[lookback:]:
            # Features
            vix = macro_df.loc[date, 'VIX']
            yield_10y = macro_df.loc[date, 'YIELD_10Y'] if 'YIELD_10Y' in macro_df.columns else 4.0
            momentum = macro_df.loc[date, 'MKT_MOM_20D'] if 'MKT_MOM_20D' in macro_df.columns else 0
            
            X_list.append([vix, yield_10y, momentum])
            
            # Target: which expert won in next 4 weeks?
            future_start = date
            future_end = macro_df.index[min(macro_df.index.get_loc(date) + lookback, len(macro_df) - 1)]
            
            best_expert = None
            best_return = -np.inf
            
            for expert, returns in expert_returns.items():
                if future_start in returns.index and future_end in returns.index:
                    ret = returns.loc[future_start:future_end].sum()
                    if ret > best_return:
                        best_return = ret
                        best_expert = expert
            
            if best_expert:
                y_list.append(self.EXPERTS.index(best_expert))
        
        if len(X_list) < 50:
            print("Not enough data for RF training")
            return
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        # Scale
        X_scaled = self.scaler.fit_transform(X)
        
        # Train Random Forest
        self.rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        self.rf_model.fit(X_scaled, y)
        
        accuracy = self.rf_model.score(X_scaled, y)
        print(f"Random Forest trained. Accuracy: {accuracy:.1%}")
        
        self.trained = True
    
    def predict_with_rf(self, vix: float, yield_10y: float, momentum: float) -> Dict[str, float]:
        """Predict using trained Random Forest."""
        if not self.trained or self.rf_model is None:
            return self.predict_weights(vix, yield_10y, momentum=momentum)[0]
        
        X = self.scaler.transform([[vix, yield_10y, momentum]])
        proba = self.rf_model.predict_proba(X)[0]
        
        # Map to expert weights
        weights = {}
        for i, expert in enumerate(self.EXPERTS):
            if i < len(proba):
                weights[expert] = proba[i]
            else:
                weights[expert] = 0.1
        
        # Normalize
        total = sum(weights.values())
        return {k: v/total for k, v in weights.items()}
    
    def evaluate_accuracy(
        self,
        macro_df: pd.DataFrame,
        expert_returns: Dict[str, pd.Series]
    ) -> Dict[str, float]:
        """
        Evaluate gating accuracy: does predicted top expert match actual best?
        """
        correct = 0
        total = 0
        
        for date in macro_df.index[:-4]:
            # Predict
            vix = macro_df.loc[date, 'VIX']
            yield_10y = macro_df.loc[date, 'YIELD_10Y'] if 'YIELD_10Y' in macro_df.columns else 4.0
            momentum = macro_df.loc[date, 'MKT_MOM_20D'] if 'MKT_MOM_20D' in macro_df.columns else 0
            
            weights, _, _ = self.predict_weights(vix, yield_10y, momentum=momentum)
            predicted_top = max(weights, key=weights.get)
            
            # Actual best
            future_date = macro_df.index[min(macro_df.index.get_loc(date) + 4, len(macro_df) - 1)]
            
            actual_best = None
            best_return = -np.inf
            for expert, returns in expert_returns.items():
                if date in returns.index:
                    ret = returns.loc[date:future_date].sum()
                    if ret > best_return:
                        best_return = ret
                        actual_best = expert
            
            if actual_best:
                correct += int(predicted_top == actual_best)
                total += 1
        
        accuracy = correct / total if total > 0 else 0
        return {'regime_accuracy': accuracy, 'samples': total}
    
    def save(self, path: str):
        """Save model."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            'rf_model': self.rf_model,
            'scaler': self.scaler,
            'trained': self.trained
        }, path)
    
    @classmethod
    def load(cls, path: str) -> 'RegimeGating':
        """Load model."""
        data = joblib.load(path)
        gating = cls(use_gemini=False)
        gating.rf_model = data['rf_model']
        gating.scaler = data['scaler']
        gating.trained = data['trained']
        return gating


def test_regime_gating():
    """Test the regime-based gating."""
    print("="*60)
    print("REGIME-BASED GATING TEST")
    print("="*60)
    
    gating = RegimeGating(use_gemini=False)
    
    test_cases = [
        (12, 4.5, 0.05, "RISK_ON (Low VIX, Positive Momentum)"),
        (18, 4.2, 0.01, "NEUTRAL (Normal VIX)"),
        (28, 4.0, -0.02, "RISK_OFF (High VIX)"),
        (35, 3.8, -0.05, "RISK_OFF (Crisis)"),
        (20, 4.5, 0.00, "NEUTRAL/RECESSION_FEAR (Inverted curve)")
    ]
    
    for vix, yield_10y, momentum, scenario in test_cases:
        # Simulate 2Y yield (normally lower than 10Y)
        yield_2y = yield_10y - 0.5 if "Inverted" not in scenario else yield_10y + 0.3
        
        weights, explanation, metrics = gating.predict_weights(
            vix, yield_10y, yield_2y, momentum
        )
        
        print(f"\n{scenario}")
        print(f"  VIX={vix}, Yield Spread={yield_10y-yield_2y:.2f}, Momentum={momentum}")
        print(f"  Regime: {metrics['regime']}")
        print(f"  Entropy: {metrics['entropy']:.3f}")
        print("  Weights:")
        for expert, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
            bar = "█" * int(weight * 30)
            print(f"    {expert:<12}: {weight:.0%} {bar}")
    
    return gating


if __name__ == "__main__":
    gating = test_regime_gating()
