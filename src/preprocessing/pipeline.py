"""
Preprocessing Module - Data Cleaning Layer
===========================================
Single Responsibility: Clean and transform data based on EDA findings

Classes:
    - MissingDataHandler: Handles missing values
    - OutlierHandler: Handles outliers
    - DataTransformer: Transforms data (returns, scaling)
    - UniverseFilter: Filters stock universe
    - PreprocessingPipeline: Orchestrates all preprocessing
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class PreprocessingConfig:
    """
    Configuration for preprocessing pipeline.
    
    Open/Closed Principle: Add new parameters without modifying existing code.
    """
    # Missing data
    ffill_max_days: int = 3
    interpolate_max_days: int = 10
    spline_max_days: int = 20
    
    # Outliers
    winsorize_lower: float = 0.005
    winsorize_upper: float = 0.995
    
    # Universe filtering
    min_coverage_pct: float = 70.0
    min_market_cap: float = 500_000_000
    min_stocks_per_sector: int = 3


class DataCleaner(ABC):
    """
    Abstract base for data cleaners.
    
    Interface Segregation: Define focused interfaces.
    """
    
    @abstractmethod
    def clean(self, data: pd.DataFrame) -> pd.DataFrame:
        pass


class MissingDataHandler(DataCleaner):
    """Handles missing values with tiered approach based on gap duration."""
    
    def __init__(self, config: PreprocessingConfig = None):
        self.config = config or PreprocessingConfig()
    
    def clean(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply tiered missing data treatment."""
        result = data.copy()
        
        # Step 1: Forward fill (short gaps: weekends, holidays)
        result = result.ffill(limit=self.config.ffill_max_days)
        
        # Step 2: Backward fill
        result = result.bfill(limit=self.config.ffill_max_days)
        
        # Step 3: Linear interpolation (medium gaps)
        result = result.interpolate(method='linear', limit=self.config.interpolate_max_days)
        
        # Step 4: Final forward fill for remaining
        result = result.ffill()
        
        return result


class OutlierHandler(DataCleaner):
    """Handles outliers using winsorization."""
    
    def __init__(self, config: PreprocessingConfig = None):
        self.config = config or PreprocessingConfig()
    
    def clean(self, data: pd.DataFrame) -> pd.DataFrame:
        """Winsorize returns and reconstruct prices."""
        returns = data.pct_change()
        
        # Calculate winsorization thresholds
        returns_flat = returns.values.flatten()
        returns_flat = returns_flat[~np.isnan(returns_flat)]
        
        lower = np.percentile(returns_flat, self.config.winsorize_lower * 100)
        upper = np.percentile(returns_flat, self.config.winsorize_upper * 100)
        
        # Clip returns
        returns_clipped = returns.clip(lower=lower, upper=upper)
        
        # Reconstruct prices from first valid price
        result = data.copy()
        for col in result.columns:
            first_valid_idx = result[col].first_valid_index()
            if first_valid_idx is not None:
                first_price = result.loc[first_valid_idx, col]
                new_prices = first_price * (1 + returns_clipped[col].fillna(0)).cumprod()
                result[col] = new_prices
        
        return result


class DataTransformer:
    """Transforms data (returns calculation, scaling)."""
    
    @staticmethod
    def to_returns(prices: pd.DataFrame) -> pd.DataFrame:
        """Convert prices to returns."""
        return prices.pct_change()
    
    @staticmethod
    def to_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
        """Convert prices to log returns."""
        return np.log(prices / prices.shift(1))
    
    @staticmethod
    def robust_scale(data: pd.DataFrame) -> pd.DataFrame:
        """Scale using median and IQR (robust to outliers)."""
        median = data.median()
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        
        return (data - median) / (iqr + 1e-10)
    
    @staticmethod
    def z_score(data: pd.DataFrame) -> pd.DataFrame:
        """Standard z-score normalization."""
        return (data - data.mean()) / (data.std() + 1e-10)


class UniverseFilter:
    """Filters stock universe based on quality criteria."""
    
    def __init__(self, config: PreprocessingConfig = None, info: Dict = None):
        self.config = config or PreprocessingConfig()
        self.info = info or {}
    
    def filter_by_coverage(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Remove stocks with insufficient data coverage."""
        coverage = (1 - data.isna().sum() / len(data)) * 100
        valid = coverage >= self.config.min_coverage_pct
        removed = coverage[~valid].index.tolist()
        return data.loc[:, valid], removed
    
    def filter_by_market_cap(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Remove stocks below market cap threshold."""
        removed = []
        valid_cols = []
        
        for col in data.columns:
            market_cap = self.info.get(col, {}).get('marketCap', 1e12)
            if market_cap >= self.config.min_market_cap:
                valid_cols.append(col)
            else:
                removed.append(col)
        
        return data[valid_cols], removed
    
    def filter_by_sector(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Remove stocks from sectors with too few stocks."""
        # Count stocks per sector
        sector_counts = {}
        stock_sectors = {}
        
        for col in data.columns:
            sector = self.info.get(col, {}).get('sector', 'Unknown')
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
            stock_sectors[col] = sector
        
        # Identify valid sectors
        valid_sectors = {s for s, c in sector_counts.items() if c >= self.config.min_stocks_per_sector}
        
        # Filter
        valid_cols = [c for c in data.columns if stock_sectors.get(c, 'Unknown') in valid_sectors]
        removed = [c for c in data.columns if c not in valid_cols]
        
        return data[valid_cols], removed


class PreprocessingPipeline:
    """
    Orchestrates the full preprocessing pipeline.
    
    Dependency Inversion: Depends on abstract DataCleaner interface.
    """
    
    def __init__(self, config: PreprocessingConfig = None, info: Dict = None):
        self.config = config or PreprocessingConfig()
        self.info = info or {}
        
        # Initialize cleaners (Dependency Injection)
        self.missing_handler = MissingDataHandler(self.config)
        self.outlier_handler = OutlierHandler(self.config)
        self.universe_filter = UniverseFilter(self.config, self.info)
        
        self.audit_log = []
    
    def run(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Run full preprocessing pipeline."""
        result = data.copy()
        
        print("\n" + "="*60)
        print("PREPROCESSING PIPELINE")
        print("="*60)
        
        initial_shape = result.shape
        
        # Step 1: Fix timezone issues
        if hasattr(result.index, 'tz') and result.index.tz is not None:
            result.index = result.index.tz_localize(None)
        
        # Step 2: Filter by coverage
        result, removed = self.universe_filter.filter_by_coverage(result)
        self._log(f"Coverage filter: removed {len(removed)} stocks")
        
        # Step 3: Handle missing data
        missing_before = result.isna().sum().sum()
        result = self.missing_handler.clean(result)
        missing_after = result.isna().sum().sum()
        self._log(f"Missing data: {missing_before:,} -> {missing_after:,}")
        
        # Step 4: Handle outliers
        result = self.outlier_handler.clean(result)
        self._log("Outliers: Winsorized at 0.5% - 99.5%")
        
        # Step 5: Filter by market cap
        result, removed = self.universe_filter.filter_by_market_cap(result)
        self._log(f"Market cap filter: removed {len(removed)} stocks")
        
        # Step 6: Filter by sector
        result, removed = self.universe_filter.filter_by_sector(result)
        self._log(f"Sector filter: removed {len(removed)} stocks")
        
        final_shape = result.shape
        
        print(f"\nUniverse: {initial_shape[1]} -> {final_shape[1]} stocks")
        print(f"Days: {final_shape[0]}")
        print("="*60)
        
        summary = {
            'initial_stocks': initial_shape[1],
            'final_stocks': final_shape[1],
            'days': final_shape[0],
            'missing_fixed': missing_before - missing_after,
            'audit_log': self.audit_log
        }
        
        return result, summary
    
    def _log(self, message: str):
        """Add message to audit log."""
        self.audit_log.append(message)
        print(f"  {message}")
