"""
EDA Module - Exploratory Data Analysis
=======================================
Single Responsibility: Analyze raw data to inform preprocessing decisions

Classes:
    - DataQualityAnalyzer: Measures 6 data quality dimensions
    - MissingAnalyzer: Analyzes missing value patterns
    - OutlierAnalyzer: Detects outliers and anomalies
    - DistributionAnalyzer: Analyzes distribution properties
    - StationarityAnalyzer: Tests for stationarity
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class DataQualityAnalyzer:
    """
    Measures 6 data quality dimensions:
    - Accuracy, Completeness, Consistency, Uniqueness, Timeliness, Validity
    
    Open/Closed: Add new dimensions without modifying existing code.
    """
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.scores = {}
    
    def accuracy(self) -> int:
        """Check for unrealistic values (negative prices, etc.)"""
        negative_prices = (self.data < 0).sum().sum()
        extreme_high = (self.data > 1e6).sum().sum()
        
        if negative_prices == 0 and extreme_high == 0:
            self.scores['accuracy'] = 100
        else:
            self.scores['accuracy'] = max(0, 100 - (negative_prices + extreme_high))
        
        return self.scores['accuracy']
    
    def completeness(self) -> int:
        """Measure missing data percentage."""
        total_cells = self.data.size
        missing_cells = self.data.isna().sum().sum()
        completeness_pct = (1 - missing_cells / total_cells) * 100
        self.scores['completeness'] = int(completeness_pct)
        return self.scores['completeness']
    
    def consistency(self) -> int:
        """Check index ordering and type uniformity."""
        is_sorted = self.data.index.is_monotonic_increasing
        no_dups = not self.data.index.has_duplicates
        
        score = 100 if (is_sorted and no_dups) else 50
        self.scores['consistency'] = score
        return score
    
    def uniqueness(self) -> int:
        """Check for duplicate rows/columns."""
        dup_cols = self.data.columns.duplicated().sum()
        dup_rows = self.data.index.duplicated().sum()
        
        score = 100 if (dup_cols == 0 and dup_rows == 0) else 80
        self.scores['uniqueness'] = score
        return score
    
    def timeliness(self, max_age_days: int = 7) -> int:
        """Check data recency."""
        try:
            latest = pd.to_datetime(self.data.index[-1])
            age = (pd.Timestamp.now() - latest).days
            score = max(0, 100 - age * 5)
        except:
            score = 90
        
        self.scores['timeliness'] = score
        return score
    
    def validity(self) -> int:
        """Check data types and formats."""
        is_datetime_index = isinstance(self.data.index, pd.DatetimeIndex)
        all_numeric = all(np.issubdtype(self.data[c].dtype, np.number) for c in self.data.columns)
        
        score = 100 if (is_datetime_index and all_numeric) else 75
        self.scores['validity'] = score
        return score
    
    def run_all(self) -> Dict[str, int]:
        """Run all quality checks and return scores."""
        self.accuracy()
        self.completeness()
        self.consistency()
        self.uniqueness()
        self.timeliness()
        self.validity()
        
        self.scores['overall'] = sum(self.scores.values()) / len(self.scores)
        return self.scores


class MissingAnalyzer:
    """
    Analyzes missing value patterns.
    
    Single Responsibility: Only analyzes, does not fix.
    """
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
    
    def get_missing_summary(self) -> Dict:
        """Get summary of missing data."""
        missing_pct = self.data.isna().sum() / len(self.data) * 100
        
        return {
            'total_missing': self.data.isna().sum().sum(),
            'total_cells': self.data.size,
            'missing_pct': self.data.isna().sum().sum() / self.data.size * 100,
            'excellent': (missing_pct < 5).sum(),
            'good': ((missing_pct >= 5) & (missing_pct < 20)).sum(),
            'fair': ((missing_pct >= 20) & (missing_pct < 50)).sum(),
            'poor': (missing_pct >= 50).sum()
        }
    
    def get_max_gaps(self, sample_n: int = 50) -> Dict[str, int]:
        """Calculate maximum consecutive gaps for each stock."""
        gaps = {}
        sample_cols = self.data.columns[:sample_n]
        
        for col in sample_cols:
            series = self.data[col]
            is_nan = series.isna()
            max_gap = 0
            current_gap = 0
            
            for val in is_nan:
                if val:
                    current_gap += 1
                    max_gap = max(max_gap, current_gap)
                else:
                    current_gap = 0
            
            gaps[col] = max_gap
        
        return gaps
    
    def get_stocks_to_remove(self, max_gap_threshold: int = 20) -> List[str]:
        """Get list of stocks to remove based on gap threshold."""
        gaps = self.get_max_gaps(sample_n=len(self.data.columns))
        return [ticker for ticker, gap in gaps.items() if gap > max_gap_threshold]


class OutlierAnalyzer:
    """
    Detects outliers in returns data.
    
    Single Responsibility: Only detects, does not fix.
    """
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.returns = data.pct_change()
    
    def get_outlier_summary(self, iqr_multiplier: float = 3.0) -> Dict:
        """Get summary of outliers using IQR method."""
        returns_flat = self.returns.values.flatten()
        returns_flat = returns_flat[~np.isnan(returns_flat)]
        
        q1 = np.percentile(returns_flat, 25)
        q3 = np.percentile(returns_flat, 75)
        iqr = q3 - q1
        
        lower = q1 - iqr_multiplier * iqr
        upper = q3 + iqr_multiplier * iqr
        
        outliers = ((returns_flat < lower) | (returns_flat > upper)).sum()
        
        return {
            'total_outliers': outliers,
            'outlier_pct': outliers / len(returns_flat) * 100,
            'min_return': returns_flat.min(),
            'max_return': returns_flat.max(),
            'iqr_lower': lower,
            'iqr_upper': upper
        }
    
    def get_extreme_returns(self, threshold: float = 0.5) -> pd.DataFrame:
        """Get instances of extreme returns (>50% by default)."""
        extreme = (self.returns.abs() > threshold).stack()
        extreme = extreme[extreme]
        return extreme


class DistributionAnalyzer:
    """
    Analyzes distribution properties (normality, fat tails).
    
    Single Responsibility: Only analyzes distributions.
    """
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.returns = data.pct_change()
    
    def get_kurtosis_stats(self) -> Dict:
        """Calculate kurtosis statistics."""
        kurtosis_vals = []
        
        for col in self.returns.columns:
            r = self.returns[col].dropna()
            if len(r) > 100:
                kurtosis_vals.append(stats.kurtosis(r))
        
        return {
            'mean_kurtosis': np.mean(kurtosis_vals),
            'median_kurtosis': np.median(kurtosis_vals),
            'max_kurtosis': np.max(kurtosis_vals),
            'pct_fat_tails': (np.array(kurtosis_vals) > 3).mean() * 100
        }
    
    def get_skewness_stats(self) -> Dict:
        """Calculate skewness statistics."""
        skew_vals = []
        
        for col in self.returns.columns:
            r = self.returns[col].dropna()
            if len(r) > 100:
                skew_vals.append(stats.skew(r))
        
        return {
            'mean_skewness': np.mean(skew_vals),
            'median_skewness': np.median(skew_vals),
            'pct_negative_skew': (np.array(skew_vals) < 0).mean() * 100
        }


class StationarityAnalyzer:
    """
    Tests for stationarity in time series.
    
    Single Responsibility: Only tests stationarity.
    """
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.returns = data.pct_change()
    
    def test_sample(self, sample_n: int = 20) -> Dict:
        """Test stationarity on sample of stocks using ADF test."""
        from statsmodels.tsa.stattools import adfuller
        
        results = {'prices_stationary': 0, 'returns_stationary': 0}
        sample_cols = np.random.choice(self.data.columns, min(sample_n, len(self.data.columns)), replace=False)
        
        for col in sample_cols:
            # Test prices
            prices = self.data[col].dropna()
            if len(prices) > 100:
                try:
                    result = adfuller(prices, maxlag=10)
                    if result[1] < 0.05:  # p-value < 0.05 = stationary
                        results['prices_stationary'] += 1
                except:
                    pass
            
            # Test returns
            rets = self.returns[col].dropna()
            if len(rets) > 100:
                try:
                    result = adfuller(rets, maxlag=10)
                    if result[1] < 0.05:
                        results['returns_stationary'] += 1
                except:
                    pass
        
        results['prices_pct'] = results['prices_stationary'] / len(sample_cols) * 100
        results['returns_pct'] = results['returns_stationary'] / len(sample_cols) * 100
        
        return results


class EDARunner:
    """
    Facade pattern: Runs all EDA analyses and generates summary.
    """
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.quality = DataQualityAnalyzer(data)
        self.missing = MissingAnalyzer(data)
        self.outliers = OutlierAnalyzer(data)
        self.distribution = DistributionAnalyzer(data)
        self.stationarity = StationarityAnalyzer(data)
    
    def run_all(self) -> Dict:
        """Run all EDA analyses."""
        results = {
            'quality': self.quality.run_all(),
            'missing': self.missing.get_missing_summary(),
            'outliers': self.outliers.get_outlier_summary(),
            'distribution': self.distribution.get_kurtosis_stats(),
            'stationarity': self.stationarity.test_sample()
        }
        
        return results
    
    def print_summary(self):
        """Print EDA summary to console."""
        results = self.run_all()
        
        print("\n" + "="*60)
        print("EDA SUMMARY")
        print("="*60)
        
        print(f"\nData Quality Score: {results['quality']['overall']:.1f}/100")
        print(f"Missing Data: {results['missing']['total_missing']:,} cells ({results['missing']['missing_pct']:.2f}%)")
        print(f"Outliers (>3 IQR): {results['outliers']['total_outliers']:,}")
        print(f"Mean Kurtosis: {results['distribution']['mean_kurtosis']:.2f} (normal=0)")
        print(f"Returns Stationary: {results['stationarity']['returns_pct']:.0f}%")
