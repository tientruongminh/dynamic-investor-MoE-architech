"""
Structural EDA Module - Chapter 5
==================================
Implements structural pattern recognition using unsupervised learning:
- Hierarchical Clustering
- Clustermap Matrix
- Sector Treemap

Theory: Market structure is hierarchical. Unsupervised learning facilitates
the discovery of 'Risk-on/Risk-off' clusters and sector rotation patterns.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import os
from typing import Dict, List, Tuple

class StructuralAnalyzer:
    """
    Analyzes market structure using clustering techniques.
    """
    
    def __init__(self, prices: pd.DataFrame, info: Dict = None):
        self.prices = prices
        self.returns = prices.pct_change().dropna()
        self.info = info or {}
        
        # Calculate correlation matrix
        self.corr_matrix = self.returns.corr()
        
    def hierarchical_clustering(self, method: str = 'ward', metric: str = 'euclidean') -> Dict:
        """
        Perform hierarchical clustering on the correlation matrix.
        
        Args:
            method: Linkage method (ward, single, complete, average)
            metric: Distance metric
            
        Returns:
            Dict containing linkage matrix and clusters
        """
        # Convert correlation to distance matrix (d = sqrt(2(1-rho)))
        dist_matrix = np.sqrt(2 * (1 - self.corr_matrix))
        
        # Perform linkage
        # We use squareform to convert distance matrix to condensed form for linkage
        from scipy.spatial.distance import squareform
        condensed_dist = squareform(dist_matrix, checks=False)
        
        Z = linkage(condensed_dist, method=method, metric=metric)
        
        # Form flat clusters (e.g., 10 clusters)
        clusters = fcluster(Z, t=10, criterion='maxclust')
        
        cluster_map = pd.Series(clusters, index=self.corr_matrix.index, name='Cluster')
        
        return {
            'linkage': Z,
            'clusters': cluster_map
        }
        
    def get_sector_data(self) -> pd.DataFrame:
        """
        Aggregate sector data including market cap and count.
        """
        data = []
        for ticker in self.prices.columns:
            inf = self.info.get(ticker, {})
            sector = inf.get('sector', 'Unknown')
            mcap = inf.get('marketCap', 0)
            data.append({'Ticker': ticker, 'Sector': sector, 'MarketCap': mcap})
            
        df = pd.DataFrame(data)
        
        # Aggregate by sector
        sector_stats = df.groupby('Sector').agg({
            'MarketCap': 'sum',
            'Ticker': 'count'
        }).rename(columns={'Ticker': 'Count'}).sort_values('MarketCap', ascending=False)
        
        return sector_stats

class StructuralPlotter:
    """Generates structural visualizations."""
    
    def __init__(self, output_dir: str = "./outputs/structural"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def plot_dendrogram(self, linkage_matrix, labels=None, filename="dendrogram.png"):
        """Plot hierarchical clustering dendrogram."""
        plt.figure(figsize=(15, 8))
        dendrogram(
            linkage_matrix,
            labels=labels,
            leaf_rotation=90,
            leaf_font_size=8,
            truncate_mode='lastp',  # Show only last p merged clusters
            p=50,
            show_contracted=True
        )
        plt.title('Hierarchical Clustering Dendrogram (Risk Structure)', fontsize=14, fontweight='bold')
        plt.xlabel('Cluster Size / Ticker')
        plt.ylabel('Distance')
        plt.tight_layout()
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, dpi=150)
        plt.close()
        return path

    def plot_clustermap(self, corr_matrix, filename="clustermap.png"):
        """
        Plot sorted clustermap matrix.
        Validates 'Risk-on Cluster' hypothesis.
        """
        # Create Clustermap
        g = sns.clustermap(
            corr_matrix,
            method='ward',
            cmap='RdBu_r',
            center=0,
            figsize=(12, 12),
            xticklabels=False,
            yticklabels=False,
            dendrogram_ratio=(.15, .15),
            cbar_pos=(0.02, 0.8, 0.03, 0.15),
            cbar_kws={'label': 'Correlation'}
        )
        
        g.fig.suptitle('Market Correlation Structure (Clustermap)', fontsize=16, fontweight='bold', y=0.98)
        
        # Annotate Risk-on Blocks intuition (conceptual)
        # Note: In a real automated plot, finding exact coordinates for annotation is hard without manual intervention,
        # so we keep appropriate titles/labels.
        
        path = os.path.join(self.output_dir, filename)
        g.savefig(path, dpi=150)
        plt.close()
        return path

    def plot_sector_treemap(self, sector_stats, filename="sector_treemap.png"):
        """
        Plot sector distribution as Treemap.
        Note: squarify is required, if not available fall back to bar chart.
        """
        try:
            import squarify
            
            plt.figure(figsize=(14, 8))
            
            # Normalize for color map
            cmap = matplotlib.cm.viridis
            mini, maxi = sector_stats['Count'].min(), sector_stats['Count'].max()
            norm = matplotlib.colors.Normalize(vmin=mini, vmax=maxi)
            colors = [cmap(norm(value)) for value in sector_stats['Count']]
            
            labels = [f"{idx}\n({row['Count']} stocks)" for idx, row in sector_stats.iterrows()]
            
            squarify.plot(
                sizes=sector_stats['MarketCap'],
                label=labels,
                color=colors,
                alpha=0.8,
                text_kwargs={'fontsize': 10, 'package': 'center', 'wrap': True}
            )
            
            plt.title('Sector Distribution by Market Cap (Treemap)', fontsize=14, fontweight='bold')
            plt.axis('off')
            
        except ImportError:
            print("squarify not installed, falling back to bar chart.")
            # Fallback to horizontal bar
            plt.figure(figsize=(12, 8))
            bars = plt.barh(sector_stats.index, sector_stats['MarketCap'], color='steelblue')
            plt.title('Sector Distribution by Market Cap', fontsize=14, fontweight='bold')
            plt.xlabel('Total Market Cap')
        
        plt.tight_layout()
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, dpi=150)
        plt.close()
        return path

def run_structural_analysis(
    prices_file: str = "./outputs/clean/processed_prices.csv",
    info_dir: str = "./data/nasdaq_574",
    output_dir: str = "./outputs/structural"
) -> Dict:
    """Run complete structural analysis."""
    print("\n" + "="*60)
    print("CHAPTER 5: STRUCTURAL EDA & CLUSTERING")
    print("="*60)
    
    # Load prices
    print("Loading data...")
    prices = pd.read_csv(prices_file, index_col=0, parse_dates=True)
    
    # Load info (mocked or loaded via shared util if implementing full loader)
    # For now, we assume simple loading or re-use existing loader logic
    # To keep it standalone/robust, we'll try to load from the main data module if possible
    # or just proceed with prices if info is complex to load here. 
    # Let's try to load info using the DataLoader from src.data if available.
    
    import sys
    sys.path.insert(0, '.')
    from src.data import DataLoader
    
    loader = DataLoader(info_dir)
    try:
        info = loader.load_info()
    except:
        print("Warning: Could not load info file. Sector analysis will be limited.")
        info = {}

    analyzer = StructuralAnalyzer(prices, info)
    plotter = StructuralPlotter(output_dir)
    
    # 1. Hierarchical Clustering
    print("1. Performing Hierarchical Clustering...")
    hc_res = analyzer.hierarchical_clustering()
    
    # 2. Sector Analysis
    print("2. Analyzing Sector Structure...")
    sector_res = analyzer.get_sector_data()
    
    # 3. Visualizations
    print("3. Generating Visualizations...")
    plots = {
        'dendrogram': plotter.plot_dendrogram(hc_res['linkage']),
        'clustermap': plotter.plot_clustermap(analyzer.corr_matrix),
        'treemap': plotter.plot_sector_treemap(sector_res)
    }
    
    print(f"Plots saved to {output_dir}")
    
    return {
        'clusters': hc_res['clusters'],
        'sector_stats': sector_res,
        'plots': plots
    }

if __name__ == "__main__":
    run_structural_analysis()
