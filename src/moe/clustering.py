"""
MoE Clustering Module - Gemini API Integration
===============================================
Phân loại 527 stocks vào 5 Expert Clusters dựa trên Gemini AI.
"""

import os
import json
import time
from typing import Dict, List
from pathlib import Path
import google.generativeai as genai
from dotenv import load_dotenv


# Load API key
load_dotenv()
API_KEY = os.getenv("API_KEY")


class GeminiClusterer:
    """
    Sử dụng Gemini để phân loại stocks vào 5 clusters:
    - Growth: High-growth tech, biotech, disruptive companies
    - Value: Mature, dividend-paying, low P/E stocks
    - Cyclical: Auto, Materials, Industrials sensitive to GDP
    - Defensive: Utilities, Staples, Healthcare - recession-resistant
    - Speculative: Pre-revenue, high beta, penny stocks
    """
    
    CLUSTERS = ['Growth', 'Value', 'Cyclical', 'Defensive', 'Speculative']
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or API_KEY
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        self.cache = {}
    
    def _create_prompt(self, ticker: str, sector: str, business_summary: str) -> str:
        """Create classification prompt."""
        return f"""You are a senior portfolio manager. Classify this stock into exactly ONE category based on its characteristics.

CATEGORIES:
1. Growth - High revenue growth (>20%), high P/E, tech/innovation focused, reinvests profits
2. Value - Low P/E, high dividend yield, stable mature business, undervalued
3. Cyclical - Highly sensitive to economic cycles (autos, materials, industrials, discretionary)
4. Defensive - Recession-resistant (utilities, staples, healthcare, essential services)
5. Speculative - Pre-revenue, high volatility (beta>2), unprofitable, early-stage

STOCK INFORMATION:
- Ticker: {ticker}
- Sector: {sector}
- Business: {business_summary[:500]}

Reply with ONLY one word from: Growth, Value, Cyclical, Defensive, Speculative
Do NOT include any explanation or punctuation."""

    def classify_stock(self, ticker: str, sector: str, business_summary: str) -> str:
        """Classify a single stock using Gemini."""
        if ticker in self.cache:
            return self.cache[ticker]
        
        prompt = self._create_prompt(ticker, sector, business_summary)
        
        try:
            response = self.model.generate_content(prompt)
            cluster = response.text.strip()
            
            # Validate response
            if cluster not in self.CLUSTERS:
                # Try to extract from response
                for c in self.CLUSTERS:
                    if c.lower() in cluster.lower():
                        cluster = c
                        break
                else:
                    cluster = 'Growth'  # Default fallback
            
            self.cache[ticker] = cluster
            return cluster
            
        except Exception as e:
            print(f"Error classifying {ticker}: {e}")
            return self._fallback_classify(sector)
    
    def _fallback_classify(self, sector: str) -> str:
        """Rule-based fallback classification."""
        sector_lower = sector.lower() if sector else ''
        
        if any(s in sector_lower for s in ['technology', 'communication']):
            return 'Growth'
        elif any(s in sector_lower for s in ['utilities', 'consumer staples', 'healthcare']):
            return 'Defensive'
        elif any(s in sector_lower for s in ['financial', 'real estate']):
            return 'Value'
        elif any(s in sector_lower for s in ['materials', 'industrials', 'consumer discretionary', 'energy']):
            return 'Cyclical'
        else:
            return 'Growth'
    
    def classify_batch(
        self, 
        stocks_info: Dict[str, dict],
        delay: float = 0.5
    ) -> Dict[str, str]:
        """
        Classify multiple stocks with rate limiting.
        
        Args:
            stocks_info: {ticker: {'sector': str, 'business_summary': str}}
            delay: Seconds between API calls
            
        Returns:
            {ticker: cluster_name}
        """
        results = {}
        total = len(stocks_info)
        
        for i, (ticker, info) in enumerate(stocks_info.items()):
            sector = info.get('sector', 'Unknown')
            summary = info.get('longBusinessSummary', info.get('business_summary', ''))
            
            cluster = self.classify_stock(ticker, sector, summary)
            results[ticker] = cluster
            
            if (i + 1) % 10 == 0:
                print(f"Classified {i+1}/{total} stocks...")
            
            time.sleep(delay)  # Rate limiting
        
        return results
    
    def save_clusters(self, clusters: Dict[str, str], output_path: str):
        """Save cluster assignments to JSON."""
        # Also create reverse mapping
        cluster_stocks = {c: [] for c in self.CLUSTERS}
        for ticker, cluster in clusters.items():
            cluster_stocks[cluster].append(ticker)
        
        output = {
            'stock_to_cluster': clusters,
            'cluster_to_stocks': cluster_stocks,
            'cluster_counts': {c: len(stocks) for c, stocks in cluster_stocks.items()}
        }
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"Saved clusters to {output_path}")
        print("Cluster distribution:")
        for c, count in output['cluster_counts'].items():
            print(f"  {c}: {count} stocks")
        
        return output


def load_stocks_info(info_dir: str = "./data/nasdaq_574/info") -> Dict[str, dict]:
    """Load stock info from JSON files."""
    stocks_info = {}
    info_path = Path(info_dir)
    
    if not info_path.exists():
        print(f"Warning: {info_dir} not found")
        return stocks_info
    
    for f in info_path.glob("*.json"):
        ticker = f.stem
        try:
            with open(f) as fp:
                stocks_info[ticker] = json.load(fp)
        except Exception as e:
            print(f"Error loading {f}: {e}")
    
    print(f"Loaded info for {len(stocks_info)} stocks")
    return stocks_info


def run_clustering(
    info_dir: str = "./data/nasdaq_574/info",
    output_path: str = "./outputs/moe/clusters.json",
    use_cache: bool = True
):
    """
    Main function to run Gemini clustering.
    """
    # Check for existing cache
    if use_cache and os.path.exists(output_path):
        print(f"Loading cached clusters from {output_path}")
        with open(output_path) as f:
            return json.load(f)
    
    # Load stock info
    stocks_info = load_stocks_info(info_dir)
    
    if not stocks_info:
        print("No stock info found!")
        return None
    
    # Initialize clusterer
    clusterer = GeminiClusterer()
    
    # Classify all stocks
    print(f"\nClassifying {len(stocks_info)} stocks using Gemini...")
    clusters = clusterer.classify_batch(stocks_info)
    
    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result = clusterer.save_clusters(clusters, output_path)
    
    return result


if __name__ == "__main__":
    result = run_clustering()
