"""
Utils Module - Shared Utilities
================================
Single Responsibility: Shared utility functions and constants

Contains:
    - NASDAQ ticker lists
    - Config classes
    - Helper functions
"""

# Large-cap NASDAQ tickers (updated 2026)
NASDAQ_LARGE_CAP = [
    # NASDAQ-100 Core
    'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'GOOG', 'META', 'TSLA', 'AVGO', 'COST',
    'ASML', 'PEP', 'AZN', 'CSCO', 'ADBE', 'NFLX', 'AMD', 'TMUS', 'INTC', 'TXN',
    'INTU', 'CMCSA', 'AMGN', 'QCOM', 'HON', 'AMAT', 'ISRG', 'BKNG', 'SBUX', 'VRTX',
    'MDLZ', 'GILD', 'ADI', 'LRCX', 'REGN', 'PANW', 'MU', 'MELI', 'KLAC', 'SNPS',
    'CDNS', 'PYPL', 'MAR', 'ORLY', 'MNST', 'CTAS', 'MRVL', 'ABNB', 'NXPI', 'AEP',
    'CSX', 'ADP', 'CHTR', 'KDP', 'FTNT', 'PAYX', 'ROP', 'PCAR', 'ADSK', 'FAST',
    'MCHP', 'KHC', 'ODFL', 'CPRT', 'IDXX', 'DXCM', 'MRNA', 'TEAM', 'EA', 'XEL',
    'VRSK', 'GEHC', 'TTD', 'CSGP', 'DLTR', 'WBD', 'ANSS', 'ON', 'ZS', 'BIIB',
    'ILMN', 'WBA', 'BKR', 'SIRI', 'DDOG', 'LCID', 'CRWD', 'WDAY', 'FANG', 'CEG',
    'CTSH', 'GFS', 'ALGN', 'ENPH', 'JD', 'ZM', 'ROST', 'EXC', 'EBAY', 'SPLK',
    
    # Technology Extended
    'CRM', 'IBM', 'ORCL', 'NOW', 'SHOP', 'SNOW', 'PLTR', 'UBER', 'NET', 'COIN',
    'OKTA', 'DOCU', 'TWLO', 'ROKU', 'SQ', 'SPOT', 'SNAP', 'PINS', 'LYFT', 'DASH',
    'RBLX', 'U', 'DBX', 'BOX', 'ZEN', 'ESTC', 'MDB', 'DKNG', 'BILL', 'HUBS',
    'PATH', 'CFLT', 'DOCN', 'APP', 'GTLB', 'AFRM', 'UPST', 'SOFI', 'HOOD', 'RIVN',
    
    # Semiconductors
    'ARM', 'MRVL', 'SWKS', 'QRVO', 'MPWR', 'WOLF', 'SLAB', 'MKSI', 'CRUS', 'RMBS',
    'POWI', 'DIOD', 'SYNA', 'SMTC', 'LSCC', 'SITM', 'ALGM', 'ACLS', 'FORM', 'OLED',
    
    # Healthcare/Biotech
    'LLY', 'UNH', 'JNJ', 'ABBV', 'MRK', 'PFE', 'TMO', 'ABT', 'DHR', 'BMY',
    'CVS', 'CI', 'MCK', 'VEEV', 'SGEN', 'ALNY', 'EXAS', 'RARE', 'NBIX', 'BMRN',
    'INCY', 'JAZZ', 'SRPT', 'UTHR', 'IONS', 'HALO', 'NTRA', 'PCVX', 'LEGN', 'IMVT',
    
    # Financials
    'JPM', 'V', 'MA', 'BAC', 'WFC', 'GS', 'MS', 'BLK', 'SCHW', 'C',
    'AXP', 'SPGI', 'ICE', 'CME', 'AON', 'MCO', 'MSCI', 'NDAQ', 'CBOE', 'MKTX',
    'TW', 'LPLA', 'RJF', 'IBKR', 'TROW', 'SEIC', 'VIRT', 'FRHC', 'SNEX', 'TREE',
    
    # Consumer
    'HD', 'WMT', 'MCD', 'NKE', 'LULU', 'TGT', 'LOW', 'TJX', 'YUM', 'DG',
    'DLTR', 'FIVE', 'ULTA', 'RH', 'WSM', 'ETSY', 'BURL', 'DECK', 'CROX', 'SKECHERS',
    'GPS', 'ANF', 'URBN', 'EXPR', 'AEO', 'BBWI', 'VSCO', 'FL', 'HIBB', 'GES',
    
    # Industrials
    'CAT', 'DE', 'UPS', 'FDX', 'LMT', 'RTX', 'BA', 'NOC', 'GD', 'GE',
    'MMM', 'EMR', 'ETN', 'ROK', 'PH', 'ITW', 'SWK', 'DOV', 'AME', 'FTV',
    'GNRC', 'TTC', 'IR', 'NDSN', 'XYL', 'IDEX', 'RBC', 'JBHT', 'EXPD', 'CHRW',
    
    # Defense
    'KTOS', 'MRCY', 'HII', 'LHX', 'TXT', 'CW', 'HXL', 'SPR', 'TDG', 'AXON',
    
    # Communications/Media
    'SATS', 'VSAT', 'IRDM', 'GSAT', 'ASTS', 'BWXT', 'MAXR', 'SPCE', 'RKT', 'VIASAT',
    'DIS', 'PARA', 'FOX', 'FOXA', 'NWSA', 'NWS', 'LYV', 'MTCH', 'IAC', 'GOOGL',
    
    # Energy/Utilities
    'XOM', 'CVX', 'COP', 'EOG', 'PXD', 'DVN', 'OXY', 'HES', 'APA', 'FANG',
    'NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'SRE', 'XEL', 'ED', 'WEC',
    
    # REITs/Real Estate
    'PLD', 'AMT', 'CCI', 'EQIX', 'PSA', 'SPG', 'O', 'DLR', 'WELL', 'AVB',
    'SBAC', 'EQR', 'VTR', 'ARE', 'ESS', 'MAA', 'UDR', 'INVH', 'CPT', 'PEAK',
    
    # Storage/Memory
    'WDC', 'STX', 'NTAP', 'PSTG', 'NEWR',
    
    # Additional Large Caps
    'LBRDK', 'LBRDA', 'SIRI', 'FWONK', 'FWONA', 'LSXMK', 'LSXMA', 'LIBERTY',
    'CHKP', 'AKAM', 'FFIV', 'JNPR', 'CIEN', 'INFN', 'COMM', 'VIAV', 'LITE',
    'PEGA', 'MANH', 'PAYC', 'PCTY', 'WEX', 'FOUR', 'GPN', 'FISV', 'FIS',
    'TRMB', 'KEYS', 'TER', 'COHR', 'NOVT', 'TECH', 'A', 'WAT', 'MTD'
]

# Clean tickers (remove known delisted/problematic)
EXCLUDED_TICKERS = [
    'ATVI', 'TWTR', 'LUMN', 'VIAC', 'DISCA', 'DISCK', 'WLTW', 'INFO', 'XLNX',
    'FISV', 'CTXS', 'NLOK', 'CERN', 'KSU', 'ANTM'
]

def get_clean_tickers() -> list:
    """Get cleaned ticker list."""
    return [t for t in NASDAQ_LARGE_CAP if t not in EXCLUDED_TICKERS]


# Config helpers
def ensure_dir(path: str):
    """Ensure directory exists."""
    import os
    os.makedirs(path, exist_ok=True)
