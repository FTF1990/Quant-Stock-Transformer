"""
Cross-Market Data Fetcher

Fetch and align stock data across multiple markets (US, Japan, China, Hong Kong)
for cross-market prediction using the SST framework.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class CrossMarketDataFetcher:
    """
    Fetch stock data across multiple markets and align timestamps
    for cross-market prediction.

    Supports:
    - US: Standard tickers (AAPL, NVDA, etc.)
    - Japan: .T suffix (6758.T for Sony)
    - China: Try both A-share code and .SS/.SZ suffixes
    - Hong Kong: .HK suffix (00700.HK for Tencent)
    """

    def __init__(self):
        self.market_suffixes = {
            'US': '',
            'JP': '.T',
            'CN': '.SS',  # Shanghai by default
            'HK': '.HK'
        }

        self.market_names = {
            'US': 'United States',
            'JP': 'Japan',
            'CN': 'China',
            'HK': 'Hong Kong'
        }

    def fetch_stock(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        market: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch data for a single stock.

        Args:
            ticker: Stock ticker (with or without suffix)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            market: Market code (US, JP, CN, HK)

        Returns:
            DataFrame with OHLCV data
        """
        # Ensure ticker has correct suffix
        if market and market in self.market_suffixes:
            suffix = self.market_suffixes[market]
            if suffix and not ticker.endswith(suffix):
                ticker = ticker + suffix

        try:
            df = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                progress=False,
                show_errors=False
            )

            if df.empty:
                # For CN stocks, try alternative suffix
                if market == 'CN' and ticker.endswith('.SS'):
                    ticker_sz = ticker.replace('.SS', '.SZ')
                    df = yf.download(
                        ticker_sz,
                        start=start_date,
                        end=end_date,
                        progress=False,
                        show_errors=False
                    )

            if not df.empty:
                # Add ticker column
                df['Ticker'] = ticker
                return df
            else:
                print(f"⚠️ No data for {ticker}")
                return pd.DataFrame()

        except Exception as e:
            print(f"❌ Error fetching {ticker}: {e}")
            return pd.DataFrame()

    def fetch_market_stocks(
        self,
        tickers: List[str],
        market: str,
        start_date: str,
        end_date: str,
        show_progress: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch multiple stocks from a single market.

        Args:
            tickers: List of ticker symbols
            market: Market code (US, JP, CN, HK)
            start_date: Start date
            end_date: End date
            show_progress: Show progress bar

        Returns:
            Dict mapping ticker to DataFrame
        """
        print(f"📥 Fetching {len(tickers)} stocks from {self.market_names.get(market, market)}...")

        data = {}
        iterator = tqdm(tickers) if show_progress else tickers

        for ticker in iterator:
            df = self.fetch_stock(ticker, start_date, end_date, market)
            if not df.empty:
                data[ticker] = df

        print(f"✅ Fetched {len(data)}/{len(tickers)} stocks successfully")
        return data

    def fetch_all_markets(
        self,
        stocks_by_market: Dict[str, List[str]],
        start_date: str,
        end_date: str,
        show_progress: bool = True
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Fetch stocks from multiple markets.

        Args:
            stocks_by_market: Dict with market codes as keys, ticker lists as values
                             e.g., {'US': ['AAPL', 'NVDA'], 'JP': ['6758.T']}
            start_date: Start date
            end_date: End date
            show_progress: Show progress bar

        Returns:
            Nested dict: {market: {ticker: DataFrame}}
        """
        all_data = {}

        for market, tickers in stocks_by_market.items():
            if not tickers:
                continue

            market_data = self.fetch_market_stocks(
                tickers,
                market,
                start_date,
                end_date,
                show_progress
            )

            all_data[market] = market_data

        return all_data

    def align_cross_market(
        self,
        source_data: Dict[str, pd.DataFrame],
        target_data: Dict[str, pd.DataFrame],
        source_market: str,
        target_market: str
    ) -> pd.DataFrame:
        """
        Align source market data with target market data for prediction.

        Logic:
        - Source market T-day closing data → Target market T+1 day data
        - Example: US closes 2024-01-15 → Predict JP on 2024-01-16

        Args:
            source_data: Dict of source market DataFrames
            target_data: Dict of target market DataFrames
            source_market: Source market code (e.g., 'US')
            target_market: Target market code (e.g., 'JP')

        Returns:
            Aligned DataFrame with source features and target labels
        """
        print(f"🔄 Aligning {source_market} → {target_market} data...")

        aligned_records = []

        # Get all available dates from source
        all_source_dates = set()
        for ticker, df in source_data.items():
            all_source_dates.update(df.index.tolist())

        all_source_dates = sorted(list(all_source_dates))

        # Get all target dates
        all_target_dates = set()
        for ticker, df in target_data.items():
            all_target_dates.update(df.index.tolist())

        for i, source_date in enumerate(all_source_dates[:-1]):
            # Find next target date
            potential_target_dates = [d for d in all_target_dates if d > source_date]

            if not potential_target_dates:
                continue

            target_date = min(potential_target_dates)

            # Build record
            record = {
                'source_date': source_date,
                'target_date': target_date,
                'days_gap': (target_date - source_date).days
            }

            # Extract source features
            for ticker, df in source_data.items():
                if source_date in df.index:
                    row = df.loc[source_date]
                    prefix = f"{source_market}_{ticker.split('.')[0]}"
                    record[f'{prefix}_close'] = row['Close']
                    record[f'{prefix}_open'] = row['Open']
                    record[f'{prefix}_high'] = row['High']
                    record[f'{prefix}_low'] = row['Low']
                    record[f'{prefix}_volume'] = row['Volume']

                    # Daily return
                    if 'Open' in row and row['Open'] > 0:
                        record[f'{prefix}_return'] = (row['Close'] / row['Open']) - 1

            # Extract target labels
            for ticker, df in target_data.items():
                if target_date in df.index:
                    row = df.loc[target_date]
                    prefix = f"{target_market}_{ticker.split('.')[0]}"

                    # Target: next day's return
                    if 'Open' in row and 'Close' in row and row['Open'] > 0:
                        record[f'{prefix}_target_return'] = (row['Close'] / row['Open']) - 1

                    # Also save close for reference
                    record[f'{prefix}_close'] = row['Close']

            aligned_records.append(record)

        aligned_df = pd.DataFrame(aligned_records)
        print(f"✅ Aligned {len(aligned_df)} trading day pairs")

        return aligned_df

    def prepare_sst_input(
        self,
        aligned_df: pd.DataFrame,
        source_market: str,
        target_stock: str,
        feature_columns: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, List[str], List[str]]:
        """
        Prepare data in SST framework format.

        Args:
            aligned_df: Aligned cross-market DataFrame
            source_market: Source market code
            target_stock: Target stock ticker prefix (e.g., '6758' for JP)
            feature_columns: List of feature types to use

        Returns:
            (prepared_df, boundary_cols, target_cols)
        """
        if feature_columns is None:
            feature_columns = ['close', 'volume', 'return']

        # Find boundary columns (all source market features)
        boundary_cols = [
            col for col in aligned_df.columns
            if col.startswith(f"{source_market}_") and
            any(feat in col for feat in feature_columns)
        ]

        # Find target columns
        target_cols = [
            col for col in aligned_df.columns
            if target_stock in col and 'target' in col
        ]

        print(f"📊 SST Input prepared:")
        print(f"   Boundary sensors: {len(boundary_cols)} features")
        print(f"   Target sensors: {len(target_cols)} labels")

        return aligned_df, boundary_cols, target_cols

    def add_technical_indicators(
        self,
        df: pd.DataFrame,
        ticker_prefix: str
    ) -> pd.DataFrame:
        """
        Add technical indicators to the DataFrame.

        Args:
            df: DataFrame with OHLCV data
            ticker_prefix: Prefix for column names (e.g., 'US_NVDA')

        Returns:
            DataFrame with added indicators
        """
        close_col = f'{ticker_prefix}_close'

        if close_col not in df.columns:
            return df

        # Moving averages
        df[f'{ticker_prefix}_MA5'] = df[close_col].rolling(5).mean()
        df[f'{ticker_prefix}_MA20'] = df[close_col].rolling(20).mean()

        # RSI
        delta = df[close_col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df[f'{ticker_prefix}_RSI'] = 100 - (100 / (1 + rs))

        # Volatility
        df[f'{ticker_prefix}_volatility'] = df[close_col].rolling(20).std()

        return df


# ============================================================================
# Convenience Functions
# ============================================================================

def fetch_from_json(
    json_path: str,
    start_date: str,
    end_date: str,
    markets: Optional[List[str]] = None
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Fetch stock data based on JSON selection file.

    Args:
        json_path: Path to stocks_selection.json
        start_date: Start date
        end_date: End date
        markets: Optional list of markets to fetch (if None, fetch all)

    Returns:
        Market data dict
    """
    import json

    with open(json_path, 'r') as f:
        stocks_data = json.load(f)

    stocks_by_market = {}

    for market, stocks in stocks_data.get('stocks', {}).items():
        if markets and market not in markets:
            continue

        tickers = [stock['ticker'] for stock in stocks]
        stocks_by_market[market] = tickers

    fetcher = CrossMarketDataFetcher()
    return fetcher.fetch_all_markets(stocks_by_market, start_date, end_date)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == '__main__':
    """Example: Fetch and align US → JP data"""

    # Example stock selection
    stocks_by_market = {
        'US': ['NVDA', 'AMD', 'INTC', 'AAPL', 'MSFT'],
        'JP': ['6758.T', '7203.T', '8035.T']  # Sony, Toyota, Tokyo Electron
    }

    # Date range
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

    # Fetch data
    fetcher = CrossMarketDataFetcher()
    all_data = fetcher.fetch_all_markets(stocks_by_market, start_date, end_date)

    # Align US → JP
    aligned_df = fetcher.align_cross_market(
        source_data=all_data['US'],
        target_data=all_data['JP'],
        source_market='US',
        target_market='JP'
    )

    print("\n" + "="*60)
    print("Aligned DataFrame Preview")
    print("="*60)
    print(aligned_df.head())
    print(f"\nShape: {aligned_df.shape}")
    print(f"Columns: {len(aligned_df.columns)}")

    # Prepare for SST
    prepared_df, boundary_cols, target_cols = fetcher.prepare_sst_input(
        aligned_df,
        source_market='US',
        target_stock='6758',  # Sony
        feature_columns=['close', 'volume', 'return']
    )

    print("\nBoundary columns:", boundary_cols[:5], "...")
    print("Target columns:", target_cols)

    # Save
    aligned_df.to_csv('outputs/us_jp_aligned.csv', index=False)
    print("\n💾 Saved to outputs/us_jp_aligned.csv")
