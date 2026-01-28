"""Yahoo Finance client for historical stock data."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional
import logging

import pandas as pd

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

from src.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class StockQuote:
    """Represents a stock quote."""
    symbol: str
    name: str
    price: float
    change: float
    change_percent: float
    volume: int
    market_cap: float
    pe_ratio: Optional[float]
    eps: Optional[float]
    dividend_yield: Optional[float]
    fifty_two_week_high: float
    fifty_two_week_low: float
    avg_volume: int
    timestamp: datetime


class YahooFinanceClient:
    """Client for fetching data from Yahoo Finance."""
    
    # TASE stocks use .TA suffix in Yahoo Finance
    TASE_SUFFIX = ".TA"
    
    def __init__(self):
        """Initialize Yahoo Finance client."""
        if not YFINANCE_AVAILABLE:
            raise ImportError(
                "yfinance is not installed. Install with: pip install yfinance"
            )
        
        self.settings = get_settings()
        self._quote_cache: dict[str, tuple[StockQuote, datetime]] = {}
        self._history_cache: dict[str, tuple[pd.DataFrame, datetime]] = {}
    
    def _normalize_symbol(self, symbol: str) -> str:
        """Ensure symbol has .TA suffix for TASE stocks.
        
        Args:
            symbol: Stock symbol (with or without .TA suffix)
            
        Returns:
            Symbol with .TA suffix.
        """
        if not symbol.endswith(self.TASE_SUFFIX):
            return f"{symbol}{self.TASE_SUFFIX}"
        return symbol
    
    def _is_cache_valid(self, cache_time: datetime) -> bool:
        """Check if cached data is still valid."""
        elapsed = (datetime.now() - cache_time).total_seconds()
        return elapsed < self.settings.cache_ttl_seconds
    
    def get_quote(self, symbol: str, force_refresh: bool = False) -> Optional[StockQuote]:
        """Get current quote for a stock.
        
        Args:
            symbol: Stock symbol (e.g., 'TEVA' or 'TEVA.TA')
            force_refresh: Force refresh from API instead of cache.
            
        Returns:
            StockQuote object or None if not found.
        """
        symbol = self._normalize_symbol(symbol)
        
        # Check cache
        if not force_refresh and symbol in self._quote_cache:
            quote, cache_time = self._quote_cache[symbol]
            if self._is_cache_valid(cache_time):
                return quote
        
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            if not info or 'regularMarketPrice' not in info:
                logger.warning(f"No data found for {symbol}")
                return None
            
            quote = StockQuote(
                symbol=symbol,
                name=info.get('longName', info.get('shortName', symbol)),
                price=info.get('regularMarketPrice', 0.0),
                change=info.get('regularMarketChange', 0.0),
                change_percent=info.get('regularMarketChangePercent', 0.0),
                volume=info.get('regularMarketVolume', 0),
                market_cap=info.get('marketCap', 0.0),
                pe_ratio=info.get('trailingPE'),
                eps=info.get('trailingEps'),
                dividend_yield=info.get('dividendYield'),
                fifty_two_week_high=info.get('fiftyTwoWeekHigh', 0.0),
                fifty_two_week_low=info.get('fiftyTwoWeekLow', 0.0),
                avg_volume=info.get('averageVolume', 0),
                timestamp=datetime.now(),
            )
            
            self._quote_cache[symbol] = (quote, datetime.now())
            return quote
            
        except Exception as e:
            logger.error(f"Error fetching quote for {symbol}: {e}")
            return None
    
    def get_historical_data(
        self,
        symbol: str,
        period: str = "6mo",
        interval: str = "1d",
        force_refresh: bool = False,
    ) -> Optional[pd.DataFrame]:
        """Get historical OHLCV data for a stock.
        
        Args:
            symbol: Stock symbol (e.g., 'TEVA' or 'TEVA.TA')
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            force_refresh: Force refresh from API instead of cache.
            
        Returns:
            DataFrame with OHLCV data or None if not found.
        """
        symbol = self._normalize_symbol(symbol)
        cache_key = f"{symbol}_{period}_{interval}"
        
        # Check cache
        if not force_refresh and cache_key in self._history_cache:
            df, cache_time = self._history_cache[cache_key]
            if self._is_cache_valid(cache_time):
                return df.copy()
        
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            
            if df.empty:
                logger.warning(f"No historical data found for {symbol}")
                return None
            
            # Ensure column names are lowercase
            df.columns = [col.lower() for col in df.columns]
            
            # Add symbol column
            df['symbol'] = symbol
            
            self._history_cache[cache_key] = (df.copy(), datetime.now())
            logger.info(f"Fetched {len(df)} rows of historical data for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return None
    
    def get_historical_data_range(
        self,
        symbol: str,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        interval: str = "1d",
    ) -> Optional[pd.DataFrame]:
        """Get historical data for a specific date range.
        
        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date (defaults to now)
            interval: Data interval
            
        Returns:
            DataFrame with OHLCV data.
        """
        symbol = self._normalize_symbol(symbol)
        end_date = end_date or datetime.now()
        
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, interval=interval)
            
            if df.empty:
                return None
            
            df.columns = [col.lower() for col in df.columns]
            df['symbol'] = symbol
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data range for {symbol}: {e}")
            return None
    
    def get_multiple_quotes(
        self, 
        symbols: list[str], 
        force_refresh: bool = False
    ) -> dict[str, Optional[StockQuote]]:
        """Get quotes for multiple stocks.
        
        Args:
            symbols: List of stock symbols
            force_refresh: Force refresh from API
            
        Returns:
            Dictionary mapping symbols to quotes.
        """
        results = {}
        for symbol in symbols:
            results[symbol] = self.get_quote(symbol, force_refresh)
        return results
    
    def get_multiple_historical(
        self,
        symbols: list[str],
        period: str = "6mo",
        interval: str = "1d",
    ) -> dict[str, Optional[pd.DataFrame]]:
        """Get historical data for multiple stocks.
        
        Args:
            symbols: List of stock symbols
            period: Data period
            interval: Data interval
            
        Returns:
            Dictionary mapping symbols to DataFrames.
        """
        results = {}
        for symbol in symbols:
            results[symbol] = self.get_historical_data(symbol, period, interval)
        return results
    
    def search_tase_stocks(self, query: str) -> list[dict]:
        """Search for TASE stocks by name or symbol.
        
        Args:
            query: Search query
            
        Returns:
            List of matching stocks.
        """
        try:
            # Use yfinance search functionality
            results = []
            
            # Try direct symbol lookup
            test_symbol = self._normalize_symbol(query.upper())
            quote = self.get_quote(test_symbol)
            if quote:
                results.append({
                    'symbol': quote.symbol,
                    'name': quote.name,
                    'price': quote.price,
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching for {query}: {e}")
            return []
    
    def get_sector_pe_average(self, sector: str) -> Optional[float]:
        """Get average P/E ratio for a sector.
        
        Note: This is a simplified implementation.
        In production, you'd want a more comprehensive sector database.
        
        Args:
            sector: Sector name
            
        Returns:
            Average P/E ratio or None.
        """
        # Sector benchmarks (approximate values for Israeli market)
        sector_pe_averages = {
            'technology': 25.0,
            'financial': 12.0,
            'healthcare': 20.0,
            'real estate': 15.0,
            'consumer': 18.0,
            'industrial': 16.0,
            'energy': 10.0,
            'materials': 14.0,
            'utilities': 15.0,
            'communication': 18.0,
        }
        
        sector_lower = sector.lower()
        for key, value in sector_pe_averages.items():
            if key in sector_lower:
                return value
        
        # Default market average
        return 17.0
