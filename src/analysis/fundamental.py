"""Fundamental analysis module for stock evaluation."""

from dataclasses import dataclass, field
from typing import Optional
import logging

from src.config import get_settings
from src.data.yahoo_client import YahooFinanceClient, StockQuote

logger = logging.getLogger(__name__)


@dataclass
class FundamentalMetrics:
    """Container for fundamental analysis metrics."""
    # Valuation
    pe_ratio: Optional[float] = None
    sector_avg_pe: Optional[float] = None
    pe_vs_sector: str = "unknown"  # undervalued, fair, overvalued
    
    # Market data
    market_cap: float = 0.0
    market_cap_category: str = "unknown"  # large, mid, small, micro
    
    # Price metrics
    current_price: float = 0.0
    fifty_two_week_high: float = 0.0
    fifty_two_week_low: float = 0.0
    price_vs_52w_high: float = 0.0  # percentage from high
    price_vs_52w_low: float = 0.0   # percentage from low
    
    # Earnings
    eps: Optional[float] = None
    
    # Dividends
    dividend_yield: Optional[float] = None
    
    # Volume/Liquidity
    avg_volume: int = 0
    volume_category: str = "unknown"  # high, medium, low
    
    # Index membership
    is_ta35: bool = False
    is_ta125: bool = False
    
    # Sector
    sector: str = ""


@dataclass
class FundamentalScore:
    """Fundamental analysis score breakdown."""
    total_score: float = 0.0  # 0-100
    valuation_score: float = 0.0
    liquidity_score: float = 0.0
    quality_score: float = 0.0
    momentum_score: float = 0.0
    signals: list[str] = field(default_factory=list)
    metrics: Optional[FundamentalMetrics] = None


class FundamentalAnalyzer:
    """Fundamental analysis engine for Israeli stocks."""
    
    # Israeli market cap thresholds (in ILS, approximate)
    MARKET_CAP_THRESHOLDS = {
        'large': 10_000_000_000,   # > 10B ILS
        'mid': 2_000_000_000,      # 2-10B ILS
        'small': 500_000_000,      # 500M-2B ILS
        'micro': 0,                # < 500M ILS
    }
    
    # Volume thresholds for liquidity
    VOLUME_THRESHOLDS = {
        'high': 1_000_000,    # > 1M shares/day
        'medium': 100_000,    # 100K-1M shares/day
        'low': 0,             # < 100K shares/day
    }
    
    # Sector P/E averages for Israeli market
    SECTOR_PE_AVERAGES = {
        'technology': 28.0,
        'financial': 11.0,
        'healthcare': 22.0,
        'real estate': 14.0,
        'consumer': 17.0,
        'industrial': 15.0,
        'energy': 9.0,
        'materials': 13.0,
        'utilities': 14.0,
        'communication': 16.0,
        'default': 16.0,
    }
    
    def __init__(self, yahoo_client: Optional[YahooFinanceClient] = None):
        """Initialize fundamental analyzer.
        
        Args:
            yahoo_client: Optional YahooFinanceClient instance.
        """
        self.settings = get_settings()
        self.yahoo = yahoo_client or YahooFinanceClient()
    
    def _get_sector_pe(self, sector: str) -> float:
        """Get average P/E for a sector.
        
        Args:
            sector: Sector name.
            
        Returns:
            Sector average P/E ratio.
        """
        sector_lower = sector.lower()
        
        for key, value in self.SECTOR_PE_AVERAGES.items():
            if key in sector_lower:
                return value
        
        return self.SECTOR_PE_AVERAGES['default']
    
    def _categorize_market_cap(self, market_cap: float) -> str:
        """Categorize market cap size.
        
        Args:
            market_cap: Market capitalization value.
            
        Returns:
            Category string.
        """
        if market_cap >= self.MARKET_CAP_THRESHOLDS['large']:
            return 'large'
        elif market_cap >= self.MARKET_CAP_THRESHOLDS['mid']:
            return 'mid'
        elif market_cap >= self.MARKET_CAP_THRESHOLDS['small']:
            return 'small'
        else:
            return 'micro'
    
    def _categorize_volume(self, avg_volume: int) -> str:
        """Categorize trading volume/liquidity.
        
        Args:
            avg_volume: Average daily volume.
            
        Returns:
            Category string.
        """
        if avg_volume >= self.VOLUME_THRESHOLDS['high']:
            return 'high'
        elif avg_volume >= self.VOLUME_THRESHOLDS['medium']:
            return 'medium'
        else:
            return 'low'
    
    def analyze(
        self, 
        symbol: str,
        quote: Optional[StockQuote] = None,
        sector: str = "",
        is_ta35: bool = False,
        is_ta125: bool = False,
    ) -> FundamentalMetrics:
        """Analyze fundamental metrics for a stock.
        
        Args:
            symbol: Stock symbol.
            quote: Optional pre-fetched quote data.
            sector: Stock sector (if known).
            is_ta35: Whether stock is in TA-35 index.
            is_ta125: Whether stock is in TA-125 index.
            
        Returns:
            FundamentalMetrics object.
        """
        # Fetch quote if not provided
        if quote is None:
            quote = self.yahoo.get_quote(symbol)
        
        if quote is None:
            logger.warning(f"Could not fetch quote for {symbol}")
            return FundamentalMetrics()
        
        metrics = FundamentalMetrics()
        
        # Valuation metrics
        metrics.pe_ratio = quote.pe_ratio
        metrics.sector = sector
        
        if sector:
            metrics.sector_avg_pe = self._get_sector_pe(sector)
            if metrics.pe_ratio and metrics.sector_avg_pe:
                ratio = metrics.pe_ratio / metrics.sector_avg_pe
                if ratio < 0.8:
                    metrics.pe_vs_sector = "undervalued"
                elif ratio > 1.2:
                    metrics.pe_vs_sector = "overvalued"
                else:
                    metrics.pe_vs_sector = "fair"
        
        # Market data
        metrics.market_cap = quote.market_cap
        metrics.market_cap_category = self._categorize_market_cap(quote.market_cap)
        
        # Price metrics
        metrics.current_price = quote.price
        metrics.fifty_two_week_high = quote.fifty_two_week_high
        metrics.fifty_two_week_low = quote.fifty_two_week_low
        
        if quote.fifty_two_week_high > 0:
            metrics.price_vs_52w_high = ((quote.price - quote.fifty_two_week_high) / 
                                          quote.fifty_two_week_high * 100)
        
        if quote.fifty_two_week_low > 0:
            metrics.price_vs_52w_low = ((quote.price - quote.fifty_two_week_low) / 
                                         quote.fifty_two_week_low * 100)
        
        # Earnings
        metrics.eps = quote.eps
        
        # Dividends
        metrics.dividend_yield = quote.dividend_yield
        
        # Volume/Liquidity
        metrics.avg_volume = quote.avg_volume
        metrics.volume_category = self._categorize_volume(quote.avg_volume)
        
        # Index membership
        metrics.is_ta35 = is_ta35
        metrics.is_ta125 = is_ta125
        
        return metrics
    
    def calculate_score(
        self,
        symbol: str,
        quote: Optional[StockQuote] = None,
        sector: str = "",
        is_ta35: bool = False,
        is_ta125: bool = False,
    ) -> FundamentalScore:
        """Calculate fundamental analysis score for a stock.
        
        Args:
            symbol: Stock symbol.
            quote: Optional pre-fetched quote data.
            sector: Stock sector.
            is_ta35: TA-35 membership.
            is_ta125: TA-125 membership.
            
        Returns:
            FundamentalScore object with breakdown.
        """
        metrics = self.analyze(symbol, quote, sector, is_ta35, is_ta125)
        score = FundamentalScore(metrics=metrics)
        signals = []
        
        # Valuation Score (max 15 points)
        if metrics.pe_ratio:
            if metrics.pe_vs_sector == "undervalued":
                score.valuation_score = 15
                signals.append(f"P/E ({metrics.pe_ratio:.1f}) below sector avg ({metrics.sector_avg_pe:.1f})")
            elif metrics.pe_vs_sector == "fair":
                score.valuation_score = 10
                signals.append("P/E in line with sector")
            elif metrics.pe_vs_sector == "overvalued":
                score.valuation_score = 3
                signals.append(f"P/E ({metrics.pe_ratio:.1f}) above sector avg")
            else:
                score.valuation_score = 7
        else:
            # No P/E available (possibly negative earnings)
            score.valuation_score = 5
            signals.append("P/E not available (check earnings)")
        
        # Liquidity Score (max 10 points)
        if metrics.volume_category == "high":
            score.liquidity_score = 10
            signals.append("High trading volume - good liquidity")
        elif metrics.volume_category == "medium":
            score.liquidity_score = 6
            signals.append("Medium trading volume")
        else:
            score.liquidity_score = 2
            signals.append("Low volume - liquidity risk")
        
        # Quality Score (max 10 points)
        quality = 0
        
        # Index membership bonus
        if metrics.is_ta35:
            quality += 5
            signals.append("TA-35 index member (blue chip)")
        elif metrics.is_ta125:
            quality += 3
            signals.append("TA-125 index member")
        
        # Market cap consideration
        if metrics.market_cap_category == "large":
            quality += 3
        elif metrics.market_cap_category == "mid":
            quality += 2
        elif metrics.market_cap_category == "small":
            quality += 1
        
        # Dividend bonus
        if metrics.dividend_yield and metrics.dividend_yield > 0.02:  # > 2%
            quality += 2
            signals.append(f"Dividend yield: {metrics.dividend_yield*100:.1f}%")
        
        score.quality_score = min(10, quality)
        
        # Momentum Score (based on price vs 52-week range, max 10 points)
        if metrics.price_vs_52w_high is not None and metrics.price_vs_52w_low is not None:
            # Calculate position in 52-week range
            range_52w = metrics.fifty_two_week_high - metrics.fifty_two_week_low
            if range_52w > 0:
                position = (metrics.current_price - metrics.fifty_two_week_low) / range_52w
                
                # We want stocks that are not at extremes
                # Ideal: 30-60% of range (recovering from lows but not overbought)
                if 0.3 <= position <= 0.6:
                    score.momentum_score = 10
                    signals.append("Price in optimal 52-week range position")
                elif position < 0.2:
                    score.momentum_score = 7
                    signals.append("Near 52-week low - potential value")
                elif position > 0.9:
                    score.momentum_score = 2
                    signals.append("Near 52-week high - stretched")
                else:
                    score.momentum_score = 5
        
        # Calculate total (scaled for 35% weight as per plan)
        raw_total = (score.valuation_score + score.liquidity_score + 
                     score.quality_score + score.momentum_score)
        max_raw = 45  # 15 + 10 + 10 + 10
        score.total_score = min(100, raw_total / max_raw * 100)
        score.signals = signals
        
        return score
    
    def compare_to_sector(
        self, 
        symbol: str, 
        sector_symbols: list[str]
    ) -> dict:
        """Compare a stock's fundamentals to sector peers.
        
        Args:
            symbol: Stock symbol to analyze.
            sector_symbols: List of peer stock symbols.
            
        Returns:
            Comparison dictionary.
        """
        target_quote = self.yahoo.get_quote(symbol)
        if not target_quote:
            return {"error": f"Could not fetch data for {symbol}"}
        
        peer_quotes = []
        for peer in sector_symbols:
            if peer != symbol:
                quote = self.yahoo.get_quote(peer)
                if quote and quote.pe_ratio:
                    peer_quotes.append(quote)
        
        if not peer_quotes:
            return {"error": "No peer data available"}
        
        # Calculate sector averages
        pe_ratios = [q.pe_ratio for q in peer_quotes if q.pe_ratio]
        avg_pe = sum(pe_ratios) / len(pe_ratios) if pe_ratios else None
        
        market_caps = [q.market_cap for q in peer_quotes if q.market_cap > 0]
        avg_market_cap = sum(market_caps) / len(market_caps) if market_caps else None
        
        return {
            "symbol": symbol,
            "pe_ratio": target_quote.pe_ratio,
            "sector_avg_pe": avg_pe,
            "pe_percentile": self._calculate_percentile(
                target_quote.pe_ratio, pe_ratios
            ) if target_quote.pe_ratio and pe_ratios else None,
            "market_cap": target_quote.market_cap,
            "sector_avg_market_cap": avg_market_cap,
            "peer_count": len(peer_quotes),
        }
    
    def _calculate_percentile(self, value: float, values: list[float]) -> float:
        """Calculate percentile of value in list."""
        if not values or value is None:
            return 50.0
        
        sorted_values = sorted(values)
        count_below = sum(1 for v in sorted_values if v < value)
        return (count_below / len(sorted_values)) * 100
