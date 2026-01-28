"""Rule-based scoring engine combining all analysis factors."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
import logging

import pandas as pd

from src.config import get_settings
from src.data.yahoo_client import YahooFinanceClient, StockQuote
from src.data.tase_client import TASEClient
from src.analysis.technical import TechnicalAnalyzer, TechnicalScore
from src.analysis.fundamental import FundamentalAnalyzer, FundamentalScore
from src.analysis.sentiment import SentimentAnalyzer, SentimentScore

logger = logging.getLogger(__name__)


class Recommendation(Enum):
    """Stock recommendation levels."""
    STRONG_BUY = "STRONG BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG SELL"


@dataclass
class StockAnalysis:
    """Complete stock analysis result."""
    symbol: str
    name: str = ""
    
    # Component scores (0-100 each)
    technical_score: float = 0.0
    fundamental_score: float = 0.0
    sentiment_score: float = 0.0
    
    # Weighted total score (0-100)
    total_score: float = 0.0
    
    # Recommendation
    recommendation: Recommendation = Recommendation.HOLD
    confidence: float = 0.0  # 0-100
    
    # Price info
    current_price: float = 0.0
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    
    # Detailed breakdowns
    technical_details: Optional[TechnicalScore] = None
    fundamental_details: Optional[FundamentalScore] = None
    sentiment_details: Optional[SentimentScore] = None
    
    # Signals summary
    bullish_signals: list[str] = field(default_factory=list)
    bearish_signals: list[str] = field(default_factory=list)
    
    # Metadata
    analysis_timestamp: datetime = field(default_factory=datetime.now)
    holding_period: str = "1-4 weeks"  # Swing trade
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'symbol': self.symbol,
            'name': self.name,
            'total_score': round(self.total_score, 1),
            'technical_score': round(self.technical_score, 1),
            'fundamental_score': round(self.fundamental_score, 1),
            'sentiment_score': round(self.sentiment_score, 1),
            'recommendation': self.recommendation.value,
            'confidence': round(self.confidence, 1),
            'current_price': self.current_price,
            'target_price': self.target_price,
            'stop_loss': self.stop_loss,
            'bullish_signals': self.bullish_signals,
            'bearish_signals': self.bearish_signals,
            'holding_period': self.holding_period,
            'timestamp': self.analysis_timestamp.isoformat(),
        }


class ScoringEngine:
    """Main scoring engine that combines all analysis components."""
    
    def __init__(
        self,
        yahoo_client: Optional[YahooFinanceClient] = None,
        tase_client: Optional[TASEClient] = None,
    ):
        """Initialize scoring engine.
        
        Args:
            yahoo_client: Optional YahooFinanceClient instance.
            tase_client: Optional TASEClient instance.
        """
        self.settings = get_settings()
        
        self.yahoo = yahoo_client or YahooFinanceClient()
        self.tase = tase_client
        
        self.technical = TechnicalAnalyzer()
        self.fundamental = FundamentalAnalyzer(self.yahoo)
        self.sentiment = SentimentAnalyzer()
        
        # Load TA-35 and TA-125 components if TASE client available
        self._ta35_symbols: list[str] = []
        self._ta125_symbols: list[str] = []
    
    async def _load_index_components(self):
        """Load TA-35 and TA-125 index components."""
        if self.tase:
            try:
                self._ta35_symbols = self.tase.get_ta35_symbols()
                self._ta125_symbols = self.tase.get_ta125_symbols()
            except Exception as e:
                logger.warning(f"Could not load index components: {e}")
    
    def _calculate_weighted_score(
        self,
        technical: float,
        fundamental: float,
        sentiment: float,
    ) -> float:
        """Calculate weighted total score.
        
        Args:
            technical: Technical score (0-100).
            fundamental: Fundamental score (0-100).
            sentiment: Sentiment score (0-100).
            
        Returns:
            Weighted total score (0-100).
        """
        weights = {
            'technical': self.settings.technical_weight,
            'fundamental': self.settings.fundamental_weight,
            'sentiment': self.settings.sentiment_weight,
        }
        
        total = (
            technical * weights['technical'] +
            fundamental * weights['fundamental'] +
            sentiment * weights['sentiment']
        )
        
        return min(100, max(0, total))
    
    def _score_to_recommendation(self, score: float) -> Recommendation:
        """Convert score to recommendation.
        
        Args:
            score: Total score (0-100).
            
        Returns:
            Recommendation enum value.
        """
        if score >= 80:
            return Recommendation.STRONG_BUY
        elif score >= 60:
            return Recommendation.BUY
        elif score >= 40:
            return Recommendation.HOLD
        elif score >= 20:
            return Recommendation.SELL
        else:
            return Recommendation.STRONG_SELL
    
    def _calculate_confidence(
        self,
        technical: TechnicalScore,
        fundamental: FundamentalScore,
        sentiment: SentimentScore,
    ) -> float:
        """Calculate confidence level based on signal consistency.
        
        Args:
            technical: Technical score details.
            fundamental: Fundamental score details.
            sentiment: Sentiment score details.
            
        Returns:
            Confidence level (0-100).
        """
        scores = [
            technical.total_score,
            fundamental.total_score,
            sentiment.total_score,
        ]
        
        # Calculate score consistency (lower variance = higher confidence)
        avg = sum(scores) / len(scores)
        variance = sum((s - avg) ** 2 for s in scores) / len(scores)
        
        # Normalize variance to confidence
        # Max variance when one is 0 and others are 100 is ~2222
        max_variance = 2222
        consistency = 1 - (variance / max_variance)
        
        # Factor in signal counts
        total_signals = len(technical.signals) + len(fundamental.signals) + len(sentiment.signals)
        signal_factor = min(1.0, total_signals / 10)
        
        # Combine factors
        confidence = (consistency * 0.6 + signal_factor * 0.4) * 100
        
        return min(100, max(0, confidence))
    
    def _calculate_targets(
        self,
        current_price: float,
        technical: TechnicalScore,
        recommendation: Recommendation,
    ) -> tuple[Optional[float], Optional[float]]:
        """Calculate target price and stop loss.
        
        Args:
            current_price: Current stock price.
            technical: Technical analysis result.
            recommendation: Stock recommendation.
            
        Returns:
            Tuple of (target_price, stop_loss).
        """
        if current_price <= 0:
            return None, None
        
        # Use ATR for volatility-based targets if available
        atr = None
        if technical.indicators and technical.indicators.atr_14:
            atr = technical.indicators.atr_14
        
        if recommendation in (Recommendation.STRONG_BUY, Recommendation.BUY):
            # Target: 5-10% gain or 2x ATR
            if atr:
                target = current_price + (atr * 2)
                stop = current_price - (atr * 1.5)
            else:
                target = current_price * 1.08  # 8% target
                stop = current_price * 0.95    # 5% stop
        elif recommendation in (Recommendation.SELL, Recommendation.STRONG_SELL):
            # For sells, reverse the logic
            target = current_price * 0.92
            stop = current_price * 1.03
        else:
            # Hold - no specific targets
            target = None
            stop = None
        
        return target, stop
    
    def _extract_signals(
        self,
        technical: TechnicalScore,
        fundamental: FundamentalScore,
        sentiment: SentimentScore,
    ) -> tuple[list[str], list[str]]:
        """Extract bullish and bearish signals.
        
        Args:
            technical: Technical analysis result.
            fundamental: Fundamental analysis result.
            sentiment: Sentiment analysis result.
            
        Returns:
            Tuple of (bullish_signals, bearish_signals).
        """
        bullish = []
        bearish = []
        
        # Technical signals
        for signal in technical.signals:
            signal_lower = signal.lower()
            if any(word in signal_lower for word in ['bullish', 'oversold', 'above', 'golden', 'breakout', 'positive']):
                bullish.append(f"[TA] {signal}")
            elif any(word in signal_lower for word in ['bearish', 'overbought', 'death', 'below', 'caution']):
                bearish.append(f"[TA] {signal}")
        
        # Fundamental signals
        for signal in fundamental.signals:
            signal_lower = signal.lower()
            if any(word in signal_lower for word in ['below', 'undervalued', 'high', 'blue chip', 'dividend', 'optimal']):
                bullish.append(f"[FA] {signal}")
            elif any(word in signal_lower for word in ['above', 'overvalued', 'low', 'risk', 'stretched']):
                bearish.append(f"[FA] {signal}")
        
        # Sentiment signals
        for signal in sentiment.signals:
            signal_lower = signal.lower()
            if any(word in signal_lower for word in ['positive', 'improving', 'consistent']):
                bullish.append(f"[Sent] {signal}")
            elif any(word in signal_lower for word in ['negative', 'declining', 'caution', 'risk', 'mixed']):
                bearish.append(f"[Sent] {signal}")
        
        return bullish[:5], bearish[:5]  # Limit to top 5 each
    
    async def analyze_stock(
        self,
        symbol: str,
        sector: str = "",
    ) -> StockAnalysis:
        """Perform complete analysis of a stock.
        
        Args:
            symbol: Stock symbol (e.g., 'TEVA.TA').
            sector: Optional sector name.
            
        Returns:
            StockAnalysis object with complete analysis.
        """
        # Normalize symbol
        if not symbol.endswith('.TA'):
            symbol = f"{symbol}.TA"
        
        analysis = StockAnalysis(symbol=symbol)
        
        try:
            # Load index components if not loaded
            if not self._ta35_symbols:
                await self._load_index_components()
            
            # Check index membership
            symbol_base = symbol.replace('.TA', '')
            is_ta35 = symbol_base in self._ta35_symbols or symbol in self._ta35_symbols
            is_ta125 = symbol_base in self._ta125_symbols or symbol in self._ta125_symbols
            
            # Fetch quote
            quote = self.yahoo.get_quote(symbol)
            if quote:
                analysis.name = quote.name
                analysis.current_price = quote.price
            
            # Fetch historical data for technical analysis
            historical = self.yahoo.get_historical_data(symbol, period="6mo")
            
            # Run analyses in parallel
            technical_task = asyncio.create_task(
                asyncio.to_thread(self.technical.calculate_score, historical)
            ) if historical is not None else None
            
            fundamental_task = asyncio.create_task(
                asyncio.to_thread(
                    self.fundamental.calculate_score,
                    symbol, quote, sector, is_ta35, is_ta125
                )
            )
            
            sentiment_task = self.sentiment.calculate_score_async(symbol)
            
            # Gather results
            results = await asyncio.gather(
                technical_task if technical_task else asyncio.sleep(0),
                fundamental_task,
                sentiment_task,
                return_exceptions=True,
            )
            
            # Process technical results
            if technical_task and not isinstance(results[0], Exception):
                tech_result = results[0]
                analysis.technical_score = tech_result.total_score
                analysis.technical_details = tech_result
            else:
                analysis.technical_score = 50  # Neutral if unavailable
            
            # Process fundamental results
            if not isinstance(results[1], Exception):
                fund_result = results[1]
                analysis.fundamental_score = fund_result.total_score
                analysis.fundamental_details = fund_result
            else:
                analysis.fundamental_score = 50
            
            # Process sentiment results
            if not isinstance(results[2], Exception):
                sent_result = results[2]
                analysis.sentiment_score = sent_result.total_score
                analysis.sentiment_details = sent_result
            else:
                analysis.sentiment_score = 50
            
            # Calculate weighted total
            analysis.total_score = self._calculate_weighted_score(
                analysis.technical_score,
                analysis.fundamental_score,
                analysis.sentiment_score,
            )
            
            # Determine recommendation
            analysis.recommendation = self._score_to_recommendation(analysis.total_score)
            
            # Calculate confidence
            if analysis.technical_details and analysis.fundamental_details and analysis.sentiment_details:
                analysis.confidence = self._calculate_confidence(
                    analysis.technical_details,
                    analysis.fundamental_details,
                    analysis.sentiment_details,
                )
            
            # Calculate targets
            if analysis.technical_details:
                analysis.target_price, analysis.stop_loss = self._calculate_targets(
                    analysis.current_price,
                    analysis.technical_details,
                    analysis.recommendation,
                )
            
            # Extract signals
            if analysis.technical_details and analysis.fundamental_details and analysis.sentiment_details:
                analysis.bullish_signals, analysis.bearish_signals = self._extract_signals(
                    analysis.technical_details,
                    analysis.fundamental_details,
                    analysis.sentiment_details,
                )
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            raise
        
        return analysis
    
    async def scan_stocks(
        self,
        symbols: list[str],
        min_score: float = 60,
    ) -> list[StockAnalysis]:
        """Scan multiple stocks and return top recommendations.
        
        Args:
            symbols: List of stock symbols to scan.
            min_score: Minimum score to include in results.
            
        Returns:
            List of StockAnalysis objects, sorted by score.
        """
        results = []
        
        for symbol in symbols:
            try:
                analysis = await self.analyze_stock(symbol)
                if analysis.total_score >= min_score:
                    results.append(analysis)
            except Exception as e:
                logger.warning(f"Failed to analyze {symbol}: {e}")
                continue
        
        # Sort by total score descending
        results.sort(key=lambda x: x.total_score, reverse=True)
        
        return results
    
    def analyze_stock_sync(self, symbol: str, sector: str = "") -> StockAnalysis:
        """Synchronous wrapper for analyze_stock.
        
        Args:
            symbol: Stock symbol.
            sector: Optional sector.
            
        Returns:
            StockAnalysis object.
        """
        return asyncio.run(self.analyze_stock(symbol, sector))
