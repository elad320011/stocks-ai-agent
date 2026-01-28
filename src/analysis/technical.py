"""Technical analysis module using pandas-ta indicators."""

from dataclasses import dataclass, field
from typing import Optional
import logging

import pandas as pd
import numpy as np

try:
    import pandas_ta as ta
    PANDAS_TA_AVAILABLE = True
except ImportError:
    PANDAS_TA_AVAILABLE = False

from src.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class TechnicalIndicators:
    """Container for technical indicator values."""
    # RSI
    rsi_14: Optional[float] = None
    rsi_signal: str = "neutral"  # oversold, neutral, overbought
    
    # MACD
    macd_value: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None
    macd_crossover: str = "none"  # bullish, bearish, none
    
    # Moving Averages
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    sma_200: Optional[float] = None
    ema_12: Optional[float] = None
    ema_26: Optional[float] = None
    
    # Price relative to MAs
    price_vs_sma20: str = "neutral"  # above, below, neutral
    price_vs_sma50: str = "neutral"
    price_vs_sma200: str = "neutral"
    
    # Crossovers
    golden_cross: bool = False  # SMA50 > SMA200
    death_cross: bool = False   # SMA50 < SMA200
    
    # Volume
    volume_sma_20: Optional[float] = None
    volume_ratio: Optional[float] = None  # current volume / avg volume
    high_volume: bool = False
    
    # Bollinger Bands
    bb_upper: Optional[float] = None
    bb_middle: Optional[float] = None
    bb_lower: Optional[float] = None
    bb_position: str = "middle"  # upper, middle, lower
    
    # ATR (volatility)
    atr_14: Optional[float] = None
    
    # Stochastic
    stoch_k: Optional[float] = None
    stoch_d: Optional[float] = None
    stoch_signal: str = "neutral"
    
    # Trend
    trend: str = "neutral"  # bullish, bearish, neutral
    trend_strength: float = 0.0  # 0-100
    
    # Divergences
    bullish_divergence: bool = False
    bearish_divergence: bool = False


@dataclass
class TechnicalScore:
    """Technical analysis score breakdown."""
    total_score: float = 0.0  # 0-100
    rsi_score: float = 0.0
    macd_score: float = 0.0
    ma_score: float = 0.0
    volume_score: float = 0.0
    trend_score: float = 0.0
    signals: list[str] = field(default_factory=list)
    indicators: Optional[TechnicalIndicators] = None


class TechnicalAnalyzer:
    """Technical analysis engine using pandas-ta."""
    
    def __init__(self):
        """Initialize technical analyzer."""
        if not PANDAS_TA_AVAILABLE:
            raise ImportError(
                "pandas-ta is not installed. Install with: pip install pandas-ta"
            )
        
        self.settings = get_settings()
    
    def calculate_indicators(self, df: pd.DataFrame) -> TechnicalIndicators:
        """Calculate all technical indicators for a stock.
        
        Args:
            df: DataFrame with OHLCV data (columns: open, high, low, close, volume)
            
        Returns:
            TechnicalIndicators object with all calculated values.
        """
        if df is None or df.empty:
            logger.warning("Empty dataframe provided")
            return TechnicalIndicators()
        
        # Ensure we have enough data
        if len(df) < 50:
            logger.warning(f"Insufficient data: {len(df)} rows (need at least 50)")
            return TechnicalIndicators()
        
        indicators = TechnicalIndicators()
        
        try:
            # Get current price
            current_price = df['close'].iloc[-1]
            
            # RSI
            rsi = ta.rsi(df['close'], length=14)
            if rsi is not None and len(rsi) > 0:
                indicators.rsi_14 = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else None
                if indicators.rsi_14:
                    if indicators.rsi_14 < 30:
                        indicators.rsi_signal = "oversold"
                    elif indicators.rsi_14 > 70:
                        indicators.rsi_signal = "overbought"
                    else:
                        indicators.rsi_signal = "neutral"
            
            # MACD
            macd_result = ta.macd(df['close'], fast=12, slow=26, signal=9)
            if macd_result is not None and not macd_result.empty:
                macd_col = [c for c in macd_result.columns if 'MACD_12_26_9' in c and 'h' not in c.lower() and 's' not in c.lower()]
                signal_col = [c for c in macd_result.columns if 'MACDs' in c]
                hist_col = [c for c in macd_result.columns if 'MACDh' in c]
                
                if macd_col:
                    indicators.macd_value = float(macd_result[macd_col[0]].iloc[-1])
                if signal_col:
                    indicators.macd_signal = float(macd_result[signal_col[0]].iloc[-1])
                if hist_col:
                    indicators.macd_histogram = float(macd_result[hist_col[0]].iloc[-1])
                    # Check for crossover
                    if len(macd_result) >= 2:
                        prev_hist = macd_result[hist_col[0]].iloc[-2]
                        curr_hist = macd_result[hist_col[0]].iloc[-1]
                        if prev_hist < 0 and curr_hist > 0:
                            indicators.macd_crossover = "bullish"
                        elif prev_hist > 0 and curr_hist < 0:
                            indicators.macd_crossover = "bearish"
            
            # Moving Averages
            sma_20 = ta.sma(df['close'], length=20)
            sma_50 = ta.sma(df['close'], length=50)
            sma_200 = ta.sma(df['close'], length=200)
            ema_12 = ta.ema(df['close'], length=12)
            ema_26 = ta.ema(df['close'], length=26)
            
            if sma_20 is not None and len(sma_20) > 0:
                indicators.sma_20 = float(sma_20.iloc[-1]) if not pd.isna(sma_20.iloc[-1]) else None
                if indicators.sma_20:
                    indicators.price_vs_sma20 = "above" if current_price > indicators.sma_20 else "below"
            
            if sma_50 is not None and len(sma_50) > 0:
                indicators.sma_50 = float(sma_50.iloc[-1]) if not pd.isna(sma_50.iloc[-1]) else None
                if indicators.sma_50:
                    indicators.price_vs_sma50 = "above" if current_price > indicators.sma_50 else "below"
            
            if sma_200 is not None and len(sma_200) > 0:
                indicators.sma_200 = float(sma_200.iloc[-1]) if not pd.isna(sma_200.iloc[-1]) else None
                if indicators.sma_200:
                    indicators.price_vs_sma200 = "above" if current_price > indicators.sma_200 else "below"
            
            if ema_12 is not None and len(ema_12) > 0:
                indicators.ema_12 = float(ema_12.iloc[-1]) if not pd.isna(ema_12.iloc[-1]) else None
            
            if ema_26 is not None and len(ema_26) > 0:
                indicators.ema_26 = float(ema_26.iloc[-1]) if not pd.isna(ema_26.iloc[-1]) else None
            
            # Golden/Death Cross
            if indicators.sma_50 and indicators.sma_200:
                indicators.golden_cross = indicators.sma_50 > indicators.sma_200
                indicators.death_cross = indicators.sma_50 < indicators.sma_200
            
            # Volume analysis
            vol_sma = ta.sma(df['volume'], length=20)
            if vol_sma is not None and len(vol_sma) > 0:
                indicators.volume_sma_20 = float(vol_sma.iloc[-1]) if not pd.isna(vol_sma.iloc[-1]) else None
                if indicators.volume_sma_20 and indicators.volume_sma_20 > 0:
                    indicators.volume_ratio = float(df['volume'].iloc[-1] / indicators.volume_sma_20)
                    indicators.high_volume = indicators.volume_ratio > 1.5
            
            # Bollinger Bands
            bbands = ta.bbands(df['close'], length=20, std=2)
            if bbands is not None and not bbands.empty:
                upper_col = [c for c in bbands.columns if 'BBU' in c]
                middle_col = [c for c in bbands.columns if 'BBM' in c]
                lower_col = [c for c in bbands.columns if 'BBL' in c]
                
                if upper_col:
                    indicators.bb_upper = float(bbands[upper_col[0]].iloc[-1])
                if middle_col:
                    indicators.bb_middle = float(bbands[middle_col[0]].iloc[-1])
                if lower_col:
                    indicators.bb_lower = float(bbands[lower_col[0]].iloc[-1])
                
                if indicators.bb_upper and indicators.bb_lower:
                    bb_range = indicators.bb_upper - indicators.bb_lower
                    if bb_range > 0:
                        position = (current_price - indicators.bb_lower) / bb_range
                        if position > 0.8:
                            indicators.bb_position = "upper"
                        elif position < 0.2:
                            indicators.bb_position = "lower"
                        else:
                            indicators.bb_position = "middle"
            
            # ATR
            atr = ta.atr(df['high'], df['low'], df['close'], length=14)
            if atr is not None and len(atr) > 0:
                indicators.atr_14 = float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else None
            
            # Stochastic
            stoch = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3)
            if stoch is not None and not stoch.empty:
                k_col = [c for c in stoch.columns if 'STOCHk' in c]
                d_col = [c for c in stoch.columns if 'STOCHd' in c]
                
                if k_col:
                    indicators.stoch_k = float(stoch[k_col[0]].iloc[-1])
                if d_col:
                    indicators.stoch_d = float(stoch[d_col[0]].iloc[-1])
                
                if indicators.stoch_k:
                    if indicators.stoch_k < 20:
                        indicators.stoch_signal = "oversold"
                    elif indicators.stoch_k > 80:
                        indicators.stoch_signal = "overbought"
            
            # Determine overall trend
            indicators.trend, indicators.trend_strength = self._calculate_trend(df, indicators)
            
            # Check for divergences
            indicators.bullish_divergence = self._check_bullish_divergence(df, indicators)
            indicators.bearish_divergence = self._check_bearish_divergence(df, indicators)
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
        
        return indicators
    
    def _calculate_trend(
        self, 
        df: pd.DataFrame, 
        indicators: TechnicalIndicators
    ) -> tuple[str, float]:
        """Calculate overall trend direction and strength.
        
        Returns:
            Tuple of (trend direction, strength 0-100)
        """
        bullish_signals = 0
        bearish_signals = 0
        total_signals = 0
        
        # Price vs MAs
        if indicators.price_vs_sma20 == "above":
            bullish_signals += 1
        elif indicators.price_vs_sma20 == "below":
            bearish_signals += 1
        total_signals += 1
        
        if indicators.price_vs_sma50 == "above":
            bullish_signals += 1
        elif indicators.price_vs_sma50 == "below":
            bearish_signals += 1
        total_signals += 1
        
        if indicators.price_vs_sma200 == "above":
            bullish_signals += 2  # Higher weight
        elif indicators.price_vs_sma200 == "below":
            bearish_signals += 2
        total_signals += 2
        
        # MACD
        if indicators.macd_crossover == "bullish":
            bullish_signals += 2
        elif indicators.macd_crossover == "bearish":
            bearish_signals += 2
        total_signals += 2
        
        # RSI
        if indicators.rsi_signal == "oversold":
            bullish_signals += 1  # Potential reversal
        elif indicators.rsi_signal == "overbought":
            bearish_signals += 1
        total_signals += 1
        
        # Golden/Death cross
        if indicators.golden_cross:
            bullish_signals += 2
        elif indicators.death_cross:
            bearish_signals += 2
        total_signals += 2
        
        if total_signals == 0:
            return "neutral", 0.0
        
        net_signal = bullish_signals - bearish_signals
        max_signal = total_signals
        
        strength = abs(net_signal) / max_signal * 100
        
        if net_signal > 1:
            return "bullish", strength
        elif net_signal < -1:
            return "bearish", strength
        else:
            return "neutral", strength
    
    def _check_bullish_divergence(
        self, 
        df: pd.DataFrame, 
        indicators: TechnicalIndicators
    ) -> bool:
        """Check for bullish divergence (price making lower lows, RSI making higher lows)."""
        if indicators.rsi_14 is None or len(df) < 20:
            return False
        
        try:
            # Look at last 20 periods
            recent_prices = df['close'].iloc[-20:]
            rsi = ta.rsi(df['close'], length=14).iloc[-20:]
            
            # Find local minima in last 20 periods
            price_min_idx = recent_prices.idxmin()
            
            # Check if current price is lower than recent minimum but RSI is higher
            if len(rsi) > 0 and not rsi.empty:
                current_rsi = rsi.iloc[-1]
                min_rsi = rsi.min()
                
                # Simplified divergence check
                if (df['close'].iloc[-1] < recent_prices.iloc[-10:].min() and 
                    current_rsi > min_rsi and 
                    indicators.rsi_signal == "oversold"):
                    return True
        except Exception:
            pass
        
        return False
    
    def _check_bearish_divergence(
        self, 
        df: pd.DataFrame, 
        indicators: TechnicalIndicators
    ) -> bool:
        """Check for bearish divergence (price making higher highs, RSI making lower highs)."""
        if indicators.rsi_14 is None or len(df) < 20:
            return False
        
        try:
            recent_prices = df['close'].iloc[-20:]
            rsi = ta.rsi(df['close'], length=14).iloc[-20:]
            
            if len(rsi) > 0 and not rsi.empty:
                current_rsi = rsi.iloc[-1]
                max_rsi = rsi.max()
                
                if (df['close'].iloc[-1] > recent_prices.iloc[-10:].max() and 
                    current_rsi < max_rsi and 
                    indicators.rsi_signal == "overbought"):
                    return True
        except Exception:
            pass
        
        return False
    
    def calculate_score(self, df: pd.DataFrame) -> TechnicalScore:
        """Calculate technical analysis score for a stock.
        
        Args:
            df: DataFrame with OHLCV data.
            
        Returns:
            TechnicalScore object with breakdown.
        """
        indicators = self.calculate_indicators(df)
        score = TechnicalScore(indicators=indicators)
        signals = []
        
        # RSI Score (max 15 points)
        if indicators.rsi_14:
            if indicators.rsi_signal == "oversold":
                score.rsi_score = 15
                signals.append("RSI oversold - potential buy signal")
            elif indicators.rsi_14 < 40:
                score.rsi_score = 10
                signals.append("RSI approaching oversold")
            elif indicators.rsi_signal == "neutral":
                score.rsi_score = 7
            elif indicators.rsi_14 > 60:
                score.rsi_score = 3
            else:
                score.rsi_score = 0
                signals.append("RSI overbought - caution")
        
        # MACD Score (max 15 points)
        if indicators.macd_crossover == "bullish":
            score.macd_score = 15
            signals.append("MACD bullish crossover")
        elif indicators.macd_histogram and indicators.macd_histogram > 0:
            score.macd_score = 10
            signals.append("MACD histogram positive")
        elif indicators.macd_crossover == "bearish":
            score.macd_score = 0
            signals.append("MACD bearish crossover")
        else:
            score.macd_score = 5
        
        # Moving Average Score (max 20 points)
        ma_score = 0
        if indicators.price_vs_sma20 == "above":
            ma_score += 5
            signals.append("Price above SMA20")
        if indicators.price_vs_sma50 == "above":
            ma_score += 5
        if indicators.price_vs_sma200 == "above":
            ma_score += 5
            signals.append("Price above SMA200 - bullish")
        if indicators.golden_cross:
            ma_score += 5
            signals.append("Golden Cross (SMA50 > SMA200)")
        elif indicators.death_cross:
            signals.append("Death Cross (SMA50 < SMA200) - bearish")
        score.ma_score = ma_score
        
        # Volume Score (max 10 points)
        if indicators.high_volume and indicators.price_vs_sma20 == "above":
            score.volume_score = 10
            signals.append("High volume breakout")
        elif indicators.high_volume:
            score.volume_score = 5
        else:
            score.volume_score = 2
        
        # Trend Score (max 10 points)
        if indicators.trend == "bullish":
            score.trend_score = min(10, indicators.trend_strength / 10)
            signals.append(f"Bullish trend (strength: {indicators.trend_strength:.0f}%)")
        elif indicators.trend == "bearish":
            score.trend_score = 0
            signals.append(f"Bearish trend (strength: {indicators.trend_strength:.0f}%)")
        else:
            score.trend_score = 5
        
        # Divergence bonus
        if indicators.bullish_divergence:
            score.trend_score = min(10, score.trend_score + 5)
            signals.append("Bullish divergence detected")
        
        # Calculate total (max ~70 points from technical, scaled to 40% weight)
        raw_total = (score.rsi_score + score.macd_score + score.ma_score + 
                     score.volume_score + score.trend_score)
        score.total_score = min(100, raw_total / 70 * 100)
        score.signals = signals
        
        return score
