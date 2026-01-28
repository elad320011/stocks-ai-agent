"""Google Gemini API integration for AI-powered stock analysis."""

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import logging
import json

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

from src.config import get_settings
from src.rules.scoring import StockAnalysis, Recommendation

logger = logging.getLogger(__name__)


@dataclass
class AIAnalysis:
    """AI-generated analysis result."""
    summary: str
    reasoning: str
    recommendation: str
    confidence: float
    key_factors: list[str]
    risks: list[str]
    catalysts: list[str]
    target_timeframe: str
    model_used: str
    generated_at: datetime


class GeminiClient:
    """Client for Google Gemini API integration."""
    
    SYSTEM_PROMPT = """You are an expert Israeli stock market analyst specializing in the Tel Aviv Stock Exchange (TASE).
Your task is to analyze stock data and provide actionable recommendations for swing trades (1-4 week holding period).

Analysis Framework:
1. Technical Analysis - Price patterns, momentum indicators, volume analysis
2. Fundamental Analysis - Valuation metrics, earnings quality, sector positioning
3. Sentiment Analysis - News flow, market sentiment, investor behavior

Output Requirements:
- Provide clear BUY/HOLD/SELL recommendation
- Explain key reasoning factors
- Identify specific risks and catalysts
- Consider Israeli market-specific factors (market hours, currency exposure, geopolitical risks)
- Target swing trade timeframe of 1-4 weeks

Always be objective and data-driven. Acknowledge uncertainty when present."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize Gemini client.
        
        Args:
            api_key: Optional Gemini API key. Uses settings if not provided.
        """
        if not GEMINI_AVAILABLE:
            raise ImportError(
                "google-generativeai is not installed. "
                "Install with: pip install google-generativeai"
            )
        
        self.settings = get_settings()
        self.api_key = api_key or self.settings.gemini_api_key
        
        if not self.api_key:
            raise ValueError("Gemini API key is required")
        
        # Configure the API
        genai.configure(api_key=self.api_key)
        
        # Initialize the model (using Gemini 2.0 Flash for free tier)
        self.model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            system_instruction=self.SYSTEM_PROMPT,
        )
        
        self._request_count = 0
        self._last_request_time: Optional[datetime] = None
    
    async def _rate_limit(self):
        """Implement rate limiting for Gemini free tier."""
        if self._last_request_time:
            elapsed = (datetime.now() - self._last_request_time).total_seconds()
            # Ensure at least 6 seconds between requests (10 RPM)
            if elapsed < 6:
                await asyncio.sleep(6 - elapsed)
        
        self._last_request_time = datetime.now()
        self._request_count += 1
    
    def _format_analysis_prompt(self, analysis: StockAnalysis) -> str:
        """Format stock analysis data into a prompt for Gemini.
        
        Args:
            analysis: StockAnalysis object with all metrics.
            
        Returns:
            Formatted prompt string.
        """
        # Build detailed prompt
        prompt_parts = [
            f"# Stock Analysis Request: {analysis.symbol}",
            f"**Company:** {analysis.name}",
            f"**Current Price:** ₪{analysis.current_price:.2f}",
            f"**Analysis Timestamp:** {analysis.analysis_timestamp.strftime('%Y-%m-%d %H:%M')}",
            "",
            "## Scoring Summary",
            f"- **Total Score:** {analysis.total_score:.1f}/100",
            f"- **Technical Score:** {analysis.technical_score:.1f}/100",
            f"- **Fundamental Score:** {analysis.fundamental_score:.1f}/100",
            f"- **Sentiment Score:** {analysis.sentiment_score:.1f}/100",
            f"- **Current Recommendation:** {analysis.recommendation.value}",
            f"- **Confidence Level:** {analysis.confidence:.1f}%",
            "",
        ]
        
        # Technical details
        if analysis.technical_details and analysis.technical_details.indicators:
            ind = analysis.technical_details.indicators
            rsi_str = f"{ind.rsi_14:.1f}" if ind.rsi_14 is not None else "N/A"
            vol_ratio_str = f"{ind.volume_ratio:.2f}" if ind.volume_ratio is not None else "N/A"
            prompt_parts.extend([
                "## Technical Indicators",
                f"- RSI (14): {rsi_str} - {ind.rsi_signal}",
                f"- MACD Crossover: {ind.macd_crossover}",
                f"- Trend: {ind.trend} (strength: {ind.trend_strength:.0f}%)",
                f"- Price vs SMA20: {ind.price_vs_sma20}",
                f"- Price vs SMA50: {ind.price_vs_sma50}",
                f"- Price vs SMA200: {ind.price_vs_sma200}",
                f"- Golden Cross: {'Yes' if ind.golden_cross else 'No'}",
                f"- Volume: {'High' if ind.high_volume else 'Normal'} (ratio: {vol_ratio_str})",
                f"- Bollinger Position: {ind.bb_position}",
                f"- Bullish Divergence: {'Yes' if ind.bullish_divergence else 'No'}",
                f"- Bearish Divergence: {'Yes' if ind.bearish_divergence else 'No'}",
                "",
            ])
        
        # Fundamental details
        if analysis.fundamental_details and analysis.fundamental_details.metrics:
            metrics = analysis.fundamental_details.metrics
            pe_str = f"{metrics.pe_ratio:.1f}" if metrics.pe_ratio is not None else "N/A"
            sector_pe_str = f"{metrics.sector_avg_pe:.1f}" if metrics.sector_avg_pe is not None else "N/A"
            prompt_parts.extend([
                "## Fundamental Metrics",
                f"- P/E Ratio: {pe_str}",
                f"- Sector Avg P/E: {sector_pe_str}",
                f"- Valuation vs Sector: {metrics.pe_vs_sector}",
                f"- Market Cap: ₪{metrics.market_cap/1e9:.2f}B ({metrics.market_cap_category})",
                f"- 52-Week Range: ₪{metrics.fifty_two_week_low:.2f} - ₪{metrics.fifty_two_week_high:.2f}",
                f"- Price vs 52W High: {metrics.price_vs_52w_high:.1f}%",
                f"- Dividend Yield: {metrics.dividend_yield*100:.1f}%" if metrics.dividend_yield else "- Dividend Yield: N/A",
                f"- TA-35 Member: {'Yes' if metrics.is_ta35 else 'No'}",
                f"- TA-125 Member: {'Yes' if metrics.is_ta125 else 'No'}",
                f"- Liquidity: {metrics.volume_category}",
                "",
            ])
        
        # Sentiment details
        if analysis.sentiment_details and analysis.sentiment_details.metrics:
            sent = analysis.sentiment_details.metrics
            prompt_parts.extend([
                "## Sentiment Analysis",
                f"- Overall Sentiment: {sent.sentiment_label} (score: {sent.overall_score:.2f})",
                f"- Articles Analyzed: {sent.total_articles}",
                f"- Positive: {sent.positive_articles}, Negative: {sent.negative_articles}, Neutral: {sent.neutral_articles}",
                f"- Trend: {sent.trend}",
                f"- Confidence: {sent.confidence:.1%}",
            ])
            if sent.recent_headlines:
                prompt_parts.append("- Recent Headlines:")
                for headline in sent.recent_headlines[:3]:
                    prompt_parts.append(f"  * {headline}")
            prompt_parts.append("")
        
        # Signals
        if analysis.bullish_signals:
            prompt_parts.append("## Bullish Signals")
            for signal in analysis.bullish_signals:
                prompt_parts.append(f"- {signal}")
            prompt_parts.append("")
        
        if analysis.bearish_signals:
            prompt_parts.append("## Bearish Signals")
            for signal in analysis.bearish_signals:
                prompt_parts.append(f"- {signal}")
            prompt_parts.append("")
        
        # Request
        prompt_parts.extend([
            "## Task",
            "Based on the above analysis, provide:",
            "1. A 2-3 sentence executive summary",
            "2. Your recommendation (STRONG BUY / BUY / HOLD / SELL / STRONG SELL)",
            "3. Key reasoning factors (3-5 points)",
            "4. Main risks to consider (2-3 points)",
            "5. Potential catalysts in next 1-4 weeks (2-3 points)",
            "6. Your confidence level (0-100%)",
            "",
            "Format your response as valid JSON with keys: summary, recommendation, key_factors, risks, catalysts, confidence",
        ])
        
        return "\n".join(prompt_parts)
    
    async def analyze(self, stock_analysis: StockAnalysis) -> AIAnalysis:
        """Generate AI-powered analysis for a stock.
        
        Args:
            stock_analysis: StockAnalysis object with all metrics.
            
        Returns:
            AIAnalysis object with AI-generated insights.
        """
        await self._rate_limit()
        
        prompt = self._format_analysis_prompt(stock_analysis)
        
        try:
            # Generate response
            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt,
                generation_config={
                    "temperature": 0.3,  # Lower temperature for more consistent analysis
                    "max_output_tokens": 1024,
                },
            )
            
            response_text = response.text
            
            # Try to parse as JSON
            try:
                # Find JSON in response
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                if json_start != -1 and json_end > json_start:
                    json_str = response_text[json_start:json_end]
                    data = json.loads(json_str)
                    
                    return AIAnalysis(
                        summary=data.get('summary', ''),
                        reasoning=response_text,
                        recommendation=data.get('recommendation', stock_analysis.recommendation.value),
                        confidence=float(data.get('confidence', stock_analysis.confidence)),
                        key_factors=data.get('key_factors', []),
                        risks=data.get('risks', []),
                        catalysts=data.get('catalysts', []),
                        target_timeframe="1-4 weeks",
                        model_used="gemini-2.0-flash",
                        generated_at=datetime.now(),
                    )
            except json.JSONDecodeError:
                logger.warning("Could not parse JSON response, using raw text")
            
            # Fallback to raw text parsing
            return AIAnalysis(
                summary=response_text[:500] if len(response_text) > 500 else response_text,
                reasoning=response_text,
                recommendation=stock_analysis.recommendation.value,
                confidence=stock_analysis.confidence,
                key_factors=stock_analysis.bullish_signals[:3],
                risks=stock_analysis.bearish_signals[:3],
                catalysts=[],
                target_timeframe="1-4 weeks",
                model_used="gemini-2.0-flash",
                generated_at=datetime.now(),
            )
            
        except Exception as e:
            logger.error(f"Error generating AI analysis: {e}")
            # Return fallback analysis
            return AIAnalysis(
                summary=f"Analysis for {stock_analysis.symbol}: Score {stock_analysis.total_score:.0f}/100, "
                        f"Recommendation: {stock_analysis.recommendation.value}",
                reasoning="AI analysis unavailable - using rule-based scoring",
                recommendation=stock_analysis.recommendation.value,
                confidence=stock_analysis.confidence * 0.8,  # Reduce confidence
                key_factors=stock_analysis.bullish_signals,
                risks=stock_analysis.bearish_signals,
                catalysts=[],
                target_timeframe="1-4 weeks",
                model_used="fallback",
                generated_at=datetime.now(),
            )
    
    async def generate_market_summary(self, analyses: list[StockAnalysis]) -> str:
        """Generate AI summary of multiple stock analyses.
        
        Args:
            analyses: List of StockAnalysis objects.
            
        Returns:
            Market summary string.
        """
        await self._rate_limit()
        
        # Build summary prompt
        prompt_parts = [
            "# Israeli Stock Market Analysis Summary",
            f"**Date:** {datetime.now().strftime('%Y-%m-%d')}",
            f"**Stocks Analyzed:** {len(analyses)}",
            "",
            "## Top Opportunities:",
        ]
        
        # Add top 5 stocks
        top_stocks = sorted(analyses, key=lambda x: x.total_score, reverse=True)[:5]
        for i, stock in enumerate(top_stocks, 1):
            prompt_parts.append(
                f"{i}. **{stock.symbol}** ({stock.name}): "
                f"Score {stock.total_score:.0f}, {stock.recommendation.value}"
            )
        
        prompt_parts.extend([
            "",
            "## Task",
            "Provide a 3-4 paragraph market summary covering:",
            "1. Overall market sentiment based on the analyzed stocks",
            "2. Key sectors or themes showing strength/weakness",
            "3. Top 2-3 actionable opportunities with brief rationale",
            "4. Key risks to watch in the coming weeks",
            "",
            "Write in a professional, concise analyst style.",
        ])
        
        prompt = "\n".join(prompt_parts)
        
        try:
            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt,
                generation_config={
                    "temperature": 0.4,
                    "max_output_tokens": 1024,
                },
            )
            return response.text
            
        except Exception as e:
            logger.error(f"Error generating market summary: {e}")
            return f"Market summary unavailable. Analyzed {len(analyses)} stocks. " \
                   f"Top pick: {top_stocks[0].symbol if top_stocks else 'N/A'}"
    
    def analyze_sync(self, stock_analysis: StockAnalysis) -> AIAnalysis:
        """Synchronous wrapper for analyze.
        
        Args:
            stock_analysis: StockAnalysis object.
            
        Returns:
            AIAnalysis object.
        """
        return asyncio.run(self.analyze(stock_analysis))
