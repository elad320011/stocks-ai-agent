"""Main AI agent orchestrator for TASE stock recommendations."""

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import logging

import httpx

from src.config import get_settings, Settings
from src.data.tase_client import TASEClient
from src.data.yahoo_client import YahooFinanceClient
from src.data.news_scraper import NewsScraper
from src.rules.scoring import ScoringEngine, StockAnalysis
from src.rules.strategies import SwingTradeStrategy, RiskLevel, PortfolioRecommendation
from src.llm.gemini_client import GeminiClient, AIAnalysis

logger = logging.getLogger(__name__)


@dataclass
class AgentResult:
    """Container for agent analysis results."""
    analyses: list[StockAnalysis]
    ai_insights: dict[str, AIAnalysis]
    market_summary: str
    portfolio: Optional[PortfolioRecommendation]
    generated_at: datetime
    total_scanned: int = 0
    source: str = "watchlist"


class StockAgent:
    """Main AI agent for TASE stock recommendations.
    
    This agent combines:
    - TASE and Yahoo Finance data
    - Technical, fundamental, and sentiment analysis
    - Rule-based scoring
    - AI-powered reasoning via Gemini
    """
    
    # Fallback watchlist when dynamic fetch fails
    FALLBACK_WATCHLIST = [
        # TA-35 Blue Chips
        "TEVA",      # Teva Pharmaceutical
        "NICE",      # NICE Systems
        "CHKP",      # Check Point Software
        "WIX",       # Wix.com
        "FVRR",      # Fiverr
        "MNDY",      # Monday.com
        "ICL",       # ICL Group
        "POLI",      # Bank Hapoalim
        "LUMI",      # Bank Leumi
        "DSCT",      # Discount Bank
        "BEZQ",      # Bezeq
        "ELCO",      # Elco
        "AZRG",      # Azrieli Group
        "MZTF",      # Mizrahi Tefahot Bank
        "OPAL",      # Opal
        "AMOT",      # Amot Investments
        "MELI",      # Melisron
        "PRTC",      # Partner Communications
        "CEL",       # Cellcom
        "DELK",      # Delek Group
    ]
    
    # Extended list of known TASE symbols (for when API is unavailable)
    EXTENDED_TASE_SYMBOLS = [
        # TA-35 Index
        "TEVA", "NICE", "CHKP", "WIX", "FVRR", "MNDY", "ICL", "POLI", "LUMI",
        "DSCT", "BEZQ", "ELCO", "AZRG", "MZTF", "OPAL", "AMOT", "MELI", "PRTC",
        "CEL", "DELK", "ESLT", "ORA", "PHOE", "MGDL", "ALHE", "GLRS", "FTAL",
        "ILDC", "ARPT", "KRNV", "BSEN", "ORL", "ELBT", "AFRE",
        # TA-90 Additional
        "NVMI", "CMCT", "AFID", "ALLG", "AMAN", "ARKO", "AURA", "AVGL", "BCOM",
        "BIG", "BIMR", "BLRX", "BRAN", "BRIL", "CAST", "CLBV", "CLIS", "CMDR",
        "DANE", "DLRL", "DRAL", "ELRN", "EMCO", "ENLT", "ENRG", "EVGN", "FNTS",
        "FRSX", "GADS", "GAIL", "GAZP", "GILM", "GNRS", "HARL", "HAYL", "HLAN",
        "HOLM", "HTCO", "IDAN", "IGLD", "ILX", "INSL", "ISCD", "ISRS", "ISTA",
        "KRDI", "KRNT", "LCTX", "LPSN", "LUZN", "LZNR", "MALT", "MAXM", "MCRN",
        "MDGS", "MDTR", "MGIC", "MISH", "MLSR", "MMAN", "MNGA", "MTRX", "MTTR",
        "MYSZ", "NAWI", "NFTA", "NVPT", "ORAD", "ORBI", "ORMP", "OVRS", "PCBT",
        "PERT", "PLTF", "PLRM", "PNAX", "POLI", "PRGO", "PTNR", "PTCH", "QNCO",
        "RLCO", "RMLI", "ROBO", "RSEL", "RVLT", "RVSN", "SFET", "SHVA", "SKLN",
        "SMTO", "SPRG", "STCM", "STIC", "STRA", "TDRN", "TIGB", "TLRD", "TMRW",
        "TPLT", "TRVL", "TSEM", "UNIT", "VILR", "VIVO", "VNTZ", "WILC", "WKME",
        "YAAC", "YBOX", "ZION",
    ]
    
    def __init__(self, settings: Optional[Settings] = None):
        """Initialize the stock agent.
        
        Args:
            settings: Optional Settings object. Uses global settings if not provided.
        """
        self.settings = settings or get_settings()
        
        # Initialize clients
        self.yahoo = YahooFinanceClient()
        
        # TASE client (may fail if no API key)
        self.tase: Optional[TASEClient] = None
        if self.settings.tase_api_key:
            try:
                self.tase = TASEClient(self.settings.tase_api_key)
            except Exception as e:
                logger.warning(f"Could not initialize TASE client: {e}")
        
        # Initialize analysis components
        self.scorer = ScoringEngine(self.yahoo, self.tase)
        self.strategy = SwingTradeStrategy()
        
        # Gemini client (may fail if no API key)
        self.gemini: Optional[GeminiClient] = None
        if self.settings.gemini_api_key:
            try:
                self.gemini = GeminiClient(self.settings.gemini_api_key)
            except Exception as e:
                logger.warning(f"Could not initialize Gemini client: {e}")
        
        self._news_scraper = NewsScraper()
    
    @property
    def is_fully_configured(self) -> bool:
        """Check if agent is fully configured with all API keys."""
        return self.gemini is not None and self.tase is not None
    
    async def get_all_tase_symbols(self, use_indices: bool = True) -> list[str]:
        """Dynamically fetch all available TASE stock symbols.
        
        Tries multiple sources in order:
        1. TASE API (if configured)
        2. Web scraping from TASE website
        3. Extended hardcoded list as fallback
        
        Args:
            use_indices: If True, prioritize TA-35/TA-125 index components.
            
        Returns:
            List of stock symbols.
        """
        symbols = []
        source = "unknown"
        
        # Try 1: TASE API
        if self.tase:
            try:
                if use_indices:
                    # Get TA-35 and TA-125 components (most liquid stocks)
                    ta35 = self.tase.get_ta35_symbols()
                    ta125 = self.tase.get_ta125_symbols()
                    symbols = list(set(ta35 + ta125))
                    source = "TASE API (indices)"
                    logger.info(f"Fetched {len(symbols)} symbols from TASE indices")
                
                if not symbols:
                    # Get all securities
                    securities = self.tase.get_securities_list()
                    symbols = [s.symbol for s in securities if s.symbol]
                    source = "TASE API (all securities)"
                    logger.info(f"Fetched {len(symbols)} symbols from TASE API")
                
                if symbols:
                    return symbols
                    
            except Exception as e:
                logger.warning(f"TASE API fetch failed: {e}")
        
        # Try 2: Scrape from TASE website or financial news
        try:
            symbols = await self._scrape_tase_symbols()
            if symbols:
                source = "web scraping"
                logger.info(f"Fetched {len(symbols)} symbols from web scraping")
                return symbols
        except Exception as e:
            logger.warning(f"Web scraping failed: {e}")
        
        # Try 3: Use extended hardcoded list
        logger.info("Using extended hardcoded TASE symbol list")
        return self.EXTENDED_TASE_SYMBOLS.copy()
    
    async def _scrape_tase_symbols(self) -> list[str]:
        """Scrape TASE symbols from public sources.
        
        Returns:
            List of stock symbols.
        """
        symbols = []
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Try to get from Globes market page
            try:
                response = await client.get(
                    "https://en.globes.co.il/en/stockslist.aspx",
                    headers={
                        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                                      'AppleWebKit/537.36'
                    }
                )
                if response.status_code == 200:
                    # Parse stock symbols from the page
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(response.text, 'lxml')
                    
                    # Look for stock links/symbols
                    for link in soup.find_all('a', href=True):
                        href = link.get('href', '')
                        if '/quote/' in href or 'stock' in href.lower():
                            text = link.get_text(strip=True)
                            if text and len(text) <= 6 and text.isalpha():
                                symbols.append(text.upper())
                                
            except Exception as e:
                logger.debug(f"Globes scrape failed: {e}")
            
            # Try Yahoo Finance Israel
            try:
                # Get some popular Israeli stocks from Yahoo
                test_symbols = ["TEVA", "NICE", "CHKP", "ICL", "POLI"]
                for sym in test_symbols:
                    if sym not in symbols:
                        symbols.append(sym)
            except Exception:
                pass
        
        return list(set(symbols))
    
    async def discover_stocks(
        self,
        min_volume: int = 100000,
        min_market_cap: float = 100_000_000,
        sectors: Optional[list[str]] = None,
    ) -> list[str]:
        """Discover tradeable TASE stocks based on criteria.
        
        Args:
            min_volume: Minimum average daily volume.
            min_market_cap: Minimum market cap in ILS.
            sectors: Optional list of sectors to filter.
            
        Returns:
            List of qualifying stock symbols.
        """
        all_symbols = await self.get_all_tase_symbols()
        qualifying = []
        
        logger.info(f"Discovering stocks from {len(all_symbols)} candidates...")
        
        for symbol in all_symbols:
            try:
                quote = self.yahoo.get_quote(symbol)
                if quote is None:
                    continue
                
                # Apply filters
                if quote.avg_volume < min_volume:
                    continue
                    
                if quote.market_cap < min_market_cap:
                    continue
                
                qualifying.append(symbol)
                
            except Exception as e:
                logger.debug(f"Could not check {symbol}: {e}")
                continue
        
        logger.info(f"Found {len(qualifying)} qualifying stocks")
        return qualifying
    
    async def analyze_stock(
        self, 
        symbol: str,
        include_ai: bool = True,
    ) -> tuple[StockAnalysis, Optional[AIAnalysis]]:
        """Analyze a single stock.
        
        Args:
            symbol: Stock symbol (e.g., 'TEVA' or 'TEVA.TA')
            include_ai: Whether to include AI analysis.
            
        Returns:
            Tuple of (StockAnalysis, AIAnalysis or None)
        """
        # Run scoring analysis
        analysis = await self.scorer.analyze_stock(symbol)
        
        # Generate AI insights if available and requested
        ai_analysis = None
        if include_ai and self.gemini:
            try:
                ai_analysis = await self.gemini.analyze(analysis)
            except Exception as e:
                logger.warning(f"AI analysis failed for {symbol}: {e}")
        
        return analysis, ai_analysis
    
    async def scan_market(
        self,
        symbols: Optional[list[str]] = None,
        min_score: float = 60,
        include_ai: bool = True,
        top_n: int = 10,
        scan_all: bool = False,
        max_stocks: int = 100,
        progress_callback=None,
    ) -> AgentResult:
        """Scan multiple stocks and get recommendations.
        
        Args:
            symbols: List of symbols to scan. If not provided and scan_all=True,
                     fetches all TASE stocks dynamically.
            min_score: Minimum score threshold.
            include_ai: Whether to include AI analysis for top picks.
            top_n: Number of top picks to include AI analysis for.
            scan_all: If True and no symbols provided, scan all TASE stocks.
            max_stocks: Maximum number of stocks to scan (for performance).
            progress_callback: Optional async callback(current, total, symbol) for progress.
            
        Returns:
            AgentResult with all analyses and insights.
        """
        source = "custom"
        
        # Determine which symbols to scan
        if symbols:
            source = "custom list"
        elif scan_all:
            # Dynamically fetch all TASE stocks
            logger.info("Fetching all TASE stocks dynamically...")
            symbols = await self.get_all_tase_symbols(use_indices=True)
            source = "TASE exchange (dynamic)"
            
            # Limit for performance
            if len(symbols) > max_stocks:
                logger.info(f"Limiting scan to {max_stocks} stocks (from {len(symbols)})")
                symbols = symbols[:max_stocks]
        else:
            # Use extended list instead of small default
            symbols = self.EXTENDED_TASE_SYMBOLS.copy()
            source = "extended watchlist"
        
        total_to_scan = len(symbols)
        logger.info(f"Scanning {total_to_scan} stocks from {source}...")
        
        # Analyze all stocks
        analyses = []
        scanned_count = 0
        
        for i, symbol in enumerate(symbols):
            try:
                if progress_callback:
                    await progress_callback(i + 1, total_to_scan, symbol)
                
                analysis = await self.scorer.analyze_stock(symbol)
                scanned_count += 1
                
                if analysis.total_score >= min_score or len(analyses) < 5:
                    analyses.append(analysis)
                    
            except Exception as e:
                logger.debug(f"Failed to analyze {symbol}: {e}")
                continue
        
        # Sort by score
        analyses.sort(key=lambda x: x.total_score, reverse=True)
        
        logger.info(f"Found {len(analyses)} stocks meeting criteria (scanned {scanned_count})")
        
        # Generate AI insights for top picks
        ai_insights = {}
        if include_ai and self.gemini:
            for analysis in analyses[:top_n]:
                try:
                    ai = await self.gemini.analyze(analysis)
                    ai_insights[analysis.symbol] = ai
                except Exception as e:
                    logger.warning(f"AI analysis failed for {analysis.symbol}: {e}")
        
        # Generate market summary
        market_summary = ""
        if include_ai and self.gemini and analyses:
            try:
                market_summary = await self.gemini.generate_market_summary(analyses)
            except Exception as e:
                logger.warning(f"Market summary generation failed: {e}")
                market_summary = self._generate_basic_summary(analyses)
        else:
            market_summary = self._generate_basic_summary(analyses)
        
        return AgentResult(
            analyses=analyses,
            ai_insights=ai_insights,
            market_summary=market_summary,
            portfolio=None,
            generated_at=datetime.now(),
            total_scanned=scanned_count,
            source=source,
        )
    
    async def build_portfolio(
        self,
        budget: float,
        risk_level: str = "moderate",
        symbols: Optional[list[str]] = None,
    ) -> PortfolioRecommendation:
        """Build a diversified portfolio recommendation.
        
        Args:
            budget: Total investment budget in ILS.
            risk_level: Risk tolerance (conservative, moderate, aggressive).
            symbols: List of symbols to consider.
            
        Returns:
            PortfolioRecommendation object.
        """
        # Convert risk level string to enum
        risk_map = {
            'conservative': RiskLevel.CONSERVATIVE,
            'moderate': RiskLevel.MODERATE,
            'aggressive': RiskLevel.AGGRESSIVE,
        }
        risk = risk_map.get(risk_level.lower(), RiskLevel.MODERATE)
        
        # Scan market for candidates
        result = await self.scan_market(symbols=symbols, min_score=50, include_ai=False)
        
        # Build portfolio
        portfolio = self.strategy.build_portfolio(result.analyses, budget, risk)
        
        return portfolio
    
    async def monitor_stock(
        self,
        symbol: str,
        interval_seconds: int = 300,
        callback=None,
    ):
        """Monitor a stock with periodic updates.
        
        Args:
            symbol: Stock symbol to monitor.
            interval_seconds: Update interval in seconds.
            callback: Optional callback function for updates.
        """
        while True:
            try:
                analysis, ai = await self.analyze_stock(symbol, include_ai=False)
                
                if callback:
                    await callback(analysis)
                else:
                    logger.info(
                        f"{symbol}: Score={analysis.total_score:.1f}, "
                        f"Rec={analysis.recommendation.value}, "
                        f"Price=₪{analysis.current_price:.2f}"
                    )
                
                await asyncio.sleep(interval_seconds)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring error for {symbol}: {e}")
                await asyncio.sleep(interval_seconds)
    
    def _generate_basic_summary(self, analyses: list[StockAnalysis]) -> str:
        """Generate basic market summary without AI.
        
        Args:
            analyses: List of stock analyses.
            
        Returns:
            Summary string.
        """
        if not analyses:
            return "No stocks analyzed."
        
        # Calculate stats
        avg_score = sum(a.total_score for a in analyses) / len(analyses)
        buy_count = sum(1 for a in analyses if "BUY" in a.recommendation.value)
        sell_count = sum(1 for a in analyses if "SELL" in a.recommendation.value)
        
        top_picks = analyses[:3]
        
        summary = [
            f"Market Scan Summary ({datetime.now().strftime('%Y-%m-%d %H:%M')})",
            f"Analyzed: {len(analyses)} stocks | Avg Score: {avg_score:.1f}",
            f"Buy Signals: {buy_count} | Sell Signals: {sell_count}",
            "",
            "Top Picks:",
        ]
        
        for i, pick in enumerate(top_picks, 1):
            summary.append(
                f"  {i}. {pick.symbol} - {pick.recommendation.value} "
                f"(Score: {pick.total_score:.0f})"
            )
        
        return "\n".join(summary)
    
    async def get_market_sentiment(self) -> dict:
        """Get overall market sentiment from news.
        
        Returns:
            Dictionary with sentiment metrics.
        """
        from src.analysis.sentiment import SentimentAnalyzer
        
        analyzer = SentimentAnalyzer(self._news_scraper)
        metrics = await analyzer.analyze_market_sentiment()
        
        return {
            'overall_score': metrics.overall_score,
            'label': metrics.sentiment_label,
            'trend': metrics.trend,
            'articles_analyzed': metrics.total_articles,
            'positive': metrics.positive_articles,
            'negative': metrics.negative_articles,
            'neutral': metrics.neutral_articles,
            'headlines': metrics.recent_headlines[:5],
        }
    
    async def close(self):
        """Clean up resources."""
        if self._news_scraper:
            await self._news_scraper.close()


# Convenience function for quick analysis
def quick_analyze(symbol: str) -> tuple[StockAnalysis, Optional[AIAnalysis]]:
    """Quick synchronous analysis of a single stock.
    
    Args:
        symbol: Stock symbol.
        
    Returns:
        Tuple of (StockAnalysis, AIAnalysis or None)
    """
    agent = StockAgent()
    
    async def _run():
        try:
            return await agent.analyze_stock(symbol)
        finally:
            await agent.close()
    
    return asyncio.run(_run())
