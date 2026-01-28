"""News scraper for Israeli financial news sources."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
import logging
import re

import httpx
from bs4 import BeautifulSoup

from src.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class NewsArticle:
    """Represents a news article."""
    title: str
    url: str
    source: str
    published_at: Optional[datetime] = None
    summary: str = ""
    symbols: list[str] = field(default_factory=list)
    sentiment_score: Optional[float] = None  # -1 to 1 scale


class NewsScraper:
    """Scraper for Israeli financial news from Globes and Calcalist."""
    
    GLOBES_EN_URL = "https://en.globes.co.il"
    GLOBES_MARKET_URL = f"{GLOBES_EN_URL}/en/market.tag"
    # Use English Globes and Yahoo Finance news instead of Calcalist (blocks scraping)
    CALCALIST_URL = "https://en.globes.co.il"
    CALCALIST_MARKETS_URL = "https://en.globes.co.il/en/businessnews"
    
    # Common Hebrew/English company name mappings
    COMPANY_MAPPINGS = {
        'teva': ['teva', 'טבע'],
        'bank hapoalim': ['hapoalim', 'הפועלים', 'poli'],
        'bank leumi': ['leumi', 'לאומי'],
        'ice': ['ice', 'אייס'],
        'elbit': ['elbit', 'אלביט'],
        'nice': ['nice', 'נייס'],
        'check point': ['checkpoint', 'צק פוינט', 'check point'],
        'wix': ['wix', 'וויקס'],
        'fiverr': ['fiverr', 'פייבר'],
        'monday': ['monday.com', 'מאנדיי'],
    }
    
    def __init__(self):
        """Initialize news scraper."""
        self.settings = get_settings()
        self._cache: dict[str, tuple[list[NewsArticle], datetime]] = {}
        self._client: Optional[httpx.AsyncClient] = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=30.0,
                headers={
                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                                  'AppleWebKit/537.36 (KHTML, like Gecko) '
                                  'Chrome/120.0.0.0 Safari/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9',
                    'Accept-Language': 'en-US,en;q=0.9,he;q=0.8',
                },
                follow_redirects=True,
            )
        return self._client
    
    async def close(self):
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
    
    def _is_cache_valid(self, cache_time: datetime) -> bool:
        """Check if cached data is still valid (15 min cache for news)."""
        elapsed = (datetime.now() - cache_time).total_seconds()
        return elapsed < 900  # 15 minutes for news
    
    async def scrape_globes(self, limit: int = 20) -> list[NewsArticle]:
        """Scrape news from Globes English market section.
        
        Args:
            limit: Maximum number of articles to return.
            
        Returns:
            List of NewsArticle objects.
        """
        cache_key = "globes"
        
        if cache_key in self._cache:
            articles, cache_time = self._cache[cache_key]
            if self._is_cache_valid(cache_time):
                return articles[:limit]
        
        articles = []
        
        try:
            client = await self._get_client()
            response = await client.get(self.GLOBES_MARKET_URL)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'lxml')
            
            # Find article elements (Globes structure)
            article_elements = soup.find_all(['article', 'div'], class_=re.compile(r'article|item|story'))
            
            for elem in article_elements[:limit * 2]:  # Get extra in case some fail
                try:
                    # Try to find title
                    title_elem = elem.find(['h2', 'h3', 'h4', 'a'], class_=re.compile(r'title|headline'))
                    if not title_elem:
                        title_elem = elem.find('a')
                    
                    if not title_elem:
                        continue
                    
                    title = title_elem.get_text(strip=True)
                    if not title or len(title) < 10:
                        continue
                    
                    # Get URL
                    url = None
                    if title_elem.name == 'a':
                        url = title_elem.get('href', '')
                    else:
                        link = elem.find('a')
                        url = link.get('href', '') if link else ''
                    
                    if url and not url.startswith('http'):
                        url = f"{self.GLOBES_EN_URL}{url}"
                    
                    # Get summary if available
                    summary_elem = elem.find(['p', 'div'], class_=re.compile(r'summary|excerpt|description'))
                    summary = summary_elem.get_text(strip=True) if summary_elem else ""
                    
                    article = NewsArticle(
                        title=title,
                        url=url or "",
                        source="Globes",
                        summary=summary,
                    )
                    
                    articles.append(article)
                    
                    if len(articles) >= limit:
                        break
                        
                except Exception as e:
                    logger.debug(f"Error parsing article element: {e}")
                    continue
            
            self._cache[cache_key] = (articles, datetime.now())
            logger.info(f"Scraped {len(articles)} articles from Globes")
            
        except Exception as e:
            logger.error(f"Error scraping Globes: {e}")
            # Return cached data if available
            if cache_key in self._cache:
                return self._cache[cache_key][0][:limit]
        
        return articles
    
    async def scrape_calcalist(self, limit: int = 20) -> list[NewsArticle]:
        """Scrape news from Calcalist Tech.
        
        Args:
            limit: Maximum number of articles to return.
            
        Returns:
            List of NewsArticle objects.
        """
        cache_key = "calcalist"
        
        if cache_key in self._cache:
            articles, cache_time = self._cache[cache_key]
            if self._is_cache_valid(cache_time):
                return articles[:limit]
        
        articles = []
        
        try:
            client = await self._get_client()
            response = await client.get(self.CALCALIST_MARKETS_URL)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'lxml')
            
            # Find article elements
            article_elements = soup.find_all(['article', 'div'], class_=re.compile(r'article|item|card'))
            
            for elem in article_elements[:limit * 2]:
                try:
                    title_elem = elem.find(['h2', 'h3', 'h4', 'a'], class_=re.compile(r'title|headline'))
                    if not title_elem:
                        title_elem = elem.find('a')
                    
                    if not title_elem:
                        continue
                    
                    title = title_elem.get_text(strip=True)
                    if not title or len(title) < 10:
                        continue
                    
                    url = None
                    if title_elem.name == 'a':
                        url = title_elem.get('href', '')
                    else:
                        link = elem.find('a')
                        url = link.get('href', '') if link else ''
                    
                    if url and not url.startswith('http'):
                        url = f"{self.CALCALIST_URL}{url}"
                    
                    article = NewsArticle(
                        title=title,
                        url=url or "",
                        source="Calcalist",
                    )
                    
                    articles.append(article)
                    
                    if len(articles) >= limit:
                        break
                        
                except Exception as e:
                    logger.debug(f"Error parsing Calcalist article: {e}")
                    continue
            
            self._cache[cache_key] = (articles, datetime.now())
            logger.info(f"Scraped {len(articles)} articles from Calcalist")
            
        except Exception as e:
            logger.error(f"Error scraping Calcalist: {e}")
            if cache_key in self._cache:
                return self._cache[cache_key][0][:limit]
        
        return articles
    
    async def get_all_news(self, limit: int = 40) -> list[NewsArticle]:
        """Get news from all sources.
        
        Args:
            limit: Maximum total articles to return.
            
        Returns:
            Combined list of articles from all sources.
        """
        # Fetch from both sources concurrently
        globes_task = self.scrape_globes(limit // 2)
        calcalist_task = self.scrape_calcalist(limit // 2)
        
        globes_articles, calcalist_articles = await asyncio.gather(
            globes_task, calcalist_task, return_exceptions=True
        )
        
        all_articles = []
        
        if isinstance(globes_articles, list):
            all_articles.extend(globes_articles)
        else:
            logger.error(f"Globes scrape failed: {globes_articles}")
        
        if isinstance(calcalist_articles, list):
            all_articles.extend(calcalist_articles)
        else:
            logger.error(f"Calcalist scrape failed: {calcalist_articles}")
        
        return all_articles[:limit]
    
    def find_mentioned_symbols(self, article: NewsArticle, known_symbols: list[str]) -> list[str]:
        """Find stock symbols mentioned in an article.
        
        Args:
            article: NewsArticle to analyze
            known_symbols: List of known TASE symbols
            
        Returns:
            List of mentioned symbols.
        """
        text = f"{article.title} {article.summary}".lower()
        found_symbols = []
        
        # Check for direct symbol mentions
        for symbol in known_symbols:
            symbol_clean = symbol.replace('.TA', '').lower()
            if symbol_clean in text:
                found_symbols.append(symbol)
        
        # Check company name mappings
        for symbol, aliases in self.COMPANY_MAPPINGS.items():
            for alias in aliases:
                if alias.lower() in text:
                    # Map to proper TASE symbol
                    tase_symbol = f"{symbol.upper().replace(' ', '')}.TA"
                    if tase_symbol not in found_symbols:
                        found_symbols.append(tase_symbol)
                    break
        
        return found_symbols
    
    async def get_news_for_symbol(
        self, 
        symbol: str, 
        days_back: int = 7
    ) -> list[NewsArticle]:
        """Get news articles mentioning a specific stock.
        
        Args:
            symbol: Stock symbol to search for
            days_back: Number of days to look back
            
        Returns:
            List of relevant articles.
        """
        all_news = await self.get_all_news(limit=50)
        
        symbol_clean = symbol.replace('.TA', '').lower()
        relevant = []
        
        for article in all_news:
            text = f"{article.title} {article.summary}".lower()
            
            # Check if symbol or company name is mentioned
            if symbol_clean in text:
                article.symbols.append(symbol)
                relevant.append(article)
                continue
            
            # Check aliases
            for company, aliases in self.COMPANY_MAPPINGS.items():
                if symbol_clean in company.lower():
                    for alias in aliases:
                        if alias.lower() in text:
                            article.symbols.append(symbol)
                            relevant.append(article)
                            break
                    break
        
        return relevant


# Synchronous wrapper for convenience
def get_news_sync(limit: int = 40) -> list[NewsArticle]:
    """Synchronous wrapper to get all news.
    
    Args:
        limit: Maximum articles to return.
        
    Returns:
        List of NewsArticle objects.
    """
    scraper = NewsScraper()
    try:
        return asyncio.run(scraper.get_all_news(limit))
    finally:
        asyncio.run(scraper.close())
