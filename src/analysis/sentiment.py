"""Sentiment analysis module for news-based stock sentiment."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import logging
import re

from src.config import get_settings
from src.data.news_scraper import NewsScraper, NewsArticle

logger = logging.getLogger(__name__)


@dataclass
class SentimentMetrics:
    """Container for sentiment analysis metrics."""
    # Overall sentiment
    overall_score: float = 0.0  # -1 (very negative) to +1 (very positive)
    sentiment_label: str = "neutral"  # very_negative, negative, neutral, positive, very_positive
    
    # Article counts
    total_articles: int = 0
    positive_articles: int = 0
    negative_articles: int = 0
    neutral_articles: int = 0
    
    # Confidence
    confidence: float = 0.0  # 0-1 based on article count and consistency
    
    # Recent trend
    trend: str = "stable"  # improving, stable, declining
    
    # Key topics
    key_topics: list[str] = field(default_factory=list)
    
    # Recent headlines
    recent_headlines: list[str] = field(default_factory=list)


@dataclass
class SentimentScore:
    """Sentiment analysis score breakdown."""
    total_score: float = 0.0  # 0-100
    news_score: float = 0.0
    volume_score: float = 0.0  # News volume/coverage
    consistency_score: float = 0.0
    signals: list[str] = field(default_factory=list)
    metrics: Optional[SentimentMetrics] = None


class SentimentAnalyzer:
    """Sentiment analysis engine for Israeli financial news."""
    
    # Positive sentiment keywords (English and Hebrew transliterated)
    POSITIVE_KEYWORDS = [
        # English
        'surge', 'soar', 'jump', 'gain', 'rise', 'rally', 'boom', 'growth',
        'profit', 'earnings beat', 'upgrade', 'outperform', 'strong', 'record',
        'breakthrough', 'expand', 'acquisition', 'deal', 'partnership', 'contract',
        'bullish', 'momentum', 'recovery', 'optimistic', 'success', 'exceed',
        'dividend', 'buyback', 'innovation', 'launch', 'approve', 'win',
        # Hebrew transliterated
        'aliya', 'tzmicha', 'revach', 'hazlacha', 
    ]
    
    # Negative sentiment keywords
    NEGATIVE_KEYWORDS = [
        # English
        'drop', 'fall', 'decline', 'plunge', 'crash', 'loss', 'miss', 'weak',
        'downgrade', 'underperform', 'cut', 'layoff', 'restructure', 'warning',
        'bearish', 'concern', 'risk', 'threat', 'lawsuit', 'investigation',
        'default', 'debt', 'bankruptcy', 'fraud', 'scandal', 'recall', 'fail',
        'slump', 'tumble', 'sink', 'struggle', 'disappointing', 'below',
        # Hebrew transliterated
        'yerida', 'hefsed', 'mashber', 'sikun',
    ]
    
    # Neutral/informational keywords (to identify news vs noise)
    NEUTRAL_KEYWORDS = [
        'report', 'announce', 'release', 'update', 'meeting', 'conference',
        'schedule', 'plan', 'expect', 'forecast', 'analyst', 'market',
    ]
    
    def __init__(self, news_scraper: Optional[NewsScraper] = None):
        """Initialize sentiment analyzer.
        
        Args:
            news_scraper: Optional NewsScraper instance.
        """
        self.settings = get_settings()
        self.scraper = news_scraper or NewsScraper()
    
    def _analyze_text_sentiment(self, text: str) -> tuple[float, list[str]]:
        """Analyze sentiment of text using keyword matching.
        
        Args:
            text: Text to analyze.
            
        Returns:
            Tuple of (sentiment score -1 to 1, list of matched keywords).
        """
        text_lower = text.lower()
        
        positive_matches = []
        negative_matches = []
        
        for keyword in self.POSITIVE_KEYWORDS:
            if keyword in text_lower:
                positive_matches.append(keyword)
        
        for keyword in self.NEGATIVE_KEYWORDS:
            if keyword in text_lower:
                negative_matches.append(keyword)
        
        total_matches = len(positive_matches) + len(negative_matches)
        
        if total_matches == 0:
            return 0.0, []
        
        # Calculate sentiment score
        score = (len(positive_matches) - len(negative_matches)) / total_matches
        
        # Normalize to -1 to 1 range with dampening
        score = max(-1.0, min(1.0, score))
        
        matched = positive_matches + [f"-{kw}" for kw in negative_matches]
        return score, matched
    
    def _score_to_label(self, score: float) -> str:
        """Convert sentiment score to label.
        
        Args:
            score: Sentiment score (-1 to 1).
            
        Returns:
            Sentiment label string.
        """
        if score >= 0.5:
            return "very_positive"
        elif score >= 0.2:
            return "positive"
        elif score <= -0.5:
            return "very_negative"
        elif score <= -0.2:
            return "negative"
        else:
            return "neutral"
    
    def analyze_article(self, article: NewsArticle) -> float:
        """Analyze sentiment of a single article.
        
        Args:
            article: NewsArticle to analyze.
            
        Returns:
            Sentiment score (-1 to 1).
        """
        # Combine title and summary for analysis
        text = f"{article.title} {article.summary}"
        
        score, keywords = self._analyze_text_sentiment(text)
        article.sentiment_score = score
        
        return score
    
    async def analyze_symbol(
        self, 
        symbol: str, 
        days_back: int = 7
    ) -> SentimentMetrics:
        """Analyze sentiment for a specific stock symbol.
        
        Args:
            symbol: Stock symbol to analyze.
            days_back: Number of days to look back.
            
        Returns:
            SentimentMetrics object.
        """
        metrics = SentimentMetrics()
        
        try:
            # Get news for symbol
            articles = await self.scraper.get_news_for_symbol(symbol, days_back)
            
            if not articles:
                # No specific news - analyze general market news
                all_news = await self.scraper.get_all_news(limit=20)
                articles = all_news[:5]  # Use top 5 general articles
            
            metrics.total_articles = len(articles)
            
            if not articles:
                return metrics
            
            # Analyze each article
            scores = []
            for article in articles:
                score = self.analyze_article(article)
                scores.append(score)
                
                if score > 0.1:
                    metrics.positive_articles += 1
                elif score < -0.1:
                    metrics.negative_articles += 1
                else:
                    metrics.neutral_articles += 1
            
            # Calculate overall score
            if scores:
                metrics.overall_score = sum(scores) / len(scores)
                metrics.sentiment_label = self._score_to_label(metrics.overall_score)
            
            # Calculate confidence based on article count and consistency
            if len(scores) >= 5:
                # More articles = higher confidence
                volume_confidence = min(1.0, len(scores) / 10)
                
                # Consistent sentiment = higher confidence
                if scores:
                    score_variance = sum((s - metrics.overall_score) ** 2 for s in scores) / len(scores)
                    consistency_confidence = max(0, 1 - score_variance)
                else:
                    consistency_confidence = 0.5
                
                metrics.confidence = (volume_confidence + consistency_confidence) / 2
            else:
                metrics.confidence = 0.3  # Low confidence with few articles
            
            # Store recent headlines
            metrics.recent_headlines = [a.title for a in articles[:5]]
            
            # Extract key topics (simplified - just use frequent words)
            all_text = " ".join(a.title for a in articles).lower()
            words = re.findall(r'\b\w{4,}\b', all_text)
            word_freq = {}
            for word in words:
                if word not in ['that', 'this', 'with', 'from', 'have', 'been']:
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            metrics.key_topics = sorted(
                word_freq.keys(), 
                key=lambda x: word_freq[x], 
                reverse=True
            )[:5]
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment for {symbol}: {e}")
        
        return metrics
    
    async def analyze_market_sentiment(self) -> SentimentMetrics:
        """Analyze overall market sentiment.
        
        Returns:
            SentimentMetrics for general market.
        """
        metrics = SentimentMetrics()
        
        try:
            articles = await self.scraper.get_all_news(limit=30)
            metrics.total_articles = len(articles)
            
            if not articles:
                return metrics
            
            scores = []
            for article in articles:
                score = self.analyze_article(article)
                scores.append(score)
                
                if score > 0.1:
                    metrics.positive_articles += 1
                elif score < -0.1:
                    metrics.negative_articles += 1
                else:
                    metrics.neutral_articles += 1
            
            if scores:
                metrics.overall_score = sum(scores) / len(scores)
                metrics.sentiment_label = self._score_to_label(metrics.overall_score)
                
                # Analyze trend (first half vs second half)
                mid = len(scores) // 2
                if mid > 0:
                    first_half = sum(scores[:mid]) / mid
                    second_half = sum(scores[mid:]) / (len(scores) - mid)
                    
                    if second_half > first_half + 0.1:
                        metrics.trend = "improving"
                    elif second_half < first_half - 0.1:
                        metrics.trend = "declining"
                    else:
                        metrics.trend = "stable"
            
            metrics.recent_headlines = [a.title for a in articles[:5]]
            
        except Exception as e:
            logger.error(f"Error analyzing market sentiment: {e}")
        
        return metrics
    
    def calculate_score(self, metrics: SentimentMetrics) -> SentimentScore:
        """Calculate sentiment score from metrics.
        
        Args:
            metrics: SentimentMetrics to score.
            
        Returns:
            SentimentScore object with breakdown.
        """
        score = SentimentScore(metrics=metrics)
        signals = []
        
        # News sentiment score (max 15 points)
        # Convert -1 to 1 range to 0 to 15
        normalized = (metrics.overall_score + 1) / 2  # 0 to 1
        score.news_score = normalized * 15
        
        if metrics.sentiment_label == "very_positive":
            signals.append("Very positive news sentiment")
        elif metrics.sentiment_label == "positive":
            signals.append("Positive news sentiment")
        elif metrics.sentiment_label == "negative":
            signals.append("Negative news sentiment - caution")
        elif metrics.sentiment_label == "very_negative":
            signals.append("Very negative news sentiment - high risk")
        
        # News volume score (max 5 points)
        if metrics.total_articles >= 10:
            score.volume_score = 5
            signals.append("High news coverage")
        elif metrics.total_articles >= 5:
            score.volume_score = 3
        elif metrics.total_articles >= 1:
            score.volume_score = 2
        else:
            score.volume_score = 1
            signals.append("Low news coverage")
        
        # Consistency score (max 5 points)
        if metrics.confidence >= 0.7:
            score.consistency_score = 5
            signals.append("Consistent sentiment signals")
        elif metrics.confidence >= 0.4:
            score.consistency_score = 3
        else:
            score.consistency_score = 1
            signals.append("Mixed sentiment signals")
        
        # Trend bonus
        if metrics.trend == "improving":
            score.news_score = min(15, score.news_score + 2)
            signals.append("Sentiment trend improving")
        elif metrics.trend == "declining":
            score.news_score = max(0, score.news_score - 2)
            signals.append("Sentiment trend declining")
        
        # Calculate total (scaled for 25% weight as per plan)
        raw_total = score.news_score + score.volume_score + score.consistency_score
        max_raw = 25  # 15 + 5 + 5
        score.total_score = min(100, raw_total / max_raw * 100)
        score.signals = signals
        
        return score
    
    async def calculate_score_async(self, symbol: str) -> SentimentScore:
        """Calculate sentiment score for a symbol.
        
        Args:
            symbol: Stock symbol.
            
        Returns:
            SentimentScore object.
        """
        metrics = await self.analyze_symbol(symbol)
        return self.calculate_score(metrics)


# Synchronous wrapper
def analyze_sentiment_sync(symbol: str) -> SentimentScore:
    """Synchronous wrapper for sentiment analysis.
    
    Args:
        symbol: Stock symbol.
        
    Returns:
        SentimentScore object.
    """
    analyzer = SentimentAnalyzer()
    
    async def _run():
        try:
            return await analyzer.calculate_score_async(symbol)
        finally:
            await analyzer.scraper.close()
    
    return asyncio.run(_run())
