"""Trading strategies for swing trades (1 week - 1 month)."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional
import logging

from src.config import get_settings
from src.rules.scoring import StockAnalysis, Recommendation

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk tolerance levels."""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


@dataclass
class PortfolioAllocation:
    """Recommended portfolio allocation."""
    symbol: str
    name: str
    allocation_percent: float
    shares: int
    investment_amount: float
    entry_price: float
    target_price: Optional[float]
    stop_loss: Optional[float]
    recommendation: Recommendation
    score: float
    holding_period: str
    risk_level: str


@dataclass
class PortfolioRecommendation:
    """Complete portfolio recommendation."""
    total_budget: float
    risk_level: RiskLevel
    allocations: list[PortfolioAllocation] = field(default_factory=list)
    cash_reserve: float = 0.0
    cash_reserve_percent: float = 0.0
    diversification_score: float = 0.0
    expected_return: float = 0.0
    max_drawdown: float = 0.0
    generated_at: datetime = field(default_factory=datetime.now)


class SwingTradeStrategy:
    """Strategy optimizer for swing trades (1-4 weeks holding period)."""
    
    # Maximum allocation to single stock based on risk level
    MAX_ALLOCATION = {
        RiskLevel.CONSERVATIVE: 0.15,  # 15%
        RiskLevel.MODERATE: 0.20,      # 20%
        RiskLevel.AGGRESSIVE: 0.30,    # 30%
    }
    
    # Minimum cash reserve based on risk level
    CASH_RESERVE = {
        RiskLevel.CONSERVATIVE: 0.20,  # 20%
        RiskLevel.MODERATE: 0.10,      # 10%
        RiskLevel.AGGRESSIVE: 0.05,    # 5%
    }
    
    # Minimum score thresholds based on risk level
    MIN_SCORE = {
        RiskLevel.CONSERVATIVE: 70,
        RiskLevel.MODERATE: 60,
        RiskLevel.AGGRESSIVE: 50,
    }
    
    # Maximum positions based on risk level
    MAX_POSITIONS = {
        RiskLevel.CONSERVATIVE: 8,
        RiskLevel.MODERATE: 6,
        RiskLevel.AGGRESSIVE: 4,
    }
    
    def __init__(self):
        """Initialize swing trade strategy."""
        self.settings = get_settings()
    
    def calculate_position_size(
        self,
        budget: float,
        price: float,
        risk_level: RiskLevel,
        score: float,
    ) -> tuple[float, int]:
        """Calculate position size based on score and risk level.
        
        Args:
            budget: Available budget for position.
            price: Stock price.
            risk_level: Risk tolerance level.
            score: Stock analysis score.
            
        Returns:
            Tuple of (investment_amount, number_of_shares).
        """
        max_alloc = self.MAX_ALLOCATION[risk_level]
        
        # Scale allocation based on score (higher score = higher allocation)
        # Score 100 = max allocation, Score 50 = half max allocation
        score_factor = (score - 50) / 50  # 0 to 1 for scores 50-100
        score_factor = max(0.3, min(1.0, score_factor))  # Clamp to 0.3-1.0
        
        allocation = budget * max_alloc * score_factor
        shares = int(allocation / price) if price > 0 else 0
        actual_investment = shares * price
        
        return actual_investment, shares
    
    def build_portfolio(
        self,
        analyses: list[StockAnalysis],
        budget: float,
        risk_level: RiskLevel = RiskLevel.MODERATE,
    ) -> PortfolioRecommendation:
        """Build diversified portfolio from stock analyses.
        
        Args:
            analyses: List of stock analyses to consider.
            budget: Total investment budget.
            risk_level: Risk tolerance level.
            
        Returns:
            PortfolioRecommendation object.
        """
        recommendation = PortfolioRecommendation(
            total_budget=budget,
            risk_level=risk_level,
        )
        
        # Filter by minimum score
        min_score = self.MIN_SCORE[risk_level]
        qualified = [a for a in analyses if a.total_score >= min_score]
        
        # Filter by recommendation (only BUY or STRONG_BUY)
        buyable = [
            a for a in qualified 
            if a.recommendation in (Recommendation.BUY, Recommendation.STRONG_BUY)
        ]
        
        if not buyable:
            # No good opportunities - keep as cash
            recommendation.cash_reserve = budget
            recommendation.cash_reserve_percent = 100
            return recommendation
        
        # Sort by score
        buyable.sort(key=lambda x: x.total_score, reverse=True)
        
        # Limit positions
        max_positions = self.MAX_POSITIONS[risk_level]
        selected = buyable[:max_positions]
        
        # Calculate cash reserve
        cash_reserve_pct = self.CASH_RESERVE[risk_level]
        cash_reserve = budget * cash_reserve_pct
        investable_budget = budget - cash_reserve
        
        # Calculate allocations
        total_invested = 0
        allocations = []
        
        # Distribute budget proportionally to scores
        total_score = sum(a.total_score for a in selected)
        
        for analysis in selected:
            # Score-weighted allocation
            weight = analysis.total_score / total_score if total_score > 0 else 1/len(selected)
            position_budget = investable_budget * weight
            
            # Apply position size limits
            investment, shares = self.calculate_position_size(
                position_budget,
                analysis.current_price,
                risk_level,
                analysis.total_score,
            )
            
            if shares > 0:
                allocation = PortfolioAllocation(
                    symbol=analysis.symbol,
                    name=analysis.name,
                    allocation_percent=round(investment / budget * 100, 1),
                    shares=shares,
                    investment_amount=investment,
                    entry_price=analysis.current_price,
                    target_price=analysis.target_price,
                    stop_loss=analysis.stop_loss,
                    recommendation=analysis.recommendation,
                    score=analysis.total_score,
                    holding_period=analysis.holding_period,
                    risk_level=risk_level.value,
                )
                allocations.append(allocation)
                total_invested += investment
        
        recommendation.allocations = allocations
        recommendation.cash_reserve = budget - total_invested
        recommendation.cash_reserve_percent = round(
            recommendation.cash_reserve / budget * 100, 1
        )
        
        # Calculate diversification score
        recommendation.diversification_score = self._calculate_diversification(allocations)
        
        # Estimate expected return and drawdown
        recommendation.expected_return = self._estimate_return(allocations)
        recommendation.max_drawdown = self._estimate_drawdown(allocations, risk_level)
        
        return recommendation
    
    def _calculate_diversification(self, allocations: list[PortfolioAllocation]) -> float:
        """Calculate portfolio diversification score (0-100).
        
        Based on number of positions and allocation balance.
        """
        if not allocations:
            return 0
        
        n = len(allocations)
        
        # Position count factor (more positions = more diversified)
        position_factor = min(1.0, n / 5)
        
        # Concentration factor (equal weights = more diversified)
        weights = [a.allocation_percent / 100 for a in allocations]
        hhi = sum(w ** 2 for w in weights)  # Herfindahl index
        # HHI of 1.0 = single position, 1/n = equal weights
        min_hhi = 1 / n if n > 0 else 1
        concentration_factor = 1 - ((hhi - min_hhi) / (1 - min_hhi)) if hhi < 1 else 0
        
        return (position_factor * 0.4 + concentration_factor * 0.6) * 100
    
    def _estimate_return(self, allocations: list[PortfolioAllocation]) -> float:
        """Estimate expected portfolio return based on targets."""
        if not allocations:
            return 0
        
        weighted_return = 0
        total_weight = 0
        
        for alloc in allocations:
            if alloc.target_price and alloc.entry_price > 0:
                expected_return = (alloc.target_price - alloc.entry_price) / alloc.entry_price
                weight = alloc.allocation_percent
                weighted_return += expected_return * weight
                total_weight += weight
        
        return weighted_return / total_weight * 100 if total_weight > 0 else 0
    
    def _estimate_drawdown(
        self, 
        allocations: list[PortfolioAllocation],
        risk_level: RiskLevel,
    ) -> float:
        """Estimate maximum portfolio drawdown."""
        if not allocations:
            return 0
        
        # Base drawdown on risk level and stop losses
        base_drawdown = {
            RiskLevel.CONSERVATIVE: 5,
            RiskLevel.MODERATE: 10,
            RiskLevel.AGGRESSIVE: 15,
        }[risk_level]
        
        # Adjust for stop losses
        weighted_stop = 0
        total_weight = 0
        
        for alloc in allocations:
            if alloc.stop_loss and alloc.entry_price > 0:
                stop_distance = (alloc.entry_price - alloc.stop_loss) / alloc.entry_price
                weight = alloc.allocation_percent
                weighted_stop += stop_distance * weight
                total_weight += weight
        
        if total_weight > 0:
            return weighted_stop / total_weight * 100
        
        return base_drawdown
    
    def get_holding_period_recommendation(
        self,
        analysis: StockAnalysis,
    ) -> tuple[int, int]:
        """Get recommended holding period in days.
        
        Args:
            analysis: Stock analysis result.
            
        Returns:
            Tuple of (min_days, max_days).
        """
        # Base holding period: 7-30 days for swing trades
        min_days = self.settings.min_holding_days
        max_days = self.settings.max_holding_days
        
        # Adjust based on technical indicators
        if analysis.technical_details and analysis.technical_details.indicators:
            indicators = analysis.technical_details.indicators
            
            # Strong trend = shorter holding (capture momentum)
            if indicators.trend_strength > 70:
                max_days = min(21, max_days)
            
            # Weak trend = longer holding (wait for development)
            elif indicators.trend_strength < 30:
                min_days = max(14, min_days)
        
        return min_days, max_days
    
    def should_exit(
        self,
        analysis: StockAnalysis,
        entry_price: float,
        current_price: float,
        days_held: int,
    ) -> tuple[bool, str]:
        """Determine if position should be exited.
        
        Args:
            analysis: Current stock analysis.
            entry_price: Original entry price.
            current_price: Current market price.
            days_held: Number of days position has been held.
            
        Returns:
            Tuple of (should_exit, reason).
        """
        # Check stop loss
        if analysis.stop_loss and current_price <= analysis.stop_loss:
            return True, "Stop loss triggered"
        
        # Check target reached
        if analysis.target_price and current_price >= analysis.target_price:
            return True, "Target price reached"
        
        # Check max holding period
        if days_held >= self.settings.max_holding_days:
            return True, "Maximum holding period reached"
        
        # Check for recommendation change
        if analysis.recommendation in (Recommendation.SELL, Recommendation.STRONG_SELL):
            return True, f"Recommendation changed to {analysis.recommendation.value}"
        
        # Check for significant score drop
        if analysis.total_score < 40:
            return True, "Score dropped below threshold"
        
        return False, ""
