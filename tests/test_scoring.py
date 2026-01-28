"""Tests for the scoring engine."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from src.rules.scoring import (
    ScoringEngine,
    StockAnalysis,
    Recommendation,
)
from src.rules.strategies import (
    SwingTradeStrategy,
    RiskLevel,
    PortfolioAllocation,
    PortfolioRecommendation,
)


@pytest.fixture
def mock_analysis():
    """Create a mock stock analysis."""
    return StockAnalysis(
        symbol="TEST.TA",
        name="Test Company",
        technical_score=70.0,
        fundamental_score=65.0,
        sentiment_score=60.0,
        total_score=66.0,
        recommendation=Recommendation.BUY,
        confidence=75.0,
        current_price=100.0,
        target_price=108.0,
        stop_loss=95.0,
        bullish_signals=["RSI oversold", "MACD bullish crossover"],
        bearish_signals=["Near 52-week high"],
    )


@pytest.fixture
def multiple_analyses():
    """Create multiple mock analyses for portfolio testing."""
    return [
        StockAnalysis(
            symbol="TOP.TA",
            name="Top Pick Ltd",
            technical_score=85.0,
            fundamental_score=80.0,
            sentiment_score=75.0,
            total_score=81.0,
            recommendation=Recommendation.STRONG_BUY,
            confidence=90.0,
            current_price=50.0,
            target_price=58.0,
            stop_loss=47.0,
        ),
        StockAnalysis(
            symbol="GOOD.TA",
            name="Good Stock Ltd",
            technical_score=70.0,
            fundamental_score=68.0,
            sentiment_score=65.0,
            total_score=68.0,
            recommendation=Recommendation.BUY,
            confidence=75.0,
            current_price=120.0,
            target_price=130.0,
            stop_loss=114.0,
        ),
        StockAnalysis(
            symbol="OK.TA",
            name="Okay Stock Ltd",
            technical_score=55.0,
            fundamental_score=60.0,
            sentiment_score=50.0,
            total_score=55.0,
            recommendation=Recommendation.HOLD,
            confidence=60.0,
            current_price=80.0,
            target_price=None,
            stop_loss=None,
        ),
        StockAnalysis(
            symbol="BAD.TA",
            name="Bad Stock Ltd",
            technical_score=30.0,
            fundamental_score=35.0,
            sentiment_score=25.0,
            total_score=30.0,
            recommendation=Recommendation.SELL,
            confidence=70.0,
            current_price=200.0,
            target_price=180.0,
            stop_loss=210.0,
        ),
    ]


class TestRecommendation:
    """Tests for Recommendation enum."""
    
    def test_recommendation_values(self):
        """Test recommendation enum values."""
        assert Recommendation.STRONG_BUY.value == "STRONG BUY"
        assert Recommendation.BUY.value == "BUY"
        assert Recommendation.HOLD.value == "HOLD"
        assert Recommendation.SELL.value == "SELL"
        assert Recommendation.STRONG_SELL.value == "STRONG SELL"


class TestStockAnalysis:
    """Tests for StockAnalysis dataclass."""
    
    def test_default_values(self):
        """Test default analysis values."""
        analysis = StockAnalysis(symbol="TEST.TA")
        
        assert analysis.symbol == "TEST.TA"
        assert analysis.name == ""
        assert analysis.total_score == 0.0
        assert analysis.recommendation == Recommendation.HOLD
        assert analysis.bullish_signals == []
    
    def test_to_dict(self, mock_analysis):
        """Test dictionary conversion."""
        result = mock_analysis.to_dict()
        
        assert result['symbol'] == "TEST.TA"
        assert result['total_score'] == 66.0
        assert result['recommendation'] == "BUY"
        assert isinstance(result['timestamp'], str)
    
    def test_holding_period(self, mock_analysis):
        """Test holding period default."""
        assert mock_analysis.holding_period == "1-4 weeks"


class TestScoringEngine:
    """Tests for ScoringEngine class."""
    
    def test_init(self):
        """Test engine initialization."""
        engine = ScoringEngine()
        assert engine is not None
        assert engine.technical is not None
        assert engine.fundamental is not None
        assert engine.sentiment is not None
    
    def test_calculate_weighted_score(self):
        """Test weighted score calculation."""
        engine = ScoringEngine()
        
        # With default weights (40%, 35%, 25%)
        score = engine._calculate_weighted_score(100, 100, 100)
        assert score == 100.0
        
        score = engine._calculate_weighted_score(0, 0, 0)
        assert score == 0.0
        
        # Partial scores
        score = engine._calculate_weighted_score(80, 70, 60)
        expected = 80 * 0.4 + 70 * 0.35 + 60 * 0.25
        assert abs(score - expected) < 0.1
    
    def test_score_to_recommendation(self):
        """Test score to recommendation conversion."""
        engine = ScoringEngine()
        
        assert engine._score_to_recommendation(90) == Recommendation.STRONG_BUY
        assert engine._score_to_recommendation(80) == Recommendation.STRONG_BUY
        assert engine._score_to_recommendation(70) == Recommendation.BUY
        assert engine._score_to_recommendation(50) == Recommendation.HOLD
        assert engine._score_to_recommendation(30) == Recommendation.SELL
        assert engine._score_to_recommendation(10) == Recommendation.STRONG_SELL


class TestSwingTradeStrategy:
    """Tests for SwingTradeStrategy class."""
    
    def test_init(self):
        """Test strategy initialization."""
        strategy = SwingTradeStrategy()
        assert strategy is not None
    
    def test_calculate_position_size(self):
        """Test position size calculation."""
        strategy = SwingTradeStrategy()
        
        # High score should get larger allocation
        amount1, shares1 = strategy.calculate_position_size(
            10000, 100.0, RiskLevel.MODERATE, 80
        )
        
        amount2, shares2 = strategy.calculate_position_size(
            10000, 100.0, RiskLevel.MODERATE, 60
        )
        
        assert amount1 > amount2
        assert shares1 > shares2
    
    def test_position_size_respects_max_allocation(self):
        """Test that position size respects max allocation limits."""
        strategy = SwingTradeStrategy()
        
        amount, shares = strategy.calculate_position_size(
            100000, 10.0, RiskLevel.MODERATE, 100
        )
        
        max_allowed = 100000 * 0.20  # 20% for moderate
        assert amount <= max_allowed
    
    def test_build_portfolio_filters_by_score(self, multiple_analyses):
        """Test that portfolio building filters by minimum score."""
        strategy = SwingTradeStrategy()
        
        # Conservative should have higher threshold
        portfolio = strategy.build_portfolio(
            multiple_analyses, 50000, RiskLevel.CONSERVATIVE
        )
        
        # Should only include stocks above 70 score threshold
        for alloc in portfolio.allocations:
            analysis = next(a for a in multiple_analyses if a.symbol == alloc.symbol)
            assert analysis.total_score >= 70
    
    def test_build_portfolio_filters_by_recommendation(self, multiple_analyses):
        """Test that portfolio only includes BUY recommendations."""
        strategy = SwingTradeStrategy()
        
        portfolio = strategy.build_portfolio(
            multiple_analyses, 50000, RiskLevel.MODERATE
        )
        
        for alloc in portfolio.allocations:
            assert alloc.recommendation in (Recommendation.BUY, Recommendation.STRONG_BUY)
    
    def test_build_portfolio_respects_cash_reserve(self, multiple_analyses):
        """Test cash reserve based on risk level."""
        strategy = SwingTradeStrategy()
        
        conservative = strategy.build_portfolio(
            multiple_analyses, 100000, RiskLevel.CONSERVATIVE
        )
        aggressive = strategy.build_portfolio(
            multiple_analyses, 100000, RiskLevel.AGGRESSIVE
        )
        
        assert conservative.cash_reserve_percent > aggressive.cash_reserve_percent
    
    def test_build_portfolio_limits_positions(self, multiple_analyses):
        """Test position count limits."""
        strategy = SwingTradeStrategy()
        
        # Create more analyses
        many_analyses = multiple_analyses * 3  # 12 analyses
        
        # Conservative should have at most 8 positions
        portfolio = strategy.build_portfolio(
            many_analyses, 100000, RiskLevel.CONSERVATIVE
        )
        
        assert len(portfolio.allocations) <= 8
    
    def test_build_portfolio_empty_when_no_buys(self):
        """Test empty portfolio when no buy signals."""
        strategy = SwingTradeStrategy()
        
        # All HOLD or SELL
        bad_analyses = [
            StockAnalysis(
                symbol="HOLD1.TA",
                total_score=45,
                recommendation=Recommendation.HOLD,
                current_price=100,
            ),
            StockAnalysis(
                symbol="SELL1.TA",
                total_score=25,
                recommendation=Recommendation.SELL,
                current_price=100,
            ),
        ]
        
        portfolio = strategy.build_portfolio(bad_analyses, 50000, RiskLevel.MODERATE)
        
        assert len(portfolio.allocations) == 0
        assert portfolio.cash_reserve == 50000
    
    def test_diversification_score(self, multiple_analyses):
        """Test diversification score calculation."""
        strategy = SwingTradeStrategy()
        
        portfolio = strategy.build_portfolio(
            multiple_analyses, 100000, RiskLevel.MODERATE
        )
        
        # Should have some diversification
        assert 0 <= portfolio.diversification_score <= 100
    
    def test_expected_return_calculation(self, multiple_analyses):
        """Test expected return calculation."""
        strategy = SwingTradeStrategy()
        
        portfolio = strategy.build_portfolio(
            multiple_analyses, 100000, RiskLevel.MODERATE
        )
        
        # Should calculate expected return
        assert portfolio.expected_return >= 0  # Can be 0 if no targets
    
    def test_should_exit_stop_loss(self, mock_analysis):
        """Test exit signal for stop loss."""
        strategy = SwingTradeStrategy()
        
        should_exit, reason = strategy.should_exit(
            mock_analysis,
            entry_price=100.0,
            current_price=94.0,  # Below stop loss of 95
            days_held=5,
        )
        
        assert should_exit is True
        assert "stop loss" in reason.lower()
    
    def test_should_exit_target_reached(self, mock_analysis):
        """Test exit signal for target reached."""
        strategy = SwingTradeStrategy()
        
        should_exit, reason = strategy.should_exit(
            mock_analysis,
            entry_price=100.0,
            current_price=110.0,  # Above target of 108
            days_held=10,
        )
        
        assert should_exit is True
        assert "target" in reason.lower()
    
    def test_should_exit_max_holding(self, mock_analysis):
        """Test exit signal for max holding period."""
        strategy = SwingTradeStrategy()
        
        should_exit, reason = strategy.should_exit(
            mock_analysis,
            entry_price=100.0,
            current_price=102.0,
            days_held=35,  # Beyond max holding
        )
        
        assert should_exit is True
        assert "holding period" in reason.lower()
    
    def test_should_not_exit_normal_conditions(self, mock_analysis):
        """Test no exit under normal conditions."""
        strategy = SwingTradeStrategy()
        
        should_exit, reason = strategy.should_exit(
            mock_analysis,
            entry_price=100.0,
            current_price=103.0,  # Between entry and target
            days_held=10,
        )
        
        assert should_exit is False
        assert reason == ""


class TestPortfolioAllocation:
    """Tests for PortfolioAllocation dataclass."""
    
    def test_allocation_fields(self):
        """Test allocation field values."""
        alloc = PortfolioAllocation(
            symbol="TEST.TA",
            name="Test Stock",
            allocation_percent=10.0,
            shares=50,
            investment_amount=5000.0,
            entry_price=100.0,
            target_price=110.0,
            stop_loss=95.0,
            recommendation=Recommendation.BUY,
            score=70.0,
            holding_period="1-4 weeks",
            risk_level="moderate",
        )
        
        assert alloc.symbol == "TEST.TA"
        assert alloc.shares == 50
        assert alloc.allocation_percent == 10.0


class TestPortfolioRecommendation:
    """Tests for PortfolioRecommendation dataclass."""
    
    def test_default_values(self):
        """Test default portfolio values."""
        portfolio = PortfolioRecommendation(
            total_budget=100000,
            risk_level=RiskLevel.MODERATE,
        )
        
        assert portfolio.total_budget == 100000
        assert portfolio.allocations == []
        assert portfolio.cash_reserve == 0.0
