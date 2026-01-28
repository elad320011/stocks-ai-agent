"""Configuration management for TASE Stock AI Agent."""

import os
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )
    
    # API Keys
    gemini_api_key: str = Field(default="", description="Google Gemini API key")
    tase_api_key: str = Field(default="", description="TASE DataWise API key")
    
    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    
    # Cache settings
    cache_ttl_seconds: int = Field(default=3600, description="Cache TTL in seconds")
    cache_dir: Path = Field(default=Path(".cache"), description="Cache directory")
    
    # Analysis settings
    technical_weight: float = Field(default=0.40, description="Technical analysis weight")
    fundamental_weight: float = Field(default=0.35, description="Fundamental analysis weight")
    sentiment_weight: float = Field(default=0.25, description="Sentiment analysis weight")
    
    # Trading parameters
    min_holding_days: int = Field(default=7, description="Minimum holding period (days)")
    max_holding_days: int = Field(default=30, description="Maximum holding period (days)")
    
    # Rate limiting
    gemini_rpm: int = Field(default=10, description="Gemini requests per minute")
    tase_rpm: int = Field(default=30, description="TASE API requests per minute")
    
    # Market settings
    market_open_hour: int = Field(default=9, description="TASE market open hour (IST)")
    market_close_hour: int = Field(default=17, description="TASE market close hour (IST)")
    market_close_minute: int = Field(default=30, description="TASE market close minute")
    
    @property
    def is_configured(self) -> bool:
        """Check if required API keys are configured."""
        return bool(self.gemini_api_key and self.tase_api_key)
    
    def validate_keys(self) -> list[str]:
        """Validate API keys and return list of missing keys."""
        missing = []
        if not self.gemini_api_key:
            missing.append("GEMINI_API_KEY")
        if not self.tase_api_key:
            missing.append("TASE_API_KEY")
        return missing


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings
