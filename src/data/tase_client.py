"""TASE DataWise API client for Israeli stock exchange data."""

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import logging

try:
    import tasepy
    TASEPY_AVAILABLE = True
except ImportError:
    TASEPY_AVAILABLE = False

from src.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class Security:
    """Represents a TASE security."""
    symbol: str
    name: str
    name_hebrew: str = ""
    isin: str = ""
    sector: str = ""
    sub_sector: str = ""
    market_cap: float = 0.0
    is_ta35: bool = False
    is_ta125: bool = False
    last_price: float = 0.0
    change_percent: float = 0.0
    volume: int = 0


@dataclass
class IndexData:
    """Represents TASE index data."""
    symbol: str
    name: str
    value: float
    change_percent: float
    timestamp: datetime = field(default_factory=datetime.now)
    components: list[str] = field(default_factory=list)


class TASEClient:
    """Client for fetching data from TASE DataWise API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize TASE client.
        
        Args:
            api_key: TASE DataWise API key. If not provided, uses settings.
        """
        self.settings = get_settings()
        self.api_key = api_key or self.settings.tase_api_key
        self._client = None
        self._securities_cache: dict[str, Security] = {}
        self._indices_cache: dict[str, IndexData] = {}
        self._cache_timestamp: Optional[datetime] = None
    
    @property
    def client(self):
        """Get or create the tasepy client."""
        if not TASEPY_AVAILABLE:
            raise ImportError(
                "tasepy is not installed. Install with: pip install tasepy"
            )
        
        if self._client is None:
            # Set API key in environment for tasepy
            if self.api_key:
                os.environ["TASE_API_KEY"] = self.api_key
            
            try:
                self._client = tasepy.quick_client()
            except Exception as e:
                logger.error(f"Failed to initialize TASE client: {e}")
                raise
        
        return self._client
    
    def _is_cache_valid(self) -> bool:
        """Check if cache is still valid."""
        if self._cache_timestamp is None:
            return False
        
        elapsed = (datetime.now() - self._cache_timestamp).total_seconds()
        return elapsed < self.settings.cache_ttl_seconds
    
    def get_securities_list(self, force_refresh: bool = False) -> list[Security]:
        """Get list of all TASE securities.
        
        Args:
            force_refresh: Force refresh from API instead of cache.
            
        Returns:
            List of Security objects.
        """
        if not force_refresh and self._is_cache_valid() and self._securities_cache:
            return list(self._securities_cache.values())
        
        try:
            # Fetch securities from TASE API
            securities_data = self.client.securities_basic.get_list()
            
            securities = []
            for sec in securities_data:
                security = Security(
                    symbol=getattr(sec, 'symbol', '') or getattr(sec, 'securityId', ''),
                    name=getattr(sec, 'securityName', '') or getattr(sec, 'name', ''),
                    name_hebrew=getattr(sec, 'securityNameHeb', ''),
                    isin=getattr(sec, 'isin', ''),
                    sector=getattr(sec, 'sector', ''),
                    sub_sector=getattr(sec, 'subSector', ''),
                )
                securities.append(security)
                self._securities_cache[security.symbol] = security
            
            self._cache_timestamp = datetime.now()
            logger.info(f"Fetched {len(securities)} securities from TASE")
            return securities
            
        except Exception as e:
            logger.error(f"Error fetching securities list: {e}")
            # Return cached data if available
            if self._securities_cache:
                logger.warning("Returning cached securities data")
                return list(self._securities_cache.values())
            raise
    
    def get_security(self, symbol: str) -> Optional[Security]:
        """Get a specific security by symbol.
        
        Args:
            symbol: Security symbol (e.g., 'TEVA')
            
        Returns:
            Security object or None if not found.
        """
        # Ensure cache is populated
        if not self._securities_cache:
            self.get_securities_list()
        
        return self._securities_cache.get(symbol)
    
    def get_indices(self, force_refresh: bool = False) -> list[IndexData]:
        """Get TASE indices data.
        
        Args:
            force_refresh: Force refresh from API instead of cache.
            
        Returns:
            List of IndexData objects.
        """
        if not force_refresh and self._is_cache_valid() and self._indices_cache:
            return list(self._indices_cache.values())
        
        try:
            # Fetch index listings
            indices_list = self.client.indices_basic.get_list()
            
            indices = []
            for idx in indices_list:
                index_data = IndexData(
                    symbol=getattr(idx, 'indexId', '') or getattr(idx, 'symbol', ''),
                    name=getattr(idx, 'indexName', '') or getattr(idx, 'name', ''),
                    value=0.0,  # Will be updated from online data
                    change_percent=0.0,
                )
                indices.append(index_data)
                self._indices_cache[index_data.symbol] = index_data
            
            # Try to get real-time rates
            try:
                online_data = self.client.indices_online.get_rates()
                for rate in online_data:
                    idx_symbol = getattr(rate, 'indexId', '') or getattr(rate, 'symbol', '')
                    if idx_symbol in self._indices_cache:
                        self._indices_cache[idx_symbol].value = getattr(rate, 'rate', 0.0)
                        self._indices_cache[idx_symbol].change_percent = getattr(rate, 'changePercent', 0.0)
                        self._indices_cache[idx_symbol].timestamp = datetime.now()
            except Exception as e:
                logger.warning(f"Could not fetch real-time index rates: {e}")
            
            logger.info(f"Fetched {len(indices)} indices from TASE")
            return list(self._indices_cache.values())
            
        except Exception as e:
            logger.error(f"Error fetching indices: {e}")
            if self._indices_cache:
                return list(self._indices_cache.values())
            raise
    
    def get_index_components(self, index_symbol: str) -> list[str]:
        """Get components of a specific index.
        
        Args:
            index_symbol: Index symbol (e.g., 'TA35')
            
        Returns:
            List of security symbols in the index.
        """
        try:
            components = self.client.indices_basic.get_components(index_symbol)
            return [getattr(c, 'symbol', '') or getattr(c, 'securityId', '') 
                    for c in components]
        except Exception as e:
            logger.error(f"Error fetching index components for {index_symbol}: {e}")
            return []
    
    def get_ta35_symbols(self) -> list[str]:
        """Get list of TA-35 index component symbols."""
        return self.get_index_components("TA35")
    
    def get_ta125_symbols(self) -> list[str]:
        """Get list of TA-125 index component symbols."""
        return self.get_index_components("TA125")
    
    def get_funds(self) -> list[dict]:
        """Get list of funds from TASE.
        
        Returns:
            List of fund data dictionaries.
        """
        try:
            funds = self.client.funds.get_list()
            return [
                {
                    'fund_id': getattr(f, 'fundId', ''),
                    'name': getattr(f, 'fundName', ''),
                    'classification': getattr(f, 'classification', ''),
                }
                for f in funds
            ]
        except Exception as e:
            logger.error(f"Error fetching funds: {e}")
            return []
    
    def is_market_open(self) -> bool:
        """Check if TASE market is currently open.
        
        TASE hours: Sunday-Thursday, 09:00-17:30 IST
        """
        now = datetime.now()
        
        # TASE is closed on Friday (4) and Saturday (5)
        if now.weekday() in (4, 5):
            return False
        
        # Check time (approximate IST check - adjust for timezone)
        market_open = now.replace(
            hour=self.settings.market_open_hour, 
            minute=0, 
            second=0
        )
        market_close = now.replace(
            hour=self.settings.market_close_hour,
            minute=self.settings.market_close_minute,
            second=0
        )
        
        return market_open <= now <= market_close
