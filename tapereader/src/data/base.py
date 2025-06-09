"""
Interface base para provedores de dados
Define o contrato que todos os providers devem seguir
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, List
from datetime import datetime
import logging
import asyncio

from ..core.models import MarketData, Trade, OrderBook


class DataProvider(ABC):
    """Interface base para todos os provedores de dados"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.connected = False
        self.last_update = {}
        
        # Event-driven attributes
        self.data_queue: Optional[asyncio.Queue] = None
        self.last_update_timestamp: Dict[str, datetime] = {}
        
    @abstractmethod
    async def connect(self) -> bool:
        """Conecta ao provedor de dados"""
        pass
        
    @abstractmethod
    async def disconnect(self) -> bool:
        """Desconecta do provedor de dados"""
        pass
        
    @abstractmethod
    async def start(self, queue: asyncio.Queue):
        """Inicia o provider em modo event-driven"""
        pass
        
    async def push_market_data(self, asset: str, data: MarketData):
        """Envia dados para a fila quando disponíveis"""
        if self.data_queue and data:
            if await self.validate_data(data):
                await self.data_queue.put((asset, data))
                self.last_update_timestamp[asset] = datetime.now()
                self.logger.debug(f"Dados enviados para fila: {asset}")
            else:
                self.logger.warning(f"Dados inválidos descartados: {asset}")
                
    async def is_connected(self) -> bool:
        """Verifica se está conectado"""
        return self.connected
        
    async def validate_data(self, data: MarketData) -> bool:
        """Valida integridade dos dados"""
        if not data:
            return False
            
        # Verifica se há trades
        if not data.trades:
            self.logger.warning(f"Sem trades para {data.asset}")
            return False
            
        # Verifica se o book está presente
        if not data.book or not data.book.bids or not data.book.asks:
            self.logger.warning(f"Book incompleto para {data.asset}")
            return False
            
        # Verifica timestamps
        now = datetime.now()
        for trade in data.trades:
            if trade.timestamp > now:
                self.logger.warning(f"Trade com timestamp futuro: {trade.timestamp}")
                return False
                
        return True
        
    def get_last_update_time(self, asset: str) -> Optional[datetime]:
        """Retorna horário da última atualização"""
        return self.last_update_timestamp.get(asset)