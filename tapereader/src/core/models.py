"""
Modelos de dados do TapeReader
Usa Pydantic para validação e serialização
"""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field, validator


class Side(str, Enum):
    """Lado da agressão ou do sinal de trading"""
    BUY = "BUY"
    SELL = "SELL"
    NEUTRAL = "NEUTRAL"


class SignalStrength(str, Enum):
    """Força do sinal baseado em confluência"""
    PREMIUM = "PREMIUM"
    STRONG = "STRONG"
    STANDARD = "STANDARD"
    WEAK = "WEAK"


class Trade(BaseModel):
    """Modelo de trade individual (Time & Sales)"""
    timestamp: datetime
    price: Decimal
    volume: int
    aggressor: Side
    
    @property
    def quantity(self) -> int:
        return self.volume
    
    @property
    def side(self) -> str:
        return self.aggressor.value.lower()
    
    class Config:
        json_encoders = {Decimal: str, datetime: lambda v: v.isoformat()}


class BookLevel(BaseModel):
    """Nível do book de ofertas"""
    price: Decimal
    volume: int
    orders: int = 1
    
    @property
    def quantity(self) -> int:
        return self.volume
    
    @validator('volume')
    def volume_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('Volume deve ser positivo')
        return v

OrderBookLevel = BookLevel


class OrderBook(BaseModel):
    """Book de ofertas completo"""
    timestamp: datetime
    bids: List[BookLevel]
    asks: List[BookLevel]
    
    @validator('bids')
    def validate_bids_order(cls, v):
        if len(v) > 1 and v != sorted(v, key=lambda x: x.price, reverse=True):
            raise ValueError('Bids devem estar ordenados (maior para menor)')
        return v
    
    @validator('asks')
    def validate_asks_order(cls, v):
        if len(v) > 1 and v != sorted(v, key=lambda x: x.price):
            raise ValueError('Asks devem estar ordenados (menor para maior)')
        return v


class MarketData(BaseModel):
    """Dados completos de mercado para análise"""
    asset: str
    timestamp: datetime
    trades: List[Trade]
    book: OrderBook
    
    @property
    def symbol(self) -> str:
        return self.asset
    
    @property
    def order_book(self) -> OrderBook:
        return self.book
    
    @validator('asset')
    def validate_asset(cls, v):
        if v not in ['DOLFUT', 'WDOFUT']:
            raise ValueError(f'Asset deve ser DOLFUT ou WDOFUT')
        return v


class BehaviorDetection(BaseModel):
    """Resultado da detecção de um comportamento"""
    behavior_type: str
    confidence: float = Field(ge=0.0, le=1.0)
    detected: bool
    direction: Optional[Side] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)
    
    @validator('confidence')
    def validate_confidence(cls, v):
        return round(v, 4)


class BehaviorSignal(BaseModel):
    """
    Sinal de comportamento detectado (usado internamente pelos behaviors).
    CORREÇÃO: Esta classe foi restaurada.
    """
    behavior_type: str
    symbol: str
    timestamp: datetime
    strength: float = Field(ge=0.0, le=1.0)
    direction: str  # 'bullish', 'bearish', 'neutral'
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def to_detection(self) -> BehaviorDetection:
        """Converte para BehaviorDetection"""
        # Mapeia a string de direção para o Enum 'Side'
        direction_map = {
            'bullish': Side.BUY,
            'bearish': Side.SELL,
            'neutral': Side.NEUTRAL
        }
        
        return BehaviorDetection(
            behavior_type=self.behavior_type,
            confidence=self.strength,
            detected=self.strength >= 0.7,
            direction=direction_map.get(self.direction.lower(), Side.NEUTRAL),
            metadata={**self.metadata, 'direction': self.direction, 'symbol': self.symbol},
            timestamp=self.timestamp
        )


class TradingSignal(BaseModel):
    """Sinal de trading completo com tracking"""
    signal_id: str
    timestamp: datetime
    direction: Side
    asset: str
    price: Decimal
    confidence: float = Field(ge=0.0, le=1.0)
    signal_strength: SignalStrength
    behaviors_detected: List[BehaviorDetection]
    primary_motivation: str
    secondary_motivations: List[str] = Field(default_factory=list)
    confluence_data: Dict[str, Any] = Field(default_factory=dict)
    market_context: Dict[str, Any] = Field(default_factory=dict)
    entry_price: Decimal
    stop_loss: Decimal
    take_profit_1: Decimal
    take_profit_2: Optional[Decimal] = None
    max_hold_time: int = 15
    risk_reward_ratio: float
    position_size_suggestion: Optional[int] = None
    status: str = "EMITTED"
    execution_data: Optional[Dict] = None
    result: Optional[Dict] = None
    strategy: str = ""
    
    class Config:
        json_encoders = {Decimal: str, datetime: lambda v: v.isoformat()}


class PriceContext(BaseModel):
    """Contexto histórico de um nível de preço"""
    price: Decimal
    visits_count: int = 0
    total_volume: int = 0
    buy_volume: int = 0
    sell_volume: int = 0
    rejections: int = 0
    breakouts: int = 0
    last_visit: Optional[datetime] = None
    avg_time_at_level: float = 0.0
    support_strength: float = 0.0
    resistance_strength: float = 0.0
    big_player_interest: bool = False
    
    def get_bias(self) -> str:
        """Retorna viés baseado no histórico"""
        if self.rejections > 3:
            return "strong_support" if self.support_strength > self.resistance_strength else "strong_resistance"
        elif self.big_player_interest:
            return "institutional_level"
        else:
            return "neutral"