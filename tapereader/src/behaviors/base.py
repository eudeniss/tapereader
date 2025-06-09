"""
Interface base para detectores de comportamento
Define o contrato que todos os detectores devem seguir
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Any
from datetime import datetime, timedelta
from decimal import Decimal
import logging

from ..core.models import MarketData, BehaviorDetection, BehaviorSignal, Trade, OrderBook, Side


class BehaviorDetector(ABC):
    """Interface base para todos os detectores de comportamento"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.enabled = config.get('enabled', True)
        
        # Parâmetros comuns
        self.min_confidence = config.get('min_confidence', 0.7)
        self.lookback_seconds = config.get('lookback_seconds', 30)
        
        # Cache para análise temporal
        self.historical_data: List[MarketData] = []
        self.max_history = 100
        
    @property
    @abstractmethod
    def behavior_type(self) -> str:
        """Tipo do comportamento detectado"""
        pass
        
    @abstractmethod
    async def detect(self, market_data: MarketData) -> BehaviorDetection:
        """
        Detecta o comportamento nos dados de mercado
        
        Args:
            market_data: Dados atuais do mercado
            
        Returns:
            BehaviorDetection com resultado da análise
        """
        pass
    
    def detect_sync(self, market_data: MarketData) -> Optional[BehaviorSignal]:
        """
        Versão síncrona do detect para compatibilidade
        Deve ser implementada pelos behaviors que não usam async
        """
        return None
        
    def update_history(self, market_data: MarketData):
        """Atualiza histórico para análise temporal"""
        self.historical_data.append(market_data)
        
        # Mantém apenas dados recentes
        if len(self.historical_data) > self.max_history:
            self.historical_data.pop(0)
            
    def get_recent_trades(self, seconds: int = None) -> List[Trade]:
        """Retorna trades recentes do histórico"""
        if seconds is None:
            seconds = self.lookback_seconds
            
        cutoff_time = datetime.now() - timedelta(seconds=seconds)
        recent_trades = []
        
        for data in reversed(self.historical_data):
            for trade in data.trades:
                if trade.timestamp >= cutoff_time:
                    recent_trades.append(trade)
                    
        return recent_trades
        
    def calculate_volume_profile(self, trades: List[Trade]) -> Dict[str, Any]:
        """Calcula perfil de volume dos trades"""
        if not trades:
            return {
                'total_volume': 0,
                'buy_volume': 0,
                'sell_volume': 0,
                'buy_ratio': 0.0,
                'avg_trade_size': 0,
                'large_trades': 0
            }
            
        total_volume = sum(t.volume for t in trades)
        buy_volume = sum(t.volume for t in trades if t.aggressor == Side.BUY)
        sell_volume = total_volume - buy_volume
        
        # Define tamanho grande baseado no ativo
        asset = self.historical_data[-1].asset if self.historical_data else 'DOLFUT'
        large_threshold = 50 if asset == 'DOLFUT' else 200
        
        return {
            'total_volume': total_volume,
            'buy_volume': buy_volume,
            'sell_volume': sell_volume,
            'buy_ratio': buy_volume / total_volume if total_volume > 0 else 0.0,
            'avg_trade_size': total_volume / len(trades),
            'large_trades': sum(1 for t in trades if t.volume >= large_threshold)
        }
        
    def analyze_price_movement(self, trades: List[Trade]) -> Dict[str, Any]:
        """Analisa movimento de preço"""
        if len(trades) < 2:
            return {
                'price_change': Decimal('0'),
                'price_range': Decimal('0'),
                'volatility': 0.0,
                'trend': 'neutral'
            }
            
        prices = [t.price for t in trades]
        first_price = prices[0]
        last_price = prices[-1]
        
        price_change = last_price - first_price
        price_range = max(prices) - min(prices)
        
        # Calcula volatilidade simples
        if len(prices) > 1:
            price_changes = [abs(prices[i] - prices[i-1]) for i in range(1, len(prices))]
            volatility = float(sum(price_changes) / len(price_changes))
        else:
            volatility = 0.0
            
        # Determina tendência
        if price_change > Decimal('1'):
            trend = 'bullish'
        elif price_change < Decimal('-1'):
            trend = 'bearish'
        else:
            trend = 'neutral'
            
        return {
            'price_change': price_change,
            'price_range': price_range,
            'volatility': volatility,
            'trend': trend
        }
        
    def analyze_book_imbalance(self, book: OrderBook) -> Dict[str, Any]:
        """Analisa desequilíbrio no book"""
        if not book.bids or not book.asks:
            return {
                'imbalance_ratio': 0.0,
                'bid_volume': 0,
                'ask_volume': 0,
                'pressure': 'neutral'
            }
            
        # Volume total em cada lado (primeiros 5 níveis)
        bid_volume = sum(level.volume for level in book.bids[:5])
        ask_volume = sum(level.volume for level in book.asks[:5])
        
        total_volume = bid_volume + ask_volume
        if total_volume == 0:
            imbalance_ratio = 0.0
        else:
            imbalance_ratio = (bid_volume - ask_volume) / total_volume
            
        # Determina pressão
        if imbalance_ratio > 0.3:
            pressure = 'buy'
        elif imbalance_ratio < -0.3:
            pressure = 'sell'
        else:
            pressure = 'neutral'
            
        return {
            'imbalance_ratio': imbalance_ratio,
            'bid_volume': bid_volume,
            'ask_volume': ask_volume,
            'pressure': pressure
        }
        
    def find_support_resistance_levels(self, trades: List[Trade], sensitivity: float = 0.5) -> Dict[str, List[Decimal]]:
        """Identifica níveis de suporte e resistência"""
        if len(trades) < 10:
            return {'support': [], 'resistance': []}
            
        prices = [t.price for t in trades]
        price_counts = {}
        
        # Agrupa preços próximos
        for price in prices:
            found = False
            for level in price_counts:
                if abs(price - level) <= Decimal(str(sensitivity)):
                    price_counts[level] += 1
                    found = True
                    break
                    
            if not found:
                price_counts[price] = 1
                
        # Identifica níveis significativos (visitados múltiplas vezes)
        significant_levels = [
            level for level, count in price_counts.items() 
            if count >= 3
        ]
        
        if not significant_levels:
            return {'support': [], 'resistance': []}
            
        current_price = trades[-1].price
        support_levels = [l for l in significant_levels if l < current_price]
        resistance_levels = [l for l in significant_levels if l > current_price]
        
        return {
            'support': sorted(support_levels, reverse=True)[:3],
            'resistance': sorted(resistance_levels)[:3]
        }
        
    def detect_unusual_activity(self, trades: List[Trade], book: OrderBook) -> Dict[str, bool]:
        """Detecta atividades incomuns"""
        volume_profile = self.calculate_volume_profile(trades)
        book_analysis = self.analyze_book_imbalance(book)
        
        # Critérios para atividade incomum
        unusual = {
            'high_volume': volume_profile['total_volume'] > self.config.get('high_volume_threshold', 1000),
            'large_trades': volume_profile['large_trades'] > self.config.get('large_trades_threshold', 5),
            'extreme_imbalance': abs(book_analysis['imbalance_ratio']) > 0.7,
            'one_sided_flow': volume_profile['buy_ratio'] > 0.8 or volume_profile['buy_ratio'] < 0.2
        }
        
        return unusual
        
    def calculate_confidence(self, signals: Dict[str, float]) -> float:
        """
        Calcula confiança final baseada em múltiplos sinais
        
        Args:
            signals: Dicionário com peso de cada sinal (0.0 a 1.0)
            
        Returns:
            Confiança final entre 0.0 e 1.0
        """
        if not signals:
            return 0.0
            
        # Média ponderada dos sinais
        total_weight = sum(abs(weight) for weight in signals.values())
        if total_weight == 0:
            return 0.0
            
        confidence = sum(signal * abs(weight) for signal, weight in signals.items()) / total_weight
        
        # Garante que está no range [0, 1]
        return max(0.0, min(1.0, confidence))
        
    def create_detection(
        self, 
        detected: bool, 
        confidence: float, 
        metadata: Dict[str, Any] = None
    ) -> BehaviorDetection:
        """Cria objeto BehaviorDetection padronizado"""
        if metadata is None:
            metadata = {}
            
        return BehaviorDetection(
            behavior_type=self.behavior_type,
            detected=detected,
            confidence=confidence,
            metadata=metadata,
            timestamp=datetime.now()
        )
    
    def signal_to_detection(self, signal: Optional[BehaviorSignal]) -> BehaviorDetection:
        """Converte BehaviorSignal para BehaviorDetection"""
        if signal is None:
            return self.create_detection(False, 0.0)
        
        return signal.to_detection()