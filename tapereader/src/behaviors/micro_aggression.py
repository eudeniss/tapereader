"""
Micro Aggression Behavior Detector
Detecta acumulação/distribuição discreta através de pequenas ordens
"""

from decimal import Decimal
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import asyncio

from .base import BehaviorDetector
from ..core.models import Trade, OrderBook, MarketData, BehaviorDetection, Side


class MicroAggressionDetector(BehaviorDetector):
    """
    Detecta micro agressões - acumulação ou distribuição discreta
    
    Características:
    - Múltiplas ordens pequenas na mesma direção
    - Evita mover o preço significativamente
    - Volume acumulado alto apesar do tamanho individual pequeno
    - Padrão persistente ao longo do tempo
    """
    
    @property
    def behavior_type(self) -> str:
        return "micro_aggression"
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.window_minutes = config.get('window_minutes', 15)
        self.min_trade_count = config.get('min_trade_count', 20)
        self.max_individual_size = config.get('max_individual_size', 50)  # DOL
        self.min_total_volume = config.get('min_total_volume', 500)  # DOL
        self.direction_threshold = config.get('direction_threshold', 0.7)  # 70% na mesma direção
        
        # Tracking
        self.micro_trades = defaultdict(list)  # symbol -> [trades]
        self.accumulation_zones = defaultdict(dict)  # symbol -> price -> stats
        
    async def detect(self, market_data: MarketData) -> BehaviorDetection:
        """Detecta padrões de micro agressão"""
        if not market_data.trades:
            return self.create_detection(False, 0.0)
            
        symbol = market_data.asset
        current_time = market_data.timestamp
        
        # Filtra trades pequenos
        self._update_micro_trades(symbol, market_data.trades, current_time)
        
        # Remove trades antigos
        self._cleanup_old_trades(symbol, current_time)
        
        # Analisa padrões
        micro_pattern = self._analyze_micro_pattern(symbol)
        
        if micro_pattern:
            # Calcula força da micro agressão
            strength = self._calculate_micro_strength(micro_pattern)
            
            if strength >= self.config.get('min_strength', 0.7):
                metadata = {
                    'trade_count': micro_pattern['trade_count'],
                    'total_volume': float(micro_pattern['total_volume']),
                    'avg_trade_size': float(micro_pattern['avg_size']),
                    'direction_ratio': micro_pattern['direction_ratio'],
                    'price_levels': micro_pattern['price_levels'],
                    'accumulation_type': micro_pattern['type'],
                    'direction': micro_pattern['direction']
                }
                
                # --- CORREÇÃO APLICADA AQUI ---
                # Mapeia a string 'bullish'/'bearish' para o Enum Side.
                direction_result = Side.BUY if micro_pattern['direction'] == 'bullish' else Side.SELL
                
                detection = self.create_detection(True, strength, metadata)
                detection.direction = direction_result
                return detection
                
        return self.create_detection(False, 0.0)
        
    def _update_micro_trades(self, symbol: str, trades: List[Trade], current_time: datetime):
        """Atualiza lista de micro trades"""
        # Define tamanho máximo baseado no símbolo
        max_size = self.max_individual_size
        if 'WDO' in symbol:
            max_size = max_size * 4  # Ajusta para mini contratos
            
        for trade in trades:
            if trade.volume <= max_size:
                self.micro_trades[symbol].append({
                    'trade': trade,
                    'timestamp': current_time,
                    'price_level': self._get_price_level(trade.price)
                })
                
    def _cleanup_old_trades(self, symbol: str, current_time: datetime):
        """Remove trades fora da janela de análise"""
        cutoff_time = current_time - timedelta(minutes=self.window_minutes)
        
        self.micro_trades[symbol] = [
            t for t in self.micro_trades[symbol]
            if t['timestamp'] > cutoff_time
        ]
        
    def _analyze_micro_pattern(self, symbol: str) -> Optional[Dict]:
        """Analisa padrão de micro agressão"""
        micro_trades = self.micro_trades[symbol]
        
        if len(micro_trades) < self.min_trade_count:
            return None
            
        # Agrupa por direção
        buy_trades = [t for t in micro_trades if t['trade'].aggressor == Side.BUY]
        sell_trades = [t for t in micro_trades if t['trade'].aggressor == Side.SELL]
        
        # Calcula volumes
        buy_volume = sum(t['trade'].volume for t in buy_trades)
        sell_volume = sum(t['trade'].volume for t in sell_trades)
        total_volume = buy_volume + sell_volume
        
        # Verifica volume mínimo
        min_volume = self.min_total_volume
        if 'WDO' in symbol:
            min_volume = min_volume * 4
            
        if total_volume < min_volume:
            return None
            
        # Determina direção predominante
        buy_ratio = float(buy_volume) / float(total_volume) if total_volume > 0 else 0.5
        
        if buy_ratio >= self.direction_threshold:
            direction = "bullish"
            dominant_trades = buy_trades
            direction_ratio = buy_ratio
        elif buy_ratio <= (1 - self.direction_threshold):
            direction = "bearish"
            dominant_trades = sell_trades
            direction_ratio = 1 - buy_ratio
        else:
            return None  # Sem direção clara
            
        # Analisa distribuição de preços
        price_distribution = self._analyze_price_distribution(dominant_trades)
        
        # Identifica tipo de acumulação
        accumulation_type = self._identify_accumulation_type(
            price_distribution, 
            dominant_trades
        )
        
        return {
            'direction': direction,
            'trade_count': len(micro_trades),
            'total_volume': total_volume,
            'avg_size': total_volume / len(micro_trades),
            'direction_ratio': direction_ratio,
            'price_levels': len(price_distribution),
            'type': accumulation_type,
            'price_distribution': price_distribution
        }
        
    def _get_price_level(self, price: Decimal) -> Decimal:
        """Agrupa preços em níveis"""
        # Arredonda para o tick mais próximo
        tick_size = Decimal('0.5')  # Padrão DOL
        return (price / tick_size).quantize(Decimal('1')) * tick_size
        
    def _analyze_price_distribution(self, trades: List[Dict]) -> Dict[Decimal, int]:
        """Analisa distribuição de trades por nível de preço"""
        distribution = defaultdict(int)
        
        for trade_data in trades:
            price_level = trade_data['price_level']
            distribution[price_level] += 1
            
        return dict(distribution)
        
    def _identify_accumulation_type(self, distribution: Dict[Decimal, int], 
                                   trades: List[Dict]) -> str:
        """Identifica o tipo de acumulação/distribuição"""
        if not distribution:
            return "unknown"
            
        # Ordena níveis de preço
        sorted_levels = sorted(distribution.keys())
        
        # Calcula concentração
        total_trades = sum(distribution.values())
        max_concentration = max(distribution.values()) / total_trades if total_trades > 0 else 0
        
        # Tipos de acumulação
        if max_concentration > 0.5:
            # Concentrado em um nível
            return "concentrated"
        elif len(sorted_levels) >= 5:
            # Distribuído em vários níveis
            price_range = float(sorted_levels[-1] - sorted_levels[0])
            avg_price = float(sum(t['trade'].price for t in trades)) / len(trades) if trades else 0
            
            if avg_price > 0 and price_range / avg_price < 0.002:  # Menos de 0.2%
                return "tight_accumulation"
            else:
                return "distributed_accumulation"
        else:
            # Padrão normal
            return "standard"
            
    def _calculate_micro_strength(self, pattern: Dict) -> float:
        """Calcula força do sinal de micro agressão"""
        score = 0.0
        
        # Pesos para cada componente
        weights = {
            'volume': 0.3,
            'consistency': 0.25,
            'persistence': 0.25,
            'stealth': 0.2
        }
        
        # 1. Volume acumulado
        volume_ratio = min(pattern['total_volume'] / (self.min_total_volume * 2), 1.0)
        score += volume_ratio * weights['volume']
        
        # 2. Consistência direcional
        consistency = (pattern['direction_ratio'] - self.direction_threshold) / (1 - self.direction_threshold) if (1 - self.direction_threshold) > 0 else 0
        score += consistency * weights['consistency']
        
        # 3. Persistência (número de trades)
        persistence = min(pattern['trade_count'] / (self.min_trade_count * 2), 1.0)
        score += persistence * weights['persistence']
        
        # 4. Stealth (quão pequenos são os trades)
        avg_size = pattern['avg_size']
        max_expected = self.max_individual_size * 0.5  # 50% do máximo é ideal
        if self.max_individual_size > 0:
            if avg_size <= max_expected:
                stealth = 1.0 - (avg_size / self.max_individual_size)
            else:
                stealth = 0.5  # Penaliza trades muito grandes
        else:
            stealth = 0.0
            
        score += stealth * weights['stealth']
        
        # Bonus por tipo de acumulação
        if pattern['type'] == 'concentrated':
            score *= 1.1  # 10% bonus
        elif pattern['type'] == 'tight_accumulation':
            score *= 1.05  # 5% bonus
            
        return min(score, 1.0)