"""
High Frequency Trading (HFT) Behavior Detector
Detecta atividade de algoritmos de alta frequência
"""

from decimal import Decimal
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from collections import defaultdict
import asyncio

from .base import BehaviorDetector
from ..core.models import Trade, OrderBook, MarketData, BehaviorDetection, Side


class HTFDetector(BehaviorDetector):
    """
    Detecta atividade de High Frequency Trading
    
    Características:
    - Múltiplas ordens pequenas em intervalos regulares
    - Spreads muito apertados
    - Cancelamentos rápidos
    - Padrões de latência consistentes
    """
    
    @property
    def behavior_type(self) -> str:
        return "htf"
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.min_frequency = config.get('min_frequency', 5)  # ordens/segundo
        self.max_interval_ms = config.get('max_interval_ms', 200)  # ms entre ordens
        self.min_pattern_length = config.get('min_pattern_length', 10)  # trades
        self.size_variance_threshold = config.get('size_variance_threshold', 0.15)  # 15%
        
        # Tracking
        self.trade_intervals = defaultdict(list)  # symbol -> [intervals]
        self.recent_patterns = defaultdict(list)   # symbol -> [patterns]
        
    async def detect(self, market_data: MarketData) -> BehaviorDetection:
        """Detecta padrões de HFT"""
        if not market_data.trades:
            return self.create_detection(False, 0.0)
            
        symbol = market_data.asset
        trades = market_data.trades
        
        # Analisa intervalos entre trades
        intervals = self._calculate_intervals(trades)
        if not intervals:
            return self.create_detection(False, 0.0)
            
        # Detecta padrões regulares
        patterns = self._find_regular_patterns(intervals, trades)
        if not patterns:
            return self.create_detection(False, 0.0)
            
        # Analisa características HFT
        best_pattern = None
        best_score = 0.0
        
        for pattern in patterns:
            hft_score = self._analyze_hft_characteristics(pattern, trades)
            
            if hft_score > best_score:
                best_score = hft_score
                best_pattern = pattern
                
        if best_score >= self.config.get('min_hft_score', 0.7):
            # Determina direção do HFT
            hft_direction_str = self._determine_hft_direction(best_pattern, trades)
            
            metadata = {
                'pattern_length': len(best_pattern['indices']),
                'avg_interval_ms': best_pattern['avg_interval'],
                'frequency': best_pattern['frequency'],
                'size_variance': best_pattern['size_variance'],
                'algorithm_type': self._identify_algorithm_type(best_pattern),
                'direction': hft_direction_str # Mantém a string para metadados
            }
            
            # --- CORREÇÃO APLICADA ---
            # Mapeia a string de direção para o Enum Side.
            direction_map = {"bullish": Side.BUY, "bearish": Side.SELL, "neutral": Side.NEUTRAL}
            direction_result = direction_map.get(hft_direction_str, Side.NEUTRAL)

            detection = self.create_detection(True, best_score, metadata)
            detection.direction = direction_result
            return detection
            
        return self.create_detection(False, 0.0)
        
    def _calculate_intervals(self, trades: List[Trade]) -> List[float]:
        """Calcula intervalos entre trades em millisegundos"""
        intervals = []
        
        for i in range(1, len(trades)):
            interval = (trades[i].timestamp - trades[i-1].timestamp).total_seconds() * 1000
            intervals.append(interval)
            
        return intervals
        
    def _find_regular_patterns(self, intervals: List[float], trades: List[Trade]) -> List[Dict]:
        """Encontra padrões regulares de intervalos"""
        patterns = []
        
        # Busca sequências com intervalos consistentes
        i = 0
        while i < len(intervals) - self.min_pattern_length:
            pattern_indices = [i]
            base_interval = intervals[i]
            
            # Tolerance de 20% no intervalo
            tolerance = base_interval * 0.2
            
            j = i + 1
            while j < len(intervals):
                if abs(intervals[j] - base_interval) <= tolerance:
                    pattern_indices.append(j + 1)  # +1 porque intervals tem len-1
                    
                    if len(pattern_indices) >= self.min_pattern_length:
                        # Calcula métricas do padrão
                        pattern_trades = [trades[idx] for idx in pattern_indices]
                        
                        pattern = {
                            'indices': pattern_indices,
                            'avg_interval': sum(intervals[idx-1] for idx in pattern_indices[1:]) / (len(pattern_indices)-1) if len(pattern_indices) > 1 else base_interval,
                            'frequency': 1000 / base_interval if base_interval > 0 else 0,  # trades/segundo
                            'size_variance': self._calculate_size_variance(pattern_trades),
                            'start_time': pattern_trades[0].timestamp,
                            'end_time': pattern_trades[-1].timestamp
                        }
                        
                        if pattern['frequency'] >= self.min_frequency:
                            patterns.append(pattern)
                            
                j += 1
                
            i = j if j > i + 1 else i + 1
            
        return patterns
        
    def _calculate_size_variance(self, trades: List[Trade]) -> float:
        """Calcula variância no tamanho das ordens"""
        if not trades:
            return 0
            
        sizes = [float(trade.volume) for trade in trades]
        avg_size = sum(sizes) / len(sizes)
        
        if avg_size == 0:
            return 0
            
        variance = sum((size - avg_size) ** 2 for size in sizes) / len(sizes)
        std_dev = variance ** 0.5
        
        return std_dev / avg_size  # Coeficiente de variação
        
    def _analyze_hft_characteristics(self, pattern: Dict, trades: List[Trade]) -> float:
        """Analisa características típicas de HFT"""
        score = 0.0
        weights = {
            'frequency': 0.3,
            'regularity': 0.25,
            'size_consistency': 0.25,
            'price_impact': 0.2
        }
        
        # 1. Alta frequência
        if pattern['frequency'] >= self.min_frequency:
            freq_score = min(pattern['frequency'] / (self.min_frequency * 2), 1.0)
            score += freq_score * weights['frequency']
            
        # 2. Regularidade dos intervalos
        if pattern['avg_interval'] <= self.max_interval_ms:
            reg_score = 1.0 - (pattern['avg_interval'] / self.max_interval_ms)
            score += reg_score * weights['regularity']
            
        # 3. Consistência no tamanho
        if pattern['size_variance'] <= self.size_variance_threshold:
            size_score = 1.0 - (pattern['size_variance'] / self.size_variance_threshold)
            score += size_score * weights['size_consistency']
            
        # 4. Baixo impacto no preço
        pattern_trades = [trades[idx] for idx in pattern['indices']]
        price_impact = self._calculate_price_impact(pattern_trades)
        
        if price_impact < 0.001:  # Menos de 0.1% de impacto
            impact_score = 1.0 - (price_impact / 0.001) if price_impact < 0.001 else 0.0
            score += impact_score * weights['price_impact']
            
        return score
        
    def _calculate_price_impact(self, trades: List[Trade]) -> float:
        """Calcula impacto no preço durante o padrão"""
        if len(trades) < 2:
            return 0
            
        first_price = float(trades[0].price)
        last_price = float(trades[-1].price)
        
        if first_price == 0:
            return 0
            
        return abs(last_price - first_price) / first_price
        
    def _determine_hft_direction(self, pattern: Dict, trades: List[Trade]) -> str:
        """Determina direção da atividade HFT"""
        pattern_trades = [trades[idx] for idx in pattern['indices']]
        
        buy_volume = sum(t.volume for t in pattern_trades if t.aggressor == Side.BUY)
        sell_volume = sum(t.volume for t in pattern_trades if t.aggressor == Side.SELL)
        
        total_volume = buy_volume + sell_volume
        if total_volume == 0:
            return "neutral"
            
        buy_ratio = float(buy_volume) / float(total_volume)
        
        if buy_ratio > 0.6:
            return "bullish"
        elif buy_ratio < 0.4:
            return "bearish"
        else:
            return "neutral"
            
    def _identify_algorithm_type(self, pattern: Dict) -> str:
        """Identifica possível tipo de algoritmo HFT"""
        freq = pattern['frequency']
        variance = pattern['size_variance']
        interval = pattern['avg_interval']
        
        # Market Making: alta frequência, baixa variância, intervalos regulares
        if freq > 10 and variance < 0.1 and interval < 100:
            return "market_maker"
            
        # Arbitrage: frequência moderada, tamanhos consistentes
        elif 5 <= freq <= 10 and variance < 0.15:
            return "arbitrage"
            
        # Momentum: frequência variável, pode ter mais variância
        elif freq > 5 and variance > 0.15:
            return "momentum"
            
        # Execution: frequência moderada, foco em execução
        else:
            return "execution"