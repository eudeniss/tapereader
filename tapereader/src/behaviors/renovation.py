"""
Order Renovation Behavior Detector
Detecta reposição sistemática de ordens no book
"""

from decimal import Decimal
from typing import List, Optional, Dict, Any, Tuple, Set
from datetime import datetime, timedelta
from collections import defaultdict, deque
import asyncio

from .base import BehaviorDetector
from ..core.models import Trade, OrderBook, MarketData, BookLevel, BehaviorDetection, Side


class RenovationDetector(BehaviorDetector):
    """
    Detecta renovação/reposição de ordens no book
    
    Características:
    - Ordens que desaparecem e reaparecem no mesmo nível
    - Manutenção de liquidez em níveis específicos
    - Padrão de refresh constante
    - Defesa de níveis importantes
    """
    
    @property
    def behavior_type(self) -> str:
        return "renovation"
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.track_levels = config.get('track_levels', 5)  # Níveis do book para monitorar
        self.renovation_threshold = config.get('renovation_threshold', 0.8)  # 80% do volume original
        self.time_window_seconds = config.get('time_window_seconds', 30)
        self.min_renovations = config.get('min_renovations', 3)
        self.significant_size = config.get('significant_size', 100)  # DOL
        
        # Tracking
        self.level_history = defaultdict(lambda: defaultdict(deque))  # symbol -> price -> history
        self.renovation_patterns = defaultdict(dict)  # symbol -> price -> stats
        self.active_levels = defaultdict(set)  # symbol -> set of prices being defended
        
    async def detect(self, market_data: MarketData) -> BehaviorDetection:
        """Detecta padrões de renovação de ordens"""
        if not market_data.book:
            return self.create_detection(False, 0.0)
            
        symbol = market_data.asset
        book = market_data.book
        current_time = market_data.timestamp
        
        # Monitora mudanças no book
        renovations = self._track_book_changes(symbol, book, current_time)
        
        # Limpa histórico antigo
        self._cleanup_old_data(symbol, current_time)
        
        # Encontra a renovação mais forte
        best_renovation = None
        best_strength = 0.0
        
        for price_level, renovation_data in renovations.items():
            strength = self._calculate_renovation_strength(renovation_data)
            
            if strength > best_strength:
                best_strength = strength
                best_renovation = {
                    'price_level': price_level,
                    'data': renovation_data
                }
                
        if best_renovation and best_strength >= self.config.get('min_strength', 0.7):
            metadata = {
                'price_level': float(best_renovation['price_level']),
                'renovation_count': best_renovation['data']['count'],
                'avg_size': float(best_renovation['data']['avg_size']),
                'total_defended': float(best_renovation['data']['total_volume']),
                'defense_duration': best_renovation['data']['duration_seconds'],
                'renovation_type': best_renovation['data']['type'],
                'direction': best_renovation['data']['direction']
            }

            # --- CORREÇÃO APLICADA AQUI ---
            # Mapeia a string 'bullish'/'bearish' para o Enum Side.
            direction_result = Side.BUY if best_renovation['data']['direction'] == 'bullish' else Side.SELL
            
            detection = self.create_detection(True, best_strength, metadata)
            detection.direction = direction_result
            return detection
            
        return self.create_detection(False, 0.0)
        
    def _track_book_changes(self, symbol: str, book: OrderBook, 
                           current_time: datetime) -> Dict[Decimal, Dict]:
        """Monitora mudanças no book e identifica renovações"""
        renovations = {}
        
        # Analisa bid side
        for i, level in enumerate(book.bids[:self.track_levels]):
            if self._is_significant_level(level, symbol):
                renovation = self._check_renovation(
                    symbol, level.price, level.volume, 'bid', current_time
                )
                if renovation:
                    renovations[level.price] = renovation
                    
        # Analisa ask side
        for i, level in enumerate(book.asks[:self.track_levels]):
            if self._is_significant_level(level, symbol):
                renovation = self._check_renovation(
                    symbol, level.price, level.volume, 'ask', current_time
                )
                if renovation:
                    renovations[level.price] = renovation
                    
        return renovations
        
    def _is_significant_level(self, level: BookLevel, symbol: str) -> bool:
        """Verifica se o nível é significativo para monitorar"""
        min_size = self.significant_size
        if 'WDO' in symbol:
            min_size = min_size * 4
            
        return level.volume >= min_size
        
    def _check_renovation(self, symbol: str, price: Decimal, current_volume: int,
                         side: str, current_time: datetime) -> Optional[Dict]:
        """Verifica se houve renovação no nível de preço"""
        # Garante que o símbolo existe no histórico
        if symbol not in self.level_history:
            self.level_history[symbol] = defaultdict(deque)
            
        history = self.level_history[symbol][price]
        
        # Adiciona estado atual
        history.append({
            'time': current_time,
            'volume': current_volume,
            'side': side
        })
        
        # Mantém apenas histórico recente
        cutoff_time = current_time - timedelta(seconds=self.time_window_seconds)
        while history and history[0]['time'] < cutoff_time:
            history.popleft()
            
        # Precisa de histórico suficiente
        if len(history) < 3:
            return None
            
        # Analisa padrões de renovação
        renovation_events = self._identify_renovation_events(history, symbol)
        
        if len(renovation_events) >= self.min_renovations:
            # Calcula estatísticas
            stats = self._calculate_renovation_stats(renovation_events, history, side)
            
            # Atualiza tracking
            if symbol not in self.renovation_patterns:
                self.renovation_patterns[symbol] = {}
            self.renovation_patterns[symbol][price] = stats
            
            if symbol not in self.active_levels:
                self.active_levels[symbol] = set()
            self.active_levels[symbol].add(price)
            
            return stats
            
        return None
        
    def _identify_renovation_events(self, history: deque, symbol: str) -> List[Dict]:
        """Identifica eventos de renovação no histórico"""
        events = []
        
        for i in range(1, len(history)):
            prev = history[i-1]
            curr = history[i]
            
            # Detecta mudanças significativas
            if prev['volume'] > 0:
                # Pattern: volume cai e depois volta
                if i < len(history) - 1:
                    next_entry = history[i+1]
                    if (curr['volume'] < prev['volume'] * 0.5 and  # Cai mais de 50%
                        next_entry['volume'] >= prev['volume'] * self.renovation_threshold):
                        
                        events.append({
                            'time': curr['time'],
                            'from_size': prev['volume'],
                            'low_size': curr['volume'],
                            'to_size': next_entry['volume'],
                            'duration': (next_entry['time'] - prev['time']).total_seconds()
                        })
                        
                # Reposição direta: aumento súbito
                elif (curr['volume'] - prev['volume']) > 0 and self._is_renovation_pattern(symbol, prev, curr):
                     events.append({
                        'time': curr['time'],
                        'from_size': prev['volume'],
                        'to_size': curr['volume'],
                        'duration': (curr['time'] - prev['time']).total_seconds()
                    })
                        
        return events
        
    def _is_renovation_pattern(self, symbol: str, prev: Dict, curr: Dict) -> bool:
        """Verifica se é um padrão de renovação válido"""
        # Deve ser o mesmo lado
        if prev.get('side') != curr.get('side'):
            return False
            
        # Tempo entre mudanças deve ser curto
        time_diff = (curr['time'] - prev['time']).total_seconds()
        if time_diff > 10:  # Mais de 10 segundos
            return False
            
        # Tamanho deve ser significativo
        min_size = self.significant_size
        if 'WDO' in symbol:
            min_size = min_size * 4
            
        return curr['volume'] >= min_size
        
    def _calculate_renovation_stats(self, events: List[Dict], history: deque, 
                                   side: str) -> Dict:
        """Calcula estatísticas das renovações"""
        if not events:
            return {}
            
        # Volumes
        total_volume = sum(e.get('to_size', e['from_size']) for e in events)
        avg_size = total_volume / len(events)
        
        # Timing
        durations = [e['duration'] for e in events if 'duration' in e]
        avg_duration = sum(durations) / len(durations) if durations else 0
        
        # Persistência
        first_time = history[0]['time']
        last_time = history[-1]['time']
        total_duration = (last_time - first_time).total_seconds()
        
        # Tipo de renovação
        renovation_type = self._classify_renovation_type(events, avg_duration)
        
        # Direção baseada no lado
        direction = "bullish" if side == 'bid' else "bearish"
        
        return {
            'count': len(events),
            'avg_size': avg_size,
            'total_volume': total_volume,
            'avg_duration': avg_duration,
            'duration_seconds': total_duration,
            'type': renovation_type,
            'direction': direction,
            'side': side
        }
        
    def _classify_renovation_type(self, events: List[Dict], avg_duration: float) -> str:
        """Classifica o tipo de renovação"""
        if not events:
            return "unknown"
            
        # Analisa padrões
        if avg_duration < 2:
            return "aggressive_defense"  # Renovação muito rápida
        elif avg_duration < 5:
            return "active_defense"      # Defesa ativa normal
        elif avg_duration < 10:
            return "passive_defense"     # Defesa mais passiva
        else:
            return "position_maintenance"  # Manutenção de posição
            
    def _calculate_renovation_strength(self, renovation_data: Dict) -> float:
        """Calcula força do sinal de renovação"""
        if not renovation_data:
            return 0.0
            
        score = 0.0
        weights = {
            'frequency': 0.3,
            'volume': 0.25,
            'persistence': 0.25,
            'aggressiveness': 0.2
        }
        
        # 1. Frequência de renovações
        count = renovation_data.get('count', 0)
        freq_score = min(count / (self.min_renovations * 2), 1.0)
        score += freq_score * weights['frequency']
        
        # 2. Volume defendido
        total_vol = renovation_data.get('total_volume', 0)
        expected_vol = self.significant_size * renovation_data.get('count', 1)
        vol_score = min(total_vol / (expected_vol * 2), 1.0) if expected_vol > 0 else 0
        score += vol_score * weights['volume']
        
        # 3. Persistência
        duration = renovation_data.get('duration_seconds', 0)
        if duration > 0:
            persistence_score = min(duration / self.time_window_seconds, 1.0)
            score += persistence_score * weights['persistence']
            
        # 4. Agressividade
        avg_duration = renovation_data.get('avg_duration', 10)
        if avg_duration > 0:
            # Quanto menor a duração, mais agressivo
            aggr_score = 1.0 - min(avg_duration / 10, 1.0)
            score += aggr_score * weights['aggressiveness']
            
        # Bonus por tipo
        reno_type = renovation_data.get('type', '')
        if reno_type == 'aggressive_defense':
            score *= 1.15
        elif reno_type == 'active_defense':
            score *= 1.1
            
        return min(score, 1.0)
        
    def _cleanup_old_data(self, symbol: str, current_time: datetime):
        """Remove dados antigos do histórico"""
        cutoff_time = current_time - timedelta(seconds=self.time_window_seconds * 2)
        
        # Verifica se o símbolo existe no histórico
        if symbol not in self.level_history:
            return
            
        # Limpa histórico de níveis
        prices_to_remove = []
        for price in list(self.level_history[symbol].keys()):
            history = self.level_history[symbol][price]
            
            # Remove entradas antigas
            while history and history[0]['time'] < cutoff_time:
                history.popleft()
                
            # Marca níveis sem histórico para remoção
            if not history:
                prices_to_remove.append(price)
        
        # Remove níveis marcados
        for price in prices_to_remove:
            del self.level_history[symbol][price]
            if symbol in self.active_levels:
                self.active_levels[symbol].discard(price)
                
        # Limpa padrões antigos
        if symbol in self.renovation_patterns:
            patterns_to_remove = []
            for price in list(self.renovation_patterns[symbol].keys()):
                if symbol not in self.active_levels or price not in self.active_levels[symbol]:
                    patterns_to_remove.append(price)
            
            # Remove padrões marcados
            for price in patterns_to_remove:
                if price in self.renovation_patterns[symbol]:
                    del self.renovation_patterns[symbol][price]