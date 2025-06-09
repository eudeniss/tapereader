"""
Recurrence Pattern Detector
Detecta padrões repetitivos no fluxo de ordens
"""

from decimal import Decimal
from typing import List, Optional, Dict, Any, Tuple, Set
from datetime import datetime, timedelta
from collections import defaultdict, deque
import hashlib
import json
import asyncio

from .base import BehaviorDetector
from ..core.models import Trade, OrderBook, MarketData, BehaviorDetection, Side


class RecurrenceDetector(BehaviorDetector):
    """
    Detecta padrões recorrentes no tape
    
    Características:
    - Sequências de trades que se repetem
    - Padrões de preço/volume similares
    - Comportamentos cíclicos
    - Assinaturas de algoritmos específicos
    """
    
    @property
    def behavior_type(self) -> str:
        return "recurrence"
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.pattern_window = config.get('pattern_window', 100)  # trades
        self.min_pattern_length = config.get('min_pattern_length', 5)
        self.similarity_threshold = config.get('similarity_threshold', 0.85)
        self.min_occurrences = config.get('min_occurrences', 3)
        self.time_window_hours = config.get('time_window_hours', 4)
        
        # Pattern storage
        self.pattern_history = defaultdict(lambda: deque(maxlen=1000))
        self.identified_patterns = defaultdict(dict)  # symbol -> pattern_hash -> info
        self.pattern_last_seen = defaultdict(dict)    # symbol -> pattern_hash -> timestamp
        
    async def detect(self, market_data: MarketData) -> BehaviorDetection:
        """Detecta padrões recorrentes"""
        if not market_data.trades or len(market_data.trades) < self.min_pattern_length:
            return self.create_detection(False, 0.0)
            
        symbol = market_data.asset
        current_time = market_data.timestamp
        
        # Extrai padrões da sequência atual
        patterns = self._extract_patterns(market_data.trades)
        
        # Procura por recorrências
        best_recurrence = None
        best_strength = 0.0
        
        for pattern in patterns:
            pattern_hash = self._hash_pattern(pattern)
            
            # Verifica se é uma recorrência
            if self._is_recurrence(symbol, pattern_hash, pattern, current_time):
                # Analisa características da recorrência
                recurrence_info = self._analyze_recurrence(symbol, pattern_hash, pattern)
                
                if recurrence_info['strength'] > best_strength:
                    best_strength = recurrence_info['strength']
                    best_recurrence = {
                        'pattern_hash': pattern_hash,
                        'pattern': pattern,
                        'info': recurrence_info
                    }
                    
        # Armazena padrões para futuras comparações
        self._store_patterns(symbol, patterns, current_time)
        
        if best_recurrence and best_strength >= self.config.get('min_strength', 0.7):
            metadata = {
                'pattern_type': best_recurrence['info']['type'],
                'pattern_length': len(best_recurrence['pattern']['trades']),
                'occurrences': best_recurrence['info']['occurrences'],
                'avg_interval': best_recurrence['info']['avg_interval'],
                'pattern_signature': best_recurrence['pattern_hash'][:8],
                'predicted_next': best_recurrence['info'].get('predicted_next'),
                'direction': best_recurrence['info']['direction']
            }

            # --- CORREÇÃO APLICADA AQUI ---
            # Mapeia a string de direção para o Enum Side.
            direction_map = {"bullish": Side.BUY, "bearish": Side.SELL, "neutral": Side.NEUTRAL}
            direction_result = direction_map.get(best_recurrence['info']['direction'], Side.NEUTRAL)
            
            detection = self.create_detection(True, best_strength, metadata)
            detection.direction = direction_result
            return detection
            
        return self.create_detection(False, 0.0)
        
    def _extract_patterns(self, trades: List[Trade]) -> List[Dict]:
        """Extrai possíveis padrões da sequência de trades"""
        patterns = []
        
        # Tenta diferentes tamanhos de padrão
        for length in range(self.min_pattern_length, min(len(trades), 20)):
            for start in range(len(trades) - length + 1):
                pattern_trades = trades[start:start + length]
                
                # Cria representação do padrão
                pattern = self._create_pattern_representation(pattern_trades)
                
                if self._is_valid_pattern(pattern):
                    patterns.append(pattern)
                    
        return patterns
        
    def _create_pattern_representation(self, trades: List[Trade]) -> Dict:
        """Cria representação normalizada do padrão"""
        if not trades:
            return {}
            
        # Normaliza para facilitar comparação
        base_price = float(trades[0].price)
        base_time = trades[0].timestamp
        
        normalized_trades = []
        for trade in trades:
            normalized = {
                'price_ratio': float(trade.price) / base_price if base_price != 0 else 0,
                'size_category': self._categorize_size(trade.volume, trades[0].asset),
                'side': trade.aggressor.value.lower(),
                'time_delta': (trade.timestamp - base_time).total_seconds()
            }
            normalized_trades.append(normalized)
            
        # Calcula características do padrão
        pattern = {
            'trades': trades,
            'normalized': normalized_trades,
            'characteristics': self._calculate_pattern_characteristics(trades),
            'signature': self._create_signature(normalized_trades)
        }
        
        return pattern
        
    def _categorize_size(self, volume: int, symbol: str) -> str:
        """Categoriza tamanho da ordem"""
        # Ajusta thresholds por símbolo
        if 'WDO' in symbol:
            small = 100
            medium = 500
            large = 1000
        else:
            small = 25
            medium = 100
            large = 200
            
        if volume <= small:
            return 'small'
        elif volume <= medium:
            return 'medium'
        elif volume <= large:
            return 'large'
        else:
            return 'xlarge'
            
    def _calculate_pattern_characteristics(self, trades: List[Trade]) -> Dict:
        """Calcula características do padrão"""
        buy_count = sum(1 for t in trades if t.aggressor == Side.BUY)
        sell_count = len(trades) - buy_count
        
        volumes = [t.volume for t in trades]
        prices = [float(t.price) for t in trades]
        
        return {
            'buy_ratio': buy_count / len(trades) if trades else 0,
            'total_volume': sum(volumes),
            'avg_volume': sum(volumes) / len(volumes) if volumes else 0,
            'price_range': max(prices) - min(prices) if prices else 0,
            'price_trend': (prices[-1] - prices[0]) / prices[0] if prices and prices[0] != 0 else 0
        }
        
    def _create_signature(self, normalized_trades: List[Dict]) -> str:
        """Cria assinatura única para o padrão"""
        # Simplifica para comparação
        simplified = []
        for trade in normalized_trades:
            simplified.append({
                'pr': round(trade['price_ratio'], 4),
                'sz': trade['size_category'],
                'sd': trade['side'][0],  # 'b' ou 's'
                'td': round(trade['time_delta'], 1)
            })
            
        return json.dumps(simplified, sort_keys=True)
        
    def _hash_pattern(self, pattern: Dict) -> str:
        """Gera hash único para o padrão"""
        return hashlib.md5(pattern['signature'].encode()).hexdigest()
        
    def _is_valid_pattern(self, pattern: Dict) -> bool:
        """Verifica se o padrão é válido para análise"""
        if not pattern or not pattern.get('trades'):
            return False
            
        chars = pattern['characteristics']
        
        # Deve ter alguma atividade significativa
        if chars['total_volume'] < 50:  # Mínimo de volume
            return False
            
        # Não pode ser muito desequilibrado
        buy_ratio = chars['buy_ratio']
        if buy_ratio == 0 or buy_ratio == 1:  # Só compras ou só vendas
            return False
            
        return True
        
    def _is_recurrence(self, symbol: str, pattern_hash: str, pattern: Dict, 
                      current_time: datetime) -> bool:
        """Verifica se é uma recorrência de padrão conhecido"""
        if pattern_hash not in self.identified_patterns[symbol]:
            return False
            
        # Verifica última ocorrência
        last_seen = self.pattern_last_seen[symbol].get(pattern_hash)
        if last_seen:
            time_since_last = (current_time - last_seen).total_seconds() / 60  # minutos
            
            # Deve ter um intervalo mínimo entre ocorrências
            if time_since_last < 1:  # Menos de 1 minuto
                return False
                
        # Verifica similaridade com padrão armazenado
        stored_pattern = self.identified_patterns[symbol][pattern_hash]
        similarity = self._calculate_similarity(pattern, stored_pattern['template'])
        
        return similarity >= self.similarity_threshold
        
    def _calculate_similarity(self, pattern1: Dict, pattern2: Dict) -> float:
        """Calcula similaridade entre dois padrões"""
        if not pattern1 or not pattern2:
            return 0.0
            
        scores = []
        
        # Compara características
        chars1 = pattern1['characteristics']
        chars2 = pattern2['characteristics']
        
        # Buy ratio
        buy_diff = abs(chars1['buy_ratio'] - chars2['buy_ratio'])
        scores.append(1.0 - buy_diff)
        
        # Volume profile
        vol_ratio = min(chars1['avg_volume'], chars2['avg_volume']) / max(chars1['avg_volume'], chars2['avg_volume']) if max(chars1['avg_volume'], chars2['avg_volume']) > 0 else 1.0
        scores.append(vol_ratio)
        
        # Price trend
        trend_diff = abs(chars1['price_trend'] - chars2['price_trend'])
        scores.append(1.0 - min(trend_diff * 10, 1.0))  # Escala para 0-1
        
        # Sequência de trades
        seq_score = self._compare_sequences(pattern1['normalized'], pattern2['normalized'])
        scores.append(seq_score)
        
        return sum(scores) / len(scores) if scores else 0.0
        
    def _compare_sequences(self, seq1: List[Dict], seq2: List[Dict]) -> float:
        """Compara sequências de trades normalizados"""
        if len(seq1) != len(seq2):
            return 0.0
            
        matches = 0
        for t1, t2 in zip(seq1, seq2):
            # Compara side e size category
            if t1['side'] == t2['side'] and t1['size_category'] == t2['size_category']:
                # Compara price ratio com tolerância
                if abs(t1['price_ratio'] - t2['price_ratio']) < 0.001:
                    matches += 1
                    
        return matches / len(seq1) if seq1 else 0.0
        
    def _analyze_recurrence(self, symbol: str, pattern_hash: str, pattern: Dict) -> Dict:
        """Analisa características da recorrência"""
        stored_info = self.identified_patterns[symbol][pattern_hash]
        
        # Atualiza contagem
        stored_info['occurrences'] += 1
        stored_info['last_occurrence'] = pattern
        
        # Calcula intervalo médio
        if 'intervals' not in stored_info:
            stored_info['intervals'] = []
            
        current_time = pattern['trades'][-1].timestamp
        if stored_info.get('last_time'):
            interval = (current_time - stored_info['last_time']).total_seconds() / 60
            stored_info['intervals'].append(interval)
            
        stored_info['last_time'] = current_time
        
        # Analisa tipo de padrão
        pattern_type = self._identify_pattern_type(stored_info)
        
        # Calcula força
        strength = self._calculate_recurrence_strength(stored_info)
        
        # Determina direção
        chars = pattern['characteristics']
        if chars['buy_ratio'] > 0.6:
            direction = "bullish"
        elif chars['buy_ratio'] < 0.4:
            direction = "bearish"
        else:
            direction = "neutral"
            
        # Tenta prever próximo movimento
        prediction = self._predict_next_move(stored_info) if stored_info['occurrences'] >= 5 else None
        
        return {
            'strength': strength,
            'direction': direction,
            'type': pattern_type,
            'occurrences': stored_info['occurrences'],
            'avg_interval': sum(stored_info['intervals']) / len(stored_info['intervals']) if stored_info['intervals'] else 0,
            'predicted_next': prediction
        }
        
    def _identify_pattern_type(self, pattern_info: Dict) -> str:
        """Identifica o tipo de padrão recorrente"""
        if pattern_info['occurrences'] < 3:
            return "emerging"
            
        # Analisa regularidade dos intervalos
        if pattern_info.get('intervals'):
            intervals = pattern_info['intervals']
            avg_interval = sum(intervals) / len(intervals) if intervals else 0
            
            if avg_interval == 0: return "irregular_pattern"
            
            # Calcula desvio padrão
            variance = sum((x - avg_interval) ** 2 for x in intervals) / len(intervals) if intervals else 0
            std_dev = variance ** 0.5
            
            if std_dev / avg_interval < 0.2:  # Baixa variação
                if avg_interval < 5:
                    return "high_frequency_algo"
                elif avg_interval < 30:
                    return "regular_algo"
                else:
                    return "scheduled_algo"
            else:
                return "irregular_pattern"
        
        return "unknown"
        
    def _calculate_recurrence_strength(self, pattern_info: Dict) -> float:
        """Calcula força do padrão recorrente"""
        score = 0.0
        
        # Número de ocorrências
        occ_score = min(pattern_info['occurrences'] / 10, 1.0)  # Max em 10
        score += occ_score * 0.4
        
        # Regularidade dos intervalos
        if pattern_info.get('intervals') and len(pattern_info['intervals']) >= 2:
            intervals = pattern_info['intervals']
            avg_interval = sum(intervals) / len(intervals)
            variance = sum((x - avg_interval) ** 2 for x in intervals) / len(intervals)
            std_dev = variance ** 0.5
            
            if avg_interval > 0:
                regularity = 1.0 - min(std_dev / avg_interval, 1.0)
                score += regularity * 0.3
                
        # Consistência do padrão
        template = pattern_info['template']
        recent = pattern_info['last_occurrence']
        consistency = self._calculate_similarity(template, recent)
        score += consistency * 0.3
        
        return min(score, 1.0)
        
    def _predict_next_move(self, pattern_info: Dict) -> Optional[Dict]:
        """Tenta prever próximo movimento baseado no histórico"""
        if not pattern_info.get('intervals') or len(pattern_info['intervals']) < 3:
            return None
            
        # Calcula intervalo esperado
        recent_intervals = pattern_info['intervals'][-5:]  # Últimos 5
        avg_interval = sum(recent_intervals) / len(recent_intervals)
        
        # Características do padrão
        chars = pattern_info['template']['characteristics']
        
        return {
            'expected_in_minutes': round(avg_interval, 1),
            'expected_direction': 'bullish' if chars['buy_ratio'] > 0.5 else 'bearish',
            'confidence': min(pattern_info['occurrences'] / 20, 0.9)  # Max 90%
        }
        
    def _store_patterns(self, symbol: str, patterns: List[Dict], current_time: datetime):
        """Armazena padrões para análise futura"""
        for pattern in patterns:
            pattern_hash = self._hash_pattern(pattern)
            
            if pattern_hash not in self.identified_patterns[symbol]:
                # Novo padrão
                self.identified_patterns[symbol][pattern_hash] = {
                    'template': pattern,
                    'first_seen': current_time,
                    'occurrences': 1,
                    'last_time': current_time
                }
            
            self.pattern_last_seen[symbol][pattern_hash] = current_time
            
        # Limpa padrões antigos
        self._cleanup_old_patterns(symbol, current_time)
        
    def _cleanup_old_patterns(self, symbol: str, current_time: datetime):
        """Remove padrões antigos do tracking"""
        cutoff_time = current_time - timedelta(hours=self.time_window_hours)
        
        # Remove padrões não vistos recentemente
        patterns_to_remove = []
        for pattern_hash, last_seen in list(self.pattern_last_seen[symbol].items()):
            if last_seen < cutoff_time:
                patterns_to_remove.append(pattern_hash)
                
        for pattern_hash in patterns_to_remove:
            if pattern_hash in self.pattern_last_seen[symbol]:
                del self.pattern_last_seen[symbol][pattern_hash]
            if pattern_hash in self.identified_patterns[symbol]:
                del self.identified_patterns[symbol][pattern_hash]