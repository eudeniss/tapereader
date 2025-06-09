"""
Sistema de Histórico e Análise Temporal
Gerencia análises históricas e padrões temporais
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta, time
from decimal import Decimal
from collections import defaultdict, deque
import statistics
import logging
from dataclasses import dataclass
from enum import Enum

from .database import DatabaseManager
from ..core.models import MarketData


class TimeFrame(Enum):
    """Timeframes para análise"""
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1w"


@dataclass
class PriceLevel:
    """Representa um nível de preço importante"""
    price: float
    strength: float
    touches: int
    rejections: int
    volume: int
    first_seen: datetime
    last_seen: datetime
    level_type: str  # support, resistance, hvn, lvn, poc


@dataclass
class MarketContext:
    """Contexto de mercado em um momento"""
    timestamp: datetime
    price: float
    trend: str  # UP, DOWN, LATERAL
    volatility: float
    volume_profile: Dict[float, int]
    key_levels: List[PriceLevel]
    market_phase: str  # accumulation, distribution, trend, range


class HistoricalAnalyzer:
    """Analisa dados históricos e identifica padrões"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Cache de análises
        self.analysis_cache = {}
        self.cache_ttl = 300  # 5 minutos
        
        # Parâmetros de análise
        self.lookback_periods = {
            TimeFrame.M1: 60,     # 60 minutos
            TimeFrame.M5: 288,    # 24 horas
            TimeFrame.M15: 672,   # 7 dias
            TimeFrame.H1: 720,    # 30 dias
            TimeFrame.D1: 365     # 1 ano
        }
        
    def get_historical_context(self, asset: str, price: float,
                             lookback_hours: int = 24) -> MarketContext:
        """Retorna contexto histórico completo de um preço"""
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=lookback_hours)
        
        # Busca dados
        candles = self.db.get_candles(asset, start_time, end_time)
        volume_profile = self.db.get_volume_profile(
            asset, start_time.date(), end_time.date()
        )
        
        # Análise de tendência
        trend = self._analyze_trend(candles)
        
        # Volatilidade
        volatility = self._calculate_volatility(candles)
        
        # Níveis importantes
        price_range = volatility * 3  # 3 desvios padrão
        key_levels = self._identify_key_levels(
            asset, price - price_range, price + price_range
        )
        
        # Fase do mercado
        market_phase = self._identify_market_phase(candles, volume_profile)
        
        return MarketContext(
            timestamp=end_time,
            price=price,
            trend=trend,
            volatility=volatility,
            volume_profile=volume_profile,
            key_levels=key_levels,
            market_phase=market_phase
        )
        
    def find_similar_patterns(self, asset: str, current_pattern: List[float],
                            lookback_days: int = 30, min_correlation: float = 0.8) -> List[Dict]:
        """Encontra padrões históricos similares"""
        cache_key = f"patterns_{asset}_{len(current_pattern)}_{lookback_days}"
        
        # Verifica cache
        if cache_key in self.analysis_cache:
            cached = self.analysis_cache[cache_key]
            if (datetime.now() - cached['timestamp']).seconds < self.cache_ttl:
                return cached['data']
                
        # Normaliza padrão atual
        normalized_current = self._normalize_pattern(current_pattern)
        
        # Busca histórico
        end_time = datetime.now()
        start_time = end_time - timedelta(days=lookback_days)
        candles = self.db.get_candles(asset, start_time, end_time)
        
        if len(candles) < len(current_pattern):
            return []
            
        # Procura padrões similares
        similar_patterns = []
        pattern_length = len(current_pattern)
        
        for i in range(len(candles) - pattern_length):
            historical_pattern = [c['close'] for c in candles[i:i + pattern_length]]
            normalized_historical = self._normalize_pattern(historical_pattern)
            
            # Calcula correlação
            correlation = self._calculate_correlation(
                normalized_current, normalized_historical
            )
            
            if correlation >= min_correlation:
                # Analisa o que aconteceu depois
                future_candles = candles[i + pattern_length:i + pattern_length + 10]
                if future_candles:
                    outcome = self._analyze_pattern_outcome(
                        historical_pattern, future_candles
                    )
                    
                    similar_patterns.append({
                        'timestamp': candles[i]['timestamp'],
                        'correlation': correlation,
                        'pattern': historical_pattern,
                        'outcome': outcome,
                        'future_move': outcome['total_move'],
                        'success': outcome['success']
                    })
                    
        # Ordena por correlação
        similar_patterns.sort(key=lambda x: x['correlation'], reverse=True)
        
        # Atualiza cache
        self.analysis_cache[cache_key] = {
            'timestamp': datetime.now(),
            'data': similar_patterns[:10]  # Top 10
        }
        
        return similar_patterns[:10]
        
    def get_intraday_profile(self, asset: str, lookback_days: int = 20) -> Dict:
        """Analisa perfil intradiário médio"""
        profiles = defaultdict(lambda: {
            'volume': [],
            'volatility': [],
            'trend_strength': []
        })
        
        end_date = datetime.now().date()
        
        for days_back in range(lookback_days):
            date = end_date - timedelta(days=days_back)
            
            # Busca dados do dia
            start = datetime.combine(date, time(9, 0))
            end = datetime.combine(date, time(18, 0))
            candles = self.db.get_candles(asset, start, end)
            
            if not candles:
                continue
                
            # Agrupa por hora
            hourly_data = defaultdict(list)
            for candle in candles:
                hour = candle['timestamp'].hour
                hourly_data[hour].append(candle)
                
            # Calcula métricas por hora
            for hour, hour_candles in hourly_data.items():
                if not hour_candles:
                    continue
                    
                # Volume total
                total_volume = sum(c['volume'] for c in hour_candles)
                profiles[hour]['volume'].append(total_volume)
                
                # Volatilidade (range médio)
                ranges = [c['high'] - c['low'] for c in hour_candles]
                avg_range = sum(ranges) / len(ranges) if ranges else 0
                profiles[hour]['volatility'].append(avg_range)
                
                # Força da tendência
                if len(hour_candles) >= 2:
                    move = hour_candles[-1]['close'] - hour_candles[0]['open']
                    strength = abs(move) / avg_range if avg_range > 0 else 0
                    profiles[hour]['trend_strength'].append(strength)
                    
        # Calcula médias
        intraday_profile = {}
        for hour in range(9, 18):
            if hour in profiles:
                intraday_profile[hour] = {
                    'avg_volume': statistics.mean(profiles[hour]['volume']) 
                                 if profiles[hour]['volume'] else 0,
                    'avg_volatility': statistics.mean(profiles[hour]['volatility'])
                                     if profiles[hour]['volatility'] else 0,
                    'avg_trend_strength': statistics.mean(profiles[hour]['trend_strength'])
                                         if profiles[hour]['trend_strength'] else 0,
                    'samples': len(profiles[hour]['volume'])
                }
                
        return intraday_profile
        
    def get_level_effectiveness(self, asset: str, level: float,
                              lookback_days: int = 30) -> Dict:
        """Analisa efetividade histórica de um nível"""
        end_time = datetime.now()
        start_time = end_time - timedelta(days=lookback_days)
        
        # Busca interações com o nível
        candles = self.db.get_candles(asset, start_time, end_time)
        
        interactions = []
        level_range = self._get_level_range(asset, level)
        
        for i, candle in enumerate(candles):
            # Verifica se tocou o nível
            if (candle['low'] <= level + level_range and 
                candle['high'] >= level - level_range):
                
                # Analisa resultado
                future_candles = candles[i+1:i+11]  # Próximos 10 candles
                if future_candles:
                    result = self._analyze_level_interaction(
                        level, candle, future_candles
                    )
                    interactions.append(result)
                    
        if not interactions:
            return {
                'level': level,
                'total_touches': 0,
                'effectiveness': 0,
                'message': 'Nível não testado no período'
            }
            
        # Calcula estatísticas
        total = len(interactions)
        holds = sum(1 for i in interactions if i['held'])
        rejections = sum(1 for i in interactions if i['rejection'])
        breakouts = sum(1 for i in interactions if i['breakout'])
        
        avg_reaction = statistics.mean([i['reaction_size'] for i in interactions])
        
        return {
            'level': level,
            'total_touches': total,
            'holds': holds,
            'rejections': rejections,
            'breakouts': breakouts,
            'hold_rate': (holds / total) * 100,
            'rejection_rate': (rejections / total) * 100,
            'breakout_rate': (breakouts / total) * 100,
            'avg_reaction': avg_reaction,
            'effectiveness': (holds + rejections) / total * 100,
            'interactions': interactions[-10:]  # Últimas 10
        }
        
    def identify_recurring_levels(self, asset: str, lookback_days: int = 90,
                                min_touches: int = 3) -> List[PriceLevel]:
        """Identifica níveis que aparecem recorrentemente"""
        end_time = datetime.now()
        start_time = end_time - timedelta(days=lookback_days)
        
        # Busca todos os níveis importantes do período
        candles = self.db.get_candles(asset, start_time, end_time)
        
        if not candles:
            return []
            
        # Identifica turning points
        turning_points = self._find_turning_points(candles)
        
        # Agrupa níveis próximos
        level_clusters = self._cluster_price_levels(
            asset, [tp['price'] for tp in turning_points]
        )
        
        # Analisa cada cluster
        recurring_levels = []
        
        for cluster_price, cluster_points in level_clusters.items():
            if len(cluster_points) >= min_touches:
                # Calcula força do nível
                strength = self._calculate_level_strength(
                    cluster_points, turning_points, candles
                )
                
                level = PriceLevel(
                    price=cluster_price,
                    strength=strength,
                    touches=len(cluster_points),
                    rejections=sum(1 for tp in turning_points 
                                 if tp['price'] in cluster_points and tp['rejected']),
                    volume=sum(tp.get('volume', 0) for tp in turning_points
                             if tp['price'] in cluster_points),
                    first_seen=min(tp['timestamp'] for tp in turning_points
                                 if tp['price'] in cluster_points),
                    last_seen=max(tp['timestamp'] for tp in turning_points
                                if tp['price'] in cluster_points),
                    level_type=self._classify_level_type(cluster_price, candles)
                )
                
                recurring_levels.append(level)
                
        # Ordena por força
        recurring_levels.sort(key=lambda x: x.strength, reverse=True)
        
        return recurring_levels
        
    def _analyze_trend(self, candles: List[Dict]) -> str:
        """Analisa tendência dos candles"""
        if len(candles) < 20:
            return "LATERAL"
            
        # Médias móveis
        sma_20 = sum(c['close'] for c in candles[-20:]) / 20
        sma_50 = sum(c['close'] for c in candles[-50:]) / 50 if len(candles) >= 50 else sma_20
        
        current_price = candles[-1]['close']
        
        # Análise de tendência
        if current_price > sma_20 > sma_50:
            # Verifica força
            price_above_sma20 = sum(1 for c in candles[-20:] if c['close'] > sma_20)
            if price_above_sma20 > 15:
                return "UP_STRONG"
            return "UP"
        elif current_price < sma_20 < sma_50:
            price_below_sma20 = sum(1 for c in candles[-20:] if c['close'] < sma_20)
            if price_below_sma20 > 15:
                return "DOWN_STRONG"
            return "DOWN"
        else:
            return "LATERAL"
            
    def _calculate_volatility(self, candles: List[Dict]) -> float:
        """Calcula volatilidade histórica"""
        if len(candles) < 2:
            return 0
            
        returns = []
        for i in range(1, len(candles)):
            if candles[i-1]['close'] > 0:
                ret = (candles[i]['close'] - candles[i-1]['close']) / candles[i-1]['close']
                returns.append(ret)
                
        if not returns:
            return 0
            
        return statistics.stdev(returns) if len(returns) > 1 else 0
        
    def _identify_key_levels(self, asset: str, price_min: float,
                           price_max: float) -> List[PriceLevel]:
        """Identifica níveis importantes na faixa"""
        # Busca do banco
        db_levels = self.db.get_important_levels(asset, price_min, price_max)
        
        levels = []
        for row in db_levels:
            level = PriceLevel(
                price=row['price'],
                strength=row['strength'],
                touches=row['touches'],
                rejections=row['rejections'],
                volume=row['volume'],
                first_seen=datetime.fromisoformat(row['first_detected']),
                last_seen=datetime.fromisoformat(row['last_updated']),
                level_type=row['level_type']
            )
            levels.append(level)
            
        return levels
        
    def _identify_market_phase(self, candles: List[Dict],
                             volume_profile: Dict) -> str:
        """Identifica fase do mercado"""
        if not candles or len(candles) < 20:
            return "undefined"
            
        # Range vs Trend
        high = max(c['high'] for c in candles[-20:])
        low = min(c['low'] for c in candles[-20:])
        range_size = high - low
        
        avg_candle = sum(c['high'] - c['low'] for c in candles[-20:]) / 20
        
        # Volume analysis
        recent_volume = sum(c['volume'] for c in candles[-10:])
        older_volume = sum(c['volume'] for c in candles[-20:-10])
        
        # Decisão
        if range_size < avg_candle * 10:  # Range apertado
            if recent_volume > older_volume * 1.5:
                return "accumulation"
            else:
                return "range"
        else:  # Range amplo
            if candles[-1]['close'] > candles[-20]['close']:
                return "uptrend"
            elif candles[-1]['close'] < candles[-20]['close']:
                return "downtrend"
            else:
                return "distribution"
                
    def _normalize_pattern(self, pattern: List[float]) -> List[float]:
        """Normaliza padrão para comparação"""
        if not pattern or len(pattern) < 2:
            return pattern
            
        min_val = min(pattern)
        max_val = max(pattern)
        range_val = max_val - min_val
        
        if range_val == 0:
            return [0.5] * len(pattern)
            
        return [(p - min_val) / range_val for p in pattern]
        
    def _calculate_correlation(self, pattern1: List[float],
                             pattern2: List[float]) -> float:
        """Calcula correlação entre padrões"""
        if len(pattern1) != len(pattern2):
            return 0
            
        # Correlação de Pearson
        mean1 = sum(pattern1) / len(pattern1)
        mean2 = sum(pattern2) / len(pattern2)
        
        numerator = sum((p1 - mean1) * (p2 - mean2) 
                       for p1, p2 in zip(pattern1, pattern2))
        
        sum_sq1 = sum((p - mean1) ** 2 for p in pattern1)
        sum_sq2 = sum((p - mean2) ** 2 for p in pattern2)
        
        denominator = (sum_sq1 * sum_sq2) ** 0.5
        
        if denominator == 0:
            return 0
            
        return numerator / denominator
        
    def _analyze_pattern_outcome(self, pattern: List[float],
                               future_candles: List[Dict]) -> Dict:
        """Analisa o resultado após um padrão"""
        if not future_candles:
            return {'success': False, 'total_move': 0}
            
        pattern_end = pattern[-1]
        
        # Movimento máximo
        max_price = max(c['high'] for c in future_candles)
        min_price = min(c['low'] for c in future_candles)
        
        up_move = ((max_price - pattern_end) / pattern_end) * 100
        down_move = ((pattern_end - min_price) / pattern_end) * 100
        
        # Direção predominante
        final_price = future_candles[-1]['close']
        total_move = ((final_price - pattern_end) / pattern_end) * 100
        
        return {
            'success': abs(total_move) > 0.5,  # Movimento > 0.5%
            'total_move': total_move,
            'max_up': up_move,
            'max_down': down_move,
            'direction': 'UP' if total_move > 0 else 'DOWN',
            'volatility': max(up_move, down_move)
        }
        
    def _get_level_range(self, asset: str, level: float) -> float:
        """Retorna range de tolerância para um nível"""
        if asset == 'DOLFUT':
            return 2.0  # 2 pontos
        else:  # WDOFUT
            return 5.0  # 5 pontos
            
    def _analyze_level_interaction(self, level: float, candle: Dict,
                                 future_candles: List[Dict]) -> Dict:
        """Analisa interação com nível"""
        # Verificar se segurou
        held = all(c['low'] >= level - self._get_level_range('', level) 
                  for c in future_candles[:5])
        
        # Verificar rejeição
        rejection = (candle['low'] <= level <= candle['high'] and
                    candle['close'] > level and
                    future_candles[0]['close'] > level)
        
        # Verificar rompimento
        breakout = any(c['close'] < level - self._get_level_range('', level)
                      for c in future_candles[:3])
        
        # Tamanho da reação
        if future_candles:
            reaction_size = abs(future_candles[0]['close'] - level)
        else:
            reaction_size = 0
            
        return {
            'timestamp': candle['timestamp'],
            'held': held,
            'rejection': rejection,
            'breakout': breakout,
            'reaction_size': reaction_size,
            'volume': candle['volume']
        }
        
    def _find_turning_points(self, candles: List[Dict]) -> List[Dict]:
        """Encontra pontos de reversão"""
        turning_points = []
        
        for i in range(1, len(candles) - 1):
            prev = candles[i-1]
            curr = candles[i]
            next = candles[i+1]
            
            # High
            if curr['high'] > prev['high'] and curr['high'] > next['high']:
                turning_points.append({
                    'timestamp': curr['timestamp'],
                    'price': curr['high'],
                    'type': 'high',
                    'volume': curr['volume'],
                    'rejected': next['close'] < curr['close']
                })
                
            # Low
            if curr['low'] < prev['low'] and curr['low'] < next['low']:
                turning_points.append({
                    'timestamp': curr['timestamp'],
                    'price': curr['low'],
                    'type': 'low',
                    'volume': curr['volume'],
                    'rejected': next['close'] > curr['close']
                })
                
        return turning_points
        
    def _cluster_price_levels(self, asset: str,
                            prices: List[float]) -> Dict[float, List[float]]:
        """Agrupa níveis de preço próximos"""
        if not prices:
            return {}
            
        clusters = defaultdict(list)
        threshold = self._get_level_range(asset, 0) * 2
        
        sorted_prices = sorted(prices)
        
        for price in sorted_prices:
            added = False
            
            for cluster_center in list(clusters.keys()):
                if abs(price - cluster_center) <= threshold:
                    clusters[cluster_center].append(price)
                    added = True
                    break
                    
            if not added:
                clusters[price].append(price)
                
        # Recalcula centros
        final_clusters = {}
        for center, prices in clusters.items():
            new_center = sum(prices) / len(prices)
            final_clusters[round(new_center, 2)] = prices
            
        return final_clusters
        
    def _calculate_level_strength(self, cluster_points: List[float],
                                turning_points: List[Dict],
                                candles: List[Dict]) -> float:
        """Calcula força de um nível"""
        touches = len(cluster_points)
        
        # Rejections
        rejections = sum(1 for tp in turning_points
                        if tp['price'] in cluster_points and tp['rejected'])
        
        # Volume
        volume = sum(tp.get('volume', 0) for tp in turning_points
                    if tp['price'] in cluster_points)
        
        # Tempo desde última visita
        last_touch = max(tp['timestamp'] for tp in turning_points
                        if tp['price'] in cluster_points)
        days_since = (datetime.now() - last_touch).days
        
        # Fórmula de força
        strength = (touches * 10 + rejections * 20) * (1 / (1 + days_since * 0.1))
        
        # Ajusta por volume
        if volume > 0:
            avg_volume = sum(c['volume'] for c in candles) / len(candles)
            volume_factor = min(volume / (avg_volume * touches), 2.0)
            strength *= volume_factor
            
        return min(strength, 100)  # Cap em 100
        
    def _classify_level_type(self, price: float, candles: List[Dict]) -> str:
        """Classifica tipo do nível"""
        if not candles:
            return "unknown"
            
        current_price = candles[-1]['close']
        
        if price > current_price:
            return "resistance"
        else:
            return "support"