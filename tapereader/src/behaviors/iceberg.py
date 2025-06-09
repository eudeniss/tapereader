"""
Detector de Iceberg
Identifica ordens grandes ocultas sendo executadas em clips menores
"""

from typing import Dict, Any, List, Tuple, Optional
from decimal import Decimal
from datetime import datetime, timedelta
from collections import defaultdict

from .base import BehaviorDetector
from ..core.models import MarketData, BehaviorDetection, OrderBook, Trade, Side


class IcebergDetector(BehaviorDetector):
    """
    Detecta ordens iceberg (ordens grandes divididas)
    
    Características:
    - Execuções repetidas do mesmo tamanho (clips)
    - Mesmo nível de preço ou muito próximo
    - Volume total muito maior que o mostrado
    - Renovação automática no book
    """
    
    @property
    def behavior_type(self) -> str:
        return "iceberg"
        
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Parâmetros
        self.clip_similarity_threshold = config.get('clip_similarity', 0.9)  # 90% similaridade
        self.min_clips = config.get('min_clips', 3)  # Mínimo de clips
        self.price_tolerance = Decimal(str(config.get('price_tolerance', 0.5)))
        self.time_window = config.get('time_window', 120)  # 2 minutos
        
        # Tamanhos típicos de clips por ativo
        self.typical_clips = {
            'DOLFUT': [25, 50, 100],  # Clips comuns
            'WDOFUT': [100, 200, 500]
        }
        
    async def detect(self, market_data: MarketData) -> BehaviorDetection:
        """Detecta padrão iceberg"""
        # Atualiza histórico
        self.update_history(market_data)
        
        if not market_data.trades:
            return self.create_detection(False, 0.0)
            
        # Analisa iceberg
        iceberg_result = self._analyze_iceberg(market_data)
        
        if iceberg_result['detected']:
            metadata = {
                'iceberg_side': iceberg_result['side'].value,
                'clip_size': iceberg_result['clip_size'],
                'clips_detected': iceberg_result['clips_count'],
                'price_level': str(iceberg_result['price_level']),
                'total_volume': iceberg_result['total_volume'],
                'estimated_remaining': iceberg_result['estimated_remaining'],
                'execution_pattern': iceberg_result['pattern'],
                'renewal_detected': iceberg_result['renewal_detected']
            }
            
            # --- CORREÇÃO APLICADA ---
            # A direção é a do lado em que o iceberg está.
            # Um iceberg de compra (no BID) é um sinal de alta.
            direction_result = iceberg_result['side']
            
            detection = self.create_detection(
                True,
                iceberg_result['confidence'],
                metadata
            )
            detection.direction = direction_result
            return detection
            
        return self.create_detection(False, 0.0)
        
    def _analyze_iceberg(self, current_data: MarketData) -> Dict[str, Any]:
        """Analisa presença de iceberg"""
        asset = current_data.asset
        
        # Agrupa trades por preço e tamanho
        trade_patterns = self._identify_trade_patterns(asset)
        
        if not trade_patterns:
            return {'detected': False}
            
        # Procura por padrões de iceberg
        iceberg_candidates = self._find_iceberg_patterns(trade_patterns)
        
        if not iceberg_candidates:
            return {'detected': False}
            
        # Analisa o candidato mais forte
        best_candidate = max(iceberg_candidates, key=lambda x: x['score'])
        
        # Verifica renovação no book
        renewal_check = self._check_order_renewal(
            current_data.book,
            best_candidate['price_level'],
            best_candidate['side']
        )
        
        # Calcula confiança
        confidence_signals = {
            'pattern_strength': best_candidate['score'],
            'clip_consistency': best_candidate['consistency'],
            'volume_significance': min(1.0, best_candidate['total_volume'] / 1000),
            'renewal_evidence': renewal_check['renewal_score']
        }
        
        confidence = self.calculate_confidence(confidence_signals)
        
        if confidence < self.min_confidence:
            return {'detected': False}
            
        # Estima tamanho restante
        estimated_remaining = self._estimate_remaining_size(best_candidate)
        
        return {
            'detected': True,
            'confidence': confidence,
            'side': best_candidate['side'],
            'clip_size': best_candidate['clip_size'],
            'clips_count': best_candidate['clips_count'],
            'price_level': best_candidate['price_level'],
            'total_volume': best_candidate['total_volume'],
            'estimated_remaining': estimated_remaining,
            'pattern': best_candidate['pattern_type'],
            'renewal_detected': renewal_check['renewal_detected']
        }
        
    def _identify_trade_patterns(self, asset: str) -> Dict[str, List[Trade]]:
        """Identifica padrões nos trades recentes"""
        recent_trades = self.get_recent_trades(self.time_window)
        
        if len(recent_trades) < self.min_clips:
            return {}
            
        # Agrupa por características similares
        patterns = defaultdict(list)
        
        for trade in recent_trades:
            # Cria chave baseada em tamanho aproximado e preço
            size_bucket = self._get_size_bucket(trade.volume, asset)
            price_bucket = self._round_to_level(trade.price)
            
            key = (size_bucket, price_bucket, trade.aggressor)
            patterns[key].append(trade)
            
        # Filtra apenas padrões com repetições
        return {k: v for k, v in patterns.items() if len(v) >= self.min_clips}
        
    def _get_size_bucket(self, volume: int, asset: str) -> int:
        """Agrupa volumes em buckets típicos"""
        # Procura clip típico mais próximo
        typical = self.typical_clips.get(asset, [25, 50, 100])
        
        closest = min(typical, key=lambda x: abs(x - volume))
        
        # Se está próximo de um clip típico (10% de tolerância)
        if abs(volume - closest) / closest <= 0.1:
            return closest
        else:
            # Arredonda para múltiplo de 5 ou 10
            if asset == 'DOLFUT':
                return round(volume / 5) * 5
            else:
                return round(volume / 10) * 10
                
    def _round_to_level(self, price: Decimal) -> Decimal:
        """Arredonda preço para nível"""
        return Decimal(str(round(float(price) / float(self.price_tolerance)) * float(self.price_tolerance)))
        
    def _find_iceberg_patterns(self, patterns: Dict[str, List[Trade]]) -> List[Dict[str, Any]]:
        """Encontra padrões que parecem iceberg"""
        candidates = []
        
        for (size_bucket, price_level, side), trades in patterns.items():
            if len(trades) < self.min_clips:
                continue
                
            # Analisa consistência dos clips
            consistency_analysis = self._analyze_clip_consistency(trades, size_bucket)
            
            if consistency_analysis['is_consistent']:
                # Analisa padrão temporal
                temporal_pattern = self._analyze_temporal_pattern(trades)
                
                # Score do padrão
                pattern_score = (
                    consistency_analysis['consistency_score'] * 0.4 +
                    temporal_pattern['regularity_score'] * 0.3 +
                    min(1.0, len(trades) / 10) * 0.3  # Mais clips = maior score
                )
                
                candidates.append({
                    'side': side,
                    'clip_size': size_bucket,
                    'clips_count': len(trades),
                    'price_level': price_level,
                    'total_volume': sum(t.volume for t in trades),
                    'consistency': consistency_analysis['consistency_score'],
                    'pattern_type': temporal_pattern['pattern_type'],
                    'score': pattern_score,
                    'trades': trades
                })
                
        return candidates
        
    def _analyze_clip_consistency(self, trades: List[Trade], expected_size: int) -> Dict[str, Any]:
        """Analisa consistência dos clips"""
        volumes = [t.volume for t in trades]
        
        # Calcula desvio dos volumes
        deviations = [abs(v - expected_size) / expected_size for v in volumes]
        avg_deviation = sum(deviations) / len(deviations)
        
        # Consistente se desvio médio < 10%
        is_consistent = avg_deviation < (1 - self.clip_similarity_threshold)
        
        # Score de consistência
        consistency_score = max(0, 1 - avg_deviation)
        
        # Verifica se há clips exatos
        exact_clips = sum(1 for v in volumes if v == expected_size)
        exact_ratio = exact_clips / len(volumes)
        
        return {
            'is_consistent': is_consistent,
            'consistency_score': consistency_score,
            'exact_ratio': exact_ratio,
            'avg_deviation': avg_deviation
        }
        
    def _analyze_temporal_pattern(self, trades: List[Trade]) -> Dict[str, Any]:
        """Analisa padrão temporal das execuções"""
        if len(trades) < 2:
            return {'pattern_type': 'single', 'regularity_score': 0}
            
        # Calcula intervalos entre trades
        intervals = []
        for i in range(1, len(trades)):
            interval = (trades[i].timestamp - trades[i-1].timestamp).total_seconds()
            intervals.append(interval)
            
        avg_interval = sum(intervals) / len(intervals)
        
        # Analisa regularidade
        if all(i < 5 for i in intervals):
            # Muito rápido - possível algo trading
            pattern_type = 'rapid'
            regularity_score = 0.8
        elif all(abs(i - avg_interval) < avg_interval * 0.5 for i in intervals):
            # Intervalos regulares - iceberg algorítmico
            pattern_type = 'algorithmic'
            regularity_score = 1.0
        else:
            # Intervalos irregulares - iceberg manual ou adaptativo
            pattern_type = 'adaptive'
            regularity_score = 0.6
            
        return {
            'pattern_type': pattern_type,
            'regularity_score': regularity_score,
            'avg_interval': avg_interval,
            'intervals': intervals
        }
        
    def _check_order_renewal(self, book: OrderBook, price_level: Decimal, side: Side) -> Dict[str, Any]:
        """Verifica se há renovação de ordem no book"""
        if not book:
            return {'renewal_detected': False, 'renewal_score': 0}
            
        # Seleciona lado relevante
        if side == Side.BUY:
            levels = book.bids
        else:
            levels = book.asks
            
        # Procura ordem no nível do iceberg
        renewal_detected = False
        renewal_score = 0
        
        for level in levels[:5]:  # Top 5 níveis
            if abs(level.price - price_level) <= self.price_tolerance:
                # Encontrou ordem no nível
                # Iceberg geralmente mostra volume pequeno/médio
                if 10 <= level.volume <= 100:  # Range típico de iceberg visível
                    renewal_detected = True
                    renewal_score = 0.8
                    
                    # Se tem apenas 1 ordem, mais provável ser iceberg
                    if level.orders == 1:
                        renewal_score = 1.0
                        
                break
                
        return {
            'renewal_detected': renewal_detected,
            'renewal_score': renewal_score
        }
        
    def _estimate_remaining_size(self, iceberg: Dict[str, Any]) -> int:
        """Estima tamanho restante do iceberg"""
        # Baseado no padrão de execução
        if iceberg['pattern_type'] == 'algorithmic':
            # Iceberg algorítmico tende a ser maior
            # Estima baseado na velocidade de execução
            avg_interval = sum(
                (iceberg['trades'][i].timestamp - iceberg['trades'][i-1].timestamp).total_seconds()
                for i in range(1, len(iceberg['trades']))
            ) / (len(iceberg['trades']) - 1)
            
            if avg_interval < 10:
                # Execução rápida - provavelmente grande
                multiplier = 5
            else:
                # Execução paciente
                multiplier = 3
                
            return iceberg['total_volume'] * multiplier
            
        else:
            # Estimativa conservadora
            return iceberg['total_volume'] * 2