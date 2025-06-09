"""
Detector de Momentum
Identifica força direcional sustentada no movimento
"""

from typing import Dict, Any, List, Optional, Tuple
from decimal import Decimal
from datetime import datetime, timedelta
import statistics

from .base import BehaviorDetector
from ..core.models import MarketData, BehaviorDetection, Trade, Side


class MomentumDetector(BehaviorDetector):
    """
    Detecta momentum (força direcional)
    
    Características:
    - Movimento consistente em uma direção
    - Volume crescente ou sustentado
    - Poucos pullbacks
    - Agressão dominante de um lado
    """
    
    @property
    def behavior_type(self) -> str:
        return "momentum"
        
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Parâmetros
        self.min_price_move = Decimal(str(config.get('min_price_move', 2.0)))  # Movimento mínimo
        self.min_directional_ratio = config.get('min_directional_ratio', 0.7)  # 70% na direção
        self.lookback_periods = config.get('lookback_periods', 5)
        self.acceleration_threshold = config.get('acceleration_threshold', 1.2)  # 20% aceleração
        
    async def detect(self, market_data: MarketData) -> BehaviorDetection:
        """Detecta momentum"""
        # Atualiza histórico
        self.update_history(market_data)
        
        if len(self.historical_data) < self.lookback_periods:
            return self.create_detection(False, 0.0)
            
        # Analisa momentum
        momentum_result = self._analyze_momentum(market_data)
        
        if momentum_result['detected']:
            metadata = {
                'momentum_direction': momentum_result['direction'].value,
                'strength': momentum_result['strength'],
                'velocity': str(momentum_result['velocity']),
                'acceleration': momentum_result['acceleration'],
                'consistency': momentum_result['consistency'],
                'volume_trend': momentum_result['volume_trend'],
                'pullback_ratio': momentum_result['pullback_ratio'],
                'current_phase': momentum_result['phase']
            }
            
            # --- CORREÇÃO APLICADA ---
            # A direção é a do próprio momentum.
            direction_result = momentum_result['direction']

            detection = self.create_detection(
                True,
                momentum_result['confidence'],
                metadata
            )
            detection.direction = direction_result
            return detection
            
        return self.create_detection(False, 0.0)
        
    def _analyze_momentum(self, current_data: MarketData) -> Dict[str, Any]:
        """Analisa presença de momentum"""
        # Coleta dados históricos
        price_series = self._build_price_series()
        
        if len(price_series) < self.lookback_periods:
            return {'detected': False}
            
        # Identifica direção dominante
        direction_analysis = self._analyze_direction(price_series)
        
        if not direction_analysis['has_direction']:
            return {'detected': False}
            
        # Analisa força do movimento
        momentum_metrics = self._calculate_momentum_metrics(
            price_series,
            direction_analysis['direction']
        )
        
        # Analisa características do volume
        volume_analysis = self._analyze_volume_momentum()
        
        # Identifica fase do momentum
        phase = self._identify_momentum_phase(
            momentum_metrics,
            volume_analysis
        )
        
        # Calcula confiança
        confidence_signals = {
            'directional_strength': direction_analysis['strength'],
            'velocity_score': momentum_metrics['velocity_score'],
            'consistency': momentum_metrics['consistency'],
            'volume_support': volume_analysis['support_score']
        }
        
        confidence = self.calculate_confidence(confidence_signals)
        
        if confidence < self.min_confidence:
            return {'detected': False}
            
        return {
            'detected': True,
            'confidence': confidence,
            'direction': direction_analysis['direction'],
            'strength': momentum_metrics['strength'],
            'velocity': momentum_metrics['velocity'],
            'acceleration': momentum_metrics['acceleration'],
            'consistency': momentum_metrics['consistency'],
            'volume_trend': volume_analysis['trend'],
            'pullback_ratio': momentum_metrics['pullback_ratio'],
            'phase': phase
        }
        
    def _build_price_series(self) -> List[Dict[str, Any]]:
        """Constrói série de preços para análise"""
        series = []
        
        for data in self.historical_data[-self.lookback_periods:]:
            if data.trades:
                # OHLC do período
                prices = [t.price for t in data.trades]
                
                period_data = {
                    'timestamp': data.timestamp,
                    'open': data.trades[0].price,
                    'high': max(prices),
                    'low': min(prices),
                    'close': data.trades[-1].price,
                    'volume': sum(t.volume for t in data.trades),
                    'trades': data.trades
                }
                
                series.append(period_data)
                
        return series
        
    def _analyze_direction(self, price_series: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analisa direção dominante do movimento"""
        if len(price_series) < 2:
            return {'has_direction': False}
            
        # Movimento total
        start_price = price_series[0]['open']
        end_price = price_series[-1]['close']
        total_move = end_price - start_price
        
        # Verifica se movimento é significativo
        if abs(total_move) < self.min_price_move:
            return {'has_direction': False}
            
        # Direção
        direction = Side.BUY if total_move > 0 else Side.SELL
        
        # Analisa consistência da direção
        moves_in_direction = 0
        total_moves = 0
        
        for i in range(1, len(price_series)):
            period_move = price_series[i]['close'] - price_series[i-1]['close']
            
            if period_move != 0:
                total_moves += 1
                if (direction == Side.BUY and period_move > 0) or \
                   (direction == Side.SELL and period_move < 0):
                    moves_in_direction += 1
                    
        directional_ratio = moves_in_direction / total_moves if total_moves > 0 else 0
        
        # Precisa de movimento direcional forte
        if directional_ratio < self.min_directional_ratio:
            return {'has_direction': False}
            
        return {
            'has_direction': True,
            'direction': direction,
            'strength': directional_ratio,
            'total_move': abs(total_move)
        }
        
    def _calculate_momentum_metrics(
        self,
        price_series: List[Dict[str, Any]],
        direction: Side
    ) -> Dict[str, Any]:
        """Calcula métricas de momentum"""
        # Velocidade (preço por período)
        velocities = []
        
        for i in range(1, len(price_series)):
            price_change = price_series[i]['close'] - price_series[i-1]['close']
            time_delta = (price_series[i]['timestamp'] - price_series[i-1]['timestamp']).total_seconds()
            
            if time_delta > 0:
                velocity = float(price_change) / time_delta
                velocities.append(velocity)
                
        avg_velocity = sum(velocities) / len(velocities) if velocities else 0
        
        # Aceleração (mudança na velocidade)
        acceleration = 0
        if len(velocities) >= 2:
            recent_velocity = sum(velocities[-2:]) / 2 if len(velocities[-2:]) > 0 else 0
            older_velocity = sum(velocities[:-2]) / len(velocities[:-2]) if len(velocities[:-2]) > 0 else 0
            
            if older_velocity != 0:
                acceleration = recent_velocity / older_velocity
                
        # Consistência (desvio padrão das velocidades)
        if len(velocities) > 1:
            velocity_std = statistics.stdev(velocities)
            consistency = max(0, 1 - (velocity_std / (abs(avg_velocity) + 0.001)))
        else:
            consistency = 0.5
            
        # Pullbacks
        pullback_count = 0
        total_periods = len(price_series) - 1
        
        for i in range(1, len(price_series)):
            period_move = price_series[i]['close'] - price_series[i-1]['close']
            
            if direction == Side.BUY and period_move < 0:
                pullback_count += 1
            elif direction == Side.SELL and period_move > 0:
                pullback_count += 1
                
        pullback_ratio = pullback_count / total_periods if total_periods > 0 else 0
        
        # Força geral
        strength = (
            abs(avg_velocity) * consistency * (1 - pullback_ratio)
        )
        
        # Normaliza velocity score
        velocity_score = min(1.0, abs(avg_velocity) / 0.1)  # 0.1 = velocidade alta
        
        return {
            'velocity': Decimal(str(avg_velocity)),
            'velocity_score': velocity_score,
            'acceleration': acceleration,
            'consistency': consistency,
            'pullback_ratio': pullback_ratio,
            'strength': strength
        }
        
    def _analyze_volume_momentum(self) -> Dict[str, Any]:
        """Analisa momentum do volume"""
        volumes = []
        
        for data in self.historical_data[-self.lookback_periods:]:
            period_volume = sum(t.volume for t in data.trades)
            volumes.append(period_volume)
            
        if len(volumes) < 2:
            return {'trend': 'neutral', 'support_score': 0.5}
            
        # Tendência do volume
        first_half_avg = sum(volumes[:len(volumes)//2]) / (len(volumes)//2) if len(volumes)//2 > 0 else 0
        second_half_avg = sum(volumes[len(volumes)//2:]) / (len(volumes) - len(volumes)//2) if (len(volumes) - len(volumes)//2) > 0 else 0
        
        if first_half_avg > 0 and second_half_avg > first_half_avg * 1.2:
            trend = 'increasing'
            support_score = 0.9
        elif first_half_avg > 0 and second_half_avg < first_half_avg * 0.8:
            trend = 'decreasing'
            support_score = 0.3
        else:
            trend = 'stable'
            support_score = 0.7
            
        # Analisa picos de volume
        avg_volume = sum(volumes) / len(volumes) if volumes else 0
        volume_spikes = sum(1 for v in volumes if v > avg_volume * 1.5)
        
        # Mais spikes = mais interesse
        if volume_spikes >= 3:
            support_score = min(1.0, support_score + 0.2)
            
        return {
            'trend': trend,
            'support_score': support_score,
            'avg_volume': avg_volume,
            'spikes': volume_spikes
        }
        
    def _identify_momentum_phase(
        self,
        momentum_metrics: Dict[str, Any],
        volume_analysis: Dict[str, Any]
    ) -> str:
        """Identifica fase atual do momentum"""
        acceleration = momentum_metrics['acceleration']
        consistency = momentum_metrics['consistency']
        volume_trend = volume_analysis['trend']
        
        # Fase inicial: Aceleração com volume crescente
        if acceleration > self.acceleration_threshold and volume_trend == 'increasing':
            return 'acceleration'
            
        # Fase madura: Velocidade constante, volume estável
        elif consistency > 0.7 and volume_trend == 'stable':
            return 'mature'
            
        # Fase de desaceleração: Perda de força
        elif acceleration < 0.8 or volume_trend == 'decreasing':
            return 'deceleration'
            
        # Fase inicial: Construindo momentum
        else:
            return 'building'