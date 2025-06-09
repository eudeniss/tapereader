"""
Detector de Exaustão
Identifica quando um movimento está perdendo força
"""

from typing import Dict, Any, List, Tuple
from decimal import Decimal
from datetime import datetime, timedelta
import statistics

from .base import BehaviorDetector
from ..core.models import MarketData, BehaviorDetection, Trade, Side


class ExhaustionDetector(BehaviorDetector):
    """
    Detecta padrões de exaustão no movimento
    
    Exaustão ocorre quando:
    - Volume diminui progressivamente
    - Movimento de preço desacelera
    - Agressão perde intensidade
    - Geralmente precede reversão
    """
    
    @property
    def behavior_type(self) -> str:
        return "exhaustion"
        
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Parâmetros específicos
        self.min_move_size = config.get('min_move_size', 3.0)  # Movimento mínimo em ticks
        self.volume_decay_threshold = config.get('volume_decay_threshold', 0.3)  # 30% decay
        self.momentum_decay_threshold = config.get('momentum_decay_threshold', 0.4)
        self.analysis_periods = config.get('analysis_periods', 3)  # Períodos para análise
        
    async def detect(self, market_data: MarketData) -> BehaviorDetection:
        """Detecta padrão de exaustão"""
        # Atualiza histórico
        self.update_history(market_data)
        
        # Precisa de histórico suficiente
        if len(self.historical_data) < 5:
            return self.create_detection(False, 0.0)
            
        # Analisa exaustão
        exhaustion_result = self._analyze_exhaustion(market_data)
        
        if exhaustion_result['detected']:
            metadata = {
                'exhaustion_type': exhaustion_result['type'],
                'movement_direction': exhaustion_result['direction'].value,
                'volume_decay': exhaustion_result['volume_decay'],
                'momentum_decay': exhaustion_result['momentum_decay'],
                'peak_price': str(exhaustion_result['peak_price']),
                'current_price': str(market_data.trades[-1].price if market_data.trades else 0),
                'stages_completed': exhaustion_result['stages']
            }
            
            # --- CORREÇÃO APLICADA ---
            # A direção da exaustão é a OPOSTA da direção do movimento.
            # Exaustão de um movimento de alta (BUY) -> sinal de reversão de BAIXA (SELL).
            direction_result = Side.SELL if exhaustion_result['direction'] == Side.BUY else Side.BUY

            detection = self.create_detection(
                detected=True,
                confidence=exhaustion_result['confidence'],
                metadata=metadata
            )
            detection.direction = direction_result
            return detection
            
        return self.create_detection(False, 0.0)
        
    def _analyze_exhaustion(self, current_data: MarketData) -> Dict[str, Any]:
        """Analisa sinais de exaustão"""
        # Divide trades em períodos para análise
        periods = self._divide_into_periods()
        
        if len(periods) < self.analysis_periods:
            return {'detected': False}
            
        # Identifica direção do movimento
        direction_result = self._identify_movement_direction(periods)
        
        if not direction_result['has_movement']:
            return {'detected': False}
            
        # Analisa decay de volume
        volume_decay = self._analyze_volume_decay(periods)
        
        # Analisa decay de momentum
        momentum_decay = self._analyze_momentum_decay(periods, direction_result['direction'])
        
        # Analisa participação (agressão)
        participation_decay = self._analyze_participation_decay(periods, direction_result['direction'])
        
        # Identifica estágio da exaustão
        exhaustion_stage = self._identify_exhaustion_stage(
            volume_decay,
            momentum_decay,
            participation_decay
        )
        
        # Calcula confiança
        confidence_signals = {
            'volume_decay': volume_decay['score'],
            'momentum_decay': momentum_decay['score'],
            'participation_decay': participation_decay['score'],
            'stage_maturity': exhaustion_stage['maturity']
        }
        
        confidence = self.calculate_confidence(confidence_signals)
        
        if confidence < self.min_confidence:
            return {'detected': False}
            
        return {
            'detected': True,
            'confidence': confidence,
            'type': exhaustion_stage['type'],
            'direction': direction_result['direction'],
            'volume_decay': volume_decay['decay_rate'],
            'momentum_decay': momentum_decay['decay_rate'],
            'peak_price': direction_result['peak_price'],
            'stages': exhaustion_stage['stages_completed']
        }
        
    def _divide_into_periods(self) -> List[List[Trade]]:
        """Divide trades históricos em períodos para análise"""
        all_trades = []
        
        # Coleta todos os trades recentes
        for data in self.historical_data[-10:]:
            all_trades.extend(data.trades)
            
        if len(all_trades) < self.analysis_periods * 5:
            return []
            
        # Divide em períodos iguais
        trades_per_period = len(all_trades) // self.analysis_periods
        periods = []
        
        for i in range(self.analysis_periods):
            start_idx = i * trades_per_period
            end_idx = start_idx + trades_per_period
            
            if i == self.analysis_periods - 1:
                # Último período pega todos os restantes
                periods.append(all_trades[start_idx:])
            else:
                periods.append(all_trades[start_idx:end_idx])
                
        return periods
        
    def _identify_movement_direction(self, periods: List[List[Trade]]) -> Dict[str, Any]:
        """Identifica se há movimento direcional claro"""
        if not periods or not periods[0]:
            return {'has_movement': False}
            
        # Preço inicial e máximos/mínimos
        start_price = periods[0][0].price
        
        all_prices = []
        for period in periods:
            all_prices.extend([t.price for t in period])
            
        max_price = max(all_prices)
        min_price = min(all_prices)
        
        # Calcula movimento total
        upward_move = float(max_price - start_price)
        downward_move = float(start_price - min_price)
        
        # Verifica se há movimento significativo
        if upward_move >= self.min_move_size:
            direction = Side.BUY
            peak_price = max_price
            movement_size = upward_move
        elif downward_move >= self.min_move_size:
            direction = Side.SELL
            peak_price = min_price
            movement_size = downward_move
        else:
            return {'has_movement': False}
            
        return {
            'has_movement': True,
            'direction': direction,
            'peak_price': peak_price,
            'movement_size': movement_size
        }
        
    def _analyze_volume_decay(self, periods: List[List[Trade]]) -> Dict[str, Any]:
        """Analisa diminuição progressiva do volume"""
        period_volumes = []
        
        for period in periods:
            total_volume = sum(t.volume for t in period)
            period_volumes.append(total_volume)
            
        if not period_volumes or period_volumes[0] == 0:
            return {'score': 0.0, 'decay_rate': 0.0}
            
        # Calcula taxa de decay
        decay_rates = []
        for i in range(1, len(period_volumes)):
            if period_volumes[i-1] > 0:
                decay = 1 - (period_volumes[i] / period_volumes[i-1])
                decay_rates.append(decay)
                
        if not decay_rates:
            return {'score': 0.0, 'decay_rate': 0.0}
            
        avg_decay = sum(decay_rates) / len(decay_rates)
        
        # Score baseado no decay progressivo
        score = 0.0
        if avg_decay > self.volume_decay_threshold:
            score = min(1.0, avg_decay / self.volume_decay_threshold)
            
            # Bonus se decay é consistente
            if all(d > 0 for d in decay_rates):
                score = min(1.0, score + 0.2)
                
        return {
            'score': score,
            'decay_rate': avg_decay,
            'period_volumes': period_volumes
        }
        
    def _analyze_momentum_decay(self, periods: List[List[Trade]], direction: Side) -> Dict[str, Any]:
        """Analisa diminuição do momentum (velocidade do movimento)"""
        period_movements = []
        
        for period in periods:
            if len(period) < 2:
                period_movements.append(0.0)
                continue
                
            # Movimento do período
            if direction == Side.BUY:
                movement = float(period[-1].price - period[0].price)
            else:
                movement = float(period[0].price - period[-1].price)
                
            # Normaliza pelo tempo
            time_span = (period[-1].timestamp - period[0].timestamp).total_seconds()
            if time_span > 0:
                momentum = movement / time_span
            else:
                momentum = 0.0
                
            period_movements.append(momentum)
            
        if not period_movements or period_movements[0] == 0:
            return {'score': 0.0, 'decay_rate': 0.0}
            
        # Calcula decay do momentum
        decay_rates = []
        for i in range(1, len(period_movements)):
            if period_movements[i-1] > 0:
                decay = 1 - (period_movements[i] / period_movements[i-1])
                decay_rates.append(decay)
                
        if not decay_rates:
            return {'score': 0.0, 'decay_rate': 0.0}
            
        avg_decay = sum(decay_rates) / len(decay_rates)
        
        # Score
        score = 0.0
        if avg_decay > self.momentum_decay_threshold:
            score = min(1.0, avg_decay / self.momentum_decay_threshold)
            
        return {
            'score': score,
            'decay_rate': avg_decay,
            'period_momentums': period_movements
        }
        
    def _analyze_participation_decay(self, periods: List[List[Trade]], direction: Side) -> Dict[str, Any]:
        """Analisa diminuição da participação (agressão)"""
        period_participations = []
        
        for period in periods:
            if not period:
                period_participations.append(0.0)
                continue
                
            # Conta trades na direção do movimento
            directional_trades = 0
            for trade in period:
                if trade.aggressor == direction:
                    directional_trades += 1
                    
            participation = directional_trades / len(period) if period else 0
            period_participations.append(participation)
            
        if not period_participations or period_participations[0] == 0:
            return {'score': 0.0, 'decay_rate': 0.0}
            
        # Calcula decay
        decay_rates = []
        for i in range(1, len(period_participations)):
            if period_participations[i-1] > 0:
                decay = 1 - (period_participations[i] / period_participations[i-1])
                decay_rates.append(decay)
                
        if not decay_rates:
            return {'score': 0.0, 'decay_rate': 0.0}
            
        avg_decay = sum(decay_rates) / len(decay_rates)
        
        # Score
        score = 0.0
        if avg_decay > 0.2:  # 20% decay na participação
            score = min(1.0, avg_decay * 2)
            
        return {
            'score': score,
            'decay_rate': avg_decay,
            'participations': period_participations
        }
        
    def _identify_exhaustion_stage(
        self,
        volume_decay: Dict[str, Any],
        momentum_decay: Dict[str, Any],
        participation_decay: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Identifica o estágio e tipo de exaustão"""
        stages_completed = 0
        
        # Estágio 1: Volume começando a cair
        if volume_decay['score'] > 0.3:
            stages_completed += 1
            
        # Estágio 2: Momentum desacelerando
        if momentum_decay['score'] > 0.4:
            stages_completed += 1
            
        # Estágio 3: Participação diminuindo
        if participation_decay['score'] > 0.3:
            stages_completed += 1
            
        # Tipo de exaustão
        if stages_completed >= 3:
            exhaustion_type = "complete"  # Exaustão completa
            maturity = 1.0
        elif stages_completed == 2:
            exhaustion_type = "developing"  # Em desenvolvimento
            maturity = 0.7
        else:
            exhaustion_type = "early"  # Inicial
            maturity = 0.4
            
        return {
            'type': exhaustion_type,
            'stages_completed': stages_completed,
            'maturity': maturity
        }