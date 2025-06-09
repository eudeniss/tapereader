"""
Detector de Stop Hunt
Identifica movimentos para acionar stops e posterior reversão
"""

from typing import Dict, Any, List, Optional, Tuple
from decimal import Decimal
from datetime import datetime, timedelta

from .base import BehaviorDetector
from ..core.models import MarketData, BehaviorDetection, Trade, Side


class StopHuntDetector(BehaviorDetector):
    """
    Detecta stop hunts (caça aos stops)
    
    Características:
    - Spike rápido além de nível importante
    - Pouco volume no spike
    - Reversão imediata após atingir stops
    - Retorno ao range original
    """
    
    @property
    def behavior_type(self) -> str:
        return "stop_hunt"
        
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Parâmetros
        self.spike_threshold = Decimal(str(config.get('spike_threshold', 3.0)))  # Ticks mínimos
        self.reversal_ratio = config.get('reversal_ratio', 0.6)  # 60% de retorno
        self.max_spike_duration = config.get('max_spike_duration', 30)  # segundos
        self.min_reversal_speed = config.get('min_reversal_speed', 0.5)  # ticks/segundo
        self.volume_threshold = config.get('volume_threshold', 0.5)  # Volume baixo no spike
        
    async def detect(self, market_data: MarketData) -> BehaviorDetection:
        """Detecta padrão de stop hunt"""
        # Atualiza histórico
        self.update_history(market_data)
        
        if len(self.historical_data) < 3:
            return self.create_detection(False, 0.0)
            
        # Analisa stop hunt
        stop_hunt_result = self._analyze_stop_hunt(market_data)
        
        if stop_hunt_result['detected']:
            metadata = {
                'hunt_type': stop_hunt_result['hunt_type'],
                'spike_size': str(stop_hunt_result['spike_size']),
                'spike_high': str(stop_hunt_result['spike_high']),
                'spike_low': str(stop_hunt_result['spike_low']),
                'reversal_percent': stop_hunt_result['reversal_percent'],
                'victims_side': stop_hunt_result['victims_side'],
                'hunt_duration': stop_hunt_result['duration_seconds'],
                'volume_anomaly': stop_hunt_result['volume_anomaly'],
                'likely_stop_level': str(stop_hunt_result['stop_level'])
            }
            
            # --- CORREÇÃO APLICADA AQUI ---
            # A direção do sinal é a da REVERSÃO esperada.
            # Bull Hunt (caça a stops de comprados) -> reversão para ALTA (BUY).
            # Bear Hunt (caça a stops de vendidos) -> reversão para BAIXA (SELL).
            direction_result = Side.BUY if stop_hunt_result['hunt_type'] == 'bull_hunt' else Side.SELL
            
            detection = self.create_detection(
                True,
                stop_hunt_result['confidence'],
                metadata
            )
            detection.direction = direction_result
            return detection
            
        return self.create_detection(False, 0.0)
        
    def _analyze_stop_hunt(self, current_data: MarketData) -> Dict[str, Any]:
        """Analisa se há stop hunt"""
        # Busca spikes recentes
        recent_spikes = self._find_price_spikes()
        
        if not recent_spikes:
            return {'detected': False}
            
        # Analisa spike mais recente
        for spike in reversed(recent_spikes):  # Do mais recente para o mais antigo
            # Verifica se houve reversão
            reversal = self._check_spike_reversal(spike, current_data)
            
            if reversal['reversed']:
                # Analisa características do stop hunt
                hunt_analysis = self._analyze_hunt_characteristics(spike, reversal)
                
                # Calcula confiança
                confidence_signals = {
                    'spike_magnitude': min(1.0, float(spike['size']) / float(self.spike_threshold * 2)),
                    'reversal_quality': reversal['reversal_quality'],
                    'volume_anomaly': hunt_analysis['volume_anomaly'],
                    'pattern_clarity': hunt_analysis['pattern_score']
                }
                
                confidence = self.calculate_confidence(confidence_signals)
                
                if confidence >= self.min_confidence:
                    return {
                        'detected': True,
                        'confidence': confidence,
                        'hunt_type': spike['direction'],
                        'spike_size': spike['size'],
                        'spike_high': spike['high'],
                        'spike_low': spike['low'],
                        'reversal_percent': reversal['reversal_percent'],
                        'victims_side': 'longs' if spike['direction'] == 'bull_hunt' else 'shorts',
                        'duration_seconds': spike['duration'],
                        'volume_anomaly': hunt_analysis['volume_anomaly'],
                        'stop_level': spike['peak_price']
                    }
                    
        return {'detected': False}
        
    def _find_price_spikes(self) -> List[Dict[str, Any]]:
        """Encontra spikes de preço suspeitos"""
        recent_trades = self.get_recent_trades(120)  # Últimos 2 minutos
        
        if len(recent_trades) < 10:
            return []
            
        spikes = []
        
        # Identifica movimentos rápidos
        for i in range(10, len(recent_trades)):
            # Janela antes do possível spike
            window_start = i - 10
            window_trades = recent_trades[window_start:i]
            
            # Calcula range normal
            normal_prices = [t.price for t in window_trades]
            normal_high = max(normal_prices)
            normal_low = min(normal_prices)
            
            # Verifica próximos trades para spike
            spike_window_end = min(i + 20, len(recent_trades))
            potential_spike_trades = recent_trades[i:spike_window_end]
            
            if not potential_spike_trades:
                continue
                
            spike_prices = [t.price for t in potential_spike_trades]
            spike_high = max(spike_prices)
            spike_low = min(spike_prices)
            
            # Detecta spike para cima (bear hunt - pega stops dos shorts)
            if spike_high > normal_high + self.spike_threshold:
                spike_size = spike_high - normal_high
                peak_price = spike_high
                direction = 'bear_hunt'
                
                # Tempo do spike
                spike_start = window_trades[-1].timestamp
                spike_peak_time = next(
                    (t.timestamp for t in potential_spike_trades if t.price >= spike_high - Decimal('0.5')),
                    spike_start
                )
                duration = (spike_peak_time - spike_start).total_seconds()
                
                if duration <= self.max_spike_duration:
                    spikes.append({
                        'timestamp': spike_start,
                        'direction': direction,
                        'size': spike_size,
                        'high': spike_high,
                        'low': normal_low,
                        'normal_high': normal_high,
                        'normal_low': normal_low,
                        'peak_price': peak_price,
                        'duration': duration,
                        'trades': potential_spike_trades,
                        'index': i
                    })
                    
            # Detecta spike para baixo (bull hunt - pega stops dos longs)
            elif spike_low < normal_low - self.spike_threshold:
                spike_size = normal_low - spike_low
                peak_price = spike_low
                direction = 'bull_hunt'
                
                spike_start = window_trades[-1].timestamp
                spike_peak_time = next(
                    (t.timestamp for t in potential_spike_trades if t.price <= spike_low + Decimal('0.5')),
                    spike_start
                )
                duration = (spike_peak_time - spike_start).total_seconds()
                
                if duration <= self.max_spike_duration:
                    spikes.append({
                        'timestamp': spike_start,
                        'direction': direction,
                        'size': spike_size,
                        'high': normal_high,
                        'low': spike_low,
                        'normal_high': normal_high,
                        'normal_low': normal_low,
                        'peak_price': peak_price,
                        'duration': duration,
                        'trades': potential_spike_trades,
                        'index': i
                    })
                    
        return spikes
        
    def _check_spike_reversal(self, spike: Dict[str, Any], current_data: MarketData) -> Dict[str, Any]:
        """Verifica se houve reversão após spike"""
        # Pega trades após o spike
        all_recent_trades = self.get_recent_trades(180)
        
        # Encontra trades após o spike
        post_spike_trades = []
        spike_end_index = spike['index'] + len(spike['trades'])
        
        if spike_end_index < len(all_recent_trades):
            post_spike_trades = all_recent_trades[spike_end_index:]
            
        if not post_spike_trades:
            # Usa trades atuais
            post_spike_trades = current_data.trades
            
        if not post_spike_trades:
            return {'reversed': False}
            
        # Analisa retorno ao range normal
        current_price = post_spike_trades[-1].price
        
        if spike['direction'] == 'bear_hunt':
            # Spike foi para cima, reversão é para baixo
            total_spike = spike['peak_price'] - spike['normal_high']
            reversal_distance = spike['peak_price'] - current_price
        else:
            # Spike foi para baixo, reversão é para cima
            total_spike = spike['normal_low'] - spike['peak_price']
            reversal_distance = current_price - spike['peak_price']
            
        if total_spike <= 0:
            return {'reversed': False}
            
        # Calcula percentual de reversão
        reversal_percent = float(reversal_distance / total_spike)
        
        # Verifica se reversão é suficiente
        if reversal_percent >= self.reversal_ratio:
            # Calcula velocidade da reversão
            reversal_speed = float('inf')
            if post_spike_trades:
                reversal_time = (post_spike_trades[-1].timestamp - spike['timestamp']).total_seconds()
                if reversal_time > 0:
                    reversal_speed = float(reversal_distance) / reversal_time
                    
                # Qualidade da reversão
                reversal_quality = min(1.0, (
                    reversal_percent / self.reversal_ratio * 0.5 +
                    min(1.0, reversal_speed / self.min_reversal_speed) * 0.5
                ))
            else:
                reversal_quality = reversal_percent / self.reversal_ratio
                
            return {
                'reversed': True,
                'reversal_percent': reversal_percent,
                'reversal_quality': reversal_quality,
                'current_price': current_price,
                'reversal_distance': reversal_distance
            }
            
        return {'reversed': False}
        
    def _analyze_hunt_characteristics(self, spike: Dict[str, Any], reversal: Dict[str, Any]) -> Dict[str, Any]:
        """Analisa características do stop hunt"""
        # Volume durante o spike
        spike_volume = sum(t.volume for t in spike['trades'])
        
        # Volume médio normal
        avg_volume = self._calculate_average_volume()
        
        # Anomalia de volume (esperamos volume BAIXO no spike)
        if avg_volume > 0 and spike['duration'] > 0:
            volume_ratio = spike_volume / (avg_volume * spike['duration'] / 60)  # Normalizado por minuto
            # Inverte: menos volume = maior anomalia (característica de stop hunt)
            volume_anomaly = max(0, 1 - volume_ratio)
        else:
            volume_anomaly = 0.5
            
        # Analisa padrão
        pattern_score = self._calculate_pattern_score(spike, reversal)
        
        # Identifica nível provável dos stops
        stop_level = self._identify_stop_level(spike)
        
        return {
            'volume_anomaly': volume_anomaly,
            'pattern_score': pattern_score,
            'spike_volume': spike_volume,
            'avg_volume': avg_volume,
            'stop_level': stop_level
        }
        
    def _calculate_average_volume(self) -> float:
        """Calcula volume médio por minuto"""
        if len(self.historical_data) < 5:
            return 100
            
        total_volume = 0
        total_time = 0
        
        for data in self.historical_data[-10:]:
            if data.trades:
                period_volume = sum(t.volume for t in data.trades)
                total_volume += period_volume
                total_time += 1
                
        if total_time > 0:
            return total_volume / total_time
        else:
            return 100
            
    def _calculate_pattern_score(self, spike: Dict[str, Any], reversal: Dict[str, Any]) -> float:
        """Calcula score do padrão de stop hunt"""
        scores = []
        
        # 1. Spike rápido
        if spike['duration'] < 10:
            scores.append(1.0)
        elif spike['duration'] < 20:
            scores.append(0.7)
        else:
            scores.append(0.4)
            
        # 2. Reversão forte
        if reversal['reversal_percent'] > 0.8:
            scores.append(1.0)
        elif reversal['reversal_percent'] > 0.6:
            scores.append(0.7)
        else:
            scores.append(0.4)
            
        # 3. Tamanho do spike adequado
        spike_ticks = float(spike['size'] / Decimal('0.5'))
        if 3 <= spike_ticks <= 10:
            scores.append(1.0)
        elif spike_ticks < 3:
            scores.append(0.5)
        else:
            scores.append(0.7)
            
        return sum(scores) / len(scores) if scores else 0.0
        
    def _identify_stop_level(self, spike: Dict[str, Any]) -> Decimal:
        """Identifica nível provável onde estavam os stops"""
        # Geralmente os stops estão logo além de níveis redondos ou suportes/resistências
        peak = spike['peak_price']
        
        # Arredonda para nível mais próximo
        return Decimal(str(round(float(peak), 0)))