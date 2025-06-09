"""
Detector de Sweep (Varredura de Liquidez)
Identifica movimentos rápidos que varrem múltiplos níveis de preço
"""

from typing import Dict, Any, List, Tuple
from decimal import Decimal
from datetime import datetime, timedelta

from .base import BehaviorDetector
from ..core.models import MarketData, BehaviorDetection, Trade, Side


class SweepDetector(BehaviorDetector):
    """
    Detecta sweeps (varreduras de liquidez)
    
    Características:
    - Movimento rápido através de múltiplos níveis
    - Alto volume em curto período
    - Geralmente limpa stops ou ordens limitadas
    - Pode indicar início de movimento forte
    """
    
    @property
    def behavior_type(self) -> str:
        return "sweep"
        
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Parâmetros específicos
        self.min_price_levels = config.get('min_price_levels', 3)  # Níveis mínimos varridos
        self.max_time_seconds = config.get('max_time_seconds', 10)  # Tempo máximo do sweep
        self.min_volume_spike = config.get('min_volume_spike', 2.0)  # Volume 2x acima média
        self.tick_size = Decimal(str(config.get('tick_size', 0.5)))
        
    async def detect(self, market_data: MarketData) -> BehaviorDetection:
        """Detecta padrão de sweep"""
        # Atualiza histórico
        self.update_history(market_data)
        
        if len(self.historical_data) < 2:
            return self.create_detection(False, 0.0)
            
        # Analisa sweep
        sweep_result = self._analyze_sweep(market_data)
        
        if sweep_result['detected']:
            metadata = {
                'sweep_direction': sweep_result['direction'].value,
                'levels_swept': sweep_result['levels_swept'],
                'price_range': str(sweep_result['price_range']),
                'time_taken': sweep_result['time_seconds'],
                'volume_ratio': sweep_result['volume_ratio'],
                'start_price': str(sweep_result['start_price']),
                'end_price': str(sweep_result['end_price']),
                'aggressive_ratio': sweep_result['aggressive_ratio']
            }
            
            # --- CORREÇÃO APLICADA ---
            # A direção do sinal é a OPOSTA da direção do sweep.
            # Sweep para baixo (SELL) -> sinal de reversão para ALTA (BUY).
            # Sweep para cima (BUY) -> sinal de reversão para BAIXA (SELL).
            direction_result = Side.BUY if sweep_result['direction'] == Side.SELL else Side.SELL

            detection = self.create_detection(
                True,
                sweep_result['confidence'],
                metadata
            )
            detection.direction = direction_result
            return detection
            
        return self.create_detection(False, 0.0)
        
    def _analyze_sweep(self, current_data: MarketData) -> Dict[str, Any]:
        """Analisa se há sweep acontecendo"""
        # Pega trades recentes
        recent_trades = self.get_recent_trades(self.max_time_seconds)
        
        if len(recent_trades) < 5:
            return {'detected': False}
            
        # Identifica movimentos rápidos
        rapid_moves = self._find_rapid_price_movements(recent_trades)
        
        if not rapid_moves:
            return {'detected': False}
            
        # Analisa o movimento mais recente
        latest_move = rapid_moves[-1]
        
        # Verifica se é um sweep válido
        if latest_move['levels_crossed'] < self.min_price_levels:
            return {'detected': False}
            
        # Calcula características do sweep
        sweep_analysis = self._analyze_sweep_characteristics(
            latest_move['trades'],
            current_data
        )
        
        # Calcula confiança
        confidence_signals = {
            'speed': latest_move['speed_score'],
            'volume_spike': sweep_analysis['volume_ratio'] / self.min_volume_spike,
            'directional_strength': sweep_analysis['directional_strength'],
            'book_impact': self._analyze_book_impact(current_data, latest_move['direction'])
        }
        
        confidence = self.calculate_confidence(confidence_signals)
        
        if confidence < self.min_confidence:
            return {'detected': False}
            
        return {
            'detected': True,
            'confidence': confidence,
            'direction': latest_move['direction'],
            'levels_swept': latest_move['levels_crossed'],
            'price_range': latest_move['price_range'],
            'time_seconds': latest_move['time_taken'],
            'volume_ratio': sweep_analysis['volume_ratio'],
            'start_price': latest_move['start_price'],
            'end_price': latest_move['end_price'],
            'aggressive_ratio': sweep_analysis['aggressive_ratio']
        }
        
    def _find_rapid_price_movements(self, trades: List[Trade]) -> List[Dict[str, Any]]:
        """Encontra movimentos rápidos de preço"""
        rapid_moves = []
        
        # Agrupa trades por janelas de tempo
        time_windows = self._create_time_windows(trades, window_size=self.max_time_seconds)
        
        for window_trades in time_windows:
            if len(window_trades) < 3:
                continue
                
            # Analisa movimento na janela
            start_price = window_trades[0].price
            end_price = window_trades[-1].price
            price_range = abs(end_price - start_price)
            
            # Conta níveis cruzados
            levels_crossed = int(price_range / self.tick_size)
            
            if levels_crossed >= self.min_price_levels:
                # Calcula velocidade
                time_taken = (window_trades[-1].timestamp - window_trades[0].timestamp).total_seconds()
                if time_taken > 0:
                    speed = float(price_range) / time_taken
                else:
                    speed = float('inf')
                    
                # Determina direção
                direction = Side.BUY if end_price > start_price else Side.SELL
                
                # Score de velocidade (normalizado)
                speed_score = min(1.0, speed / (float(self.tick_size) * 2))  # 2 ticks/segundo = score 1.0
                
                rapid_moves.append({
                    'trades': window_trades,
                    'start_price': start_price,
                    'end_price': end_price,
                    'price_range': price_range,
                    'levels_crossed': levels_crossed,
                    'time_taken': time_taken,
                    'speed': speed,
                    'speed_score': speed_score,
                    'direction': direction
                })
                
        return rapid_moves
        
    def _create_time_windows(self, trades: List[Trade], window_size: int) -> List[List[Trade]]:
        """Cria janelas de tempo deslizantes"""
        if not trades:
            return []
            
        windows = []
        
        for i in range(len(trades)):
            window_start = trades[i].timestamp
            window_end = window_start + timedelta(seconds=window_size)
            
            # Coleta trades na janela
            window_trades = []
            for j in range(i, len(trades)):
                if trades[j].timestamp <= window_end:
                    window_trades.append(trades[j])
                else:
                    break
                    
            if len(window_trades) >= 3:
                windows.append(window_trades)
                
        return windows
        
    def _analyze_sweep_characteristics(
        self,
        sweep_trades: List[Trade],
        current_data: MarketData
    ) -> Dict[str, Any]:
        """Analisa características detalhadas do sweep"""
        # Volume
        sweep_volume = sum(t.volume for t in sweep_trades)
        
        # Volume médio histórico
        historical_avg = self._calculate_historical_average_volume()
        volume_ratio = sweep_volume / historical_avg if historical_avg > 0 else 1.0
        
        # Agressividade
        aggressive_trades = 0
        total_aggressive_volume = 0
        
        # Determina direção do sweep
        start_price = sweep_trades[0].price
        end_price = sweep_trades[-1].price
        sweep_direction = Side.BUY if end_price > start_price else Side.SELL
        
        for trade in sweep_trades:
            if trade.aggressor == sweep_direction:
                aggressive_trades += 1
                total_aggressive_volume += trade.volume
                
        aggressive_ratio = aggressive_trades / len(sweep_trades) if sweep_trades else 0
        
        # Força direcional
        directional_strength = (total_aggressive_volume / sweep_volume) if sweep_volume > 0 else 0
        
        return {
            'volume_ratio': volume_ratio,
            'aggressive_ratio': aggressive_ratio,
            'directional_strength': directional_strength,
            'total_volume': sweep_volume,
            'aggressive_volume': total_aggressive_volume
        }
        
    def _calculate_historical_average_volume(self) -> float:
        """Calcula volume médio histórico por período"""
        if len(self.historical_data) < 10:
            return 100  # Default
            
        # Volume médio dos últimos períodos
        period_volumes = []
        
        for data in self.historical_data[-20:]:
            period_volume = sum(t.volume for t in data.trades)
            if period_volume > 0:
                period_volumes.append(period_volume)
                
        if period_volumes:
            return sum(period_volumes) / len(period_volumes)
        else:
            return 100
            
    def _analyze_book_impact(self, market_data: MarketData, sweep_direction: Side) -> float:
        """Analisa impacto do sweep no book"""
        if not market_data.book:
            return 0.5
            
        # Verifica se o lado varrido está com pouca liquidez
        if sweep_direction == Side.BUY:
            # Sweep comprador varreu asks
            relevant_levels = market_data.book.asks[:5]
        else:
            # Sweep vendedor varreu bids
            relevant_levels = market_data.book.bids[:5]
            
        if not relevant_levels:
            return 0.8  # Book vazio indica sweep bem-sucedido
            
        # Calcula liquidez remanescente
        total_volume = sum(level.volume for level in relevant_levels)
        
        # Volume médio por nível
        avg_volume_per_level = total_volume / len(relevant_levels) if relevant_levels else 0
        
        # Se volume está baixo, sweep foi efetivo
        historical_avg = self._calculate_historical_average_volume()
        
        if historical_avg > 0:
            liquidity_ratio = avg_volume_per_level / (historical_avg / 10)  # Assumindo 10 níveis normalmente
            
            # Inverte: menos liquidez = maior impacto
            impact_score = max(0, 1 - liquidity_ratio)
        else:
            impact_score = 0.5
            
        return impact_score