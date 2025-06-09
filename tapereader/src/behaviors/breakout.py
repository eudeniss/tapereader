"""
Detector de Breakout
Identifica rompimentos de níveis importantes
"""

from typing import Dict, Any, List, Optional, Tuple
from decimal import Decimal
from datetime import datetime, timedelta

from .base import BehaviorDetector
from ..core.models import MarketData, BehaviorDetection, Trade, Side


class BreakoutDetector(BehaviorDetector):
    """
    Detecta breakouts (rompimentos)
    
    Características:
    - Rompimento de nível importante
    - Aumento de volume no rompimento
    - Momentum após o rompimento
    - Possível retest do nível
    """
    
    @property
    def behavior_type(self) -> str:
        return "breakout"
        
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Parâmetros
        self.consolidation_periods = config.get('consolidation_periods', 10)
        self.breakout_threshold = Decimal(str(config.get('breakout_threshold', 1.0)))
        self.volume_spike_ratio = config.get('volume_spike_ratio', 1.5)
        self.confirmation_bars = config.get('confirmation_bars', 3)
        self.retest_tolerance = Decimal(str(config.get('retest_tolerance', 0.5)))
        
    async def detect(self, market_data: MarketData) -> BehaviorDetection:
        """Detecta breakout"""
        # Atualiza histórico
        self.update_history(market_data)
        
        if len(self.historical_data) < self.consolidation_periods:
            return self.create_detection(False, 0.0)
            
        # Analisa breakout
        breakout_result = self._analyze_breakout(market_data)
        
        if breakout_result['detected']:
            metadata = {
                'breakout_type': breakout_result['type'],
                'breakout_direction': breakout_result['direction'].value,
                'level_broken': str(breakout_result['level']),
                'breakout_strength': breakout_result['strength'],
                'volume_confirmation': breakout_result['volume_confirmed'],
                'consolidation_time': breakout_result['consolidation_periods'],
                'breakout_size': str(breakout_result['breakout_size']),
                'retest_status': breakout_result['retest_status']
            }
            
            # --- CORREÇÃO APLICADA ---
            # A direção é a do próprio breakout.
            direction_result = breakout_result['direction']

            detection = self.create_detection(
                True,
                breakout_result['confidence'],
                metadata
            )
            detection.direction = direction_result
            return detection
            
        return self.create_detection(False, 0.0)
        
    def _analyze_breakout(self, current_data: MarketData) -> Dict[str, Any]:
        """Analisa se há breakout"""
        # Identifica consolidação prévia
        consolidation = self._find_consolidation_range()
        
        if not consolidation['found']:
            return {'detected': False}
            
        # Verifica se houve rompimento
        breakout = self._check_breakout(consolidation, current_data)
        
        if not breakout['occurred']:
            return {'detected': False}
            
        # Analisa qualidade do breakout
        quality_analysis = self._analyze_breakout_quality(
            breakout,
            consolidation,
            current_data
        )
        
        # Verifica retest (se aplicável)
        retest_analysis = self._check_retest(breakout, current_data)
        
        # Calcula confiança
        confidence_signals = {
            'breakout_clarity': breakout['clarity'],
            'volume_confirmation': quality_analysis['volume_score'],
            'momentum_continuation': quality_analysis['momentum_score'],
            'pattern_quality': consolidation['quality']
        }
        
        confidence = self.calculate_confidence(confidence_signals)
        
        if confidence < self.min_confidence:
            return {'detected': False}
            
        return {
            'detected': True,
            'confidence': confidence,
            'type': self._classify_breakout_type(consolidation),
            'direction': breakout['direction'],
            'level': breakout['level'],
            'strength': quality_analysis['strength'],
            'volume_confirmed': quality_analysis['volume_confirmed'],
            'consolidation_periods': consolidation['duration'],
            'breakout_size': breakout['size'],
            'retest_status': retest_analysis['status']
        }
        
    def _find_consolidation_range(self) -> Dict[str, Any]:
        """Encontra range de consolidação"""
        lookback_data = self.historical_data[-self.consolidation_periods:]
        
        if len(lookback_data) < self.consolidation_periods // 2:
            return {'found': False}
            
        # Coleta todos os preços
        all_prices = []
        for data in lookback_data:
            all_prices.extend([t.price for t in data.trades])
            
        if not all_prices:
            return {'found': False}
            
        # Define range
        high = max(all_prices)
        low = min(all_prices)
        range_size = high - low
        
        # Verifica se é uma consolidação válida (range estreito)
        avg_price = sum(all_prices) / len(all_prices) if all_prices else Decimal('0')
        if avg_price == 0: return {'found': False}
        range_percent = float(range_size / avg_price)
        
        # Consolidação se range < 2% do preço
        if range_percent > 0.02:
            return {'found': False}
            
        # Analisa distribuição de preços no range
        price_distribution = self._analyze_price_distribution(all_prices, high, low)
        
        # Qualidade da consolidação
        quality = self._calculate_consolidation_quality(
            price_distribution,
            len(lookback_data)
        )
        
        return {
            'found': True,
            'high': high,
            'low': low,
            'range': range_size,
            'center': (high + low) / 2,
            'duration': len(lookback_data),
            'quality': quality,
            'distribution': price_distribution
        }
        
    def _analyze_price_distribution(
        self,
        prices: List[Decimal],
        high: Decimal,
        low: Decimal
    ) -> Dict[str, float]:
        """Analisa como os preços estão distribuídos no range"""
        if high == low:
            return {'upper_third': 0.33, 'middle_third': 0.34, 'lower_third': 0.33}
            
        range_size = high - low
        upper_threshold = high - range_size / 3
        lower_threshold = low + range_size / 3
        
        upper_count = sum(1 for p in prices if p >= upper_threshold)
        lower_count = sum(1 for p in prices if p <= lower_threshold)
        middle_count = len(prices) - upper_count - lower_count
        
        total = len(prices)
        
        return {
            'upper_third': upper_count / total,
            'middle_third': middle_count / total,
            'lower_third': lower_count / total
        }
        
    def _calculate_consolidation_quality(
        self,
        distribution: Dict[str, float],
        periods: int
    ) -> float:
        """Calcula qualidade da consolidação"""
        # Boa consolidação tem distribuição equilibrada
        ideal_distribution = 0.333
        distribution_score = 1 - sum(
            abs(v - ideal_distribution) for v in distribution.values()
        ) / 2
        
        # Mais períodos = melhor consolidação
        duration_score = min(1.0, periods / 20)
        
        quality = (distribution_score * 0.6 + duration_score * 0.4)
        
        return quality
        
    def _check_breakout(
        self,
        consolidation: Dict[str, Any],
        current_data: MarketData
    ) -> Dict[str, Any]:
        """Verifica se houve breakout"""
        if not current_data.trades:
            return {'occurred': False}
            
        current_price = current_data.trades[-1].price
        
        # Breakout para cima
        if current_price > consolidation['high'] + self.breakout_threshold:
            return {
                'occurred': True,
                'direction': Side.BUY,
                'level': consolidation['high'],
                'size': current_price - consolidation['high'],
                'clarity': float(
                    (current_price - consolidation['high']) / 
                    consolidation['range']
                ) if consolidation['range'] > 0 else 1.0
            }
            
        # Breakout para baixo
        elif current_price < consolidation['low'] - self.breakout_threshold:
            return {
                'occurred': True,
                'direction': Side.SELL,
                'level': consolidation['low'],
                'size': consolidation['low'] - current_price,
                'clarity': float(
                    (consolidation['low'] - current_price) / 
                    consolidation['range']
                ) if consolidation['range'] > 0 else 1.0
            }
            
        return {'occurred': False}
        
    def _analyze_breakout_quality(
        self,
        breakout: Dict[str, Any],
        consolidation: Dict[str, Any],
        current_data: MarketData
    ) -> Dict[str, Any]:
        """Analisa qualidade do breakout"""
        # Volume no breakout
        recent_trades = self.get_recent_trades(60)
        breakout_trades = [
            t for t in recent_trades
            if (breakout['direction'] == Side.BUY and t.price > consolidation['high']) or
               (breakout['direction'] == Side.SELL and t.price < consolidation['low'])
        ]
        
        if breakout_trades:
            breakout_volume = sum(t.volume for t in breakout_trades)
            
            # Volume médio durante consolidação
            consolidation_avg_volume = self._calculate_average_volume_in_range(
                consolidation['high'],
                consolidation['low']
            )
            
            volume_ratio = (
                breakout_volume / consolidation_avg_volume 
                if consolidation_avg_volume > 0 else 1.0
            )
            
            volume_confirmed = volume_ratio >= self.volume_spike_ratio
            volume_score = min(1.0, volume_ratio / self.volume_spike_ratio)
        else:
            volume_confirmed = False
            volume_score = 0.0
            
        # Momentum após breakout
        momentum_score = self._calculate_post_breakout_momentum(
            breakout,
            current_data
        )
        
        # Força geral
        strength = (volume_score * 0.5 + momentum_score * 0.5)
        
        return {
            'volume_confirmed': volume_confirmed,
            'volume_score': volume_score,
            'momentum_score': momentum_score,
            'strength': strength
        }
        
    def _calculate_average_volume_in_range(
        self,
        high: Decimal,
        low: Decimal
    ) -> float:
        """Calcula volume médio durante consolidação"""
        total_volume = 0
        periods = 0
        
        for data in self.historical_data[-self.consolidation_periods:]:
            period_volume = 0
            for trade in data.trades:
                if low <= trade.price <= high:
                    period_volume += trade.volume
                    
            if period_volume > 0:
                total_volume += period_volume
                periods += 1
                
        return total_volume / periods if periods > 0 else 100
        
    def _calculate_post_breakout_momentum(
        self,
        breakout: Dict[str, Any],
        current_data: MarketData
    ) -> float:
        """Calcula momentum após breakout"""
        # Verifica se preço continua na direção do breakout
        recent_prices = [t.price for t in current_data.trades[-self.confirmation_bars:]]
        
        if not recent_prices:
            return 0.0
            
        if breakout['direction'] == Side.BUY:
            # Quantos trades estão acima do nível rompido
            above_level = sum(1 for p in recent_prices if p > breakout['level'])
            momentum_score = above_level / len(recent_prices)
        else:
            # Quantos trades estão abaixo do nível rompido
            below_level = sum(1 for p in recent_prices if p < breakout['level'])
            momentum_score = below_level / len(recent_prices)
            
        return momentum_score
        
    def _check_retest(
        self,
        breakout: Dict[str, Any],
        current_data: MarketData
    ) -> Dict[str, Any]:
        """Verifica se houve retest do nível rompido"""
        recent_trades = self.get_recent_trades(120)  # 2 minutos
        
        # Procura por retorno ao nível
        retest_trades = []
        
        for trade in recent_trades:
            distance = abs(trade.price - breakout['level'])
            if distance <= self.retest_tolerance:
                retest_trades.append(trade)
                
        if not retest_trades:
            return {'status': 'no_retest', 'successful': None}
            
        # Verifica se retest foi bem-sucedido
        if breakout['direction'] == Side.BUY:
            # Para breakout de alta, retest bem-sucedido se preço não voltou abaixo
            failed = any(t.price < breakout['level'] - self.breakout_threshold 
                        for t in retest_trades)
            successful = not failed
        else:
            # Para breakout de baixa, retest bem-sucedido se preço não voltou acima
            failed = any(t.price > breakout['level'] + self.breakout_threshold 
                        for t in retest_trades)
            successful = not failed
            
        return {
            'status': 'retest_occurred',
            'successful': successful,
            'retest_count': len(retest_trades)
        }
        
    def _classify_breakout_type(self, consolidation: Dict[str, Any]) -> str:
        """Classifica tipo de breakout baseado no padrão"""
        distribution = consolidation['distribution']
        
        # Triângulo ascendente: mais tempo no topo
        if distribution['upper_third'] > 0.5:
            return 'ascending_triangle'
            
        # Triângulo descendente: mais tempo no fundo
        elif distribution['lower_third'] > 0.5:
            return 'descending_triangle'
            
        # Retângulo: distribuição equilibrada
        elif distribution['middle_third'] > 0.4:
            return 'rectangle'
            
        # Padrão simétrico
        else:
            return 'symmetric'