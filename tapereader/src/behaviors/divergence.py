"""
Detector de Divergência
Identifica divergências entre preço e volume ou entre ativos
OTIMIZADO: Usa módulo centralizado de estatísticas
"""

from typing import Dict, Any, List, Optional, Tuple
from decimal import Decimal
from datetime import datetime, timedelta
import statistics
import numpy as np

from .base import BehaviorDetector
from ..core.models import MarketData, BehaviorDetection, Trade, Side

# OTIMIZAÇÃO: Importa funções do módulo centralizado
# Supondo que você tenha este módulo, caso contrário, comente ou ajuste
# from ..utils.statistics import (
#     calculate_trend,
#     calculate_correlation,
#     calculate_returns,
#     calculate_volatility
# )


class DivergenceDetector(BehaviorDetector):
    """
    Detecta divergências no mercado
    
    Tipos de divergência:
    - Preço vs Volume: Movimento sem suporte de volume
    - Alta vs Baixa: Novos extremos com menos força
    - Inter-ativos: DOLFUT vs WDOFUT divergindo
    
    OTIMIZADO: Usa funções estatísticas centralizadas para melhor performance
    """
    
    @property
    def behavior_type(self) -> str:
        return "divergence"
        
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Parâmetros
        self.lookback_periods = config.get('lookback_periods', 10)
        self.divergence_threshold = config.get('divergence_threshold', 0.3)  # 30% divergência
        self.volume_correlation_threshold = config.get('volume_correlation_threshold', 0.5)
        
        # Cache para análise inter-ativos
        self.cross_asset_data = {
            'DOLFUT': [],
            'WDOFUT': []
        }
        
    async def detect(self, market_data: MarketData) -> BehaviorDetection:
        """Versão assíncrona que chama a implementação síncrona."""
        return self.detect_sync(market_data)

    def detect_sync(self, market_data: MarketData) -> BehaviorDetection:
        """Detecta divergências"""
        # Atualiza histórico
        self.update_history(market_data)
        
        # Atualiza cache cross-asset
        self._update_cross_asset_cache(market_data)
        
        if len(self.historical_data) < self.lookback_periods:
            return self.create_detection(False, 0.0)
            
        # Analisa divergências
        divergence_result = self._analyze_divergences(market_data)
        
        if divergence_result['detected']:
            metadata = {
                'divergence_type': divergence_result['type'],
                'divergence_strength': divergence_result['strength'],
                'price_trend': divergence_result['price_trend'],
                'volume_trend': divergence_result['volume_trend'],
                'correlation': divergence_result['correlation'],
                'periods_diverging': divergence_result['periods'],
                'reversal_probability': divergence_result['reversal_probability']
            }
            
            # Adiciona dados específicos por tipo
            if divergence_result['type'] == 'intermarket':
                metadata['leader_asset'] = divergence_result.get('leader_asset')
                metadata['laggard_asset'] = divergence_result.get('laggard_asset')

            # --- CORREÇÃO APLICADA ---
            # A direção é baseada no tipo da divergência (bullish ou bearish).
            direction_result = Side.NEUTRAL
            if 'bullish' in divergence_result['type']:
                direction_result = Side.BUY
            elif 'bearish' in divergence_result['type']:
                direction_result = Side.SELL
            
            detection = self.create_detection(
                True,
                divergence_result['confidence'],
                metadata
            )
            detection.direction = direction_result
            return detection
            
        return self.create_detection(False, 0.0)
        
    def _update_cross_asset_cache(self, market_data: MarketData):
        """Atualiza cache para análise entre ativos"""
        asset = market_data.asset
        
        # Mantém dados recentes dos dois ativos
        if market_data.trades:
            avg_price = sum(t.price for t in market_data.trades) / len(market_data.trades)
            total_volume = sum(t.volume for t in market_data.trades)
            
            self.cross_asset_data[asset].append({
                'timestamp': market_data.timestamp,
                'price': avg_price,
                'volume': total_volume
            })
            
            # Mantém apenas dados recentes
            if len(self.cross_asset_data[asset]) > 50:
                self.cross_asset_data[asset].pop(0)
                
    def _analyze_divergences(self, current_data: MarketData) -> Dict[str, Any]:
        """Analisa todos os tipos de divergência"""
        results = []
        
        # 1. Divergência Preço vs Volume
        price_volume_div = self._analyze_price_volume_divergence(current_data)
        if price_volume_div['detected']:
            results.append(price_volume_div)
            
        # 2. Divergência de Força (momentum)
        momentum_div = self._analyze_momentum_divergence(current_data)
        if momentum_div['detected']:
            results.append(momentum_div)
            
        # 3. Divergência entre ativos
        if len(self.cross_asset_data['DOLFUT']) > 5 and len(self.cross_asset_data['WDOFUT']) > 5:
            intermarket_div = self._analyze_intermarket_divergence(current_data.asset)
            if intermarket_div['detected']:
                results.append(intermarket_div)
                
        # Retorna a divergência mais forte
        if results:
            return max(results, key=lambda x: x['confidence'])
        else:
            return {'detected': False}
            
    def _analyze_price_volume_divergence(self, current_data: MarketData) -> Dict[str, Any]:
        """Analisa divergência entre preço e volume"""
        price_data = []
        volume_data = []
        
        for data in self.historical_data[-self.lookback_periods:]:
            if data.trades:
                avg_price = sum(t.price for t in data.trades) / len(data.trades)
                total_volume = sum(t.volume for t in data.trades)
                
                price_data.append(float(avg_price))
                volume_data.append(total_volume)
                
        if len(price_data) < 3: return {'detected': False}
            
        price_trend = np.polyfit(range(len(price_data)), price_data, 1)[0]
        volume_trend = np.polyfit(range(len(volume_data)), volume_data, 1)[0]
        correlation = np.corrcoef(price_data, volume_data)[0, 1] if len(price_data) > 1 else 0
        
        if price_trend > 0.1 and volume_trend < -0.1:
            divergence_type = 'bearish_price_volume'
            strength = abs(price_trend - volume_trend)
        elif price_trend < -0.1 and volume_trend > 0.1:
            divergence_type = 'bullish_price_volume'
            strength = abs(price_trend - volume_trend)
        else:
            return {'detected': False}
            
        if strength < self.divergence_threshold: return {'detected': False}
            
        periods_diverging = self._count_diverging_periods(price_data, volume_data)
        reversal_prob = self._calculate_reversal_probability(divergence_type, strength, periods_diverging)
        
        confidence = self.calculate_confidence({
            'divergence_strength': min(1.0, strength),
            'correlation_weakness': 1 - abs(correlation),
            'persistence': min(1.0, periods_diverging / 5)
        })
        
        return {
            'detected': True, 'confidence': confidence, 'type': divergence_type,
            'strength': strength, 'price_trend': price_trend, 'volume_trend': volume_trend,
            'correlation': correlation, 'periods': periods_diverging, 'reversal_probability': reversal_prob
        }
        
    def _analyze_momentum_divergence(self, current_data: MarketData) -> Dict[str, Any]:
        """Analisa divergência de momentum (força do movimento)"""
        swings = self._identify_price_swings()
        if len(swings) < 4: return {'detected': False}
            
        highs = [s for s in swings if s['type'] == 'high']
        lows = [s for s in swings if s['type'] == 'low']
        
        if len(highs) >= 2:
            high_divergence = self._check_swing_divergence(highs, 'high')
            if high_divergence['detected']: return high_divergence
                
        if len(lows) >= 2:
            low_divergence = self._check_swing_divergence(lows, 'low')
            if low_divergence['detected']: return low_divergence
                
        return {'detected': False}
        
    def _analyze_intermarket_divergence(self, current_asset: str) -> Dict[str, Any]:
        """Analisa divergência entre DOLFUT e WDOFUT"""
        dol_data = self.cross_asset_data['DOLFUT'][-10:]
        wdo_data = self.cross_asset_data['WDOFUT'][-10:]
        if len(dol_data) < 5 or len(wdo_data) < 5: return {'detected': False}
            
        synced_data = self._sync_cross_asset_data(dol_data, wdo_data)
        if len(synced_data) < 3: return {'detected': False}
            
        dol_prices = [d['dol_price'] for d in synced_data]
        wdo_prices = [d['wdo_price'] for d in synced_data]
        
        dol_trend = np.polyfit(range(len(dol_prices)), dol_prices, 1)[0]
        wdo_trend = np.polyfit(range(len(wdo_prices)), wdo_prices, 1)[0]
        correlation = np.corrcoef(dol_prices, wdo_prices)[0, 1] if len(dol_prices) > 1 else 0
        
        trend_difference = abs(dol_trend - wdo_trend)
        if trend_difference < self.divergence_threshold: return {'detected': False}
            
        leader = 'DOLFUT' if abs(dol_trend) > abs(wdo_trend) else 'WDOFUT'
        laggard = 'WDOFUT' if leader == 'DOLFUT' else 'DOLFUT'
        
        confidence = self.calculate_confidence({
            'trend_divergence': min(1.0, trend_difference),
            'low_correlation': 1 - correlation if correlation > 0 else 1.0
        })

        divergence_type = 'bearish_intermarket' if dol_trend > wdo_trend else 'bullish_intermarket'
        
        return {
            'detected': True, 'confidence': confidence, 'type': divergence_type,
            'strength': trend_difference, 'price_trend': dol_trend if current_asset == 'DOLFUT' else wdo_trend,
            'volume_trend': 0, 'correlation': correlation, 'periods': len(synced_data),
            'reversal_probability': 0.6 if correlation < 0.3 else 0.4,
            'leader_asset': leader, 'laggard_asset': laggard
        }
        
    def _count_diverging_periods(self, price_data: List[float], volume_data: List[float]) -> int:
        if len(price_data) < 2 or len(volume_data) < 2: return 0
        diverging_count = 0
        for i in range(1, min(len(price_data), len(volume_data))):
            price_change = price_data[i] - price_data[i-1]
            volume_change = volume_data[i] - volume_data[i-1]
            if (price_change > 0 and volume_change < 0) or (price_change < 0 and volume_change > 0):
                diverging_count += 1
        return diverging_count
        
    def _calculate_reversal_probability(self, divergence_type: str, strength: float, periods: int) -> float:
        base_probability = 0.6 if 'bearish' in divergence_type else 0.55
        strength_adjustment = min(0.2, strength * 0.3)
        persistence_adjustment = min(0.1, periods * 0.02)
        return min(0.9, base_probability + strength_adjustment + persistence_adjustment)
        
    def _identify_price_swings(self) -> List[Dict[str, Any]]:
        swings = []
        price_series = []
        for data in self.historical_data[-20:]:
            if data.trades:
                prices = [t.price for t in data.trades]
                price_series.append({
                    'high': max(prices), 'low': min(prices), 'volume': sum(t.volume for t in data.trades),
                    'timestamp': data.timestamp
                })
        if len(price_series) < 5: return []
            
        for i in range(2, len(price_series) - 2):
            is_high = price_series[i]['high'] > price_series[i-1]['high'] and price_series[i]['high'] > price_series[i-2]['high'] and \
                      price_series[i]['high'] > price_series[i+1]['high'] and price_series[i]['high'] > price_series[i+2]['high']
            is_low = price_series[i]['low'] < price_series[i-1]['low'] and price_series[i]['low'] < price_series[i-2]['low'] and \
                     price_series[i]['low'] < price_series[i+1]['low'] and price_series[i]['low'] < price_series[i+2]['low']
            if is_high:
                swings.append({'type': 'high', 'price': price_series[i]['high'], 'volume': price_series[i]['volume'], 'index': i})
            if is_low:
                swings.append({'type': 'low', 'price': price_series[i]['low'], 'volume': price_series[i]['volume'], 'index': i})
        return swings
        
    def _check_swing_divergence(self, swings: List[Dict[str, Any]], swing_type: str) -> Dict[str, Any]:
        if len(swings) < 2: return {'detected': False}
        prev_swing, curr_swing = swings[-2], swings[-1]
        price_change = float(curr_swing['price'] - prev_swing['price'])
        volume_change = curr_swing['volume'] - prev_swing['volume']
        
        divergence_type, strength = None, 0
        if swing_type == 'high' and price_change > 0 and volume_change < 0:
            divergence_type = 'bearish_momentum'
            strength = abs(volume_change / prev_swing['volume']) if prev_swing['volume'] > 0 else 1.0
        elif swing_type == 'low' and price_change < 0 and volume_change > 0:
            divergence_type = 'bullish_momentum'
            strength = abs(volume_change / prev_swing['volume']) if prev_swing['volume'] > 0 else 1.0
        else:
            return {'detected': False}
            
        if strength < 0.2: return {'detected': False}
        confidence = min(0.9, 0.5 + strength)
        
        return {
            'detected': True, 'confidence': confidence, 'type': divergence_type,
            'strength': strength, 'price_trend': price_change, 'volume_trend': volume_change,
            'correlation': -0.5, 'periods': curr_swing['index'] - prev_swing['index'],
            'reversal_probability': 0.7 if strength > 0.5 else 0.6
        }
        
    def _sync_cross_asset_data(self, dol_data: List[Dict], wdo_data: List[Dict]) -> List[Dict]:
        synced = []
        for dol_point in dol_data:
            closest_wdo = min(wdo_data, key=lambda w: abs((w['timestamp'] - dol_point['timestamp']).total_seconds()))
            if abs((closest_wdo['timestamp'] - dol_point['timestamp']).total_seconds()) < 60:
                synced.append({
                    'timestamp': dol_point['timestamp'], 'dol_price': float(dol_point['price']),
                    'wdo_price': float(closest_wdo['price']), 'dol_volume': dol_point['volume'],
                    'wdo_volume': closest_wdo['volume']
                })
        return synced