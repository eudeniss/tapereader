"""
Classificador de Regime de Mercado
Identifica o regime atual do mercado para ajustar estratégias dinamicamente
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import deque
from enum import Enum

from src.core.logger import get_logger
from src.core.models import BookLevel, Side


class MarketRegime(str, Enum):
    """Tipos de regime de mercado"""
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    RANGING = "RANGING"
    VOLATILE = "VOLATILE"
    BREAKOUT = "BREAKOUT"
    CHOPPY = "CHOPPY"
    UNKNOWN = "UNKNOWN"


class RegimeClassifier:
    """Classifica o regime de mercado atual"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        
        # Histórico de dados
        self.price_history = deque(maxlen=1000)
        self.volume_history = deque(maxlen=1000)
        self.volatility_history = deque(maxlen=100)
        
        # Estado atual
        self.current_regime = MarketRegime.UNKNOWN
        self.regime_confidence = 0.0
        self.last_classification = None
        
        # Parâmetros
        self.lookback_period = 30  # minutos
        self.trend_threshold = 0.002  # 0.2%
        self.volatility_window = 20
        
    def classify(self, market_data: Dict) -> MarketRegime:
        """Classifica o regime de mercado baseado nos dados atuais"""
        try:
            # Extrai dados relevantes
            prices = self._extract_prices(market_data)
            volumes = self._extract_volumes(market_data)
            book_imbalance = self._calculate_book_imbalance(market_data)
            
            if not prices or len(prices) < 10:
                return self.current_regime  # Mantém regime atual se dados insuficientes
            
            # Calcula métricas
            trend_strength, trend_direction = self._calculate_trend(prices)
            volatility = self._calculate_volatility(prices)
            volume_profile = self._analyze_volume_profile(volumes)
            price_action_quality = self._analyze_price_action(prices)
            
            # Armazena volatilidade
            self.volatility_history.append(volatility)
            
            # Classifica regime
            regime = self._determine_regime(
                trend_strength,
                trend_direction,
                volatility,
                volume_profile,
                price_action_quality,
                book_imbalance
            )
            
            # Atualiza estado
            if regime != self.current_regime:
                self.logger.info(
                    f"Regime mudou: {self.current_regime.value} → {regime.value} "
                    f"(Confiança: {self.regime_confidence:.1%})"
                )
            
            self.current_regime = regime
            self.last_classification = datetime.now()
            
            return regime
            
        except Exception as e:
            self.logger.error(f"Erro ao classificar regime: {e}")
            return MarketRegime.UNKNOWN
    
    def _extract_prices(self, market_data: Dict) -> List[float]:
        """Extrai preços dos dados de mercado"""
        prices = []
        
        # De trades recentes
        if 'price_data' in market_data:
            for trade in market_data['price_data'][-100:]:
                if isinstance(trade, dict) and 'price' in trade and trade['price'] is not None:
                    try:
                        prices.append(float(trade['price']))
                    except (ValueError, TypeError):
                        pass
        
        # De market depth (last_price)
        if 'book_data' in market_data:
            last_price = market_data['book_data'].get('last_price')
            if last_price is not None:
                try:
                    prices.append(float(last_price))
                except (ValueError, TypeError):
                    pass
                    
        return prices
    
    def _extract_volumes(self, market_data: Dict) -> List[float]:
        """Extrai volumes dos dados de mercado"""
        volumes = []
        
        if 'volume_data' in market_data:
            if isinstance(market_data['volume_data'], dict):
                for level, vol in market_data['volume_data'].items():
                    if isinstance(vol, (int, float)):
                        volumes.append(float(vol))
            elif isinstance(market_data['volume_data'], list):
                volumes = [float(v) for v in market_data['volume_data'] if isinstance(v, (int, float))]
                
        return volumes
    
    def _calculate_book_imbalance(self, market_data: Dict) -> float:
        """Calcula desbalanceamento do book"""
        try:
            book = market_data.get('book_data')
            if not book:
                return 0.0
            
            bids = book.get('bids', [])
            asks = book.get('asks', [])

            # Acessa o atributo '.volume' do objeto BookLevel
            bid_volume = sum(level.volume for level in bids[:5])
            ask_volume = sum(level.volume for level in asks[:5])
            
            total_volume = bid_volume + ask_volume
            if total_volume == 0:
                return 0.0
                
            imbalance = (bid_volume - ask_volume) / total_volume
            return imbalance
            
        except Exception as e:
            self.logger.debug(f"Erro ao calcular imbalance do book: {e}")
            return 0.0
    
    def _calculate_trend(self, prices: List[float]) -> Tuple[float, int]:
        """Calcula força e direção da tendência"""
        if len(prices) < 2:
            return 0.0, 0
            
        x = np.arange(len(prices))
        y = np.array(prices)
        
        A = np.vstack([x, np.ones(len(x))]).T
        slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
        
        avg_price = np.mean(prices)
        normalized_slope = slope / avg_price if avg_price > 0 else 0
        
        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        if abs(normalized_slope) < self.trend_threshold:
            direction = 0
        else:
            direction = 1 if normalized_slope > 0 else -1
            
        return r_squared, direction
    
    def _calculate_volatility(self, prices: List[float]) -> float:
        """Calcula volatilidade normalizada"""
        if len(prices) < 2:
            return 0.0
            
        returns = np.diff(prices) / prices[:-1]
        volatility = np.std(returns) if len(returns) > 0 else 0.0
        
        return volatility
    
    def _analyze_volume_profile(self, volumes: List[float]) -> Dict:
        """Analisa perfil de volume"""
        if not volumes:
            return {'avg': 0, 'trend': 0, 'spikes': 0}
            
        avg_volume = np.mean(volumes)
        
        if len(volumes) > 5:
            recent_avg = np.mean(volumes[-5:])
            older_avg = np.mean(volumes[:-5])
            volume_trend = (recent_avg - older_avg) / older_avg if older_avg > 0 else 0
        else:
            volume_trend = 0
            
        if len(volumes) > 10:
            volume_std = np.std(volumes)
            spikes = sum(1 for v in volumes if v > avg_volume + 2 * volume_std)
        else:
            spikes = 0
            
        return {'avg': avg_volume, 'trend': volume_trend, 'spikes': spikes}
    
    def _analyze_price_action(self, prices: List[float]) -> float:
        """Analisa qualidade da ação do preço"""
        if len(prices) < 10:
            return 0.5
            
        changes = np.diff(prices)
        if len(changes) < 2: return 0.5

        direction_changes = sum(1 for i in range(1, len(changes)) if changes[i] * changes[i-1] < 0)
        quality = 1 - (direction_changes / (len(changes) -1))
        
        return quality
    
    def _determine_regime(
        self,
        trend_strength: float,
        trend_direction: int,
        volatility: float,
        volume_profile: Dict,
        price_action_quality: float,
        book_imbalance: float
    ) -> MarketRegime:
        """Determina regime baseado nas métricas"""
        
        avg_volatility = np.mean(list(self.volatility_history)) if self.volatility_history else volatility
        relative_volatility = volatility / avg_volatility if avg_volatility > 0 else 1.0
        
        if relative_volatility > 2.0:
            self.regime_confidence = min(0.9, relative_volatility / 3.0)
            return MarketRegime.VOLATILE
            
        if (trend_strength > 0.7 and 
            volume_profile['trend'] > 0.5 and 
            abs(book_imbalance) > 0.3):
            self.regime_confidence = trend_strength
            return MarketRegime.BREAKOUT
            
        if trend_strength > 0.6 and trend_direction != 0:
            self.regime_confidence = trend_strength
            return MarketRegime.TRENDING_UP if trend_direction > 0 else MarketRegime.TRENDING_DOWN
                
        if relative_volatility < 0.7 and trend_strength < 0.3:
            self.regime_confidence = 1 - trend_strength
            return MarketRegime.RANGING
            
        if price_action_quality < 0.4:
            self.regime_confidence = 1 - price_action_quality
            return MarketRegime.CHOPPY
            
        self.regime_confidence = 0.5
        return MarketRegime.RANGING
    
    def get_regime_adjustments(self, regime: MarketRegime) -> Dict:
        """Retorna ajustes recomendados para o regime"""
        adjustments = {
            MarketRegime.TRENDING_UP: {'favored_behaviors': ['momentum', 'breakout', 'institutional'], 'avoid_behaviors': ['mean_reversion', 'fade'], 'confidence_multiplier': 1.1, 'stop_distance_multiplier': 1.2, 'target_multiplier': 1.3},
            MarketRegime.TRENDING_DOWN: {'favored_behaviors': ['momentum', 'breakout', 'exhaustion'], 'avoid_behaviors': ['mean_reversion', 'fade'], 'confidence_multiplier': 1.1, 'stop_distance_multiplier': 1.2, 'target_multiplier': 1.3},
            MarketRegime.RANGING: {'favored_behaviors': ['support_resistance', 'absorption', 'fade'], 'avoid_behaviors': ['breakout', 'momentum'], 'confidence_multiplier': 0.95, 'stop_distance_multiplier': 0.8, 'target_multiplier': 0.9},
            MarketRegime.VOLATILE: {'favored_behaviors': ['stop_hunt', 'exhaustion', 'divergence'], 'avoid_behaviors': ['breakout'], 'confidence_multiplier': 0.9, 'stop_distance_multiplier': 1.5, 'target_multiplier': 1.1},
            MarketRegime.BREAKOUT: {'favored_behaviors': ['breakout', 'momentum', 'institutional'], 'avoid_behaviors': ['fade', 'mean_reversion'], 'confidence_multiplier': 1.2, 'stop_distance_multiplier': 1.3, 'target_multiplier': 1.5},
            MarketRegime.CHOPPY: {'favored_behaviors': ['fade', 'mean_reversion'], 'avoid_behaviors': ['momentum', 'breakout'], 'confidence_multiplier': 0.8, 'stop_distance_multiplier': 0.7, 'target_multiplier': 0.8},
            MarketRegime.UNKNOWN: {'favored_behaviors': [], 'avoid_behaviors': [], 'confidence_multiplier': 0.7, 'stop_distance_multiplier': 1.0, 'target_multiplier': 1.0}
        }
        return adjustments.get(regime, adjustments[MarketRegime.UNKNOWN])
    
    def get_statistics(self) -> Dict:
        """Retorna estatísticas do classificador"""
        return {
            'current_regime': self.current_regime.value,
            'regime_confidence': round(self.regime_confidence, 3),
            'last_classification': self.last_classification.isoformat() if self.last_classification else None,
            'price_history_size': len(self.price_history),
            'avg_volatility': round(np.mean(list(self.volatility_history)), 6) if self.volatility_history else 0
        }