"""
Detector de Absorção
Identifica quando um lado está absorvendo agressão sem deixar o preço mover
"""

from typing import Dict, Any, List
from decimal import Decimal
from datetime import datetime, timedelta

from .base import BehaviorDetector
from ..core.models import MarketData, BehaviorDetection, OrderBook, Trade, Side


class AbsorptionDetector(BehaviorDetector):
    """
    Detecta padrões de absorção no mercado
    
    Absorção ocorre quando:
    - Alto volume de agressão em uma direção
    - Preço não se move proporcionalmente
    - Indica grande player defendendo nível
    """
    
    @property
    def behavior_type(self) -> str:
        return "absorption"
        
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Parâmetros específicos para absorção
        self.min_volume_ratio = config.get('min_volume_ratio', 2.0)  # Volume 2x acima da média
        self.max_price_change = config.get('max_price_change', 1.0)  # Máximo movimento em ticks
        self.min_trades = config.get('min_trades', 10)
        self.absorption_window = config.get('absorption_window', 20)  # segundos
        
    async def detect(self, market_data: MarketData) -> BehaviorDetection:
        """Detecta padrão de absorção"""
        # Atualiza histórico
        self.update_history(market_data)
        
        # Precisa de dados suficientes
        if len(self.historical_data) < 3:
            return self.create_detection(False, 0.0)
            
        # Analisa trades recentes
        recent_trades = self.get_recent_trades(self.absorption_window)
        
        if len(recent_trades) < self.min_trades:
            return self.create_detection(False, 0.0)
            
        # Detecta absorção
        absorption_result = self._analyze_absorption(recent_trades, market_data)
        
        if absorption_result['detected']:
            metadata = {
                'absorption_side': absorption_result['side'].value, # Usar .value para serialização
                'volume_absorbed': absorption_result['volume_absorbed'],
                'price_resistance': absorption_result['price_resistance'],
                'absorption_price': str(absorption_result['price_level']),
                'aggressor_volume': absorption_result['aggressor_volume'],
                'defender_strength': absorption_result['defender_strength']
            }
            
            # --- CORREÇÃO APLICADA ---
            # A direção é a do lado que está absorvendo (o lado defensor).
            # Se o lado BUY está absorvendo, a expectativa é de alta.
            direction_result = absorption_result['side']

            detection = self.create_detection(
                detected=True, 
                confidence=absorption_result['confidence'],
                metadata=metadata
            )
            detection.direction = direction_result
            return detection
            
        return self.create_detection(False, 0.0)
        
    def _analyze_absorption(self, trades: List[Trade], current_data: MarketData) -> Dict[str, Any]:
        """Analisa se há absorção acontecendo"""
        # Calcula perfil de volume
        volume_profile = self.calculate_volume_profile(trades)
        
        # Verifica se há agressão unidirecional forte
        buy_ratio = volume_profile['buy_ratio']
        
        if buy_ratio > 0.7:  # Forte agressão compradora
            aggressor_side = Side.BUY
            absorption_side = Side.SELL # Vendedores absorvendo
        elif buy_ratio < 0.3:  # Forte agressão vendedora
            aggressor_side = Side.SELL
            absorption_side = Side.BUY # Compradores absorvendo
        else:
            # Sem agressão unidirecional clara
            return {'detected': False}
            
        # Analisa movimento de preço
        price_movement = self._analyze_price_stability(trades, aggressor_side)
        
        # Verifica se preço está contido apesar da agressão
        if not price_movement['stable']:
            return {'detected': False}
            
        # Analisa book para confirmar absorção
        book_analysis = self._analyze_absorption_in_book(
            current_data.book, 
            absorption_side,
            price_movement['center_price']
        )
        
        # Calcula confiança
        confidence_signals = {
            'aggression_strength': (buy_ratio if aggressor_side == Side.BUY else 1 - buy_ratio),
            'price_stability': price_movement['stability_score'],
            'book_presence': book_analysis['defender_presence'],
            'volume_anomaly': self._calculate_volume_anomaly(volume_profile)
        }
        
        confidence = self.calculate_confidence(confidence_signals)
        
        # Precisa de confiança mínima
        if confidence < self.min_confidence:
            return {'detected': False}
            
        return {
            'detected': True,
            'confidence': confidence,
            'side': absorption_side,
            'aggressor_side': aggressor_side,
            'volume_absorbed': volume_profile['total_volume'],
            'aggressor_volume': (
                volume_profile['buy_volume'] if aggressor_side == Side.BUY 
                else volume_profile['sell_volume']
            ),
            'price_resistance': price_movement['stability_score'],
            'price_level': price_movement['center_price'],
            'defender_strength': book_analysis['defender_presence']
        }
        
    def _analyze_price_stability(self, trades: List[Trade], aggressor_side: Side) -> Dict[str, Any]:
        """Analisa se preço está estável apesar da agressão"""
        if not trades:
            return {'stable': False, 'stability_score': 0.0}
            
        prices = [t.price for t in trades]
        
        # Calcula range de preço
        price_range = max(prices) - min(prices)
        
        # Centro do range (onde a absorção está ocorrendo)
        center_price = min(prices) + price_range / 2
        
        # Verifica quantos trades aconteceram perto do centro
        near_center = 0
        tolerance = Decimal(str(self.max_price_change))
        
        for trade in trades:
            if abs(trade.price - center_price) <= tolerance:
                near_center += 1
                
        # Score de estabilidade
        stability_ratio = near_center / len(trades)
        
        # Penaliza se houve movimento na direção da agressão
        directional_movement = 0
        for i in range(1, len(trades)):
            if aggressor_side == Side.BUY and trades[i].price > trades[i-1].price:
                directional_movement += 1
            elif aggressor_side == Side.SELL and trades[i].price < trades[i-1].price:
                directional_movement += 1
                
        movement_penalty = directional_movement / len(trades)
        stability_score = stability_ratio * (1 - movement_penalty * 0.5)
        
        # Considera estável se:
        # 1. Range pequeno
        # 2. Maioria dos trades perto do centro
        # 3. Pouco movimento direcional
        is_stable = (
            float(price_range) <= self.max_price_change * 2 and
            stability_score > 0.6
        )
        
        return {
            'stable': is_stable,
            'stability_score': stability_score,
            'center_price': center_price,
            'price_range': price_range
        }
        
    def _analyze_absorption_in_book(
        self, 
        book: OrderBook, 
        absorption_side: Side,
        price_level: Decimal
    ) -> Dict[str, Any]:
        """Analisa presença de absorvedor no book"""
        if absorption_side == Side.BUY:
            # Absorvedor está comprando (bids)
            levels = book.bids
        else:
            # Absorvedor está vendendo (asks)
            levels = book.asks
            
        if not levels:
            return {'defender_presence': 0.0}
            
        # Procura por ordens grandes perto do nível de absorção
        large_orders = []
        total_volume = 0
        
        for level in levels[:5]:  # Top 5 níveis
            total_volume += level.volume
            
            # Verifica se está perto do nível de absorção
            if abs(level.price - price_level) <= Decimal('1.0'):
                large_orders.append(level)
                
        if not large_orders:
            return {'defender_presence': 0.0}
            
        # Calcula força do defensor
        defender_volume = sum(level.volume for level in large_orders)
        
        # Verifica se há concentração (poucos orders com muito volume)
        concentration = 0.0
        for level in large_orders:
            if level.orders <= 2:  # 1-2 ordens grandes
                concentration += 0.5
                
        # Score de presença do defensor
        volume_ratio = defender_volume / total_volume if total_volume > 0 else 0
        defender_presence = min(1.0, volume_ratio + concentration)
        
        return {
            'defender_presence': defender_presence,
            'defender_volume': defender_volume,
            'large_orders': len(large_orders)
        }
        
    def _calculate_volume_anomaly(self, volume_profile: Dict[str, Any]) -> float:
        """Calcula anomalia de volume (muito acima do normal)"""
        # Verifica histórico de volume
        if len(self.historical_data) < 10:
            return 0.5  # Neutro se não há histórico
            
        # Calcula volume médio histórico
        historical_volumes = []
        for data in self.historical_data[-20:]:  # Últimos 20 períodos
            hist_volume = sum(t.volume for t in data.trades)
            if hist_volume > 0:
                historical_volumes.append(hist_volume)
                
        if not historical_volumes:
            return 0.5
            
        avg_volume = sum(historical_volumes) / len(historical_volumes)
        current_volume = volume_profile['total_volume']
        
        # Razão de volume atual vs médio
        if avg_volume > 0:
            volume_ratio = current_volume / avg_volume
            
            # Converte para score 0-1
            # Ratio 2+ = score 1.0
            anomaly_score = min(1.0, max(0.0, (volume_ratio - 1.0)))
        else:
            anomaly_score = 0.5
            
        return anomaly_score