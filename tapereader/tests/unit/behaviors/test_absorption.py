"""
Testes unitários para o detector de absorção
"""

import pytest
from datetime import datetime
from decimal import Decimal

from src.behaviors.absorption import AbsorptionDetector
from src.core.models import MarketData, Side, Trade, OrderBook, BookLevel


class TestAbsorptionDetector:
    """Testes do detector de absorção"""
    
    @pytest.fixture
    def detector(self, test_config):
        """Cria detector de absorção"""
        config = test_config['behaviors']['absorption']
        return AbsorptionDetector(config)
    
    def test_initialization(self, detector):
        """Testa inicialização do detector"""
        assert detector.behavior_type == "absorption"
        assert detector.min_confidence == 0.75
        assert detector.lookback_seconds == 30
    
    @pytest.mark.asyncio
    async def test_detect_buy_absorption(self, detector, absorption_scenario_trades, sample_order_book):
        """Testa detecção de absorção compradora"""
        # Cria book com grande comprador
        book_with_absorption = sample_order_book
        book_with_absorption.bids[0].volume = 500  # Grande bid absorvendo
        
        market_data = MarketData(
            asset='DOLFUT',
            timestamp=datetime.now(),
            trades=absorption_scenario_trades,
            book=book_with_absorption
        )
        
        # Atualiza histórico
        detector.update_history(market_data)
        
        # Detecta
        result = await detector.detect(market_data)
        
        assert result.detected == True
        assert result.confidence >= 0.75
        assert result.metadata['absorption_side'] == Side.BUY
    
    @pytest.mark.asyncio
    async def test_no_absorption_normal_market(self, detector, sample_market_data):
        """Testa que não detecta absorção em mercado normal"""
        result = await detector.detect(sample_market_data)
        
        assert result.detected == False
        assert result.confidence == 0.0
    
    def test_analyze_price_stability(self, detector):
        """Testa análise de estabilidade de preço"""
        trades = [
            Trade(timestamp=datetime.now(), price=Decimal('5750.00'), volume=10, aggressor=Side.BUY),
            Trade(timestamp=datetime.now(), price=Decimal('5750.00'), volume=10, aggressor=Side.SELL),
            Trade(timestamp=datetime.now(), price=Decimal('5750.50'), volume=10, aggressor=Side.BUY),
            Trade(timestamp=datetime.now(), price=Decimal('5750.00'), volume=10, aggressor=Side.SELL),
        ]
        
        result = detector._analyze_price_stability(trades, Side.BUY)
        
        assert result['stable'] == True
        assert result['price_range'] == Decimal('0.50')
        assert 0 < result['stability_score'] <= 1
    
    def test_calculate_volume_anomaly(self, detector, sample_market_data):
        """Testa cálculo de anomalia de volume"""
        # Popula histórico
        for _ in range(10):
            detector.update_history(sample_market_data)
        
        volume_profile = {
            'total_volume': 1000,  # Volume muito alto
            'buy_volume': 800,
            'sell_volume': 200
        }
        
        anomaly = detector._calculate_volume_anomaly(volume_profile)
        
        assert 0 <= anomaly <= 1
        assert anomaly > 0.5  # Volume anômalo