"""
Testes do analisador de confluência
"""

import pytest
from datetime import datetime
from decimal import Decimal

from src.core.models import MarketData, BehaviorDetection, Side


class TestConfluenceAnalyzer:
    """Testes do analisador de confluência"""
    
    def test_update_market_data(self, confluence_analyzer, sample_market_data):
        """Testa atualização de dados de mercado"""
        initial_size = len(confluence_analyzer.price_history['DOLFUT'])
        
        confluence_analyzer.update_market_data('DOLFUT', sample_market_data)
        
        assert len(confluence_analyzer.price_history['DOLFUT']) == initial_size + 1
    
    def test_correlation_calculation(self, confluence_analyzer):
        """Testa cálculo de correlação"""
        # Popula histórico com dados correlacionados
        for i in range(20):
            timestamp = datetime.now()
            
            # DOLFUT
            dol_data = {
                'timestamp': timestamp,
                'price': 5750.0 + i,
                'volume': 100
            }
            confluence_analyzer.price_history['DOLFUT'].append(dol_data)
            
            # WDOFUT - perfeitamente correlacionado
            wdo_data = {
                'timestamp': timestamp,
                'price': 5750.0 + i,
                'volume': 400
            }
            confluence_analyzer.price_history['WDOFUT'].append(wdo_data)
        
        correlation = confluence_analyzer._calculate_price_correlation()
        
        assert correlation > 0.95  # Alta correlação
    
    def test_behavior_confluence(self, confluence_analyzer):
        """Testa análise de confluência de comportamentos"""
        primary_behaviors = [
            BehaviorDetection(
                behavior_type='absorption',
                confidence=0.85,
                detected=True,
                metadata={'absorption_side': Side.BUY}
            ),
            BehaviorDetection(
                behavior_type='momentum',
                confidence=0.80,
                detected=True,
                metadata={'momentum_direction': Side.BUY}
            )
        ]
        
        secondary_behaviors = [
            BehaviorDetection(
                behavior_type='absorption',
                confidence=0.82,
                detected=True,
                metadata={'absorption_side': Side.BUY}
            ),
            BehaviorDetection(
                behavior_type='exhaustion',
                confidence=0.75,
                detected=True,
                metadata={'direction': Side.SELL}  # Conflitante
            )
        ]
        
        result = confluence_analyzer._analyze_behavior_confluence(
            primary_behaviors,
            secondary_behaviors
        )
        
        assert len(result['matching']) == 1  # Apenas absorption combina
        assert len(result['conflicting']) == 0  # Exhaustion não está em primary
        assert 0 <= result['match_score'] <= 1
    
    def test_confluence_boost_calculation(self, confluence_analyzer):
        """Testa cálculo de boost de confiança"""
        from src.strategies.confluence import ConfluenceLevel
        
        behavior_confluence = {
            'match_score': 0.8,
            'conflicting': []
        }
        
        # Confluência PREMIUM
        boost = confluence_analyzer._calculate_confidence_boost(
            ConfluenceLevel.PREMIUM,
            behavior_confluence
        )
        assert boost == 0.15
        
        # Confluência WEAK
        boost = confluence_analyzer._calculate_confidence_boost(
            ConfluenceLevel.WEAK,
            behavior_confluence
        )
        assert boost == 0.0
        
        # Com conflitos
        behavior_confluence['conflicting'] = ['some_behavior']
        boost = confluence_analyzer._calculate_confidence_boost(
            ConfluenceLevel.PREMIUM,
            behavior_confluence
        )
        assert boost == 0.075  # 50% de penalidade