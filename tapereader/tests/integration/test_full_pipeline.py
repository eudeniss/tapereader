"""
Testes de integração do pipeline completo
"""

import pytest
import asyncio
from datetime import datetime
from decimal import Decimal

from src.behaviors.manager import BehaviorManager
from src.strategies.confluence import ConfluenceAnalyzer
from src.strategies.decision_matrix import DecisionMatrix
from src.strategies.risk_manager import RiskManager
from src.core.tracking import SignalTracker
from src.core.models import MarketData


@pytest.mark.integration
class TestFullPipeline:
    """Testes do pipeline completo de geração de sinais"""
    
    @pytest.fixture
    def pipeline_components(self, test_config):
        """Cria todos os componentes do pipeline"""
        signal_tracker = SignalTracker()
        
        return {
            'behavior_manager': BehaviorManager(test_config),
            'confluence_analyzer': ConfluenceAnalyzer(test_config),
            'decision_matrix': DecisionMatrix(test_config, signal_tracker),
            'risk_manager': RiskManager(test_config),
            'signal_tracker': signal_tracker
        }
    
    @pytest.mark.asyncio
    async def test_signal_generation_flow(self, pipeline_components, sample_market_data):
        """Testa fluxo completo de geração de sinal"""
        behavior_manager = pipeline_components['behavior_manager']
        confluence_analyzer = pipeline_components['confluence_analyzer']
        decision_matrix = pipeline_components['decision_matrix']
        risk_manager = pipeline_components['risk_manager']
        
        # 1. Detecta comportamentos
        behaviors_dol = await behavior_manager.analyze(sample_market_data)
        
        # Simula dados do WDOFUT
        wdo_data = sample_market_data.copy()
        wdo_data.asset = 'WDOFUT'
        behaviors_wdo = await behavior_manager.analyze(wdo_data)
        
        # 2. Analisa confluência
        confluence_analyzer.update_market_data('DOLFUT', sample_market_data)
        confluence_analyzer.update_market_data('WDOFUT', wdo_data)
        
        confluence_result = confluence_analyzer.analyze_confluence(
            'DOLFUT',
            behaviors_dol,
            behaviors_wdo
        )
        
        # 3. Gera sinal
        market_context = {
            'current_price': Decimal('5750.00'),
            'atr': Decimal('2.5'),
            'session_volume': 1000,
            'current_volatility': 0.01,
            'avg_volatility': 0.008,
            'current_spread': Decimal('0.5'),
            'avg_spread': Decimal('0.5')
        }
        
        signal = decision_matrix.generate_signal(
            'DOLFUT',
            behaviors_dol,
            confluence_result,
            market_context
        )
        
        # 4. Valida risco (se sinal foi gerado)
        if signal:
            approved, reason, adjusted_signal = risk_manager.validate_signal(
                signal,
                market_context
            )
            
            # Verifica resultado
            assert isinstance(approved, bool)
            if approved:
                assert adjusted_signal is not None
                assert adjusted_signal.signal_id != ""
            else:
                assert reason is not None
    
    @pytest.mark.asyncio
    async def test_multiple_scenarios(self, pipeline_components):
        """Testa múltiplos cenários de mercado"""
        behavior_manager = pipeline_components['behavior_manager']
        
        scenarios = [
            'normal_market',
            'high_volatility',
            'absorption_pattern',
            'institutional_flow'
        ]
        
        results = []
        
        for scenario in scenarios:
            # Gera dados para o cenário
            market_data = self._generate_scenario_data(scenario)
            
            # Processa
            behaviors = await behavior_manager.analyze(market_data)
            
            results.append({
                'scenario': scenario,
                'behaviors_detected': len(behaviors),
                'behavior_types': [b.behavior_type for b in behaviors]
            })
        
        # Verifica que diferentes cenários geram diferentes resultados
        behavior_sets = [set(r['behavior_types']) for r in results]
        assert len(set(map(tuple, behavior_sets))) > 1  # Pelo menos 2 diferentes
    
    def _generate_scenario_data(self, scenario: str) -> MarketData:
        """Gera dados de mercado para diferentes cenários"""
        # Implementação simplificada
        # Em produção, isso geraria dados mais realistas
        
        base_trades = []
        base_time = datetime.now()
        
        if scenario == 'absorption_pattern':
            # Muita agressão sem movimento de preço
            for i in range(20):
                base_trades.append(Trade(
                    timestamp=base_time - timedelta(seconds=20-i),
                    price=Decimal('5750.00'),
                    volume=50,
                    aggressor=Side.SELL
                ))
        else:
            # Mercado normal
            for i in range(10):
                base_trades.append(Trade(
                    timestamp=base_time - timedelta(seconds=10-i),
                    price=Decimal('5750.00') + Decimal(str(i * 0.5)),
                    volume=20,
                    aggressor=Side.BUY if i % 2 == 0 else Side.SELL
                ))
        
        return MarketData(
            asset='DOLFUT',
            timestamp=base_time,
            trades=base_trades,
            book=self._generate_basic_book()
        )
    
    def _generate_basic_book(self) -> OrderBook:
        """Gera book básico"""
        return OrderBook(
            timestamp=datetime.now(),
            bids=[
                BookLevel(price=Decimal('5749.50'), volume=100),
                BookLevel(price=Decimal('5749.00'), volume=150),
            ],
            asks=[
                BookLevel(price=Decimal('5750.50'), volume=100),
                BookLevel(price=Decimal('5751.00'), volume=150),
            ]
        )