"""
Exemplo de testes unitários usando Injeção de Dependências
Mostra como a DI facilita criar testes isolados
"""

import unittest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

# Adiciona o diretório raiz ao path
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.models import MarketData, Trade, TradingSignal, BehaviorDetection
from src.strategies import ConfluenceAnalyzer, DecisionMatrix, RiskManager, RegimeClassifier
from src.behaviors import BehaviorManager
from src.core import SignalTracker
from main import TapeReaderSystem, bootstrap_components


class TestTapeReaderWithDI(unittest.TestCase):
    """Testes do TapeReader usando Injeção de Dependências"""
    
    def setUp(self):
        """Configura mocks para cada teste"""
        # Cria mocks dos componentes
        self.mock_confluence = Mock(spec=ConfluenceAnalyzer)
        self.mock_decision = Mock(spec=DecisionMatrix)
        self.mock_risk = Mock(spec=RiskManager)
        self.mock_tracker = Mock(spec=SignalTracker)
        self.mock_regime = Mock(spec=RegimeClassifier)
        self.mock_behaviors = Mock(spec=BehaviorManager)
        
        # Configura comportamento padrão dos mocks
        self.mock_regime.current_regime = "UNKNOWN"
        self.mock_behaviors.analyze = AsyncMock(return_value=[])
        self.mock_confluence.analyze_confluence = Mock(return_value={
            'level': 'STANDARD',
            'correlation': 0.7,
            'confidence_boost': 0.05
        })
        
        # Cria sistema com mocks
        self.system = TapeReaderSystem(
            confluence_analyzer=self.mock_confluence,
            decision_matrix=self.mock_decision,
            risk_manager=self.mock_risk,
            signal_tracker=self.mock_tracker,
            regime_classifier=self.mock_regime,
            behavior_manager=self.mock_behaviors,
            console=None,
            use_console=False
        )
        
    def test_initialization(self):
        """Testa se o sistema inicializa corretamente com DI"""
        self.assertIsNotNone(self.system)
        self.assertEqual(self.system.confluence_analyzer, self.mock_confluence)
        self.assertEqual(self.system.decision_matrix, self.mock_decision)
        self.assertEqual(self.system.risk_manager, self.mock_risk)
        self.assertEqual(self.system.signal_tracker, self.mock_tracker)
        self.assertEqual(self.system.regime_classifier, self.mock_regime)
        self.assertEqual(self.system.behavior_manager, self.mock_behaviors)
        
    def test_factory_method(self):
        """Testa o factory method create_from_config"""
        with patch('main.bootstrap_components') as mock_bootstrap:
            # Configura o mock para retornar componentes
            mock_bootstrap.return_value = {
                'confluence_analyzer': self.mock_confluence,
                'decision_matrix': self.mock_decision,
                'risk_manager': self.mock_risk,
                'signal_tracker': self.mock_tracker,
                'regime_classifier': self.mock_regime,
                'behavior_manager': self.mock_behaviors,
                'console': None
            }
            
            # Cria sistema usando factory
            system = TapeReaderSystem.create_from_config()
            
            # Verifica se foi criado corretamente
            self.assertIsNotNone(system)
            mock_bootstrap.assert_called_once_with("config/strategies.yaml")
            
    async def test_process_market_data_no_behaviors(self):
        """Testa processamento quando não há comportamentos detectados"""
        # Cria dados de mercado mock
        market_data = MarketData(
            asset="DOLFUT",
            timestamp=datetime.now(),
            trades=[Trade(price=5750.0, volume=10, timestamp=datetime.now())],
            book=Mock()
        )
        
        # Configura mock para não retornar comportamentos
        self.mock_behaviors.analyze.return_value = []
        
        # Processa dados
        await self.system.process_market_data("DOLFUT", market_data)
        
        # Verifica se os métodos foram chamados corretamente
        self.mock_behaviors.analyze.assert_called_once()
        self.mock_confluence.update_market_data.assert_called_once_with("DOLFUT", market_data)
        self.mock_confluence.update_behaviors.assert_called_once_with("DOLFUT", [])
        
        # Não deve tentar gerar sinal se não há comportamentos
        self.mock_decision.generate_signal.assert_not_called()
        
    async def test_process_market_data_with_signal_generation(self):
        """Testa geração de sinal quando há comportamentos"""
        # Cria dados mock
        market_data = MarketData(
            asset="DOLFUT",
            timestamp=datetime.now(),
            trades=[Trade(price=5750.0, volume=100, timestamp=datetime.now())],
            book=Mock()
        )
        
        behavior = BehaviorDetection(
            behavior_type="momentum",
            confidence=0.85,
            timestamp=datetime.now()
        )
        
        signal = TradingSignal(
            asset="DOLFUT",
            direction="BUY",
            entry_price=5750.0,
            stop_loss=5740.0,
            take_profit_1=5760.0,
            confidence=0.85,
            strategy="momentum_breakout",
            signal_id="TEST_SIGNAL_001"
        )
        
        # Configura mocks
        self.mock_behaviors.analyze.return_value = [behavior]
        self.mock_behaviors.get_active_behaviors.return_value = []
        self.mock_decision.generate_signal.return_value = signal
        self.mock_risk.validate_signal.return_value = (True, "OK", signal)
        
        # Processa dados
        await self.system.process_market_data("DOLFUT", market_data)
        
        # Verifica chamadas
        self.mock_behaviors.analyze.assert_called_once()
        self.mock_decision.generate_signal.assert_called_once()
        self.mock_risk.validate_signal.assert_called_once_with(signal, self.system.market_context["DOLFUT"])
        
        # Verifica se o sinal foi adicionado
        self.assertIn("TEST_SIGNAL_001", self.system.active_signals)
        
    def test_regime_change_adjustment(self):
        """Testa ajuste do sistema quando regime muda"""
        # Configura mock do regime classifier
        from src.strategies.regime_classifier import MarketRegime
        self.mock_regime.get_regime_adjustments.return_value = {
            'favored_behaviors': ['momentum', 'breakout'],
            'avoid_behaviors': ['fade'],
            'confidence_multiplier': 1.1,
            'stop_distance_multiplier': 1.2,
            'target_multiplier': 1.3
        }
        
        # Executa ajuste
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(
            self.system._adjust_system_for_regime(MarketRegime.TRENDING_UP)
        )
        
        # Verifica se o método foi chamado
        self.mock_regime.get_regime_adjustments.assert_called_once_with(MarketRegime.TRENDING_UP)
        
    def test_get_system_status(self):
        """Testa obtenção de status do sistema"""
        # Configura mocks para retornar estatísticas
        self.mock_tracker.get_statistics.return_value = {
            'strategy_performance': {
                'momentum_breakout': {
                    'win_rate': 0.65,
                    'total_pnl': 1250.0,
                    'profit_factor': 2.1,
                    'total_trades': 20
                }
            }
        }
        
        self.mock_risk.get_risk_status.return_value = {
            'daily_pnl': 250.0,
            'open_positions': 1,
            'risk_level': 'LOW'
        }
        
        self.mock_behaviors.get_statistics.return_value = {
            'detections_count': 150,
            'active_behaviors': 3
        }
        
        # Obtém status
        status = self.system.get_system_status()
        
        # Verifica estrutura do status
        self.assertIn('timestamp', status)
        self.assertIn('active_signals', status)
        self.assertIn('risk_status', status)
        self.assertIn('behavior_stats', status)
        self.assertIn('signal_performance', status)
        self.assertIn('market_regime', status)
        
        # Verifica valores
        self.assertEqual(status['risk_status']['daily_pnl'], 250.0)
        self.assertEqual(status['behavior_stats']['detections_count'], 150)
        
    async def test_stop_loss_triggered(self):
        """Testa acionamento de stop loss"""
        # Cria sinal ativo
        signal = TradingSignal(
            asset="DOLFUT",
            direction="BUY",
            entry_price=5750.0,
            stop_loss=5740.0,
            take_profit_1=5760.0,
            confidence=0.85,
            strategy="test_strategy",
            signal_id="TEST_SL_001"
        )
        
        self.system.active_signals["TEST_SL_001"] = signal
        
        # Cria dados com preço abaixo do stop
        market_data = MarketData(
            asset="DOLFUT",
            timestamp=datetime.now(),
            trades=[Trade(price=5739.0, volume=50, timestamp=datetime.now())],
            book=Mock()
        )
        
        # Configura mock do tracker
        self.mock_tracker.update_signal_status = Mock()
        
        # Atualiza sinais
        await self.system._update_active_signals("DOLFUT", market_data)
        
        # Verifica se o sinal foi fechado
        self.assertNotIn("TEST_SL_001", self.system.active_signals)
        
        # Verifica se o tracker foi atualizado
        self.mock_tracker.update_signal_status.assert_called_once()
        
    def tearDown(self):
        """Limpa após cada teste"""
        self.system = None


class TestBootstrap(unittest.TestCase):
    """Testa a função bootstrap"""
    
    @patch('main.StrategyLoader')
    @patch('main.RegimeClassifier')
    @patch('main.BehaviorManager')
    def test_bootstrap_components(self, mock_behavior_mgr, mock_regime, mock_loader):
        """Testa criação de componentes via bootstrap"""
        # Configura mocks
        mock_loader_instance = Mock()
        mock_loader.return_value = mock_loader_instance
        mock_loader_instance.confluence_analyzer = Mock()
        mock_loader_instance.decision_matrix = Mock()
        mock_loader_instance.risk_manager = Mock()
        mock_loader_instance.signal_tracker = Mock()
        
        # Executa bootstrap
        components = bootstrap_components()
        
        # Verifica se todos os componentes foram criados
        self.assertIn('strategy_loader', components)
        self.assertIn('confluence_analyzer', components)
        self.assertIn('decision_matrix', components)
        self.assertIn('risk_manager', components)
        self.assertIn('signal_tracker', components)
        self.assertIn('regime_classifier', components)
        self.assertIn('behavior_manager', components)
        
        # Verifica chamadas
        mock_loader.assert_called_once_with("config/strategies.yaml")
        mock_regime.assert_called_once()
        mock_behavior_mgr.assert_called_once()


if __name__ == '__main__':
    unittest.main()