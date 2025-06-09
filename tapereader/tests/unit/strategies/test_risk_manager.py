"""
Testes do gerenciador de risco
"""

import pytest
from datetime import datetime, time
from decimal import Decimal

from src.core.models import TradingSignal, Side, SignalStrength


class TestRiskManager:
    """Testes do gerenciador de risco"""
    
    @pytest.fixture
    def sample_signal(self):
        """Sinal de exemplo para testes"""
        return TradingSignal(
            signal_id="TEST_001",
            timestamp=datetime.now(),
            direction=Side.BUY,
            asset='DOLFUT',
            price=Decimal('5750.00'),
            confidence=0.85,
            signal_strength=SignalStrength.STRONG,
            behaviors_detected=[],
            primary_motivation="Test signal",
            entry_price=Decimal('5750.50'),
            stop_loss=Decimal('5748.00'),
            take_profit_1=Decimal('5755.00'),
            risk_reward_ratio=2.0,
            position_size_suggestion=10
        )
    
    def test_trading_time_validation(self, risk_manager):
        """Testa validação de horário"""
        # Mock do horário atual
        import datetime as dt
        
        # Durante horário de trading
        with pytest.patch('datetime.datetime') as mock_datetime:
            mock_datetime.now.return_value = dt.datetime.combine(
                dt.date.today(),
                time(10, 30)  # 10:30 AM
            )
            mock_datetime.side_effect = lambda *args, **kw: dt.datetime(*args, **kw)
            
            valid, reason = risk_manager._validate_trading_time()
            assert valid == True
            assert reason is None
        
        # Antes da abertura
        with pytest.patch('datetime.datetime') as mock_datetime:
            mock_datetime.now.return_value = dt.datetime.combine(
                dt.date.today(),
                time(8, 0)  # 8:00 AM
            )
            mock_datetime.side_effect = lambda *args, **kw: dt.datetime(*args, **kw)
            
            valid, reason = risk_manager._validate_trading_time()
            assert valid == False
            assert "ainda não abriu" in reason
    
    def test_loss_limit_validation(self, risk_manager):
        """Testa validação de limite de perda"""
        # Sem perdas
        risk_manager.daily_pnl = Decimal('0')
        valid, reason = risk_manager._validate_loss_limits()
        assert valid == True
        
        # Próximo do limite
        risk_manager.daily_pnl = Decimal('-0.015')  # -1.5%
        valid, reason = risk_manager._validate_loss_limits()
        assert valid == True
        
        # Limite atingido
        risk_manager.daily_pnl = Decimal('-0.025')  # -2.5%
        valid, reason = risk_manager._validate_loss_limits()
        assert valid == False
        assert "Limite diário atingido" in reason
    
    def test_position_size_adjustment(self, risk_manager, sample_signal):
        """Testa ajuste de tamanho de posição"""
        from src.strategies.risk_manager import RiskLevel
        
        # Risco baixo - sem ajuste
        with pytest.patch.object(risk_manager, '_calculate_current_risk_level', return_value=RiskLevel.LOW):
            adjusted = risk_manager._adjust_position_size(sample_signal, {})
            assert adjusted.position_size_suggestion == 10
        
        # Risco alto - reduz 50%
        with pytest.patch.object(risk_manager, '_calculate_current_risk_level', return_value=RiskLevel.HIGH):
            adjusted = risk_manager._adjust_position_size(sample_signal, {})
            assert adjusted.position_size_suggestion == 5
    
    def test_circuit_breaker(self, risk_manager):
        """Testa circuit breaker"""
        # Ativa circuit breaker
        risk_manager._activate_circuit_breaker(30)
        
        assert risk_manager.circuit_breaker_active == True
        assert risk_manager.circuit_breaker_until is not None
        
        # Verifica se está ativo
        assert risk_manager._is_circuit_breaker_active() == True