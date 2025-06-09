"""
Configurações e fixtures globais para testes
"""

import pytest
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
from decimal import Decimal
import yaml

# Adiciona src ao path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import Config, AppConfig, SystemConfig, LoggingConfig
from src.core.models import Trade, OrderBook, BookLevel, MarketData, Side
from src.behaviors.manager import BehaviorManager
from src.strategies.confluence import ConfluenceAnalyzer
from src.strategies.decision_matrix import DecisionMatrix
from src.strategies.risk_manager import RiskManager
from src.core.tracking import SignalTracker


@pytest.fixture
def event_loop():
    """Cria event loop para testes assíncronos"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_config():
    """Configuração de teste"""
    return {
        'app': {
            'name': 'TapeReader Test',
            'version': '2.0-test'
        },
        'modes': {
            'test': {
                'data_source': 'mock_dynamic',
                'log_level': 'DEBUG',
                'update_interval_ms': 100
            }
        },
        'system': {
            'min_confidence': 0.80,
            'max_signals_per_minute': 5
        },
        'logging': {
            'base_dir': 'tests/logs',
            'format': '[%(asctime)s] %(levelname)s - %(message)s'
        },
        'excel': {
            'ranges': {
                'dolfut': {
                    'time_trades': 'B4:E103',
                    'order_book': 'N4:Q103'
                },
                'wdofut': {
                    'time_trades': 'H4:K103',
                    'order_book': 'T4:W103'
                }
            }
        },
        'behaviors': {
            'absorption': {
                'enabled': True,
                'min_confidence': 0.75,
                'lookback_seconds': 30
            },
            'exhaustion': {
                'enabled': True,
                'min_confidence': 0.70,
                'lookback_seconds': 60
            }
        }
    }


@pytest.fixture
def signal_tracker():
    """Tracker de sinais para testes"""
    return SignalTracker()


@pytest.fixture
def behavior_manager(test_config):
    """Manager de comportamentos para testes"""
    return BehaviorManager(test_config)


@pytest.fixture
def confluence_analyzer(test_config):
    """Analisador de confluência para testes"""
    return ConfluenceAnalyzer(test_config)


@pytest.fixture
def decision_matrix(test_config, signal_tracker):
    """Matriz de decisão para testes"""
    return DecisionMatrix(test_config, signal_tracker)


@pytest.fixture
def risk_manager(test_config):
    """Gerenciador de risco para testes"""
    return RiskManager(test_config)