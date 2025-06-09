"""
Módulo de Estratégias do TapeReader
Exporta todas as classes de estratégias e componentes relacionados
"""

from .strategy_loader import StrategyLoader
from .confluence import ConfluenceAnalyzer
from .decision_matrix import DecisionMatrix
from .risk_manager import RiskManager
from .regime_classifier import RegimeClassifier, MarketRegime

__all__ = [
    'StrategyLoader',
    'ConfluenceAnalyzer',
    'DecisionMatrix',
    'RiskManager',
    'RegimeClassifier',
    'MarketRegime'
]