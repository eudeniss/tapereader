"""
Utilitários do TapeReader
Módulos auxiliares para cálculos e operações comuns
"""

# Exporta funções principais do módulo statistics
from .statistics import (
    calculate_trend,
    calculate_correlation,
    calculate_returns,
    calculate_volatility,
    calculate_sma,
    calculate_ema,
    calculate_rsi,
    calculate_bollinger_bands,
    calculate_sharpe_ratio,
    calculate_max_drawdown,
    calculate_beta,
    detect_outliers,
    clear_cache,
    get_cache_stats
)

__all__ = [
    'calculate_trend',
    'calculate_correlation',
    'calculate_returns',
    'calculate_volatility',
    'calculate_sma',
    'calculate_ema',
    'calculate_rsi',
    'calculate_bollinger_bands',
    'calculate_sharpe_ratio',
    'calculate_max_drawdown',
    'calculate_beta',
    'detect_outliers',
    'clear_cache',
    'get_cache_stats'
]