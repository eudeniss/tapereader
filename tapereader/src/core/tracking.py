"""
Sistema de rastreamento de sinais
Gera IDs únicos e rastreia lifecycle dos sinais
Inclui rastreamento de P&L por estratégia e regime de mercado
COM PERSISTÊNCIA DE ESTADO
"""

import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import json
from pathlib import Path
from collections import defaultdict
import numpy as np

from .models import TradingSignal
from .logger import get_logger


class SignalStatus(str, Enum):
    """Estados possíveis de um sinal"""
    ANALYZING = "ANALYZING"
    EMITTED = "EMITTED"
    CONFIRMED = "CONFIRMED"
    EXECUTED = "EXECUTED"
    PARTIAL = "PARTIAL"
    CLOSED = "CLOSED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"


class StrategyPerformance:
    """Rastreia performance de uma estratégia específica"""
    
    def __init__(self, strategy_name: str):
        self.strategy_name = strategy_name
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0
        self.gross_profit = 0.0
        self.gross_loss = 0.0
        self.max_drawdown = 0.0
        self.current_drawdown = 0.0
        self.peak_pnl = 0.0
        self.trades_by_regime = defaultdict(lambda: {'total': 0, 'wins': 0, 'pnl': 0.0})
        self.recent_trades = []  # Últimas 20 trades para cálculo de momentum
        self.confidence_adjustments = []  # Histórico de ajustes
        
    def add_trade(self, pnl: float, regime: str, confidence: float):
        """Adiciona resultado de uma trade"""
        self.total_trades += 1
        self.total_pnl += pnl
        
        # Atualiza estatísticas
        if pnl > 0:
            self.winning_trades += 1
            self.gross_profit += pnl
        else:
            self.losing_trades += 1
            self.gross_loss += abs(pnl)
            
        # Atualiza drawdown
        if self.total_pnl > self.peak_pnl:
            self.peak_pnl = self.total_pnl
        self.current_drawdown = self.peak_pnl - self.total_pnl
        self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
        
        # Rastreia por regime
        regime_stats = self.trades_by_regime[regime]
        regime_stats['total'] += 1
        if pnl > 0:
            regime_stats['wins'] += 1
        regime_stats['pnl'] += pnl
        
        # Mantém últimas 20 trades
        self.recent_trades.append({
            'pnl': pnl,
            'regime': regime,
            'confidence': confidence,
            'timestamp': datetime.now()
        })
        if len(self.recent_trades) > 20:
            self.recent_trades.pop(0)
    
    @property
    def win_rate(self) -> float:
        """Taxa de acerto da estratégia"""
        if self.total_trades == 0:
            return 0.5  # Assume 50% se não há histórico
        return self.winning_trades / self.total_trades
    
    @property
    def profit_factor(self) -> float:
        """Fator de lucro (gross profit / gross loss)"""
        if self.gross_loss == 0:
            return float('inf') if self.gross_profit > 0 else 0
        return self.gross_profit / self.gross_loss
    
    @property
    def expectancy(self) -> float:
        """Expectância matemática por trade"""
        if self.total_trades == 0:
            return 0
        return self.total_pnl / self.total_trades
    
    @property
    def recent_momentum(self) -> float:
        """Momentum recente baseado nas últimas trades"""
        if len(self.recent_trades) < 5:
            return 0
        
        # Calcula win rate recente
        recent_wins = sum(1 for t in self.recent_trades[-10:] if t['pnl'] > 0)
        recent_win_rate = recent_wins / min(10, len(self.recent_trades))
        
        # Compara com win rate histórico
        momentum = (recent_win_rate - self.win_rate) * 2
        return max(-0.2, min(0.2, momentum))  # Limita entre -20% e +20%
    
    def get_regime_performance(self, regime: str) -> Dict:
        """Retorna performance em um regime específico"""
        if regime not in self.trades_by_regime:
            return {'win_rate': 0.5, 'expectancy': 0, 'total_trades': 0}
        
        stats = self.trades_by_regime[regime]
        win_rate = stats['wins'] / stats['total'] if stats['total'] > 0 else 0.5
        expectancy = stats['pnl'] / stats['total'] if stats['total'] > 0 else 0
        
        return {
            'win_rate': win_rate,
            'expectancy': expectancy,
            'total_trades': stats['total'],
            'total_pnl': stats['pnl']
        }
    
    def calculate_confidence_adjustment(self, base_confidence: float, current_regime: str) -> float:
        """Calcula ajuste de confiança baseado em performance"""
        # Performance geral
        general_adjustment = 0.0
        
        # Baseado em win rate
        if self.total_trades >= 10:
            if self.win_rate > 0.6:
                general_adjustment += 0.05
            elif self.win_rate < 0.4:
                general_adjustment -= 0.05
        
        # Baseado em profit factor
        if self.profit_factor > 2.0:
            general_adjustment += 0.03
        elif self.profit_factor < 1.0:
            general_adjustment -= 0.03
        
        # Baseado em drawdown
        if self.total_pnl > 0 and self.current_drawdown > 0:
            dd_ratio = self.current_drawdown / self.peak_pnl
            if dd_ratio > 0.2:  # Drawdown > 20%
                general_adjustment -= 0.05
        
        # Performance no regime atual
        regime_adjustment = 0.0
        regime_perf = self.get_regime_performance(current_regime)
        
        if regime_perf['total_trades'] >= 5:
            if regime_perf['win_rate'] > 0.6:
                regime_adjustment += 0.05
            elif regime_perf['win_rate'] < 0.4:
                regime_adjustment -= 0.05
        
        # Momentum recente
        momentum_adjustment = self.recent_momentum
        
        # Calcula ajuste total
        total_adjustment = general_adjustment + regime_adjustment + momentum_adjustment
        total_adjustment = max(-0.15, min(0.15, total_adjustment))  # Limita entre -15% e +15%
        
        # Registra ajuste
        self.confidence_adjustments.append({
            'timestamp': datetime.now(),
            'regime': current_regime,
            'general': general_adjustment,
            'regime': regime_adjustment,
            'momentum': momentum_adjustment,
            'total': total_adjustment
        })
        
        # Retorna confiança ajustada
        adjusted_confidence = base_confidence + total_adjustment
        return max(0.1, min(0.99, adjusted_confidence))  # Mantém entre 10% e 99%
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte performance para dicionário serializável"""
        return {
            'strategy_name': self.strategy_name,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'total_pnl': self.total_pnl,
            'gross_profit': self.gross_profit,
            'gross_loss': self.gross_loss,
            'max_drawdown': self.max_drawdown,
            'current_drawdown': self.current_drawdown,
            'peak_pnl': self.peak_pnl,
            'trades_by_regime': dict(self.trades_by_regime),
            'recent_trades': [
                {
                    'pnl': t['pnl'],
                    'regime': t['regime'],
                    'confidence': t['confidence'],
                    'timestamp': t['timestamp'].isoformat()
                }
                for t in self.recent_trades
            ]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StrategyPerformance':
        """Cria StrategyPerformance a partir de dicionário"""
        perf = cls(data['strategy_name'])
        perf.total_trades = data.get('total_trades', 0)
        perf.winning_trades = data.get('winning_trades', 0)
        perf.losing_trades = data.get('losing_trades', 0)
        perf.total_pnl = data.get('total_pnl', 0)
        perf.gross_profit = data.get('gross_profit', 0)
        perf.gross_loss = data.get('gross_loss', 0)
        perf.max_drawdown = data.get('max_drawdown', 0)
        perf.current_drawdown = data.get('current_drawdown', 0)
        perf.peak_pnl = data.get('peak_pnl', 0)
        
        # Restaura trades por regime
        for regime, stats in data.get('trades_by_regime', {}).items():
            perf.trades_by_regime[regime] = stats
        
        # Restaura trades recentes
        for trade_data in data.get('recent_trades', []):
            perf.recent_trades.append({
                'pnl': trade_data['pnl'],
                'regime': trade_data['regime'],
                'confidence': trade_data['confidence'],
                'timestamp': datetime.fromisoformat(trade_data['timestamp'])
            })
        
        return perf


class SignalTracker:
    """Rastreia o ciclo de vida completo dos sinais com persistência"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.active_signals: Dict[str, TradingSignal] = {}
        self.signal_history: List[TradingSignal] = []
        self.signal_counter = 0
        
        # NOVO: Rastreamento de performance por estratégia
        self.strategy_performance: Dict[str, StrategyPerformance] = {}
        self.current_regime = "UNKNOWN"
        self.regime_history = []
        
        # Arquivo de persistência
        self.performance_file = Path("data/strategy_performance.json")
        self.load_performance_history()
        
    def generate_signal_id(self, signal: TradingSignal) -> str:
        """Gera ID único para o sinal"""
        # Formato: SIG_YYYYMMDD_HHMMSS_DIR_PRICE_CONF_HASH
        now = datetime.now()
        unique_hash = str(uuid.uuid4())[:6]
        
        signal_id = (
            f"SIG_{now:%Y%m%d}_{now:%H%M%S}_"
            f"{signal.direction}_{signal.price}_"
            f"{int(signal.confidence*100)}_{unique_hash}"
        )
        
        return signal_id
        
    def create_tracked_signal(self, signal: TradingSignal) -> TradingSignal:
        """Adiciona tracking ID ao sinal"""
        if not signal.signal_id:
            signal.signal_id = self.generate_signal_id(signal)
            
        signal.status = SignalStatus.EMITTED
        self.active_signals[signal.signal_id] = signal
        self.signal_counter += 1
        
        # NOVO: Registra estratégia se não existir
        if signal.strategy not in self.strategy_performance:
            self.strategy_performance[signal.strategy] = StrategyPerformance(signal.strategy)
        
        self.logger.info(
            f"Sinal criado: {signal.signal_id} - "
            f"Estratégia: {signal.strategy} - "
            f"Total ativo: {len(self.active_signals)}"
        )
        
        return signal
        
    def update_signal_status(
        self, 
        signal_id: str, 
        new_status: SignalStatus, 
        data: Dict = None
    ):
        """Atualiza status do sinal"""
        if signal_id not in self.active_signals:
            self.logger.warning(f"Sinal não encontrado: {signal_id}")
            return
            
        signal = self.active_signals[signal_id]
        old_status = signal.status
        signal.status = new_status.value
        
        # Adiciona dados adicionais
        if data:
            if new_status == SignalStatus.EXECUTED:
                signal.execution_data = data
            elif new_status == SignalStatus.CLOSED:
                signal.result = data
                # NOVO: Registra P&L da estratégia
                self._record_trade_result(signal, data)
                
        self.logger.info(
            f"Sinal {signal_id}: {old_status} → {new_status.value}"
        )
        
        # Se finalizado, move para histórico
        final_statuses = [
            SignalStatus.CLOSED,
            SignalStatus.CANCELLED,
            SignalStatus.EXPIRED
        ]
        
        if new_status in final_statuses:
            self.signal_history.append(signal)
            del self.active_signals[signal_id]
            self.logger.info(
                f"Sinal finalizado: {signal_id} - "
                f"Histórico: {len(self.signal_history)} sinais"
            )
            
    def _record_trade_result(self, signal: TradingSignal, result_data: Dict):
        """Registra resultado da trade na performance da estratégia"""
        if 'pnl' not in result_data:
            self.logger.warning(f"Resultado sem P&L para sinal {signal.signal_id}")
            return
            
        pnl = result_data['pnl']
        strategy = signal.strategy
        
        # Registra na performance da estratégia
        if strategy in self.strategy_performance:
            perf = self.strategy_performance[strategy]
            perf.add_trade(pnl, self.current_regime, signal.confidence)
            
            self.logger.info(
                f"P&L registrado para {strategy}: {pnl:+.2f} - "
                f"Total P&L: {perf.total_pnl:+.2f} - "
                f"Win Rate: {perf.win_rate:.1%}"
            )
            
            # Salva histórico
            self.save_performance_history()
            
    def update_market_regime(self, new_regime: str):
        """Atualiza regime de mercado atual"""
        if new_regime != self.current_regime:
            self.logger.info(f"Regime de mercado: {self.current_regime} → {new_regime}")
            self.current_regime = new_regime
            self.regime_history.append({
                'regime': new_regime,
                'timestamp': datetime.now()
            })
            
    def get_strategy_confidence(self, strategy_name: str, base_confidence: float) -> float:
        """Retorna confiança ajustada para uma estratégia"""
        if strategy_name not in self.strategy_performance:
            return base_confidence
            
        perf = self.strategy_performance[strategy_name]
        adjusted = perf.calculate_confidence_adjustment(base_confidence, self.current_regime)
        
        if abs(adjusted - base_confidence) > 0.01:
            self.logger.info(
                f"Confiança ajustada para {strategy_name}: "
                f"{base_confidence:.1%} → {adjusted:.1%} "
                f"(Regime: {self.current_regime})"
            )
            
        return adjusted
        
    def get_active_signals(self) -> List[TradingSignal]:
        """Retorna lista de sinais ativos"""
        return list(self.active_signals.values())
        
    def get_signal_by_id(self, signal_id: str) -> Optional[TradingSignal]:
        """Busca sinal por ID"""
        # Primeiro nos ativos
        if signal_id in self.active_signals:
            return self.active_signals[signal_id]
            
        # Depois no histórico
        for signal in self.signal_history:
            if signal.signal_id == signal_id:
                return signal
                
        return None
        
    def save_signal(self, signal: TradingSignal):
        """Salva sinal para análise posterior"""
        # Usa o sistema de logger para salvar
        from .logger import get_logger_system
        logger_system = get_logger_system()
        if logger_system:
            logger_system.log_signal(signal)
            
    def expire_old_signals(self, max_age_minutes: int = 15):
        """Expira sinais antigos não executados"""
        now = datetime.now()
        expired = []
        
        for signal_id, signal in self.active_signals.items():
            age = (now - signal.timestamp).total_seconds() / 60
            
            if age > max_age_minutes and signal.status == SignalStatus.EMITTED:
                expired.append(signal_id)
                
        for signal_id in expired:
            self.update_signal_status(signal_id, SignalStatus.EXPIRED)
            self.logger.info(f"Sinal expirado: {signal_id}")
            
    def get_statistics(self) -> Dict:
        """Retorna estatísticas dos sinais"""
        total_signals = self.signal_counter
        active_signals = len(self.active_signals)
        
        # Analisa histórico
        stats = {
            'total_signals': total_signals,
            'active_signals': active_signals,
            'historical_signals': len(self.signal_history),
            'current_regime': self.current_regime
        }
        
        if self.signal_history:
            # Por status
            status_counts = {}
            for signal in self.signal_history:
                status = signal.status
                status_counts[status] = status_counts.get(status, 0) + 1
                
            # Por direção
            buy_signals = sum(1 for s in self.signal_history if s.direction == "BUY")
            sell_signals = sum(1 for s in self.signal_history if s.direction == "SELL")
            
            # Confiança média
            avg_confidence = sum(s.confidence for s in self.signal_history) / len(self.signal_history)
            
            stats.update({
                'status_distribution': status_counts,
                'buy_signals': buy_signals,
                'sell_signals': sell_signals,
                'average_confidence': round(avg_confidence, 2)
            })
            
        # NOVO: Estatísticas por estratégia
        strategy_stats = {}
        for name, perf in self.strategy_performance.items():
            strategy_stats[name] = {
                'total_trades': perf.total_trades,
                'win_rate': round(perf.win_rate, 3),
                'profit_factor': round(perf.profit_factor, 2),
                'total_pnl': round(perf.total_pnl, 2),
                'expectancy': round(perf.expectancy, 2),
                'max_drawdown': round(perf.max_drawdown, 2),
                'current_regime_performance': perf.get_regime_performance(self.current_regime)
            }
            
        stats['strategy_performance'] = strategy_stats
        
        return stats
        
    def save_performance_history(self):
        """Salva histórico de performance em arquivo"""
        try:
            # Cria diretório se não existir
            self.performance_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Prepara dados para serialização
            data = {
                'last_update': datetime.now().isoformat(),
                'current_regime': self.current_regime,
                'strategies': {}
            }
            
            for name, perf in self.strategy_performance.items():
                data['strategies'][name] = perf.to_dict()
                
            # Salva arquivo
            with open(self.performance_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Erro ao salvar performance: {e}")
            
    def load_performance_history(self):
        """Carrega histórico de performance do arquivo"""
        try:
            if not self.performance_file.exists():
                return
                
            with open(self.performance_file, 'r') as f:
                data = json.load(f)
                
            # Restaura regime
            self.current_regime = data.get('current_regime', 'UNKNOWN')
            
            # Restaura performance das estratégias
            for name, perf_data in data.get('strategies', {}).items():
                self.strategy_performance[name] = StrategyPerformance.from_dict(perf_data)
                
            self.logger.info(
                f"Performance histórica carregada: "
                f"{len(self.strategy_performance)} estratégias"
            )
            
        except Exception as e:
            self.logger.error(f"Erro ao carregar performance: {e}")
            
    def get_best_performing_strategies(self, regime: Optional[str] = None) -> List[Tuple[str, float]]:
        """Retorna estratégias ordenadas por performance"""
        if regime is None:
            regime = self.current_regime
            
        strategies_scores = []
        
        for name, perf in self.strategy_performance.items():
            # Score baseado em múltiplos fatores
            score = 0.0
            
            # Win rate geral (peso 30%)
            score += perf.win_rate * 0.3
            
            # Profit factor (peso 20%)
            if perf.profit_factor > 0:
                score += min(perf.profit_factor / 3, 1.0) * 0.2
                
            # Performance no regime atual (peso 30%)
            regime_perf = perf.get_regime_performance(regime)
            if regime_perf['total_trades'] >= 3:
                score += regime_perf['win_rate'] * 0.3
                
            # Momentum recente (peso 20%)
            momentum = perf.recent_momentum
            score += (0.5 + momentum) * 0.2
            
            strategies_scores.append((name, score))
            
        # Ordena por score decrescente
        strategies_scores.sort(key=lambda x: x[1], reverse=True)
        
        return strategies_scores
    
    # NOVO: Métodos de Persistência
    def get_state(self) -> Dict[str, Any]:
        """Retorna estado completo para serialização"""
        state = {
            'signal_counter': self.signal_counter,
            'current_regime': self.current_regime,
            'active_signals': {
                signal_id: {
                    'signal_id': signal.signal_id,
                    'asset': signal.asset,
                    'direction': signal.direction.value if hasattr(signal.direction, 'value') else signal.direction,
                    'entry_price': float(signal.entry_price),
                    'stop_loss': float(signal.stop_loss),
                    'take_profit_1': float(signal.take_profit_1),
                    'confidence': signal.confidence,
                    'strategy': signal.strategy,
                    'timestamp': signal.timestamp.isoformat(),
                    'status': signal.status
                }
                for signal_id, signal in self.active_signals.items()
            },
            'signal_history_summary': {
                'count': len(self.signal_history),
                'last_10': [
                    {
                        'signal_id': s.signal_id,
                        'timestamp': s.timestamp.isoformat(),
                        'status': s.status,
                        'pnl': s.result.get('pnl', 0) if hasattr(s, 'result') and s.result else 0
                    }
                    for s in self.signal_history[-10:]
                ]
            },
            'strategy_performance': {
                name: perf.to_dict()
                for name, perf in self.strategy_performance.items()
            },
            'regime_history': [
                {
                    'regime': r['regime'],
                    'timestamp': r['timestamp'].isoformat()
                }
                for r in self.regime_history[-10:]  # Últimos 10 regimes
            ]
        }
        
        return state
    
    def load_state(self, state: Dict[str, Any]):
        """Carrega estado a partir de dicionário"""
        try:
            # Restaura contador
            self.signal_counter = state.get('signal_counter', 0)
            
            # Restaura regime
            self.current_regime = state.get('current_regime', 'UNKNOWN')
            
            # Restaura sinais ativos
            self.active_signals.clear()
            for signal_id, signal_data in state.get('active_signals', {}).items():
                try:
                    # Recria TradingSignal
                    signal = TradingSignal(
                        asset=signal_data['asset'],
                        direction=signal_data['direction'],
                        entry_price=signal_data['entry_price'],
                        stop_loss=signal_data['stop_loss'],
                        take_profit_1=signal_data['take_profit_1'],
                        confidence=signal_data['confidence'],
                        strategy=signal_data['strategy']
                    )
                    signal.signal_id = signal_data['signal_id']
                    signal.timestamp = datetime.fromisoformat(signal_data['timestamp'])
                    signal.status = signal_data.get('status', SignalStatus.EMITTED)
                    
                    self.active_signals[signal_id] = signal
                except Exception as e:
                    self.logger.error(f"Erro ao restaurar sinal {signal_id}: {e}")
            
            # Restaura performance das estratégias
            self.strategy_performance.clear()
            for name, perf_data in state.get('strategy_performance', {}).items():
                try:
                    self.strategy_performance[name] = StrategyPerformance.from_dict(perf_data)
                except Exception as e:
                    self.logger.error(f"Erro ao restaurar performance de {name}: {e}")
            
            # Restaura histórico de regime
            self.regime_history = []
            for regime_data in state.get('regime_history', []):
                self.regime_history.append({
                    'regime': regime_data['regime'],
                    'timestamp': datetime.fromisoformat(regime_data['timestamp'])
                })
            
            self.logger.info(
                f"Estado do SignalTracker restaurado: "
                f"{len(self.active_signals)} sinais ativos, "
                f"{len(self.strategy_performance)} estratégias"
            )
            
        except Exception as e:
            self.logger.error(f"Erro ao carregar estado do SignalTracker: {e}")