"""
Formatador de sinais e dados para o console
"""

from decimal import Decimal
from datetime import datetime
from typing import Dict, Any, List

from src.core.models import TradingSignal, BehaviorDetection, Side


class SignalFormatter:
    """Formata sinais para exibição"""
    
    @staticmethod
    def format_price(price: Decimal, decimals: int = 2) -> str:
        """Formata preço com decimais fixas"""
        return f"{float(price):,.{decimals}f}"
    
    @staticmethod
    def format_percentage(value: float) -> str:
        """Formata percentual"""
        return f"{value:.1%}"
    
    @staticmethod
    def format_pnl(pnl: Decimal) -> str:
        """Formata P&L com cor"""
        value = float(pnl)
        if value > 0:
            return f"[green]+{value:,.2f}[/green]"
        elif value < 0:
            return f"[red]{value:,.2f}[/red]"
        else:
            return f"[white]{value:,.2f}[/white]"
    
    @staticmethod
    def get_direction_emoji(direction: Side) -> str:
        """Retorna emoji para direção"""
        return "🟢" if direction == Side.BUY else "🔴"
    
    @staticmethod
    def get_signal_color(signal: TradingSignal) -> str:
        """Retorna cor baseada no sinal"""
        if signal.direction == Side.BUY:
            return "green"
        else:
            return "red"
    
    @staticmethod
    def format_signal_summary(signal: TradingSignal) -> Dict[str, str]:
        """Formata resumo do sinal"""
        return {
            'direction': f"{SignalFormatter.get_direction_emoji(signal.direction)} {signal.direction.value}",
            'price': SignalFormatter.format_price(signal.entry_price),
            'confidence': SignalFormatter.format_percentage(signal.confidence),
            'stop': SignalFormatter.format_price(signal.stop_loss),
            'target': SignalFormatter.format_price(signal.take_profit_1),
            'risk_reward': f"{signal.risk_reward_ratio:.1f}:1",
            'behaviors': ", ".join(signal.behaviors_detected),
            'timestamp': signal.timestamp.strftime("%H:%M:%S")
        }
    
    @staticmethod
    def format_behavior_summary(behaviors: List[BehaviorDetection]) -> str:
        """Formata resumo de comportamentos"""
        if not behaviors:
            return "Nenhum comportamento detectado"
        
        summary = []
        for behavior in behaviors:
            confidence = SignalFormatter.format_percentage(behavior.confidence)
            summary.append(f"{behavior.behavior_type} ({confidence})")
        
        return " | ".join(summary)
    
    @staticmethod
    def format_market_stats(market_data: Dict[str, Any]) -> Dict[str, str]:
        """Formata estatísticas de mercado"""
        return {
            'price': SignalFormatter.format_price(
                Decimal(str(market_data.get('current_price', 0)))
            ),
            'volume': f"{market_data.get('session_volume', 0):,}",
            'trades': f"{market_data.get('trades_count', 0):,}",
            'volatility': f"{market_data.get('current_volatility', 0):.4f}",
            'last_update': market_data.get('last_update', datetime.now()).strftime("%H:%M:%S")
        }


class TableFormatter:
    """Formata dados em tabelas"""
    
    @staticmethod
    def format_active_signals(signals: List[TradingSignal], 
                            current_prices: Dict[str, Decimal]) -> List[Dict[str, str]]:
        """Formata sinais ativos para tabela"""
        rows = []
        
        for signal in signals:
            current = current_prices.get(signal.asset, signal.entry_price)
            
            # Calcula P&L
            if signal.direction == Side.BUY:
                pnl = current - signal.entry_price
            else:
                pnl = signal.entry_price - current
            
            # Calcula status
            if signal.direction == Side.BUY:
                if current >= signal.take_profit_1:
                    status = "[green]ALVO[/green]"
                elif current <= signal.stop_loss:
                    status = "[red]STOP[/red]"
                else:
                    status = "[yellow]ATIVO[/yellow]"
            else:
                if current <= signal.take_profit_1:
                    status = "[green]ALVO[/green]"
                elif current >= signal.stop_loss:
                    status = "[red]STOP[/red]"
                else:
                    status = "[yellow]ATIVO[/yellow]"
            
            rows.append({
                'time': signal.timestamp.strftime("%H:%M"),
                'asset': signal.asset,
                'direction': f"{SignalFormatter.get_direction_emoji(signal.direction)} {signal.direction.value}",
                'entry': SignalFormatter.format_price(signal.entry_price),
                'current': SignalFormatter.format_price(current),
                'pnl': SignalFormatter.format_pnl(pnl),
                'status': status
            })
        
        return rows
    
    @staticmethod
    def format_performance_stats(stats: Dict[str, Any]) -> Dict[str, str]:
        """Formata estatísticas de performance"""
        return {
            'total_signals': str(stats.get('total_signals', 0)),
            'win_rate': SignalFormatter.format_percentage(stats.get('win_rate', 0)),
            'profit_factor': f"{stats.get('profit_factor', 0):.2f}",
            'avg_win': SignalFormatter.format_price(Decimal(str(stats.get('avg_win', 0)))),
            'avg_loss': SignalFormatter.format_price(Decimal(str(stats.get('avg_loss', 0)))),
            'best_trade': SignalFormatter.format_price(Decimal(str(stats.get('best_trade', 0)))),
            'worst_trade': SignalFormatter.format_price(Decimal(str(stats.get('worst_trade', 0)))),
            'total_pnl': SignalFormatter.format_pnl(Decimal(str(stats.get('total_pnl', 0))))
        }