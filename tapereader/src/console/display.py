"""
Console Display Básico para TapeReader
Implementação mínima para visualizar sinais
"""

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from datetime import datetime
from decimal import Decimal
from typing import List, Dict, Any
import asyncio

from src.core.models import TradingSignal, BehaviorDetection
from src.console.formatter import SignalFormatter, TableFormatter
from src.console.templates import ConsoleTemplates


class TapeReaderConsole:
    """Console visual do TapeReader"""
    
    def __init__(self):
        self.console = Console()
        self.active_signals: List[TradingSignal] = []
        self.recent_behaviors: Dict[str, List[BehaviorDetection]] = {
            'DOLFUT': [],
            'WDOFUT': []
        }
        self.market_data: Dict[str, Dict[str, Any]] = {
            'DOLFUT': {},
            'WDOFUT': {}
        }
        
    def display_signal(self, signal: TradingSignal):
        """Exibe um novo sinal"""
        # Formata dados do sinal
        signal_data = SignalFormatter.format_signal_summary(signal)
        
        # Cria painel usando template
        panel = ConsoleTemplates.signal_alert_panel(signal_data, f"ENTRADA - {signal.asset}")
        
        # Exibe com destaque
        self.console.print("")
        self.console.print(panel)
        self.console.print("")
        
        # Adiciona aos sinais ativos
        self.active_signals.append(signal)
        
        # Toca alerta sonoro (opcional)
        self.console.bell()
        
    def update_market_data(self, asset: str, data: Dict[str, Any]):
        """Atualiza dados de mercado"""
        self.market_data[asset] = data
        
    def update_behaviors(self, asset: str, behaviors: List[BehaviorDetection]):
        """Atualiza comportamentos detectados"""
        self.recent_behaviors[asset] = behaviors[-5:]  # Últimos 5
        
    def display_full_status(self):
        """Exibe status completo usando templates"""
        self.console.clear()
        
        # Header
        self.console.print(ConsoleTemplates.header_panel())
        self.console.print("")
        
        # Market Overview
        dol_stats = SignalFormatter.format_market_stats(self.market_data['DOLFUT'])
        wdo_stats = SignalFormatter.format_market_stats(self.market_data['WDOFUT'])
        market_table = ConsoleTemplates.market_overview_table(dol_stats, wdo_stats)
        self.console.print(market_table)
        self.console.print("")
        
        # Active Signals
        current_prices = {
            'DOLFUT': Decimal(str(self.market_data['DOLFUT'].get('current_price', 0))),
            'WDOFUT': Decimal(str(self.market_data['WDOFUT'].get('current_price', 0)))
        }
        
        signals_data = TableFormatter.format_active_signals(
            self.active_signals[-5:],  # Últimos 5 sinais
            current_prices
        )
        signals_table = ConsoleTemplates.active_signals_table(signals_data)
        self.console.print(signals_table)
        
        # Behaviors Summary
        behaviors_summary = {
            'DOLFUT': [b.behavior_type for b in self.recent_behaviors['DOLFUT']],
            'WDOFUT': [b.behavior_type for b in self.recent_behaviors['WDOFUT']]
        }
        
        if any(behaviors_summary.values()):
            self.console.print("")
            behaviors_panel = ConsoleTemplates.behavior_summary_panel(behaviors_summary)
            self.console.print(behaviors_panel)
    
    async def start_live_display(self, refresh_rate: float = 1.0):
        """Inicia display ao vivo com atualização automática"""
        with Live(console=self.console, refresh_per_second=1/refresh_rate) as live:
            while True:
                # Cria layout atualizado
                layout = Layout()
                layout.split_column(
                    Layout(name="header", size=5),
                    Layout(name="main", ratio=1)
                )
                
                # Header
                layout["header"].update(ConsoleTemplates.header_panel())
                
                # Main content em duas colunas
                main_layout = Layout()
                main_layout.split_row(
                    Layout(name="left", ratio=1),
                    Layout(name="right", ratio=1)
                )
                
                # Coluna esquerda - Market e Signals
                left_content = Layout()
                left_content.split_column(
                    Layout(self._create_market_table(), size=10),
                    Layout(self._create_signals_table())
                )
                main_layout["left"].update(left_content)
                
                # Coluna direita - Behaviors e Stats
                right_content = Layout()
                right_content.split_column(
                    Layout(self._create_behaviors_panel(), size=8),
                    Layout(self._create_stats_panel())
                )
                main_layout["right"].update(right_content)
                
                layout["main"].update(main_layout)
                
                # Atualiza display
                live.update(layout)
                await asyncio.sleep(refresh_rate)
    
    def _create_market_table(self) -> Table:
        """Cria tabela de mercado para live display"""
        dol_stats = SignalFormatter.format_market_stats(self.market_data['DOLFUT'])
        wdo_stats = SignalFormatter.format_market_stats(self.market_data['WDOFUT'])
        return ConsoleTemplates.market_overview_table(dol_stats, wdo_stats)
    
    def _create_signals_table(self) -> Table:
        """Cria tabela de sinais para live display"""
        current_prices = {
            'DOLFUT': Decimal(str(self.market_data['DOLFUT'].get('current_price', 0))),
            'WDOFUT': Decimal(str(self.market_data['WDOFUT'].get('current_price', 0)))
        }
        
        signals_data = TableFormatter.format_active_signals(
            self.active_signals[-10:],  # Últimos 10 sinais
            current_prices
        )
        return ConsoleTemplates.active_signals_table(signals_data)
    
    def _create_behaviors_panel(self) -> Panel:
        """Cria painel de comportamentos para live display"""
        behaviors_summary = {
            'DOLFUT': [b.behavior_type for b in self.recent_behaviors['DOLFUT']],
            'WDOFUT': [b.behavior_type for b in self.recent_behaviors['WDOFUT']]
        }
        return ConsoleTemplates.behavior_summary_panel(behaviors_summary)
    
    def _create_stats_panel(self) -> Panel:
        """Cria painel de estatísticas para live display"""
        # Simula estatísticas (em produção, viria do signal tracker)
        stats = {
            'total_signals': str(len(self.active_signals)),
            'win_rate': "0.0%",  # Calculado em tempo real
            'profit_factor': "0.00",
            'avg_win': "0.00",
            'avg_loss': "0.00",
            'best_trade': "0.00",
            'worst_trade': "0.00",
            'total_pnl': "0.00"
        }
        return ConsoleTemplates.performance_panel(stats)


# Exemplo de uso
if __name__ == "__main__":
    import asyncio
    from src.core.models import Side
    
    async def demo():
        console = TapeReaderConsole()
        
        # Simula dados de mercado
        console.update_market_data('DOLFUT', {
            'current_price': Decimal('5750.50'),
            'session_volume': 1250,
            'trades_count': 450,
            'current_volatility': 0.0012,
            'last_update': datetime.now()
        })
        
        console.update_market_data('WDOFUT', {
            'current_price': Decimal('5751.00'),
            'session_volume': 4800,
            'trades_count': 1200,
            'current_volatility': 0.0015,
            'last_update': datetime.now()
        })
        
        # Simula comportamentos
        behaviors = [
            BehaviorDetection(
                behavior_type='absorption',
                confidence=0.85,
                detected=True,
                metadata={'absorption_side': Side.BUY}
            ),
            BehaviorDetection(
                behavior_type='exhaustion',
                confidence=0.78,
                detected=True,
                metadata={'direction': Side.SELL}
            )
        ]
        console.update_behaviors('DOLFUT', behaviors)
        
        # Simula sinal
        signal = TradingSignal(
            signal_id="DEMO_001",
            timestamp=datetime.now(),
            direction=Side.BUY,
            asset="DOLFUT",
            price=Decimal("5750.00"),
            confidence=0.85,
            behaviors_detected=["absorption", "exhaustion"],
            primary_motivation="Reversão após absorção + exaustão vendedora",
            entry_price=Decimal("5750.50"),
            stop_loss=Decimal("5748.00"),
            take_profit_1=Decimal("5755.00"),
            risk_reward_ratio=2.0,
            position_size_suggestion=10
        )
        
        # Exibe sinal
        console.display_signal(signal)
        
        # Espera um pouco
        await asyncio.sleep(2)
        
        # Exibe status completo
        console.display_full_status()
        
        # Demonstra display ao vivo (por 10 segundos)
        print("\n\nIniciando display ao vivo por 10 segundos...")
        
        # Cria task para atualizar dados
        async def update_data():
            for i in range(10):
                # Simula mudança de preço
                new_price = Decimal('5750.50') + Decimal(str(i * 0.5))
                console.update_market_data('DOLFUT', {
                    'current_price': new_price,
                    'session_volume': 1250 + i * 50,
                    'trades_count': 450 + i * 10,
                    'current_volatility': 0.0012,
                    'last_update': datetime.now()
                })
                await asyncio.sleep(1)
        
        # Executa display ao vivo e atualizações em paralelo
        try:
            await asyncio.gather(
                console.start_live_display(refresh_rate=0.5),
                update_data()
            )
        except asyncio.CancelledError:
            pass
        except KeyboardInterrupt:
            print("\n\nDisplay ao vivo encerrado.")
    
    # Executa demo
    asyncio.run(demo())