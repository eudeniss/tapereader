"""
Templates visuais para o console do TapeReader
"""

from decimal import Decimal
from rich.panel import Panel
from rich.table import Table
from rich.columns import Columns
from rich.text import Text
from rich.box import ROUNDED, DOUBLE
from typing import Dict, List, Any


class ConsoleTemplates:
    """Templates pré-definidos para o console"""
    
    @staticmethod
    def header_panel() -> Panel:
        """Cria painel de cabeçalho"""
        header_text = """[bold cyan]TAPEREADER PROFESSIONAL v2.0[/bold cyan]
[dim]Sistema de Análise de Fluxo de Ordens[/dim]
[yellow]Meta: 80%+ de Acurácia[/yellow]"""
        
        return Panel(
            header_text,
            title="[bold]📊 Tape Reader[/bold]",
            border_style="bright_blue",
            box=DOUBLE,
            expand=False
        )
    
    @staticmethod
    def signal_alert_panel(signal_data: Dict[str, str], signal_type: str = "ENTRADA") -> Panel:
        """Cria painel de alerta de sinal"""
        color = "green" if "BUY" in signal_data['direction'] else "red"
        
        content = f"""
[bold]{signal_data['direction']}[/bold] @ [cyan]{signal_data['price']}[/cyan]
[dim]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/dim]
📊 Confiança: [yellow]{signal_data['confidence']}[/yellow]
🎯 Alvo: [green]{signal_data['target']}[/green] | 🛑 Stop: [red]{signal_data['stop']}[/red]
📈 R:R: [cyan]{signal_data['risk_reward']}[/cyan]
🔍 Comportamentos: [dim]{signal_data['behaviors']}[/dim]
⏰ Hora: [white]{signal_data['timestamp']}[/white]
"""
        
        return Panel(
            content.strip(),
            title=f"[bold]🚨 SINAL DE {signal_type}[/bold]",
            border_style=color,
            box=ROUNDED,
            expand=False
        )
    
    @staticmethod
    def market_overview_table(dol_stats: Dict[str, str], wdo_stats: Dict[str, str]) -> Table:
        """Cria tabela de visão geral do mercado"""
        table = Table(
            title="[bold]📈 Visão Geral do Mercado[/bold]",
            show_header=True,
            header_style="bold cyan",
            box=ROUNDED
        )
        
        # Colunas
        table.add_column("Métrica", style="dim", width=15)
        table.add_column("DOLFUT", style="yellow", justify="right")
        table.add_column("WDOFUT", style="yellow", justify="right")
        
        # Linhas
        table.add_row("Preço", dol_stats['price'], wdo_stats['price'])
        table.add_row("Volume", dol_stats['volume'], wdo_stats['volume'])
        table.add_row("Trades", dol_stats['trades'], wdo_stats['trades'])
        table.add_row("Volatilidade", dol_stats['volatility'], wdo_stats['volatility'])
        table.add_row("Atualização", dol_stats['last_update'], wdo_stats['last_update'])
        
        return table
    
    @staticmethod
    def active_signals_table(signals_data: List[Dict[str, str]]) -> Table:
        """Cria tabela de sinais ativos"""
        table = Table(
            title="[bold]🎯 Sinais Ativos[/bold]",
            show_header=True,
            header_style="bold cyan",
            box=ROUNDED
        )
        
        # Colunas
        table.add_column("Hora", style="dim", width=8)
        table.add_column("Ativo", style="yellow")
        table.add_column("Direção", justify="center")
        table.add_column("Entrada", style="cyan", justify="right")
        table.add_column("Atual", style="white", justify="right")
        table.add_column("P&L", justify="right")
        table.add_column("Status", justify="center")
        
        # Adiciona linhas
        for signal in signals_data:
            table.add_row(
                signal['time'],
                signal['asset'],
                signal['direction'],
                signal['entry'],
                signal['current'],
                signal['pnl'],
                signal['status']
            )
        
        if not signals_data:
            table.add_row(
                "-", "-", "-", "-", "-", "-", "[dim]Sem sinais[/dim]"
            )
        
        return table
    
    @staticmethod
    def performance_panel(stats: Dict[str, str]) -> Panel:
        """Cria painel de performance"""
        content = f"""
[bold]📊 Performance da Sessão[/bold]
[dim]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/dim]

Total de Sinais: [cyan]{stats['total_signals']}[/cyan]
Taxa de Acerto: [green]{stats['win_rate']}[/green]
Profit Factor: [yellow]{stats['profit_factor']}[/yellow]

Média Ganho: [green]{stats['avg_win']}[/green]
Média Perda: [red]{stats['avg_loss']}[/red]

Melhor Trade: [green]{stats['best_trade']}[/green]
Pior Trade: [red]{stats['worst_trade']}[/red]

[bold]P&L Total: {stats['total_pnl']}[/bold]
"""
        
        return Panel(
            content.strip(),
            border_style="bright_blue",
            box=ROUNDED,
            expand=False
        )
    
    @staticmethod
    def behavior_summary_panel(behaviors: Dict[str, List[str]]) -> Panel:
        """Cria painel de resumo de comportamentos"""
        lines = ["[bold]🔍 Comportamentos Detectados[/bold]", ""]
        
        for asset, behavior_list in behaviors.items():
            if behavior_list:
                behaviors_str = " | ".join(behavior_list)
                lines.append(f"[yellow]{asset}:[/yellow] {behaviors_str}")
            else:
                lines.append(f"[yellow]{asset}:[/yellow] [dim]Nenhum[/dim]")
        
        return Panel(
            "\n".join(lines),
            border_style="blue",
            box=ROUNDED,
            expand=False
        )
    
    @staticmethod
    def risk_status_panel(risk_data: Dict[str, Any]) -> Panel:
        """Cria painel de status de risco"""
        level_colors = {
            'LOW': 'green',
            'MEDIUM': 'yellow',
            'HIGH': 'orange',
            'EXTREME': 'red'
        }
        
        risk_level = risk_data.get('current_risk_level', 'LOW')
        color = level_colors.get(risk_level, 'white')
        
        content = f"""
[bold]🛡️ Gestão de Risco[/bold]
[dim]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/dim]

Nível de Risco: [{color}]{risk_level}[/{color}]
P&L Diário: {SignalFormatter.format_pnl(Decimal(str(risk_data.get('daily_pnl', 0))))}
Posições Abertas: [cyan]{risk_data.get('open_positions', 0)}[/cyan]/{risk_data.get('max_positions', 3)}

Circuit Breaker: [{'red]ATIVO' if risk_data.get('circuit_breaker_active') else 'green]INATIVO'}[/]
"""
        
        return Panel(
            content.strip(),
            border_style=color,
            box=ROUNDED,
            expand=False
        )
    
    @staticmethod
    def create_full_layout(components: Dict[str, Any]) -> Columns:
        """Cria layout completo com múltiplos componentes"""
        columns = []
        
        # Coluna esquerda - Market Overview e Active Signals
        left_column = [
            components.get('market_overview'),
            Text(""),  # Espaço
            components.get('active_signals')
        ]
        
        # Coluna direita - Performance e Risk
        right_column = [
            components.get('performance'),
            Text(""),  # Espaço
            components.get('risk_status'),
            Text(""),  # Espaço
            components.get('behaviors')
        ]
        
        return Columns([left_column, right_column], expand=True)


# Importar formatter para uso nos templates
from src.console.formatter import SignalFormatter, TableFormatter