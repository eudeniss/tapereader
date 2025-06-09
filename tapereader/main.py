"""
TapeReader Professional v2.0 - Arquivo Principal
Sistema completo de análise de fluxo de ordens
COM PONDERAÇÃO DINÂMICA, INJEÇÃO DE DEPENDÊNCIAS E PERSISTÊNCIA DE ESTADO
"""

import asyncio
import logging
import argparse
import sys
import os
import json
import orjson
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

try:
    from src.strategies import StrategyLoader
    from src.behaviors import BehaviorManager
    from src.core.models import MarketData, BehaviorDetection, TradingSignal
    # NOVO: Importa RegimeClassifier
    from src.strategies.regime_classifier import RegimeClassifier, MarketRegime
    from src.strategies import ConfluenceAnalyzer, DecisionMatrix, RiskManager
    from src.core import SignalTracker
    # CORREÇÃO: Importar o loader de configuração principal
    from src.core.config import load_config
except ImportError as e:
    print(f"❌ ERRO ao importar módulos principais: {e}")
    print("Verifique se a estrutura de pastas está correta:")
    print("  tapereader/")
    print("  └── src/")
    print("      ├── strategies/")
    print("      ├── behaviors/")
    print("      └── core/")
    exit(1)

# Importação opcional do console
try:
    from src.console.display import TapeReaderConsole
    CONSOLE_AVAILABLE = True
except ImportError:
    print("⚠️ AVISO: Console visual não disponível. Usando console simplificado.")
    CONSOLE_AVAILABLE = False
    
    # Console simplificado como fallback
    class TapeReaderConsole:
        """Console simplificado para quando o módulo completo não está disponível"""
        
        def __init__(self):
            self.active_signals = []
            self.market_data = {'DOLFUT': {}, 'WDOFUT': {}}
            self.recent_behaviors = {'DOLFUT': [], 'WDOFUT': []}
            
        def display_signal(self, signal):
            """Exibe sinal de forma simplificada"""
            direction = "🟢 COMPRA" if signal.direction.value == "BUY" else "🔴 VENDA"
            print(f"\n{'='*60}")
            print(f"🚨 NOVO SINAL - {signal.asset}")
            print(f"{direction} @ {signal.entry_price}")
            print(f"Confiança: {signal.confidence:.1%}")
            print(f"Stop: {signal.stop_loss} | Alvo: {signal.take_profit_1}")
            print(f"Motivo: {signal.primary_motivation}")
            print(f"{'='*60}\n")
            self.active_signals.append(signal)
            
        def update_market_data(self, asset, data):
            """Atualiza dados de mercado"""
            self.market_data[asset] = data
            
        def update_behaviors(self, asset, behaviors):
            """Atualiza comportamentos"""
            self.recent_behaviors[asset] = behaviors[-5:]
            
        def display_full_status(self):
            """Exibe status simplificado"""
            print(f"\r[{datetime.now().strftime('%H:%M:%S')}] ", end="")
            print(f"DOL: {self.market_data['DOLFUT'].get('current_price', 'N/A')} ", end="")
            print(f"WDO: {self.market_data['WDOFUT'].get('current_price', 'N/A')} ", end="")
            print(f"Sinais: {len(self.active_signals)} ", end="")
            sys.stdout.flush()


# NOVO: Função de bootstrap para criar componentes
def bootstrap_components(config_path: str = "config/strategies.yaml") -> Dict[str, Any]:
    """
    Função de bootstrap para criar todos os componentes com injeção de dependências
    
    Returns:
        Dict com todos os componentes instanciados
    """
    logger = logging.getLogger('Bootstrap')
    logger.info("Inicializando componentes via bootstrap...")
    
    # 1. Carrega configuração de estratégias
    strategy_loader = StrategyLoader(config_path)
    
    # 2. Extrai componentes individuais
    confluence_analyzer = strategy_loader.confluence_analyzer
    decision_matrix = strategy_loader.decision_matrix
    risk_manager = strategy_loader.risk_manager
    signal_tracker = strategy_loader.signal_tracker
    
    # 3. Cria classificador de regime
    regime_classifier = RegimeClassifier()
    
    # 4. Carrega configuração de behaviors
    behaviors_config = {
        'behaviors': {
            'absorption': {'enabled': True, 'min_confidence': 0.7},
            'exhaustion': {'enabled': True, 'min_confidence': 0.7},
            'institutional': {'enabled': True, 'min_confidence': 0.75},
            'support_resistance': {'enabled': True, 'min_confidence': 0.7},
            'sweep': {'enabled': True, 'min_confidence': 0.7},
            'stop_hunt': {'enabled': True, 'min_confidence': 0.75},
            'iceberg': {'enabled': True, 'min_confidence': 0.75},
            'momentum': {'enabled': True, 'min_confidence': 0.7},
            'breakout': {'enabled': True, 'min_confidence': 0.75},
            'divergence': {'enabled': True, 'min_confidence': 0.7},
            'htf': {'enabled': True, 'min_confidence': 0.75},
            'micro_aggression': {'enabled': True, 'min_confidence': 0.7},
            'recurrence': {'enabled': True, 'min_confidence': 0.75},
            'renovation': {'enabled': True, 'min_confidence': 0.7}
        }
    }
    
    # 5. Cria behavior manager
    behavior_manager = BehaviorManager(behaviors_config)
    
    # 6. Cria console se disponível
    console = TapeReaderConsole() if CONSOLE_AVAILABLE else None
    
    logger.info("✅ Todos os componentes criados com sucesso")
    
    return {
        'strategy_loader': strategy_loader,
        'confluence_analyzer': confluence_analyzer,
        'decision_matrix': decision_matrix,
        'risk_manager': risk_manager,
        'signal_tracker': signal_tracker,
        'regime_classifier': regime_classifier,
        'behavior_manager': behavior_manager,
        'console': console,
        'behaviors_config': behaviors_config
    }


class TapeReaderSystem:
    """Sistema principal do TapeReader com Injeção de Dependências e Persistência"""
    
    def __init__(
        self,
        confluence_analyzer: ConfluenceAnalyzer,
        decision_matrix: DecisionMatrix,
        risk_manager: RiskManager,
        signal_tracker: SignalTracker,
        regime_classifier: RegimeClassifier,
        behavior_manager: BehaviorManager,
        console: Optional[TapeReaderConsole] = None,
        use_console: bool = True
    ):
        """
        Inicializa o sistema com componentes injetados
        
        Args:
            confluence_analyzer: Analisador de confluência
            decision_matrix: Matriz de decisão
            risk_manager: Gerenciador de risco
            signal_tracker: Rastreador de sinais
            regime_classifier: Classificador de regime
            behavior_manager: Gerenciador de comportamentos
            console: Console visual (opcional)
            use_console: Se deve usar console visual
        """
        # Configurar logging
        self.logger = logging.getLogger('TapeReader')
        self.logger.info("Inicializando TapeReader Professional v2.0 com DI")
        
        # Componentes injetados
        self.confluence_analyzer = confluence_analyzer
        self.decision_matrix = decision_matrix
        self.risk_manager = risk_manager
        self.signal_tracker = signal_tracker
        self.regime_classifier = regime_classifier
        self.behavior_manager = behavior_manager
        self.current_regime = MarketRegime.UNKNOWN
        
        # Console visual
        self.use_console = use_console and console is not None
        self.console = console
        
        # Estado do sistema
        self.active_signals: Dict[str, TradingSignal] = {}
        self.market_context: Dict[str, Dict[str, Any]] = {
            'DOLFUT': {},
            'WDOFUT': {}
        }
        
        # Event-driven queue
        self.data_queue = asyncio.Queue()
        
        # Estado do sistema
        self.running = True
        
        # NOVO: Configurações de persistência
        self.state_file = Path("data/system_state.json")
        self.autosave_interval = 300  # 5 minutos
        self.last_save_time = datetime.now()
        
        # NOVO: Carrega estado anterior se existir
        self._load_state()
        
        self.logger.info("Sistema inicializado com sucesso via DI")
    
    # NOVO: Factory method alternativo para criar com config
    @classmethod
    def create_from_config(cls, config_path: str = "config/strategies.yaml", use_console: bool = True):
        """
        Factory method para criar sistema a partir de arquivo de config
        
        Args:
            config_path: Caminho para arquivo YAML
            use_console: Se deve usar console visual
            
        Returns:
            TapeReaderSystem configurado
        """
        components = bootstrap_components(config_path)
        
        return cls(
            confluence_analyzer=components['confluence_analyzer'],
            decision_matrix=components['decision_matrix'],
            risk_manager=components['risk_manager'],
            signal_tracker=components['signal_tracker'],
            regime_classifier=components['regime_classifier'],
            behavior_manager=components['behavior_manager'],
            console=components['console'],
            use_console=use_console
        )
    
    # NOVO: Métodos de Persistência de Estado
    def _save_state(self):
        """Salva estado completo do sistema"""
        try:
            # Prepara estado para serialização
            state = {
                'timestamp': datetime.now().isoformat(),
                'version': '2.0',
                'system': {
                    'current_regime': self.current_regime.value,
                    'running': self.running,
                    'active_signals': {
                        signal_id: {
                            'asset': signal.asset,
                            'direction': signal.direction.value,
                            'entry_price': float(signal.entry_price),
                            'stop_loss': float(signal.stop_loss),
                            'take_profit_1': float(signal.take_profit_1),
                            'confidence': signal.confidence,
                            'strategy': signal.strategy,
                            'signal_id': signal.signal_id,
                            'timestamp': signal.timestamp.isoformat()
                        }
                        for signal_id, signal in self.active_signals.items()
                    },
                    'market_context': {
                        asset: {
                            'current_price': ctx.get('current_price'),
                            'session_volume': ctx.get('session_volume'),
                            'trades_count': ctx.get('trades_count'),
                            'current_volatility': ctx.get('current_volatility'),
                            'last_update': ctx.get('last_update').isoformat() if ctx.get('last_update') else None
                        }
                        for asset, ctx in self.market_context.items()
                    }
                },
                'components': {
                    'risk_manager': self.risk_manager.get_state() if hasattr(self.risk_manager, 'get_state') else {},
                    'signal_tracker': self.signal_tracker.get_state() if hasattr(self.signal_tracker, 'get_state') else {},
                    'regime_classifier': {
                        'current_regime': self.current_regime.value,
                        'confidence': self.regime_classifier.regime_confidence if hasattr(self.regime_classifier, 'regime_confidence') else 0
                    }
                }
            }
            
            # Cria diretório se não existir
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Faz backup rotativo antes de salvar
            self._rotate_backups()
            
            # Salva usando orjson para performance
            with open(self.state_file, 'wb') as f:
                f.write(orjson.dumps(state, option=orjson.OPT_INDENT_2))
            
            self.last_save_time = datetime.now()
            self.logger.debug(f"Estado salvo em {self.state_file}")
            
        except Exception as e:
            self.logger.error(f"Erro ao salvar estado: {e}")
    
    def _load_state(self):
        """Carrega estado anterior do sistema"""
        try:
            if not self.state_file.exists():
                self.logger.info("Nenhum estado anterior encontrado")
                return
            
            with open(self.state_file, 'rb') as f:
                state = orjson.loads(f.read())
            
            # Valida versão
            if state.get('version') != '2.0':
                self.logger.warning("Versão de estado incompatível, ignorando")
                return
            
            # Restaura estado do sistema
            system_state = state.get('system', {})
            
            # Restaura regime
            try:
                self.current_regime = MarketRegime(system_state.get('current_regime', 'UNKNOWN'))
            except:
                self.current_regime = MarketRegime.UNKNOWN
            
            # Restaura contexto de mercado
            for asset, ctx in system_state.get('market_context', {}).items():
                if asset in self.market_context:
                    self.market_context[asset].update({
                        'current_price': ctx.get('current_price'),
                        'session_volume': ctx.get('session_volume', 0),
                        'trades_count': ctx.get('trades_count', 0),
                        'current_volatility': ctx.get('current_volatility', 0),
                        'last_update': datetime.fromisoformat(ctx['last_update']) if ctx.get('last_update') else None
                    })
            
            # Restaura componentes
            components_state = state.get('components', {})
            
            # Risk Manager
            if hasattr(self.risk_manager, 'load_state') and 'risk_manager' in components_state:
                self.risk_manager.load_state(components_state['risk_manager'])
            
            # Signal Tracker
            if hasattr(self.signal_tracker, 'load_state') and 'signal_tracker' in components_state:
                self.signal_tracker.load_state(components_state['signal_tracker'])
            
            # Restaura sinais ativos (validados)
            self._restore_active_signals(system_state.get('active_signals', {}))
            
            self.logger.info(
                f"Estado restaurado de {state['timestamp']} - "
                f"{len(self.active_signals)} sinais ativos"
            )
            
        except Exception as e:
            self.logger.error(f"Erro ao carregar estado: {e}")
    
    def _restore_active_signals(self, signals_data: Dict[str, Dict]):
        """Restaura e valida sinais ativos"""
        now = datetime.now()
        restored_count = 0
        expired_count = 0
        
        for signal_id, signal_data in signals_data.items():
            try:
                # Recria objeto TradingSignal
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
                
                # Valida idade do sinal (máximo 1 hora)
                age = now - signal.timestamp
                if age < timedelta(hours=1):
                    self.active_signals[signal_id] = signal
                    restored_count += 1
                else:
                    expired_count += 1
                    
            except Exception as e:
                self.logger.error(f"Erro ao restaurar sinal {signal_id}: {e}")
        
        if restored_count > 0 or expired_count > 0:
            self.logger.info(
                f"Sinais: {restored_count} restaurados, {expired_count} expirados"
            )
    
    def _rotate_backups(self):
        """Mantém backups rotativos do estado"""
        try:
            max_backups = 5
            backup_dir = self.state_file.parent / "backups"
            backup_dir.mkdir(exist_ok=True)
            
            # Lista backups existentes
            backups = sorted(backup_dir.glob("system_state_*.json"))
            
            # Remove backups antigos
            while len(backups) >= max_backups:
                oldest = backups.pop(0)
                oldest.unlink()
            
            # Cria novo backup
            if self.state_file.exists():
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = backup_dir / f"system_state_{timestamp}.json"
                
                with open(self.state_file, 'rb') as src:
                    with open(backup_path, 'wb') as dst:
                        dst.write(src.read())
                        
        except Exception as e:
            self.logger.error(f"Erro ao rotacionar backups: {e}")
    
    async def autosave_state(self):
        """Task assíncrona para salvar estado periodicamente"""
        while self.running:
            try:
                await asyncio.sleep(self.autosave_interval)
                
                # Verifica se passou tempo suficiente
                if (datetime.now() - self.last_save_time).total_seconds() >= self.autosave_interval:
                    self._save_state()
                    self.logger.info("✅ Autosave realizado")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Erro no autosave: {e}")
    
    def shutdown(self):
        """Desliga o sistema salvando o estado"""
        self.logger.info("Iniciando shutdown do sistema...")
        
        # Para o loop principal
        self.running = False
        
        # Salva estado final
        self._save_state()
        
        # Salva estado de emergência dos componentes
        if hasattr(self.risk_manager, 'emergency_save'):
            self.risk_manager.emergency_save()
        
        self.logger.info("✅ Sistema desligado com sucesso")
        
    async def process_data(self):
        """Consome dados da fila e processa (event-driven)"""
        while self.running:
            try:
                # Aguarda dados na fila
                asset, market_data = await self.data_queue.get()
                
                # NOVO: Classifica regime de mercado antes de processar
                await self._classify_and_update_regime(asset, market_data)
                
                # Processa os dados
                await self.process_market_data(asset, market_data)
                
                # Marca como processado
                self.data_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Erro ao processar dados da fila: {e}")
                
    async def _classify_and_update_regime(self, asset: str, market_data: MarketData):
        """NOVO: Classifica e atualiza regime de mercado"""
        try:
            # Prepara dados para classificação
            classification_data = {
                'price_data': [
                    {'price': t.price, 'volume': t.volume, 'timestamp': t.timestamp}
                    for t in market_data.trades[-100:]
                ] if market_data.trades else [],
                'volume_data': market_data.book.volume_profile if hasattr(market_data, 'book') and hasattr(market_data.book, 'volume_profile') else {},
                'book_data': {
                    'bids': market_data.book.bids if hasattr(market_data, 'book') else [],
                    'asks': market_data.book.asks if hasattr(market_data, 'book') else [],
                    'last_price': market_data.trades[-1].price if market_data.trades else None
                }
            }
            
            # Classifica regime
            new_regime = self.regime_classifier.classify(classification_data)
            
            # Atualiza se mudou
            if new_regime != self.current_regime:
                self.logger.info(f"🔄 Regime de mercado mudou: {self.current_regime} → {new_regime}")
                self.current_regime = new_regime
                
                # Atualiza tracker
                if hasattr(self.signal_tracker, 'update_market_regime'):
                    self.signal_tracker.update_market_regime(new_regime.value)
                
                # Ajusta sistema para novo regime
                await self._adjust_system_for_regime(new_regime)
                
        except Exception as e:
            self.logger.error(f"Erro ao classificar regime: {e}")
                
    async def process_market_data(self, asset: str, market_data: MarketData):
        """
        Processa dados de mercado e gera sinais
        
        Este é o método principal que:
        1. Detecta comportamentos
        2. Analisa confluência
        3. Gera sinais
        4. Valida com risk manager
        """
        self.logger.debug(f"Processando dados para {asset}")
        
        # 1. Atualiza contexto de mercado
        self._update_market_context(asset, market_data)
        
        # 2. Detecta comportamentos
        behaviors = await self.behavior_manager.analyze(market_data)
        
        if behaviors:
            self.logger.info(
                f"{asset}: {len(behaviors)} comportamentos detectados - "
                f"{[b.behavior_type for b in behaviors]}"
            )
            
            # Atualiza console
            if self.use_console:
                self.console.update_behaviors(asset, behaviors)
        
        # 3. Atualiza análise de confluência
        self.confluence_analyzer.update_market_data(asset, market_data)
        self.confluence_analyzer.update_behaviors(asset, behaviors)
        
        # 4. Se há comportamentos, tenta gerar sinal
        if behaviors:
            await self._try_generate_signal(asset, behaviors)
            
        # 5. Atualiza sinais ativos
        await self._update_active_signals(asset, market_data)
        
    async def _try_generate_signal(self, asset: str, behaviors: List[BehaviorDetection]):
        """Tenta gerar sinal de trading"""
        # Pega comportamentos do outro ativo para confluência
        other_asset = 'WDOFUT' if asset == 'DOLFUT' else 'DOLFUT'
        other_behaviors = self.behavior_manager.get_active_behaviors(other_asset)
        
        # Analisa confluência
        confluence = self.confluence_analyzer.analyze_confluence(
            asset,
            behaviors,
            other_behaviors
        )
        
        self.logger.info(
            f"Confluência {asset}: {confluence['level']} "
            f"(correlação: {confluence['correlation']:.2f}, "
            f"boost: {confluence['confidence_boost']:.2f})"
        )
        
        # Tenta gerar sinal
        signal = self.decision_matrix.generate_signal(
            asset,
            behaviors,
            confluence,
            self.market_context[asset]
        )
        
        if signal:
            # NOVO: Ajusta confiança baseado em performance histórica
            if hasattr(self.signal_tracker, 'get_strategy_confidence'):
                original_confidence = signal.confidence
                adjusted_confidence = self.signal_tracker.get_strategy_confidence(
                    signal.strategy,
                    original_confidence
                )
                
                if adjusted_confidence != original_confidence:
                    self.logger.info(
                        f"📊 Confiança ajustada para {signal.strategy}: "
                        f"{original_confidence:.1%} → {adjusted_confidence:.1%} "
                        f"(Regime: {self.current_regime})"
                    )
                    signal.confidence = adjusted_confidence
            
            # Valida com risk manager
            approved, reason, adjusted_signal = self.risk_manager.validate_signal(
                signal,
                self.market_context[asset]
            )
            
            if approved:
                self.logger.info(
                    f"✅ SINAL APROVADO: {asset} {signal.direction.value} "
                    f"@ {signal.entry_price} - Confiança: {signal.confidence:.2%}"
                )
                
                # Adiciona aos sinais ativos
                self.active_signals[signal.signal_id] = adjusted_signal
                self.risk_manager.add_open_position(adjusted_signal)
                
                # Notifica
                await self._notify_signal(adjusted_signal)
            else:
                self.logger.warning(
                    f"❌ Sinal rejeitado: {reason}"
                )
                
    async def _update_active_signals(self, asset: str, market_data: MarketData):
        """Atualiza sinais ativos (stop loss, take profit, etc)"""
        current_price = market_data.trades[-1].price if market_data.trades else None
        
        if not current_price:
            return
            
        for signal_id, signal in list(self.active_signals.items()):
            if signal.asset != asset:
                continue
                
            # Verifica stop loss
            if signal.direction == 'BUY' and current_price <= signal.stop_loss:
                pnl = current_price - signal.entry_price
                self.logger.warning(
                    f"🛑 STOP LOSS atingido: {signal_id} - P&L: {pnl}"
                )
                await self._close_position(signal_id, current_price, pnl)
                
            elif signal.direction == 'SELL' and current_price >= signal.stop_loss:
                pnl = signal.entry_price - current_price
                self.logger.warning(
                    f"🛑 STOP LOSS atingido: {signal_id} - P&L: {pnl}"
                )
                await self._close_position(signal_id, current_price, pnl)
                
            # Verifica take profit
            elif signal.direction == 'BUY' and current_price >= signal.take_profit_1:
                pnl = current_price - signal.entry_price
                self.logger.info(
                    f"🎯 TAKE PROFIT atingido: {signal_id} - P&L: {pnl}"
                )
                await self._close_position(signal_id, current_price, pnl)
                
            elif signal.direction == 'SELL' and current_price <= signal.take_profit_1:
                pnl = signal.entry_price - current_price
                self.logger.info(
                    f"🎯 TAKE PROFIT atingido: {signal_id} - P&L: {pnl}"
                )
                await self._close_position(signal_id, current_price, pnl)
                
    async def _close_position(self, signal_id: str, exit_price, pnl):
        """Fecha uma posição"""
        if signal_id in self.active_signals:
            signal = self.active_signals[signal_id]
            
            # NOVO: Atualiza tracking com resultado
            if hasattr(self.signal_tracker, 'update_signal_status'):
                from src.core.tracking import SignalStatus
                self.signal_tracker.update_signal_status(
                    signal_id,
                    SignalStatus.CLOSED,
                    {'pnl': pnl, 'exit_price': exit_price}
                )
            
            # Atualiza risk manager
            self.risk_manager.update_position_result(signal_id, pnl, closed=True)
            
            # Remove dos ativos
            del self.active_signals[signal_id]
            
    async def _adjust_system_for_regime(self, regime: MarketRegime):
        """NOVO: Ajusta sistema baseado no regime"""
        adjustments = self.regime_classifier.get_regime_adjustments(regime)
        
        # Ajusta comportamentos favorecidos/evitados
        for detector_name, detector in self.behavior_manager.detectors.items():
            if hasattr(detector, 'adjust_for_regime'):
                detector.adjust_for_regime(regime.value, adjustments)
        
        # Ajusta parâmetros da DecisionMatrix se necessário
        if hasattr(self.decision_matrix, 'set_regime_multipliers'):
            self.decision_matrix.set_regime_multipliers(adjustments)
            
        self.logger.info(
            f"Sistema ajustado para regime {regime}: "
            f"Favorece {adjustments['favored_behaviors']}, "
            f"Evita {adjustments['avoid_behaviors']}"
        )
            
    async def _notify_signal(self, signal: TradingSignal):
        """Notifica sobre novo sinal"""
        self.logger.info(f"📢 Notificando sinal: {signal.signal_id}")
        
        # Exibe no console visual
        if self.use_console:
            self.console.display_signal(signal)
        
    def _update_market_context(self, asset: str, market_data: MarketData):
        """Atualiza contexto de mercado"""
        if market_data.trades:
            current_price = market_data.trades[-1].price
            volume = sum(t.volume for t in market_data.trades)
            
            self.market_context[asset].update({
                'current_price': current_price,
                'session_volume': self.market_context[asset].get('session_volume', 0) + volume,
                'last_update': datetime.now(),
                'trades_count': self.market_context[asset].get('trades_count', 0) + len(market_data.trades)
            })
            
            # Calcula volatilidade simples
            if len(market_data.trades) > 1:
                prices = [t.price for t in market_data.trades]
                price_changes = [
                    abs(float(prices[i] - prices[i-1]))
                    for i in range(1, len(prices))
                ]
                volatility = sum(price_changes) / len(price_changes)
                self.market_context[asset]['current_volatility'] = volatility
                
        # Atualiza console
        if self.use_console:
            self.console.update_market_data(asset, self.market_context[asset])
                
    def get_system_status(self) -> Dict[str, Any]:
        """Retorna status completo do sistema"""
        # MODIFICADO: Inclui performance por estratégia
        performance_stats = {} 
        if hasattr(self.signal_tracker, 'get_statistics'):
            tracker_stats = self.signal_tracker.get_statistics()
            performance_stats = tracker_stats.get('strategy_performance', {})
            
        return {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'active_signals': len(self.active_signals),
            'active_signals_detail': [
                {
                    'id': s.signal_id,
                    'asset': s.asset,
                    'direction': s.direction.value,
                    'entry': float(s.entry_price),
                    'confidence': s.confidence
                }
                for s in self.active_signals.values()
            ],
            'risk_status': self.risk_manager.get_risk_status(),
            'behavior_stats': self.behavior_manager.get_statistics(),
            'confluence_stats': self.confluence_analyzer.get_statistics(),
            'decision_stats': self.decision_matrix.get_statistics(),
            'signal_performance': performance_stats,
            # NOVO: Adiciona regime atual
            'market_regime': {
                'current': self.current_regime.value,
                'confidence': self.regime_classifier.regime_confidence,
                'stats': self.regime_classifier.get_statistics()
            }
        }
    
    # NOVO: Método para mostrar estatísticas periodicamente
    async def show_performance_statistics(self):
        """Exibe estatísticas de performance periodicamente"""
        while self.running:
            try:
                await asyncio.sleep(300)  # A cada 5 minutos
                
                if hasattr(self.signal_tracker, 'get_statistics'):
                    stats = self.signal_tracker.get_statistics()
                    
                    self.logger.info("\n" + "="*60)
                    self.logger.info("📊 ESTATÍSTICAS DE PERFORMANCE")
                    self.logger.info("="*60)
                    
                    # Performance por estratégia
                    if 'strategy_performance' in stats:
                        self.logger.info("\n📈 Por Estratégia:")
                        for strategy, perf in stats['strategy_performance'].items():
                            self.logger.info(
                                f"  {strategy}: "
                                f"Win Rate: {perf['win_rate']:.1%} | "
                                f"P&L: {perf['total_pnl']:+.2f} | "
                                f"PF: {perf['profit_factor']:.2f} | "
                                f"Trades: {perf['total_trades']}"
                            )
                    
                    # Melhores estratégias no regime atual
                    if hasattr(self.signal_tracker, 'get_best_performing_strategies'):
                        best = self.signal_tracker.get_best_performing_strategies()
                        if best:
                            self.logger.info(f"\n🏆 Top 3 no regime {self.current_regime}:")
                            for i, (name, score) in enumerate(best[:3], 1):
                                self.logger.info(f"  {i}. {name} (Score: {score:.2f})")
                    
                    self.logger.info("="*60 + "\n")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Erro ao mostrar estatísticas: {e}")
        
    async def reload_strategies(self):
        """Recarrega estratégias do arquivo de configuração"""
        self.logger.info("Recarregando estratégias...")
        
        # Salva estado atual antes de recarregar
        self._save_state()
        
        # Com DI, precisa recriar componentes
        try:
            components = bootstrap_components()
            
            # Atualiza referências
            self.confluence_analyzer = components['confluence_analyzer']
            self.decision_matrix = components['decision_matrix']
            self.risk_manager = components['risk_manager']
            
            # Notifica componentes sobre reload
            self.logger.info("✅ Estratégias recarregadas com sucesso")
            
            # Emite evento de reload
            if hasattr(self, '_config_reload_callbacks'):
                for callback in self._config_reload_callbacks:
                    try:
                        await callback()
                    except Exception as e:
                        self.logger.error(f"Erro em callback de reload: {e}")
            
            return True
        except Exception as e:
            self.logger.error(f"❌ Falha ao recarregar estratégias: {e}")
            return False
    
    async def watch_config_changes(self):
        """Monitora mudanças na configuração e faz hot-reload"""
        try:
            from src.utils.config_watcher import ConfigWatcher
            
            self.logger.info("🔍 Iniciando monitoramento de configuração...")
            
            # Cria watcher
            watcher = ConfigWatcher(
                watch_paths=['config/'],
                file_patterns=['*.yaml', '*.yml'],
                callback=self._on_config_change
            )
            
            # Inicia monitoramento
            watcher.start()
            
            # Mantém watcher rodando
            while self.running:
                await asyncio.sleep(1)
                
            # Para watcher ao encerrar
            watcher.stop()
            
        except ImportError:
            self.logger.warning("watchdog não instalado - hot-reload desabilitado")
            self.logger.info("Instale com: pip install watchdog>=4.0.0")
        except Exception as e:
            self.logger.error(f"Erro no monitoramento de config: {e}")
    
    async def _on_config_change(self, filepath: Path):
        """Callback quando arquivo de config muda"""
        self.logger.info(f"📝 Mudança detectada em: {filepath}")
        
        # Aguarda um pouco para garantir que o arquivo foi salvo completamente
        await asyncio.sleep(0.5)
        
        # Valida se é um arquivo de estratégias
        if filepath.name == 'strategies.yaml':
            self.logger.info("🔄 Recarregando estratégias...")
            
            # Faz backup da config atual
            backup_file = filepath.parent / f".{filepath.stem}_backup.yaml"
            try:
                import shutil
                shutil.copy2(filepath, backup_file)
            except:
                pass
            
            # Tenta recarregar
            success = await self.reload_strategies()
            
            if success:
                self.logger.info("✅ Hot-reload concluído com sucesso!")
                
                # Notifica console se disponível
                if self.use_console and hasattr(self.console, 'show_notification'):
                    self.console.show_notification(
                        "🔄 Configuração recarregada",
                        "success"
                    )
            else:
                self.logger.error("❌ Falha no hot-reload - mantendo config anterior")
                
                # Tenta restaurar backup
                if backup_file.exists():
                    try:
                        shutil.copy2(backup_file, filepath)
                        self.logger.info("📋 Backup restaurado")
                    except:
                        pass
    
    def register_config_reload_callback(self, callback):
        """Registra callback para ser chamado após reload de config"""
        if not hasattr(self, '_config_reload_callbacks'):
            self._config_reload_callbacks = []
        self._config_reload_callbacks.append(callback)


async def run_production(args):
    """Executa em modo produção com dados reais"""
    try:
        # Tenta a importação do arquivo renomeado se o original não existir
        from src.data.excel_reader import ExcelRTDReader as ExcelRTDProvider
    except ImportError:
        print("❌ ERRO: Módulo de leitura Excel (excel_reader.py) não encontrado!")
        print("Verifique se o arquivo src/data/excel_reader.py existe.")
        return
    
    # Inicializa sistema
    print("\n🚀 Inicializando TapeReader Professional v2.0...")
    print(f"   Modo: {args.mode.upper()}")
    print(f"   Console: {'Visual' if CONSOLE_AVAILABLE else 'Simplificado'}")
    print(f"   Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("")

    # CORREÇÃO: Carrega a configuração completa uma vez
    try:
        full_config = load_config()
        full_config.current_mode = 'production'
    except Exception as e:
        print(f"❌ ERRO ao carregar configuração: {e}")
        logging.error(f"Erro ao carregar configuração: {e}")
        return
    
    # NOVO: Usa factory method
    system = TapeReaderSystem.create_from_config("config/strategies.yaml", use_console=True)
    
    # Configura provider de dados
    # CORREÇÃO FINAL: Passa o objeto de configuração completo para o DataProvider.
    data_provider = ExcelRTDProvider(full_config)
    
    try:
        await data_provider.connect()
    except Exception as e:
        print(f"\n❌ ERRO ao conectar ao Excel: {e}")
        print("\nVerifique:")
        print("1. Se o arquivo rtd_tapeReading.xlsx existe na pasta raiz do projeto")
        print("2. Se o Excel está instalado e aberto com a planilha")
        print("3. Se os dados RTD estão funcionando na planilha")
        print("\n💡 Dica: Execute em modo teste primeiro:")
        print("   python main.py --mode test")
        return
    
    print("\n" + "="*60)
    print("TAPEREADER PROFESSIONAL v2.0 - MODO PRODUÇÃO")
    print("="*60)
    print("\nSistema iniciado. Monitorando fluxo de ordens...")
    print("Pressione Ctrl+C para parar\n")
    
    # Inicia provider em modo event-driven
    await data_provider.start(system.data_queue)
    
    # Tasks paralelas
    async def update_display():
        """Atualiza display periodicamente"""
        if not system.use_console:
            while True:
                await asyncio.sleep(10)
                status = system.get_system_status()
                logging.info(f"Status: {status['active_signals']} sinais ativos")
        else:
            while True:
                try:
                    system.console.display_full_status()
                    await asyncio.sleep(2)
                except Exception as e:
                    logging.error(f"Erro ao atualizar display: {e}")
                    await asyncio.sleep(1)
    
    try:
        await asyncio.gather(
            system.process_data(),  # Consome da fila
            update_display(),
            system.show_performance_statistics(),  # NOVO: Task de estatísticas
            system.autosave_state(),  # NOVO: Task de autosave
            system.watch_config_changes()  # NOVO: Hot-reload de config
        )
    except KeyboardInterrupt:
        print("\n\nEncerrando sistema...")
        system.shutdown()  # NOVO: Usa shutdown ao invés de system.running = False
        await data_provider.disconnect()
        status = system.get_system_status()
        print("\n=== ESTATÍSTICAS DA SESSÃO ===")
        print(f"Sinais gerados: {status['decision_stats'].get('signals_generated', 0)}")
        print(f"P&L Total (simulado): {status['risk_status']['daily_pnl']:.2f}")
        
        # NOVO: Mostra performance por estratégia
        if 'signal_performance' in status and status['signal_performance']:
            print("\n=== PERFORMANCE POR ESTRATÉGIA ===")
            for strategy, perf in status['signal_performance'].items():
                print(f"{strategy}: Win Rate: {perf['win_rate']:.1%} | P&L: {perf['total_pnl']:+.2f}")


async def run_test(args):
    """Executa em modo teste com mock dinâmico"""
    try:
        from src.data.mock_dynamic import MockDynamicProvider
    except ImportError:
        print("❌ ERRO: Módulo de mock (mock_dynamic.py) não encontrado!")
        print("Verifique se o arquivo src/data/mock_dynamic.py existe.")
        return
    
    print("\n🚀 Inicializando TapeReader Professional v2.0...")
    print(f"   Modo: {args.mode.upper()}")
    print(f"   Console: {'Visual' if CONSOLE_AVAILABLE else 'Simplificado'}")
    print(f"   Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("")

    # CORREÇÃO: Carrega a configuração completa uma vez
    try:
        full_config = load_config()
        full_config.current_mode = 'test'
    except Exception as e:
        print(f"❌ ERRO ao carregar configuração: {e}")
        logging.error(f"Erro ao carregar configuração: {e}")
        return

    # NOVO: Usa factory method
    system = TapeReaderSystem.create_from_config("config/strategies.yaml", use_console=True)
    
    # CORREÇÃO: Passa a configuração completa para o MockDynamicProvider também
    data_provider = MockDynamicProvider(full_config)
    
    await data_provider.connect()
    
    print("\n" + "="*60)
    print("TAPEREADER PROFESSIONAL v2.0 - MODO TESTE")
    print("="*60)
    print("\nExecutando cenários de teste automaticamente...")
    
    # Inicia provider em modo event-driven
    await data_provider.start(system.data_queue)
    
    async def monitor_scenarios():
        """Monitora o progresso dos cenários"""
        last_scenario = None
        while data_provider.generating:
            if data_provider.current_scenario:
                current_scenario_name = data_provider.current_scenario.type.value
                if current_scenario_name != last_scenario:
                    total = data_provider.scenario_stats['total']
                    tested = data_provider.scenario_stats['tested'] + 1
                    print(f"\n📊 Testando cenário ({tested}/{total}): {current_scenario_name}")
                    last_scenario = current_scenario_name
            await asyncio.sleep(1)

    async def update_test_display():
        """Atualiza display durante testes"""
        if not system.use_console:
            return
        while True:
            system.console.display_full_status()
            await asyncio.sleep(1)
    
    try:
        # Tasks paralelas
        tasks = [
            asyncio.create_task(system.process_data()),  # Consome da fila
            asyncio.create_task(monitor_scenarios()),
            asyncio.create_task(system.show_performance_statistics()),  # NOVO
            asyncio.create_task(system.autosave_state()),  # NOVO: Autosave também em testes
            asyncio.create_task(system.watch_config_changes())  # NOVO: Hot-reload em testes
        ]
        
        if system.use_console:
            tasks.append(asyncio.create_task(update_test_display()))
        
        # Aguarda até todos os cenários serem testados
        while data_provider.scenario_stats['tested'] < data_provider.scenario_stats['total']:
            await asyncio.sleep(1)
        
        # Para o sistema
        system.shutdown()  # NOVO: Usa shutdown
        
        # Cancela tasks
        for task in tasks:
            task.cancel()
            
    except Exception as e:
        logging.error(f"Erro durante testes: {e}")
    
    await data_provider.disconnect()
    
    status = system.get_system_status()
    print("\n\n" + "="*24 + " TESTES CONCLUÍDOS " + "="*23)
    print(f"Sinais totais gerados nos testes: {status['decision_stats'].get('signals_generated', 0)}")
    print("Verifique os logs para detalhes das detecções em cada cenário.")
    print("="*60)


# NOVO: Exemplo de uso da DI para testes unitários
def create_test_system():
    """
    Exemplo de como criar sistema para testes com mocks
    """
    from unittest.mock import Mock
    
    # Cria mocks dos componentes
    mock_confluence = Mock(spec=ConfluenceAnalyzer)
    mock_decision = Mock(spec=DecisionMatrix)
    mock_risk = Mock(spec=RiskManager)
    mock_tracker = Mock(spec=SignalTracker)
    mock_regime = Mock(spec=RegimeClassifier)
    mock_behaviors = Mock(spec=BehaviorManager)
    
    # Cria sistema com mocks
    test_system = TapeReaderSystem(
        confluence_analyzer=mock_confluence,
        decision_matrix=mock_decision,
        risk_manager=mock_risk,
        signal_tracker=mock_tracker,
        regime_classifier=mock_regime,
        behavior_manager=mock_behaviors,
        console=None,
        use_console=False
    )
    
    return test_system, {
        'confluence': mock_confluence,
        'decision': mock_decision,
        'risk': mock_risk,
        'tracker': mock_tracker,
        'regime': mock_regime,
        'behaviors': mock_behaviors
    }


def main():
    """Função principal com parsing de argumentos"""
    parser = argparse.ArgumentParser(
        description='TapeReader Professional v2.0'
    )
    parser.add_argument(
        '--mode',
        choices=['production', 'test'],
        default='test',
        help='Modo de execução (production/test)'
    )
    
    args = parser.parse_args()
    
    # Cria diretórios necessários
    os.makedirs('logs', exist_ok=True)
    os.makedirs('logs/analysis', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Configura logging baseado no modo
    log_level = logging.DEBUG if args.mode == 'test' else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/tapereader.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    # Executa modo apropriado
    try:
        if args.mode == 'production':
            asyncio.run(run_production(args))
        else:
            asyncio.run(run_test(args))
    except KeyboardInterrupt:
        print("\n\nSistema encerrado pelo usuário.")
    except Exception as e:
        print(f"\n❌ Erro fatal: {e}")
        logging.exception("Erro fatal no sistema")


if __name__ == "__main__":
    main()