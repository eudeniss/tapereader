"""
Matriz de Decisão - Versão Melhorada e Refatorada
Combina comportamentos e confluência para gerar sinais de trading.
OTIMIZAÇÃO: Seleciona o MELHOR sinal disponível, não apenas o primeiro válido.
REATORADO: A direção do sinal agora é lida diretamente do comportamento,
             conforme o Guia de Migração para desacoplamento.
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from decimal import Decimal
from dataclasses import dataclass
import logging

# Supondo que estes modelos existam em seus módulos correspondentes
from ..core.models import BehaviorDetection, TradingSignal, Side, SignalStrength
from ..core.tracking import SignalTracker
from .confluence import ConfluenceLevel


@dataclass
class StrategyRule:
    """Define a estrutura para uma regra de estratégia de trading."""
    name: str
    required_behaviors: List[str]
    optional_behaviors: List[str]
    min_confidence: float
    base_confidence: float
    description: str
    entry_offset: Decimal
    stop_multiplier: float
    target_multiplier: float
    max_hold_time: int  # Tempo máximo de hold configurável


@dataclass
class CandidateSignal:
    """Representa um sinal candidato para seleção"""
    signal: TradingSignal
    rule: StrategyRule
    score: float  # Score composto para seleção


class DecisionMatrix:
    """
    Matriz de decisão que transforma comportamentos de mercado em sinais de trading,
    com regras carregadas dinamicamente a partir da configuração.
    MELHORADO: Agora seleciona o melhor sinal entre todos os candidatos válidos.
    """

    def __init__(self, config: Dict[str, Any], signal_tracker: SignalTracker):
        self.config = config
        self.signal_tracker = signal_tracker
        self.logger = logging.getLogger(__name__)

        # Parâmetros gerais
        self.min_final_confidence = config.get('min_final_confidence', 0.75)
        self.max_signals_per_minute = config.get('max_signals_per_minute', 2)
        self.cooldown_seconds = config.get('cooldown_seconds', 30)

        # Parâmetros de ATR
        self.atr_period = config.get('atr_period', 14)
        self.default_atr = {k: Decimal(str(v)) for k, v in config.get('default_atr', {
            'DOLFUT': '2.5',
            'WDOFUT': '2.5'
        }).items()}

        # Configurações de Risco/Retorno
        rr_config = config.get('risk_reward_config', {})
        self.min_risk_reward_ratio = rr_config.get('min_risk_reward_ratio', 1.5)

        # Carrega configuração de risco por trade
        self.trading_risk_config = config.get('trading_risk', {})

        # Regras de estratégia (carregadas do config)
        self.strategy_rules = self._initialize_rules()

        # NOVO: Configuração de seleção de sinais
        selection_config = config.get('signal_selection', {})
        self.selection_weights = {
            'confidence': selection_config.get('confidence_weight', 0.4),
            'risk_reward': selection_config.get('risk_reward_weight', 0.3),
            'confluence': selection_config.get('confluence_weight', 0.2),
            'behaviors': selection_config.get('behaviors_weight', 0.1)
        }

        # Cache de sinais recentes
        self.recent_signals: List[TradingSignal] = []

        # Estatísticas
        self.stats = {
            'signals_generated': 0,
            'signals_by_strategy': {},
            'confidence_distribution': [],
            'candidate_signals_evaluated': 0,
            'selection_reasons': {}
        }
        self.logger.info(f"DecisionMatrix inicializada com {len(self.strategy_rules)} regras.")

    def _initialize_rules(self) -> List[StrategyRule]:
        """Inicializa as regras de estratégia a partir do dicionário de configuração."""
        rules = []
        all_rules_config = {
            **self.config.get('strategy_rules', {}),
            **self.config.get('advanced_strategy_rules', {})
        }

        for rule_name, cfg in all_rules_config.items():
            try:
                rule = StrategyRule(
                    name=cfg.get('name', rule_name),
                    required_behaviors=cfg.get('required_behaviors', []),
                    optional_behaviors=cfg.get('optional_behaviors', []),
                    min_confidence=float(cfg.get('min_confidence', 0.70)),
                    base_confidence=float(cfg.get('base_confidence', 0.80)),
                    description=cfg.get('description', 'N/A'),
                    entry_offset=Decimal(str(cfg.get('entry_offset', '0.0'))),
                    stop_multiplier=float(cfg.get('stop_multiplier', 1.5)),
                    target_multiplier=float(cfg.get('target_multiplier', 2.5)),
                    max_hold_time=int(cfg.get('max_hold_time', 15))
                )
                rules.append(rule)
                self.logger.debug(f"Regra '{rule.name}' carregada com sucesso.")
            except (KeyError, TypeError, ValueError) as e:
                self.logger.error(f"Erro ao carregar a regra '{rule_name}': {e}. Verifique a configuração.")

        if not rules:
            self.logger.warning("Nenhuma regra de estratégia encontrada na configuração.")
            return []

        return rules

    def generate_signal(
        self,
        asset: str,
        behaviors: List[BehaviorDetection],
        confluence_analysis: Dict[str, Any],
        market_context: Dict[str, Any]
    ) -> Optional[TradingSignal]:
        """
        Processa os dados de mercado e comportamentos para gerar o MELHOR sinal de trading.
        MELHORADO: Agora avalia todos os sinais candidatos e seleciona o melhor
        """
        if not self._check_signal_cooldown(asset):
            return None
        if not self._check_signal_limit():
            self.logger.warning("Limite de sinais por minuto atingido.")
            return None

        behavior_map = {b.behavior_type: b for b in behaviors}
        self.logger.debug(f"Comportamentos detectados para avaliação: {list(behavior_map.keys())}")

        # NOVO: Coletar todos os sinais candidatos válidos
        candidate_signals = self._collect_candidate_signals(
            asset, behavior_map, confluence_analysis, market_context
        )

        if not candidate_signals:
            self.logger.debug("Nenhum sinal candidato válido encontrado.")
            return None

        # NOVO: Selecionar o melhor sinal
        best_candidate = self._select_best_signal(candidate_signals)
        
        if best_candidate:
            self.logger.info(
                f"Melhor sinal selecionado: '{best_candidate.rule.name}' "
                f"para {asset} (score: {best_candidate.score:.3f})"
            )
            
            # Registrar estatísticas de seleção
            self._record_selection_stats(candidate_signals, best_candidate)
            
            # Criar sinal rastreado
            tracked_signal = self.signal_tracker.create_tracked_signal(best_candidate.signal)
            self._update_internal_state(tracked_signal, best_candidate.rule.name)
            return tracked_signal

        return None

    def _collect_candidate_signals(
        self,
        asset: str,
        behaviors: Dict[str, BehaviorDetection],
        confluence: Dict[str, Any],
        context: Dict[str, Any]
    ) -> List[CandidateSignal]:
        """
        NOVO: Coleta todos os sinais candidatos válidos de todas as regras
        """
        candidates = []
        
        for rule in self.strategy_rules:
            signal = self._evaluate_rule(rule, behaviors, confluence, context, asset)
            if signal:
                # Calcular score para o sinal
                score = self._calculate_signal_score(signal, rule, confluence, behaviors)
                
                candidate = CandidateSignal(
                    signal=signal,
                    rule=rule,
                    score=score
                )
                candidates.append(candidate)
                
                self.logger.debug(
                    f"Candidato válido: {rule.name} "
                    f"(confiança: {signal.confidence:.2%}, score: {score:.3f})"
                )
        
        self.stats['candidate_signals_evaluated'] += len(candidates)
        return candidates

    def _calculate_signal_score(
        self,
        signal: TradingSignal,
        rule: StrategyRule,
        confluence: Dict[str, Any],
        behaviors: Dict[str, BehaviorDetection]
    ) -> float:
        """
        NOVO: Calcula um score composto para classificar sinais
        Considera múltiplos fatores com pesos configuráveis
        """
        scores = {}
        
        # 1. Score de Confiança
        scores['confidence'] = signal.confidence
        
        # 2. Score de Risco/Retorno
        rr_score = min(1.0, signal.risk_reward_ratio / 3.0)  # Normalizado até R/R 3:1
        scores['risk_reward'] = rr_score
        
        # 3. Score de Confluência
        confluence_level = confluence.get('level', ConfluenceLevel.WEAK)
        confluence_scores = {
            ConfluenceLevel.PREMIUM: 1.0,
            ConfluenceLevel.STRONG: 0.8,
            ConfluenceLevel.STANDARD: 0.6,
            ConfluenceLevel.WEAK: 0.3
        }
        scores['confluence'] = confluence_scores.get(confluence_level, 0.3)
        
        # 4. Score de Comportamentos (quantidade e qualidade)
        behavior_score = len(behaviors) / 10.0  # Normalizado para até 10 comportamentos
        # Bonus por comportamentos críticos
        critical_behaviors = ['institutional', 'absorption', 'iceberg']
        critical_count = sum(1 for b in critical_behaviors if b in behaviors)
        behavior_score += critical_count * 0.1
        scores['behaviors'] = min(1.0, behavior_score)
        
        # Calcular score final ponderado
        final_score = 0.0
        for component, weight in self.selection_weights.items():
            final_score += scores.get(component, 0.0) * weight
            
        # Bonus adicional por combinações específicas
        if self._has_premium_combination(behaviors):
            final_score *= 1.1  # 10% de bonus
            
        return min(1.0, final_score)

    def _has_premium_combination(self, behaviors: Dict[str, BehaviorDetection]) -> bool:
        """
        NOVO: Verifica se há combinações premium de comportamentos
        """
        premium_combinations = [
            ['institutional', 'absorption'],
            ['absorption', 'iceberg'],
            ['institutional', 'iceberg'],
            ['stop_hunt', 'exhaustion'],
            ['sweep', 'momentum']
        ]
        
        for combo in premium_combinations:
            if all(b in behaviors for b in combo):
                return True
        return False

    def _select_best_signal(self, candidates: List[CandidateSignal]) -> Optional[CandidateSignal]:
        """
        NOVO: Seleciona o melhor sinal entre os candidatos
        Usa critérios de desempate inteligentes
        """
        if not candidates:
            return None
            
        # Ordenar por score (maior primeiro)
        sorted_candidates = sorted(candidates, key=lambda c: c.score, reverse=True)
        
        # Se há um claro vencedor (diferença > 10%), retornar
        if len(sorted_candidates) >= 2:
            score_diff = sorted_candidates[0].score - sorted_candidates[1].score
            if score_diff > 0.1:
                return sorted_candidates[0]
        
        # Desempate por critérios adicionais
        top_candidates = [c for c in sorted_candidates if c.score >= sorted_candidates[0].score - 0.05]
        
        # Critério 1: Maior risco/retorno
        best_by_rr = max(top_candidates, key=lambda c: c.signal.risk_reward_ratio)
        
        # Critério 2: Menor stop loss (menor risco)
        best_by_risk = min(
            top_candidates,
            key=lambda c: abs(c.signal.entry_price - c.signal.stop_loss)
        )
        
        # Se o mesmo sinal ganha em ambos os critérios, retornar
        if best_by_rr == best_by_risk:
            return best_by_rr
            
        # Caso contrário, retornar o de maior score original
        return sorted_candidates[0]

    def _record_selection_stats(self, candidates: List[CandidateSignal], selected: CandidateSignal):
        """
        NOVO: Registra estatísticas sobre o processo de seleção
        """
        # Registrar motivo da seleção
        reason = f"{selected.rule.name} (score: {selected.score:.3f})"
        self.stats['selection_reasons'][reason] = self.stats['selection_reasons'].get(reason, 0) + 1
        
        # Log detalhado se múltiplos candidatos
        if len(candidates) > 1:
            self.logger.info(
                f"Seleção entre {len(candidates)} candidatos: "
                f"[{', '.join(f'{c.rule.name}({c.score:.2f})' for c in candidates)}]"
            )

    def _evaluate_rule(
        self,
        rule: StrategyRule,
        behaviors: Dict[str, BehaviorDetection],
        confluence: Dict[str, Any],
        context: Dict[str, Any],
        asset: str
    ) -> Optional[TradingSignal]:
        """Avalia uma única regra de estratégia e constrói um sinal se os critérios forem atendidos."""
        # 1. Verificar comportamentos obrigatórios
        if not all(required in behaviors for required in rule.required_behaviors):
            return None

        # 2. Calcular confiança
        confidence = self._calculate_confidence(rule, behaviors, confluence)
        if confidence < max(rule.min_confidence, self.min_final_confidence):
            return None

        # 3. Determinar direção (AJUSTADO)
        direction = self._determine_signal_direction(behaviors, rule)
        if not direction or direction == Side.NEUTRAL: # Ignora direções neutras
            self.logger.debug(f"Não foi possível determinar uma direção clara para a regra '{rule.name}'.")
            return None

        # 4. Calcular parâmetros do trade
        current_price = context.get('current_price', Decimal('0'))
        atr = context.get('atr', self.default_atr.get(asset, Decimal('2.5')))
        
        entry_price = self._calculate_entry_price(current_price, direction, rule.entry_offset)
        stop_loss = self._calculate_stop_loss(entry_price, direction, rule.stop_multiplier, atr)
        targets = self._calculate_targets(entry_price, direction, rule.target_multiplier, atr)

        # 5. Validar Risco/Retorno
        risk_reward_ratio = self._calculate_risk_reward(entry_price, stop_loss, targets['target_1'])
        if risk_reward_ratio < self.min_risk_reward_ratio:
            self.logger.debug(f"Sinal para regra '{rule.name}' descartado devido a R/R baixo: {risk_reward_ratio:.2f}")
            return None

        # 6. Construir o sinal
        return TradingSignal(
            signal_id="",
            timestamp=datetime.now(),
            direction=direction,
            asset=asset,
            price=current_price,
            confidence=round(confidence, 4),
            signal_strength=self._determine_signal_strength(confidence),
            behaviors_detected=list(behaviors.values()),
            primary_motivation=rule.description,
            secondary_motivations=[b_type for b_type in behaviors if b_type not in rule.required_behaviors],
            confluence_data=confluence,
            market_context=context,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit_1=targets['target_1'],
            take_profit_2=targets.get('target_2'),
            max_hold_time=rule.max_hold_time,
            risk_reward_ratio=risk_reward_ratio,
            position_size_suggestion=self._calculate_position_size(
                asset, entry_price, stop_loss, context.get('account_balance', 100000.0)
            )
        )

    def _calculate_confidence(
        self, rule: StrategyRule, behaviors: Dict[str, BehaviorDetection], confluence: Dict[str, Any]
    ) -> float:
        """Calcula a pontuação de confiança para uma combinação de regra e comportamentos."""
        confidence = rule.base_confidence
        for optional in rule.optional_behaviors:
            if optional in behaviors:
                confidence += 0.03
        confidence += confluence.get('confidence_boost', 0.0)
        return min(confidence, 1.0)

    def _determine_signal_direction(
        self, behaviors: Dict[str, BehaviorDetection], rule: StrategyRule
    ) -> Optional[Side]:
        """
        Determina a direção do sinal com base na direção explícita dos comportamentos.
        Esta função foi simplificada para ler o atributo 'direction' dos behaviors,
        eliminando a necessidade de lógica de interpretação interna.
        """
        directions = []
        # Itera apenas sobre os comportamentos obrigatórios para definir a direção
        for bh_type in rule.required_behaviors:
            behavior = behaviors.get(bh_type)
            # Acessa diretamente o atributo 'direction' do comportamento
            if behavior and hasattr(behavior, 'direction') and behavior.direction:
                directions.append(behavior.direction)
        
        if not directions:
            return None
        
        # Contagem de direções para resolver conflitos
        buy_count = directions.count(Side.BUY)
        sell_count = directions.count(Side.SELL)

        if buy_count > sell_count:
            return Side.BUY
        if sell_count > buy_count:
            return Side.SELL
        
        # Se houver empate ou apenas um comportamento, usa a direção dele
        if buy_count == sell_count and buy_count > 0:
             self.logger.warning(f"Conflito de direção para regra '{rule.name}'. Behaviors com direções opostas.")
             return Side.NEUTRAL # Retorna NEUTRO para indicar conflito

        return directions[0]

    def _calculate_entry_price(self, current_price: Decimal, direction: Side, offset: Decimal) -> Decimal:
        """Calcula o preço de entrada com base na direção e offset."""
        return current_price + offset if direction == Side.BUY else current_price - offset

    def _calculate_stop_loss(self, entry_price: Decimal, direction: Side, multiplier: float, atr: Decimal) -> Decimal:
        """Calcula o preço de stop loss baseado em ATR."""
        stop_distance = atr * Decimal(str(multiplier))
        return entry_price - stop_distance if direction == Side.BUY else entry_price + stop_distance

    def _calculate_targets(self, entry_price: Decimal, direction: Side, multiplier: float, atr: Decimal) -> Dict[str, Decimal]:
        """Calcula os preços de alvos (take profit) baseados em ATR."""
        total_distance = atr * Decimal(str(multiplier))
        if direction == Side.BUY:
            target_1 = entry_price + (total_distance * Decimal('0.6'))
            target_2 = entry_price + total_distance
        else:
            target_1 = entry_price - (total_distance * Decimal('0.6'))
            target_2 = entry_price - total_distance
        return {'target_1': target_1, 'target_2': target_2}

    def _calculate_risk_reward(self, entry: Decimal, stop: Decimal, target: Decimal) -> float:
        """Calcula a relação risco/retorno para o primeiro alvo."""
        risk = abs(entry - stop)
        reward = abs(target - entry)
        return float(reward / risk) if risk > 0 else 0.0

    def _calculate_position_size(self, asset: str, entry_price: Decimal, stop_loss: Decimal, account_balance: float) -> int:
        """Calcula o tamanho da posição sugerido (baseado em risco fixo)."""
        risk_per_trade_percent = self.trading_risk_config.get('risk_per_trade', 0.01)
        risk_amount = account_balance * risk_per_trade_percent
        
        risk_per_contract = abs(entry_price - stop_loss)
        if risk_per_contract == 0: return 1

        point_value_config = self.trading_risk_config.get('asset_point_value', {})
        point_value = point_value_config.get(asset, 10.0)
        risk_per_contract_monetary = float(risk_per_contract) * point_value

        if risk_per_contract_monetary == 0: return 1
        
        contracts = int(risk_amount / risk_per_contract_monetary)
        
        max_contracts_config = self.trading_risk_config.get('max_position_size', {})
        max_contracts = max_contracts_config.get(asset, 50)
        return max(1, min(contracts, max_contracts))

    def _determine_signal_strength(self, confidence: float) -> SignalStrength:
        """Categoriza a força do sinal com base na pontuação de confiança final."""
        if confidence >= 0.90: return SignalStrength.PREMIUM
        if confidence >= 0.85: return SignalStrength.STRONG
        return SignalStrength.STANDARD

    def _check_signal_cooldown(self, asset: str) -> bool:
        """Verifica se um novo sinal para um ativo específico está violando o período de cooldown."""
        now = datetime.now()
        cooldown_limit = now - timedelta(seconds=self.cooldown_seconds)
        for signal in reversed(self.recent_signals):
            if signal.timestamp < cooldown_limit: break
            if signal.asset == asset:
                self.logger.debug(f"Cooldown ativo para {asset}. Último sinal foi às {signal.timestamp}.")
                return False
        return True

    def _check_signal_limit(self) -> bool:
        """Verifica se o número de sinais no último minuto excede o limite configurado."""
        now = datetime.now()
        limit_cutoff = now - timedelta(minutes=1)
        recent_count = sum(1 for s in self.recent_signals if s.timestamp > limit_cutoff)
        return recent_count < self.max_signals_per_minute

    def _update_internal_state(self, signal: TradingSignal, strategy_name: str):
        """Atualiza o cache de sinais e as estatísticas internas."""
        self.recent_signals.append(signal)
        cutoff = datetime.now() - timedelta(minutes=5)
        self.recent_signals = [s for s in self.recent_signals if s.timestamp > cutoff]

        self.stats['signals_generated'] += 1
        self.stats['signals_by_strategy'][strategy_name] = self.stats['signals_by_strategy'].get(strategy_name, 0) + 1
        self.stats['confidence_distribution'].append(signal.confidence)
        if len(self.stats['confidence_distribution']) > 200:
            self.stats['confidence_distribution'].pop(0)

    def reload_rules(self, new_config: Dict[str, Any]):
        """Recarrega as regras de estratégia e outros parâmetros a partir de uma nova configuração."""
        self.logger.info("Iniciando recarregamento de regras e configuração...")
        old_count = len(self.strategy_rules)
        self.config.update(new_config)
        
        self.min_final_confidence = self.config.get('min_final_confidence', self.min_final_confidence)
        rr_config = self.config.get('risk_reward_config', {})
        self.min_risk_reward_ratio = rr_config.get('min_risk_reward_ratio', self.min_risk_reward_ratio)
        self.trading_risk_config = self.config.get('trading_risk', self.trading_risk_config)
        
        # NOVO: Atualizar pesos de seleção
        selection_config = self.config.get('signal_selection', {})
        self.selection_weights.update({
            'confidence': selection_config.get('confidence_weight', self.selection_weights['confidence']),
            'risk_reward': selection_config.get('risk_reward_weight', self.selection_weights['risk_reward']),
            'confluence': selection_config.get('confluence_weight', self.selection_weights['confluence']),
            'behaviors': selection_config.get('behaviors_weight', self.selection_weights['behaviors'])
        })

        self.strategy_rules = self._initialize_rules()
        new_count = len(self.strategy_rules)
        self.logger.info(f"Regras recarregadas. Antes: {old_count}, Agora: {new_count}")

    def get_statistics(self) -> Dict[str, Any]:
        """Retorna as estatísticas de desempenho da matriz de decisão."""
        dist = self.stats['confidence_distribution']
        avg_confidence = sum(dist) / len(dist) if dist else 0.0
        
        # NOVO: Estatísticas de seleção
        top_selection_reasons = sorted(
            self.stats['selection_reasons'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        return {
            'total_signals': self.stats['signals_generated'],
            'signals_by_strategy': self.stats['signals_by_strategy'],
            'average_confidence': round(avg_confidence, 3),
            'active_rules': len(self.strategy_rules),
            'recent_signals_cache_size': len(self.recent_signals),
            'candidate_signals_evaluated': self.stats['candidate_signals_evaluated'],
            'avg_candidates_per_signal': (
                self.stats['candidate_signals_evaluated'] / max(1, self.stats['signals_generated'])
            ),
            'top_selection_reasons': top_selection_reasons,
            'selection_weights': self.selection_weights
        }

    def get_active_rules(self) -> List[Dict[str, Any]]:
        """Retorna uma lista formatada das regras de estratégia atualmente ativas."""
        return [{
            'name': rule.name,
            'description': rule.description,
            'required_behaviors': rule.required_behaviors,
            'optional_behaviors': rule.optional_behaviors,
            'confidence_range': f"{rule.min_confidence:.0%} - {rule.base_confidence:.0%}",
            'base_risk_reward': f"1:{rule.target_multiplier / rule.stop_multiplier:.1f}" if rule.stop_multiplier > 0 else "N/A"
        } for rule in self.strategy_rules]