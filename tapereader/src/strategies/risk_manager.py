"""
Gerenciador de Risco - Versão Atualizada com Persistência
Valida sinais e aplica controles de risco
Agora lê configurações do arquivo YAML e possui persistência de estado
"""

from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta, time
from decimal import Decimal
import logging
import json
from pathlib import Path
from enum import Enum

from ..core.models import TradingSignal, Side


class RiskLevel(str, Enum):
    """Níveis de risco"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    EXTREME = "EXTREME"


class RiskManager:
    """
    Gerencia risco dos sinais de trading
    Versão atualizada que lê configurações do YAML e persiste estado
    
    Responsabilidades:
    - Validar sinais antes da execução
    - Aplicar limites de risco
    - Gerenciar exposição total
    - Controlar horários de operação
    - Persistir estado para recuperação
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Parâmetros de risco
        self.max_daily_loss = config.get('max_daily_loss', 0.02)  # 2% da conta
        self.max_position_risk = config.get('max_position_risk', 0.01)  # 1% por posição
        self.max_concurrent_positions = config.get('max_concurrent_positions', 3)
        self.max_correlation_exposure = config.get('max_correlation_exposure', 0.8)
        
        # Horários de operação - agora lê do config
        trading_hours_config = config.get('trading_hours', {})
        self.trading_hours = {
            'start': self._parse_time(trading_hours_config.get('start', '09:05')),
            'end': self._parse_time(trading_hours_config.get('end', '17:30')),
            'lunch_start': self._parse_time(trading_hours_config.get('lunch_start', '12:00')),
            'lunch_end': self._parse_time(trading_hours_config.get('lunch_end', '13:00'))
        }
        
        # Filtros especiais
        self.avoid_news_window = config.get('avoid_news_window', 5)  # minutos
        self.min_volume_threshold = config.get('min_volume_threshold', 100)
        
        # Circuit breakers do config
        cb_config = config.get('circuit_breakers', {})
        self.consecutive_losses_limit = cb_config.get('consecutive_losses_limit', 5)
        self.circuit_breaker_duration = cb_config.get('circuit_breaker_duration', 30)
        self.daily_loss_circuit_breaker = cb_config.get('daily_loss_circuit_breaker', 0.03)
        
        # Risk adjustments
        self.risk_adjustments = config.get('risk_adjustments', {
            'LOW': {'position_multiplier': 1.0, 'confidence_threshold': 0.80},
            'MEDIUM': {'position_multiplier': 0.75, 'confidence_threshold': 0.83},
            'HIGH': {'position_multiplier': 0.50, 'confidence_threshold': 0.86},
            'EXTREME': {'position_multiplier': 0.25, 'confidence_threshold': 0.90}
        })
        
        # Estado do risco
        self.daily_pnl = Decimal('0')
        self.open_positions = []
        self.daily_trades = 0
        self.consecutive_losses = 0
        self.last_reset = datetime.now().date()
        
        # Circuit breakers
        self.circuit_breaker_active = False
        self.circuit_breaker_until = None
        
        # Estatísticas
        self.stats = {
            'signals_approved': 0,
            'signals_rejected': 0,
            'rejections_by_reason': {},
            'risk_adjustments': 0
        }
        
        # NOVO: Arquivo de estado de emergência
        self.emergency_state_file = Path("data/risk_manager_emergency.json")
        
        self.logger.info(
            f"RiskManager inicializado - "
            f"Trading: {self.trading_hours['start'].strftime('%H:%M')} - "
            f"{self.trading_hours['end'].strftime('%H:%M')}"
        )
        
    def _parse_time(self, time_str: str) -> time:
        """Converte string HH:MM para objeto time"""
        try:
            return datetime.strptime(time_str, '%H:%M').time()
        except ValueError:
            # Fallback para formato com segundos
            try:
                return datetime.strptime(time_str, '%H:%M:%S').time()
            except ValueError:
                # Retorna default se falhar
                self.logger.warning(f"Formato de hora inválido: {time_str}, usando 09:00")
                return time(9, 0)
    
    def validate_signal(
        self,
        signal: TradingSignal,
        market_context: Dict[str, Any]
    ) -> Tuple[bool, Optional[str], Optional[TradingSignal]]:
        """
        Valida sinal aplicando todos os filtros de risco
        
        Returns:
            Tuple[aprovado, motivo_rejeição, sinal_ajustado]
        """
        # Reset diário se necessário
        self._check_daily_reset()
        
        # Verifica circuit breaker
        if self._is_circuit_breaker_active():
            self._update_statistics('rejected', 'circuit_breaker_active')
            return False, "Circuit breaker ativo", None
            
        # 1. Validação de horário
        time_valid, time_reason = self._validate_trading_time()
        if not time_valid:
            self._update_statistics('rejected', time_reason)
            return False, time_reason, None
            
        # 2. Validação de limites de perda
        loss_valid, loss_reason = self._validate_loss_limits()
        if not loss_valid:
            self._update_statistics('rejected', loss_reason)
            return False, loss_reason, None
            
        # 3. Validação de exposição
        exposure_valid, exposure_reason = self._validate_exposure(signal)
        if not exposure_valid:
            self._update_statistics('rejected', exposure_reason)
            return False, exposure_reason, None
            
        # 4. Validação de contexto de mercado
        context_valid, context_reason = self._validate_market_context(
            signal,
            market_context
        )
        if not context_valid:
            self._update_statistics('rejected', context_reason)
            return False, context_reason, None
            
        # 5. Ajusta tamanho da posição se necessário
        adjusted_signal = self._adjust_position_size(signal, market_context)
        
        # 6. Validação final de risco/retorno
        rr_valid, rr_reason = self._validate_risk_reward(adjusted_signal)
        if not rr_valid:
            self._update_statistics('rejected', rr_reason)
            return False, rr_reason, None
            
        # Sinal aprovado
        self._update_statistics('approved', 'all_checks_passed')
        
        return True, None, adjusted_signal
        
    def _check_daily_reset(self):
        """Verifica se precisa resetar contadores diários"""
        today = datetime.now().date()
        
        if today > self.last_reset:
            self.logger.info("Resetando contadores diários de risco")
            self.daily_pnl = Decimal('0')
            self.daily_trades = 0
            self.last_reset = today
            
            # Reset circuit breaker se expirou
            if self.circuit_breaker_until and datetime.now() > self.circuit_breaker_until:
                self.circuit_breaker_active = False
                self.circuit_breaker_until = None
                
    def _is_circuit_breaker_active(self) -> bool:
        """Verifica se circuit breaker está ativo"""
        if self.circuit_breaker_active:
            if self.circuit_breaker_until and datetime.now() < self.circuit_breaker_until:
                return True
            else:
                # Expirou
                self.circuit_breaker_active = False
                self.circuit_breaker_until = None
                self.logger.info("Circuit breaker expirou")
                
        return False
        
    def _validate_trading_time(self) -> Tuple[bool, Optional[str]]:
        """Valida se está em horário de operação"""
        current_time = datetime.now().time()
        
        # Antes da abertura
        if current_time < self.trading_hours['start']:
            return False, "Mercado ainda não abriu"
            
        # Após fechamento
        if current_time > self.trading_hours['end']:
            return False, "Mercado fechado"
            
        # Horário de almoço (opcional)
        if (self.trading_hours['lunch_start'] <= current_time <= 
            self.trading_hours['lunch_end']):
            # Permite apenas fechamento de posições
            return False, "Horário de almoço - apenas fechamento"
            
        return True, None
        
    def _validate_loss_limits(self) -> Tuple[bool, Optional[str]]:
        """Valida limites de perda"""
        # Verifica perda diária
        daily_loss_pct = abs(float(self.daily_pnl))
        
        if daily_loss_pct >= self.max_daily_loss:
            return False, f"Limite diário atingido: {daily_loss_pct:.2%}"
            
        # Verifica circuit breaker por perda excessiva
        if daily_loss_pct >= self.daily_loss_circuit_breaker:
            self._activate_circuit_breaker(self.circuit_breaker_duration)
            return False, f"Circuit breaker por perda: {daily_loss_pct:.2%}"
            
        # Verifica sequência de perdas
        if self.consecutive_losses >= self.consecutive_losses_limit:
            # Ativa circuit breaker
            self._activate_circuit_breaker(self.circuit_breaker_duration)
            return False, f"{self.consecutive_losses} perdas consecutivas - circuit breaker ativado"
            
        return True, None
        
    def _validate_exposure(self, signal: TradingSignal) -> Tuple[bool, Optional[str]]:
        """Valida exposição total"""
        # Conta posições abertas
        open_count = len(self.open_positions)
        
        if open_count >= self.max_concurrent_positions:
            return False, f"Limite de posições atingido ({open_count}/{self.max_concurrent_positions})"
            
        # Verifica correlação entre posições
        if self._check_correlation_risk(signal):
            return False, "Risco de correlação muito alto"
            
        return True, None
        
    def _validate_market_context(
        self,
        signal: TradingSignal,
        context: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """Valida contexto de mercado"""
        # Volume mínimo
        current_volume = context.get('session_volume', 0)
        if current_volume < self.min_volume_threshold:
            return False, f"Volume baixo: {current_volume}"
            
        # Volatilidade extrema
        volatility = context.get('current_volatility', 0)
        avg_volatility = context.get('avg_volatility', 1)
        
        if avg_volatility > 0 and volatility > avg_volatility * 3:
            return False, "Volatilidade extrema detectada"
            
        # Proximidade de notícias
        next_news = context.get('next_news_minutes')
        if next_news and next_news <= self.avoid_news_window:
            return False, f"Notícia em {next_news} minutos"
            
        # Spread muito largo
        spread = context.get('current_spread', Decimal('0'))
        avg_spread = context.get('avg_spread', Decimal('0.5'))
        
        if avg_spread > 0 and spread > avg_spread * 3:
            return False, f"Spread muito largo: {spread}"
            
        return True, None
        
    def _validate_risk_reward(self, signal: TradingSignal) -> Tuple[bool, Optional[str]]:
        """Valida relação risco/retorno"""
        # Pega configuração do R/R do config se disponível
        rr_config = self.config.get('risk_reward_config', {})
        min_rr = rr_config.get('min_risk_reward_ratio', 1.5)
        
        if signal.risk_reward_ratio < min_rr:
            return False, f"R/R insuficiente: {signal.risk_reward_ratio:.1f} < {min_rr}"
            
        # Verifica se stop está muito próximo (pode ser stopado por ruído)
        stop_distance = abs(signal.entry_price - signal.stop_loss)
        min_stop_distance = Decimal(str(rr_config.get('min_stop_distance', 1.0)))
        
        if stop_distance < min_stop_distance:
            return False, f"Stop muito próximo: {stop_distance}"
            
        # Verifica se stop está muito longe
        max_stop_distance = Decimal(str(rr_config.get('max_stop_distance', 5.0)))
        
        if stop_distance > max_stop_distance:
            return False, f"Stop muito distante: {stop_distance}"
            
        return True, None
        
    def _adjust_position_size(
        self,
        signal: TradingSignal,
        context: Dict[str, Any]
    ) -> TradingSignal:
        """Ajusta tamanho da posição baseado no risco"""
        # Calcula risco atual
        current_risk = self._calculate_current_risk_level(context)
        
        # Usa ajustes do config
        risk_config = self.risk_adjustments.get(current_risk.value, {})
        factor = risk_config.get('position_multiplier', 0.5)
        
        # Verifica se confiança está adequada para o nível de risco
        min_confidence = risk_config.get('confidence_threshold', 0.80)
        
        if signal.confidence < min_confidence:
            self.logger.warning(
                f"Confiança {signal.confidence:.2f} abaixo do mínimo "
                f"{min_confidence:.2f} para risco {current_risk}"
            )
            factor *= 0.5  # Reduz ainda mais
        
        if factor < 1.0:
            # Cria cópia do sinal com tamanho ajustado
            adjusted_signal = signal.copy()
            
            if signal.position_size_suggestion:
                original_size = signal.position_size_suggestion
                adjusted_signal.position_size_suggestion = max(
                    1,  # Mínimo 1 contrato
                    int(original_size * factor)
                )
                
                self.logger.info(
                    f"Posição ajustada por risco {current_risk}: "
                    f"{original_size} -> {adjusted_signal.position_size_suggestion} "
                    f"(fator: {factor:.2f})"
                )
            
            self.stats['risk_adjustments'] += 1
            
            return adjusted_signal
            
        return signal
        
    def _calculate_current_risk_level(self, context: Dict[str, Any]) -> RiskLevel:
        """Calcula nível de risco atual do mercado"""
        risk_score = 0
        
        # Fatores de risco
        
        # 1. Perda diária atual
        daily_loss_pct = abs(float(self.daily_pnl))
        
        if daily_loss_pct > self.max_daily_loss * 0.75:
            risk_score += 3
        elif daily_loss_pct > self.max_daily_loss * 0.5:
            risk_score += 2
        elif daily_loss_pct > self.max_daily_loss * 0.25:
            risk_score += 1
            
        # 2. Volatilidade
        current_vol = context.get('current_volatility', 0)
        avg_vol = context.get('avg_volatility', 1)
        
        if avg_vol > 0:
            vol_ratio = current_vol / avg_vol
            
            if vol_ratio > 2.5:
                risk_score += 3
            elif vol_ratio > 2:
                risk_score += 2
            elif vol_ratio > 1.5:
                risk_score += 1
            
        # 3. Horário
        current_time = datetime.now().time()
        
        # Primeira hora
        if current_time < time(10, 0):
            risk_score += 1
            
        # Última hora
        if current_time > time(16, 30):
            risk_score += 2
            
        # 4. Número de posições abertas
        position_ratio = len(self.open_positions) / self.max_concurrent_positions
        
        if position_ratio >= 0.8:
            risk_score += 2
        elif position_ratio >= 0.6:
            risk_score += 1
            
        # 5. Perdas consecutivas
        if self.consecutive_losses >= 4:
            risk_score += 3
        elif self.consecutive_losses >= 3:
            risk_score += 2
        elif self.consecutive_losses >= 2:
            risk_score += 1
            
        # Determina nível
        if risk_score >= 8:
            return RiskLevel.EXTREME
        elif risk_score >= 5:
            return RiskLevel.HIGH
        elif risk_score >= 3:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
            
    def _check_correlation_risk(self, signal: TradingSignal) -> bool:
        """Verifica risco de correlação entre posições"""
        if not self.open_positions:
            return False
            
        # Conta direção das posições abertas
        same_direction = sum(
            1 for pos in self.open_positions
            if pos.direction == signal.direction
        )
        
        # Se todas na mesma direção, risco alto
        correlation = same_direction / len(self.open_positions)
        
        return correlation > self.max_correlation_exposure
        
    def _activate_circuit_breaker(self, minutes: int):
        """Ativa circuit breaker por período determinado"""
        self.circuit_breaker_active = True
        self.circuit_breaker_until = datetime.now() + timedelta(minutes=minutes)
        
        self.logger.warning(
            f"⚠️ Circuit breaker ativado até {self.circuit_breaker_until:%H:%M:%S}"
        )
        
    def update_position_result(
        self,
        signal_id: str,
        pnl: Decimal,
        closed: bool = True
    ):
        """Atualiza resultado de uma posição"""
        # Atualiza P&L diário
        self.daily_pnl += pnl
        
        # Atualiza sequência de ganhos/perdas
        if pnl < 0:
            self.consecutive_losses += 1
            self.logger.warning(
                f"Perda #{self.consecutive_losses} - P&L: {pnl:.2f}"
            )
        else:
            if self.consecutive_losses > 0:
                self.logger.info(
                    f"Sequência de {self.consecutive_losses} perdas quebrada"
                )
            self.consecutive_losses = 0
            
        # Remove de posições abertas se fechada
        if closed:
            self.open_positions = [
                p for p in self.open_positions
                if p.signal_id != signal_id
            ]
            
        self.logger.info(
            f"Posição {signal_id} atualizada - P&L: {pnl:.2f}, "
            f"P&L Diário: {self.daily_pnl:.2f} ({float(self.daily_pnl):.2%})"
        )
        
    def add_open_position(self, signal: TradingSignal):
        """Adiciona posição à lista de abertas"""
        self.open_positions.append(signal)
        self.daily_trades += 1
        
        self.logger.info(
            f"Nova posição: {signal.asset} {signal.direction} - "
            f"Total abertas: {len(self.open_positions)}"
        )
        
    def _update_statistics(self, result: str, reason: str = None):
        """Atualiza estatísticas"""
        if result == 'approved':
            self.stats['signals_approved'] += 1
        else:
            self.stats['signals_rejected'] += 1
            
            if reason:
                if reason not in self.stats['rejections_by_reason']:
                    self.stats['rejections_by_reason'][reason] = 0
                self.stats['rejections_by_reason'][reason] += 1
                
    def get_risk_status(self) -> Dict[str, Any]:
        """Retorna status atual do risco"""
        current_risk = self._calculate_current_risk_level({})
        
        return {
            'daily_pnl': float(self.daily_pnl),
            'daily_pnl_pct': float(self.daily_pnl) * 100,
            'open_positions': len(self.open_positions),
            'open_positions_detail': [
                {
                    'asset': pos.asset,
                    'direction': pos.direction.value,
                    'entry': float(pos.entry_price)
                }
                for pos in self.open_positions
            ],
            'daily_trades': self.daily_trades,
            'consecutive_losses': self.consecutive_losses,
            'circuit_breaker': self.circuit_breaker_active,
            'circuit_breaker_until': (
                self.circuit_breaker_until.strftime('%H:%M:%S')
                if self.circuit_breaker_until else None
            ),
            'current_risk_level': current_risk.value,
            'position_multiplier': self.risk_adjustments.get(
                current_risk.value, {}
            ).get('position_multiplier', 1.0)
        }
        
    def get_statistics(self) -> Dict[str, Any]:
        """Retorna estatísticas do gerenciador de risco"""
        total_signals = (
            self.stats['signals_approved'] + 
            self.stats['signals_rejected']
        )
        
        approval_rate = (
            self.stats['signals_approved'] / total_signals
            if total_signals > 0 else 0
        )
        
        return {
            'total_signals_analyzed': total_signals,
            'signals_approved': self.stats['signals_approved'],
            'signals_rejected': self.stats['signals_rejected'],
            'approval_rate': round(approval_rate, 3),
            'risk_adjustments': self.stats['risk_adjustments'],
            'top_rejection_reasons': sorted(
                self.stats['rejections_by_reason'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:5],
            'trading_hours': {
                'start': self.trading_hours['start'].strftime('%H:%M'),
                'end': self.trading_hours['end'].strftime('%H:%M'),
                'lunch_start': self.trading_hours['lunch_start'].strftime('%H:%M'),
                'lunch_end': self.trading_hours['lunch_end'].strftime('%H:%M')
            }
        }
        
    def reload_config(self, new_config: Dict[str, Any]):
        """Recarrega configurações"""
        self.config.update(new_config)
        
        # Atualiza parâmetros
        self.max_daily_loss = new_config.get('max_daily_loss', self.max_daily_loss)
        self.max_position_risk = new_config.get('max_position_risk', self.max_position_risk)
        self.max_concurrent_positions = new_config.get('max_concurrent_positions', self.max_concurrent_positions)
        
        # Atualiza horários se presentes
        if 'trading_hours' in new_config:
            for key, value in new_config['trading_hours'].items():
                if key in self.trading_hours:
                    self.trading_hours[key] = self._parse_time(value)
                    
        self.logger.info("Configurações do RiskManager recarregadas")
    
    # NOVO: Métodos de Persistência
    def get_state(self) -> Dict[str, Any]:
        """Retorna estado completo para serialização"""
        state = {
            'timestamp': datetime.now().isoformat(),
            'daily_pnl': float(self.daily_pnl),
            'daily_trades': self.daily_trades,
            'consecutive_losses': self.consecutive_losses,
            'last_reset': self.last_reset.isoformat(),
            'circuit_breaker': {
                'active': self.circuit_breaker_active,
                'until': self.circuit_breaker_until.isoformat() if self.circuit_breaker_until else None
            },
            'open_positions': [
                {
                    'signal_id': pos.signal_id,
                    'asset': pos.asset,
                    'direction': pos.direction.value if hasattr(pos.direction, 'value') else pos.direction,
                    'entry_price': float(pos.entry_price),
                    'stop_loss': float(pos.stop_loss),
                    'take_profit_1': float(pos.take_profit_1),
                    'confidence': pos.confidence,
                    'strategy': pos.strategy,
                    'timestamp': pos.timestamp.isoformat()
                }
                for pos in self.open_positions
            ],
            'statistics': {
                'signals_approved': self.stats['signals_approved'],
                'signals_rejected': self.stats['signals_rejected'],
                'rejections_by_reason': dict(self.stats['rejections_by_reason']),
                'risk_adjustments': self.stats['risk_adjustments']
            }
        }
        
        return state
    
    def load_state(self, state: Dict[str, Any]):
        """Carrega estado a partir de dicionário"""
        try:
            # Restaura P&L e contadores
            self.daily_pnl = Decimal(str(state.get('daily_pnl', 0)))
            self.daily_trades = state.get('daily_trades', 0)
            self.consecutive_losses = state.get('consecutive_losses', 0)
            
            # Restaura data do último reset
            if 'last_reset' in state:
                self.last_reset = datetime.fromisoformat(state['last_reset']).date()
            
            # Restaura circuit breaker
            cb_state = state.get('circuit_breaker', {})
            self.circuit_breaker_active = cb_state.get('active', False)
            if cb_state.get('until'):
                self.circuit_breaker_until = datetime.fromisoformat(cb_state['until'])
            
            # Restaura posições abertas
            self.open_positions = []
            for pos_data in state.get('open_positions', []):
                try:
                    # Recria TradingSignal
                    signal = TradingSignal(
                        asset=pos_data['asset'],
                        direction=pos_data['direction'],
                        entry_price=pos_data['entry_price'],
                        stop_loss=pos_data['stop_loss'],
                        take_profit_1=pos_data['take_profit_1'],
                        confidence=pos_data['confidence'],
                        strategy=pos_data['strategy']
                    )
                    signal.signal_id = pos_data['signal_id']
                    signal.timestamp = datetime.fromisoformat(pos_data['timestamp'])
                    
                    self.open_positions.append(signal)
                except Exception as e:
                    self.logger.error(f"Erro ao restaurar posição: {e}")
            
            # Restaura estatísticas
            stats_data = state.get('statistics', {})
            self.stats['signals_approved'] = stats_data.get('signals_approved', 0)
            self.stats['signals_rejected'] = stats_data.get('signals_rejected', 0)
            self.stats['rejections_by_reason'] = stats_data.get('rejections_by_reason', {})
            self.stats['risk_adjustments'] = stats_data.get('risk_adjustments', 0)
            
            self.logger.info(
                f"Estado do RiskManager restaurado: "
                f"P&L: {self.daily_pnl:.2f}, "
                f"{len(self.open_positions)} posições abertas"
            )
            
        except Exception as e:
            self.logger.error(f"Erro ao carregar estado do RiskManager: {e}")
    
    def emergency_save(self):
        """Salva estado de emergência em arquivo separado"""
        try:
            # Cria diretório se não existir
            self.emergency_state_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Prepara estado compacto
            emergency_state = {
                'timestamp': datetime.now().isoformat(),
                'reason': 'emergency_save',
                'daily_pnl': float(self.daily_pnl),
                'consecutive_losses': self.consecutive_losses,
                'open_positions_count': len(self.open_positions),
                'open_positions': [
                    {
                        'signal_id': pos.signal_id,
                        'asset': pos.asset,
                        'direction': pos.direction.value if hasattr(pos.direction, 'value') else pos.direction,
                        'entry_price': float(pos.entry_price)
                    }
                    for pos in self.open_positions
                ],
                'circuit_breaker_active': self.circuit_breaker_active
            }
            
            # Salva arquivo
            with open(self.emergency_state_file, 'w') as f:
                json.dump(emergency_state, f, indent=2)
            
            self.logger.info(f"Estado de emergência salvo em {self.emergency_state_file}")
            
        except Exception as e:
            self.logger.error(f"Erro ao salvar estado de emergência: {e}")