"""
Strategy Loader - Versão Melhorada com Hot-Reload
Carrega e gerencia todas as estratégias a partir do arquivo YAML
Suporta hot-reload com validação e rollback
"""

import yaml
import logging
import json
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Callable
from datetime import datetime
from copy import deepcopy

from .confluence import ConfluenceAnalyzer
from .decision_matrix import DecisionMatrix
from .risk_manager import RiskManager
from ..core.tracking import SignalTracker


class ConfigValidationError(Exception):
    """Erro de validação de configuração"""
    pass


class StrategyLoader:
    """Carrega e gerencia estratégias do arquivo de configuração com hot-reload"""
    
    def __init__(self, config_path: str = "config/strategies.yaml"):
        self.config_path = Path(config_path)
        self.logger = logging.getLogger(__name__)
        self.config = {}
        
        # Componentes de estratégia
        self.confluence_analyzer = None
        self.decision_matrix = None
        self.risk_manager = None
        self.signal_tracker = None
        
        # Timestamp da última carga
        self.last_loaded = None
        self.load_count = 0
        
        # Histórico de configurações (para rollback)
        self.config_history = []
        self.max_history_size = 5
        
        # Callbacks de reload
        self.reload_callbacks: List[Callable] = []
        
        # Validadores customizados
        self.validators: Dict[str, Callable] = {}
        
        # Carrega configuração inicial
        self.load_config()
        
    def load_config(self) -> bool:
        """Carrega configuração do arquivo YAML"""
        try:
            if not self.config_path.exists():
                self.logger.error(f"Arquivo de configuração não encontrado: {self.config_path}")
                return False
            
            # Faz backup antes de carregar
            self._backup_current_config()
            
            with open(self.config_path, 'r', encoding='utf-8') as f:
                new_config = yaml.safe_load(f)
            
            # Valida nova configuração
            self._validate_full_config(new_config)
            
            # Atualiza configuração
            self.config = new_config
            self.last_loaded = datetime.now()
            self.load_count += 1
            
            self.logger.info(
                f"Configuração carregada de {self.config_path} "
                f"(load #{self.load_count})"
            )
            
            # Inicializa componentes
            self._initialize_components()
            
            # Adiciona ao histórico
            self._add_to_history(new_config)
            
            return True
            
        except ConfigValidationError as e:
            self.logger.error(f"Configuração inválida: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Erro ao carregar configuração: {e}")
            return False
    
    def _validate_full_config(self, config: Dict[str, Any]):
        """Valida configuração completa"""
        # Validação estrutural básica
        self._validate_structure(config)
        
        # Validações específicas por seção
        self._validate_strategies_section(config.get('strategies', {}))
        self._validate_rules_section(config.get('strategy_rules', {}))
        self._validate_risk_section(config.get('strategies', {}).get('risk_manager', {}))
        
        # Validadores customizados
        for name, validator in self.validators.items():
            try:
                validator(config)
            except Exception as e:
                raise ConfigValidationError(f"Validador '{name}' falhou: {e}")
    
    def _validate_structure(self, config: Dict[str, Any]):
        """Valida estrutura básica da configuração"""
        required_sections = ['strategies', 'strategy_rules']
        
        for section in required_sections:
            if section not in config:
                raise ConfigValidationError(f"Seção obrigatória ausente: {section}")
        
        # Valida subseções de strategies
        strategies = config['strategies']
        required_strategies = ['confluence', 'decision_matrix', 'risk_manager']
        
        for strategy in required_strategies:
            if strategy not in strategies:
                raise ConfigValidationError(f"Estratégia obrigatória ausente: {strategy}")
    
    def _validate_strategies_section(self, strategies: Dict[str, Any]):
        """Valida seção de estratégias"""
        # Valida confluence
        confluence = strategies.get('confluence', {})
        if 'correlation_window' in confluence:
            if not 1 <= confluence['correlation_window'] <= 1000:
                raise ConfigValidationError(
                    "correlation_window deve estar entre 1 e 1000"
                )
        
        # Valida decision_matrix
        decision = strategies.get('decision_matrix', {})
        if 'confidence_weights' in decision:
            weights = decision['confidence_weights']
            total = sum(weights.values())
            if abs(total - 1.0) > 0.01:
                raise ConfigValidationError(
                    f"Soma dos pesos de confiança deve ser 1.0 (atual: {total})"
                )
    
    def _validate_rules_section(self, rules: Dict[str, Any]):
        """Valida seção de regras"""
        for rule_id, rule in rules.items():
            valid, error = self.validate_rule(rule)
            if not valid:
                raise ConfigValidationError(f"Regra '{rule_id}' inválida: {error}")
    
    def _validate_risk_section(self, risk_config: Dict[str, Any]):
        """Valida configuração de risco"""
        # Valida limites
        if 'max_daily_loss' in risk_config:
            if not 0 < risk_config['max_daily_loss'] <= 0.1:
                raise ConfigValidationError(
                    "max_daily_loss deve estar entre 0 e 0.1 (10%)"
                )
        
        if 'max_concurrent_positions' in risk_config:
            if not 1 <= risk_config['max_concurrent_positions'] <= 10:
                raise ConfigValidationError(
                    "max_concurrent_positions deve estar entre 1 e 10"
                )
    
    def _initialize_components(self):
        """Inicializa todos os componentes de estratégia"""
        try:
            strategies_config = self.config['strategies']
            
            # Signal Tracker (preserva histórico se existir)
            if self.signal_tracker is None:
                self.signal_tracker = SignalTracker()
            
            # Confluence Analyzer
            self.confluence_analyzer = ConfluenceAnalyzer(
                strategies_config['confluence']
            )
            
            # Decision Matrix com regras
            decision_config = {
                **strategies_config['decision_matrix'],
                'strategy_rules': self.config.get('strategy_rules', {}),
                'advanced_strategy_rules': self.config.get('advanced_strategy_rules', {}),
                'risk_reward_config': self.config.get('risk_reward_config', {})
            }
            
            self.decision_matrix = DecisionMatrix(
                decision_config,
                self.signal_tracker
            )
            
            # Risk Manager
            risk_config = {
                **strategies_config['risk_manager'],
                'circuit_breakers': self.config.get('circuit_breakers', {}),
                'risk_adjustments': self.config.get('risk_adjustments', {})
            }
            
            # Se já existe, preserva estado
            if self.risk_manager and hasattr(self.risk_manager, 'get_state'):
                old_state = self.risk_manager.get_state()
                self.risk_manager = RiskManager(risk_config)
                if hasattr(self.risk_manager, 'load_state'):
                    self.risk_manager.load_state(old_state)
            else:
                self.risk_manager = RiskManager(risk_config)
            
            self.logger.info("Todos os componentes inicializados com sucesso")
            
            # Notifica callbacks
            self._notify_reload_callbacks()
            
        except Exception as e:
            self.logger.error(f"Erro ao inicializar componentes: {e}")
            raise
    
    def reload_config(self, validate_only: bool = False) -> bool:
        """
        Recarrega configuração do arquivo
        
        Args:
            validate_only: Se True, apenas valida sem aplicar mudanças
            
        Returns:
            True se sucesso, False caso contrário
        """
        self.logger.info("Recarregando configuração...")
        
        # Guarda config antiga para rollback
        old_config = deepcopy(self.config)
        old_components = self._save_components_state()
        
        try:
            # Carrega nova config
            with open(self.config_path, 'r', encoding='utf-8') as f:
                new_config = yaml.safe_load(f)
            
            # Valida nova configuração
            self._validate_full_config(new_config)
            
            if validate_only:
                self.logger.info("Validação concluída - configuração válida")
                return True
            
            # Detecta mudanças
            changes = self._detect_changes(old_config, new_config)
            if not changes:
                self.logger.info("Nenhuma mudança detectada")
                return True
            
            # Aplica nova configuração
            self.config = new_config
            self.last_loaded = datetime.now()
            self.load_count += 1
            
            # Reinicializa componentes
            self._initialize_components()
            
            # Adiciona ao histórico
            self._add_to_history(new_config)
            
            # Log das mudanças
            self.logger.info(
                f"Configuração recarregada com sucesso - "
                f"{len(changes)} mudanças aplicadas"
            )
            for change in changes[:5]:  # Mostra até 5 mudanças
                self.logger.debug(f"  - {change}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Falha ao recarregar: {e}")
            
            # Rollback
            try:
                self.logger.info("Executando rollback...")
                self.config = old_config
                self._restore_components_state(old_components)
                self._initialize_components()
                self.logger.info("Rollback concluído")
            except Exception as rollback_error:
                self.logger.critical(f"Falha no rollback: {rollback_error}")
            
            return False
    
    def _detect_changes(self, old_config: Dict, new_config: Dict) -> List[str]:
        """Detecta mudanças entre configurações"""
        changes = []
        
        def compare_dicts(d1, d2, path=""):
            for key in set(d1.keys()) | set(d2.keys()):
                current_path = f"{path}.{key}" if path else key
                
                if key not in d1:
                    changes.append(f"Adicionado: {current_path}")
                elif key not in d2:
                    changes.append(f"Removido: {current_path}")
                elif isinstance(d1[key], dict) and isinstance(d2[key], dict):
                    compare_dicts(d1[key], d2[key], current_path)
                elif d1[key] != d2[key]:
                    changes.append(f"Modificado: {current_path}")
        
        compare_dicts(old_config, new_config)
        return changes
    
    def _save_components_state(self) -> Dict[str, Any]:
        """Salva estado dos componentes para rollback"""
        state = {}
        
        if self.risk_manager and hasattr(self.risk_manager, 'get_state'):
            state['risk_manager'] = self.risk_manager.get_state()
        
        if self.signal_tracker and hasattr(self.signal_tracker, 'get_state'):
            state['signal_tracker'] = self.signal_tracker.get_state()
        
        return state
    
    def _restore_components_state(self, state: Dict[str, Any]):
        """Restaura estado dos componentes"""
        if 'risk_manager' in state and self.risk_manager and hasattr(self.risk_manager, 'load_state'):
            self.risk_manager.load_state(state['risk_manager'])
        
        if 'signal_tracker' in state and self.signal_tracker and hasattr(self.signal_tracker, 'load_state'):
            self.signal_tracker.load_state(state['signal_tracker'])
    
    def _backup_current_config(self):
        """Faz backup da configuração atual"""
        if not self.config_path.exists():
            return
        
        backup_dir = self.config_path.parent / "backups"
        backup_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = backup_dir / f"{self.config_path.stem}_backup_{timestamp}.yaml"
        
        try:
            shutil.copy2(self.config_path, backup_file)
            self.logger.debug(f"Backup criado: {backup_file}")
            
            # Limpa backups antigos
            self._cleanup_old_backups(backup_dir)
        except Exception as e:
            self.logger.warning(f"Falha ao criar backup: {e}")
    
    def _cleanup_old_backups(self, backup_dir: Path, keep_count: int = 10):
        """Remove backups antigos"""
        backups = sorted(backup_dir.glob("*_backup_*.yaml"))
        
        if len(backups) > keep_count:
            for backup in backups[:-keep_count]:
                try:
                    backup.unlink()
                    self.logger.debug(f"Backup antigo removido: {backup}")
                except:
                    pass
    
    def _add_to_history(self, config: Dict[str, Any]):
        """Adiciona configuração ao histórico"""
        self.config_history.append({
            'timestamp': datetime.now(),
            'config': deepcopy(config),
            'load_count': self.load_count
        })
        
        # Mantém apenas últimas N configurações
        if len(self.config_history) > self.max_history_size:
            self.config_history.pop(0)
    
    def _notify_reload_callbacks(self):
        """Notifica callbacks registrados sobre reload"""
        for callback in self.reload_callbacks:
            try:
                callback(self)
            except Exception as e:
                self.logger.error(f"Erro em callback de reload: {e}")
    
    def register_reload_callback(self, callback: Callable):
        """Registra callback para ser chamado após reload"""
        self.reload_callbacks.append(callback)
    
    def register_validator(self, name: str, validator: Callable):
        """Registra validador customizado"""
        self.validators[name] = validator
    
    def validate_rule(self, rule_config: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Valida uma regra de estratégia"""
        required_fields = [
            'name', 'required_behaviors', 'min_confidence',
            'base_confidence', 'description', 'entry_offset',
            'stop_multiplier', 'target_multiplier'
        ]
        
        for field in required_fields:
            if field not in rule_config:
                return False, f"Campo obrigatório ausente: {field}"
        
        # Valida tipos
        if not isinstance(rule_config['required_behaviors'], list):
            return False, "required_behaviors deve ser uma lista"
        
        if not rule_config['required_behaviors']:
            return False, "Pelo menos um comportamento obrigatório é necessário"
        
        # Valida ranges
        if not 0 <= rule_config['min_confidence'] <= 1:
            return False, "min_confidence deve estar entre 0 e 1"
        
        if not 0 <= rule_config['base_confidence'] <= 1:
            return False, "base_confidence deve estar entre 0 e 1"
        
        if rule_config['min_confidence'] > rule_config['base_confidence']:
            return False, "min_confidence não pode ser maior que base_confidence"
        
        return True, None
    
    def update_config(self, updates: Dict[str, Any]) -> bool:
        """Atualiza configuração e salva no arquivo"""
        try:
            # Faz backup
            self._backup_current_config()
            
            # Aplica atualizações
            new_config = deepcopy(self.config)
            self._deep_update(new_config, updates)
            
            # Valida nova configuração
            self._validate_full_config(new_config)
            
            # Salva no arquivo
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(new_config, f, default_flow_style=False, sort_keys=False)
            
            # Recarrega
            return self.reload_config()
            
        except Exception as e:
            self.logger.error(f"Erro ao atualizar configuração: {e}")
            return False
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict):
        """Atualiza dicionário recursivamente"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def rollback_config(self, steps: int = 1) -> bool:
        """
        Volta para uma configuração anterior
        
        Args:
            steps: Número de versões para voltar
            
        Returns:
            True se sucesso
        """
        if not self.config_history or steps > len(self.config_history):
            self.logger.error("Histórico insuficiente para rollback")
            return False
        
        try:
            # Pega configuração do histórico
            historical = self.config_history[-(steps + 1)]
            old_config = historical['config']
            
            # Aplica configuração histórica
            self.config = deepcopy(old_config)
            self._initialize_components()
            
            # Salva no arquivo
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
            
            self.logger.info(
                f"Rollback realizado - voltou {steps} versões "
                f"para {historical['timestamp'].strftime('%H:%M:%S')}"
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Erro no rollback: {e}")
            return False
    
    def get_config_section(self, section: str) -> Optional[Dict[str, Any]]:
        """Retorna uma seção específica da configuração"""
        return self.config.get(section)
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Retorna informações sobre as estratégias carregadas"""
        info = {
            'config_path': str(self.config_path),
            'last_loaded': self.last_loaded.strftime('%Y-%m-%d %H:%M:%S') if self.last_loaded else None,
            'load_count': self.load_count,
            'history_size': len(self.config_history),
            'components': {
                'confluence': self.confluence_analyzer is not None,
                'decision_matrix': self.decision_matrix is not None,
                'risk_manager': self.risk_manager is not None,
                'signal_tracker': self.signal_tracker is not None
            }
        }
        
        # Adiciona info das regras
        if self.decision_matrix:
            info['active_rules'] = self.decision_matrix.get_active_rules()
        
        # Adiciona status do risk manager
        if self.risk_manager:
            info['risk_status'] = self.risk_manager.get_risk_status()
        
        return info
    
    def export_config(self, output_path: Optional[Path] = None) -> Path:
        """
        Exporta configuração atual
        
        Args:
            output_path: Caminho de saída (opcional)
            
        Returns:
            Path do arquivo exportado
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = Path(f"config_export_{timestamp}.yaml")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
        
        self.logger.info(f"Configuração exportada para: {output_path}")
        return output_path