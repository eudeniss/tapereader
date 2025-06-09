"""
Sistema de configuração do TapeReader
Carrega e valida configurações YAML
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, validator
import logging


class ExcelConfig(BaseModel):
    """Configuração do Excel"""
    file_path: str
    ranges: Dict[str, Dict[str, str]]
    column_mapping: Dict[str, Dict[str, int]]


class CacheConfig(BaseModel):
    """Configuração de cache"""
    enabled: bool = True
    ttl_seconds: int = 300
    max_size: int = 1000
    redis_url: Optional[str] = None


class StrategyConfig(BaseModel):
    """Configuração de estratégias"""
    enabled_strategies: List[str] = []
    default_strategy: str = "momentum"
    parameters: Dict[str, Dict[str, Any]] = {}


class RiskConfig(BaseModel):
    """Configuração de gerenciamento de risco"""
    max_position_size: float = 0.1
    max_daily_loss: float = 0.02
    stop_loss_percentage: float = 0.05
    position_sizing_method: str = "fixed"
    
    @validator('position_sizing_method')
    def validate_sizing_method(cls, v):
        valid_methods = ['fixed', 'kelly', 'volatility_based']
        if v not in valid_methods:
            raise ValueError(f'position_sizing_method deve ser um de: {valid_methods}')
        return v


class DatabaseConfig(BaseModel):
    """Configuração de banco de dados"""
    enabled: bool = False
    url: str = "sqlite:///tapereader.db"
    pool_size: int = 5
    echo: bool = False
    auto_migrate: bool = True


class MockConfig(BaseModel):
    """Configuração para dados mock"""
    scenarios: Dict[str, Dict[str, Any]] = {}
    default_scenario: str = "balanced"
    random_seed: Optional[int] = None
    tick_interval_ms: int = 500


class BehaviorConfig(BaseModel):
    """Configuração de comportamentos"""
    patterns: Dict[str, Dict[str, Any]] = {}
    thresholds: Dict[str, float] = {}
    timeframes: List[str] = ["1min", "5min", "15min"]


class ModeConfig(BaseModel):
    """Configuração de um modo de operação"""
    data_source: str
    log_level: str = "INFO"
    debug: bool = False
    update_interval_ms: int = 250
    excel_file: Optional[str] = None
    scenarios: Optional[List[str]] = None
    
    @validator('data_source')
    def validate_data_source(cls, v):
        valid_sources = ['excel_rtd', 'mock_dynamic', 'mock_static']
        if v not in valid_sources:
            raise ValueError(f'data_source deve ser um de: {valid_sources}')
        return v


class SystemConfig(BaseModel):
    """Configuração do sistema"""
    min_confidence: float = Field(default=0.80, ge=0.0, le=1.0)
    max_signals_per_minute: int = Field(default=3, ge=1, le=10)


class LoggingConfig(BaseModel):
    """Configuração de logging"""
    base_dir: str = "../logs"
    format: str = "[%(asctime)s] %(levelname)s - %(name)s - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"


class AppConfig(BaseModel):
    """Configuração principal da aplicação"""
    name: str = "TapeReader Professional"
    version: str = "2.0"
    base_path: str = Field(default_factory=lambda: os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Config(BaseModel):
    """Configuração completa do sistema"""
    app: AppConfig
    modes: Dict[str, ModeConfig]
    system: SystemConfig
    logging: LoggingConfig
    excel: ExcelConfig
    
    # Configurações opcionais (podem não estar presentes)
    cache: Optional[CacheConfig] = None
    strategies: Optional[StrategyConfig] = None
    risk: Optional[RiskConfig] = None
    database: Optional[DatabaseConfig] = None
    mock: Optional[MockConfig] = None
    behaviors: Optional[BehaviorConfig] = None
    
    # Modo atual (definido em runtime)
    current_mode: Optional[str] = None
    
    def get_mode_config(self) -> ModeConfig:
        """Retorna configuração do modo atual"""
        if not self.current_mode:
            self.current_mode = os.getenv('MODE', 'production')
        
        if self.current_mode not in self.modes:
            raise ValueError(f"Modo '{self.current_mode}' não encontrado na configuração")
            
        return self.modes[self.current_mode]
    
    def get_cache_config(self) -> CacheConfig:
        """Retorna configuração de cache com valores padrão se não definida"""
        return self.cache or CacheConfig()
    
    def get_risk_config(self) -> RiskConfig:
        """Retorna configuração de risco com valores padrão se não definida"""
        return self.risk or RiskConfig()
    
    def get_database_config(self) -> DatabaseConfig:
        """Retorna configuração de banco de dados com valores padrão se não definida"""
        return self.database or DatabaseConfig()


def merge_configs(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    """
    Mescla duas configurações de forma recursiva
    
    Args:
        base: Configuração base
        update: Configuração para atualizar/adicionar
        
    Returns:
        Configuração mesclada
    """
    result = base.copy()
    
    for key, value in update.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Merge recursivo para dicionários
            result[key] = merge_configs(result[key], value)
        else:
            # Substitui o valor
            result[key] = value
    
    return result


def load_config(config_path: str = None) -> Config:
    """
    Carrega configuração do arquivo YAML e arquivos adicionais
    
    Args:
        config_path: Caminho para o arquivo de configuração principal
        
    Returns:
        Objeto Config validado
        
    Raises:
        FileNotFoundError: Se o arquivo principal não for encontrado
        yaml.YAMLError: Se houver erro ao processar YAML
        ValidationError: Se a configuração for inválida
    """
    if config_path is None:
        # Caminho padrão relativo ao arquivo atual
        base_dir = Path(__file__).parent.parent.parent
        config_path = base_dir / "config" / "main.yaml"
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Arquivo de configuração não encontrado: {config_path}")
    
    try:
        # Carrega o YAML principal
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        
        # Diretório base para arquivos de configuração
        config_dir = config_path.parent
        
        # Lista de arquivos de configuração adicionais
        config_files = [
            'excel.yaml',
            'behaviors.yaml',
            'cache.yaml',
            'strategies.yaml',
            'risk.yaml',
            'database.yaml',
            'mock.yaml'
        ]
        
        # Carrega e mescla arquivos adicionais
        for config_file in config_files:
            config_file_path = config_dir / config_file
            if config_file_path.exists():
                try:
                    with open(config_file_path, 'r', encoding='utf-8') as f:
                        file_data = yaml.safe_load(f)
                        if file_data:
                            # Mescla com a configuração principal
                            config_data = merge_configs(config_data, file_data)
                            logging.debug(f"Carregado arquivo de configuração: {config_file}")
                except Exception as e:
                    logging.warning(f"Erro ao carregar {config_file}: {e}")
        
        # Substitui variáveis de ambiente
        config_str = yaml.dump(config_data)
        config_str = os.path.expandvars(config_str)
        config_data = yaml.safe_load(config_str)
        
        # Cria objeto de configuração validado
        config = Config(**config_data)
        
        # Define modo atual
        config.current_mode = os.getenv('MODE', 'production')
        
        # Valida que o modo existe
        if config.current_mode not in config.modes:
            available_modes = list(config.modes.keys())
            raise ValueError(
                f"Modo '{config.current_mode}' não encontrado. "
                f"Modos disponíveis: {available_modes}"
            )
        
        logging.info(f"Configuração carregada - Modo: {config.current_mode}")
        
        return config
        
    except yaml.YAMLError as e:
        raise ValueError(f"Erro ao processar arquivo YAML: {e}")
    except Exception as e:
        raise RuntimeError(f"Erro ao carregar configuração: {e}")


def reload_config() -> Config:
    """
    Recarrega a configuração global
    
    Returns:
        Nova configuração carregada
    """
    global _config
    _config = load_config()
    logging.info("Configuração recarregada")
    return _config


# Singleton para configuração global
_config: Optional[Config] = None


def get_config() -> Config:
    """
    Retorna configuração global (singleton)
    
    Returns:
        Objeto de configuração global
    """
    global _config
    if _config is None:
        _config = load_config()
    return _config


def set_mode(mode: str) -> None:
    """
    Define o modo de operação atual
    
    Args:
        mode: Nome do modo
        
    Raises:
        ValueError: Se o modo não existir
    """
    config = get_config()
    if mode not in config.modes:
        raise ValueError(f"Modo '{mode}' não existe na configuração")
    
    config.current_mode = mode
    os.environ['MODE'] = mode
    logging.info(f"Modo alterado para: {mode}")


def get_current_mode() -> str:
    """
    Retorna o modo de operação atual
    
    Returns:
        Nome do modo atual
    """
    config = get_config()
    return config.current_mode or os.getenv('MODE', 'production')


# Funções utilitárias para acessar configurações específicas rapidamente
def get_excel_config() -> ExcelConfig:
    """Retorna configuração do Excel"""
    return get_config().excel


def get_system_config() -> SystemConfig:
    """Retorna configuração do sistema"""
    return get_config().system


def get_logging_config() -> LoggingConfig:
    """Retorna configuração de logging"""
    return get_config().logging


def get_app_config() -> AppConfig:
    """Retorna configuração da aplicação"""
    return get_config().app