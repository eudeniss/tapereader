"""
Sistema de logging do TapeReader
Configura logs para arquivo e console com diferentes níveis
OTIMIZADO: Usa orjson para serialização rápida
"""

import os
import sys
import logging
import logging.handlers
from datetime import datetime
from pathlib import Path
from typing import Optional
import json

# OTIMIZAÇÃO: Importa orjson para serialização rápida
try:
    import orjson
    ORJSON_AVAILABLE = True
except ImportError:
    ORJSON_AVAILABLE = False
    logging.warning("orjson não disponível - usando json padrão (mais lento)")

from pythonjsonlogger import jsonlogger


class SignalFilter(logging.Filter):
    """Filtro para capturar apenas logs de sinais"""
    def filter(self, record):
        return hasattr(record, 'signal') and record.signal


class TapeReaderLogger:
    """Sistema de logging customizado para TapeReader"""
    
    def __init__(self, config):
        self.config = config
        self.log_dir = Path(config.logging.base_dir).resolve()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Cria subdiretório de análise
        self.analysis_dir = self.log_dir / "analysis"
        self.analysis_dir.mkdir(exist_ok=True)
        
        # Configuração base
        self.setup_logging()
        
    def setup_logging(self):
        """Configura o sistema de logging"""
        # Remove handlers existentes
        root = logging.getLogger()
        for handler in root.handlers[:]:
            root.removeHandler(handler)
        
        # Formato padrão
        formatter = logging.Formatter(
            self.config.logging.format,
            datefmt=self.config.logging.date_format
        )
        
        # Formato JSON para análise
        json_formatter = jsonlogger.JsonFormatter()
        
        # Nível base
        mode_config = self.config.get_mode_config()
        log_level = getattr(logging, mode_config.log_level.upper())
        root.setLevel(log_level)
        
        # Handler para console
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(log_level)
        root.addHandler(console_handler)
        
        # Handler para arquivo principal
        app_log_path = self.log_dir / "app.log"
        app_handler = logging.handlers.RotatingFileHandler(
            app_log_path,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        app_handler.setFormatter(formatter)
        app_handler.setLevel(log_level)
        root.addHandler(app_handler)
        
        # Handler para sinais
        signals_log_path = self.log_dir / "signals.log"
        signals_handler = logging.handlers.RotatingFileHandler(
            signals_log_path,
            maxBytes=5*1024*1024,  # 5MB
            backupCount=10,
            encoding='utf-8'
        )
        signals_handler.setFormatter(formatter)
        signals_handler.addFilter(SignalFilter())
        signals_handler.setLevel(logging.INFO)
        root.addHandler(signals_handler)
        
        # Handler para debug (apenas em modo debug)
        if mode_config.debug:
            debug_log_path = self.log_dir / "debug.log"
            debug_handler = logging.handlers.RotatingFileHandler(
                debug_log_path,
                maxBytes=20*1024*1024,  # 20MB
                backupCount=3,
                encoding='utf-8'
            )
            debug_handler.setFormatter(formatter)
            debug_handler.setLevel(logging.DEBUG)
            root.addHandler(debug_handler)
        
        logging.info("Sistema de logging configurado")
        logging.info(f"Logs em: {self.log_dir}")
        if ORJSON_AVAILABLE:
            logging.info("✅ Usando orjson para serialização rápida")
        
    def log_signal(self, signal):
        """Log especial para sinais de trading"""
        logger = logging.getLogger(__name__)
        
        # Log simples
        logger.info(
            f"SINAL {signal.direction} @ {signal.price} - "
            f"Confiança: {signal.confidence:.0%}",
            extra={'signal': True}
        )
        
        # Salva análise detalhada
        self.save_signal_analysis(signal)
        
    def save_signal_analysis(self, signal):
        """
        Salva análise completa do sinal em JSON
        OTIMIZADO: Usa orjson para serialização 3-10x mais rápida
        """
        # Diretório por data
        date_dir = self.analysis_dir / datetime.now().strftime("%Y-%m-%d")
        date_dir.mkdir(exist_ok=True)
        
        # Arquivo do sinal
        signal_file = date_dir / f"{signal.signal_id}.json"
        
        # Dados completos
        signal_data = signal.dict()
        
        # Adiciona metadados
        signal_data['_metadata'] = {
            'saved_at': datetime.now().isoformat(),
            'mode': self.config.current_mode,
            'version': self.config.app.version
        }
        
        if ORJSON_AVAILABLE:
            # OTIMIZAÇÃO: Usa orjson para serialização rápida
            # orjson.OPT_INDENT_2 para manter formatação legível
            # orjson.OPT_SORT_KEYS para ordenar chaves
            # orjson.OPT_NON_STR_KEYS para suportar chaves não-string
            json_bytes = orjson.dumps(
                signal_data,
                option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS | orjson.OPT_NON_STR_KEYS,
                default=self._orjson_default
            )
            
            # Escreve bytes diretamente (mais rápido)
            with open(signal_file, 'wb') as f:
                f.write(json_bytes)
        else:
            # Fallback para json padrão se orjson não estiver disponível
            with open(signal_file, 'w', encoding='utf-8') as f:
                json.dump(signal_data, f, indent=2, ensure_ascii=False, default=str)
            
        logging.debug(f"Análise salva: {signal_file}")
        
    def _orjson_default(self, obj):
        """
        Função de serialização customizada para orjson
        Converte tipos não suportados nativamente
        """
        # Datetime para ISO string
        if isinstance(obj, datetime):
            return obj.isoformat()
        
        # Path para string
        if isinstance(obj, Path):
            return str(obj)
        
        # Decimal para float (cuidado com precisão)
        if hasattr(obj, 'to_eng_string'):  # Decimal
            return float(obj)
        
        # Enum para value
        if hasattr(obj, 'value'):
            return obj.value
        
        # Objetos com método dict()
        if hasattr(obj, 'dict'):
            return obj.dict()
        
        # Fallback para string
        return str(obj)


def setup_logging(config) -> TapeReaderLogger:
    """Configura e retorna o sistema de logging"""
    logger_system = TapeReaderLogger(config)
    return logger_system


# Logger global
_logger_system: Optional[TapeReaderLogger] = None


def get_logger(name: str = None) -> logging.Logger:
    """Retorna um logger configurado"""
    if name is None:
        name = __name__
    return logging.getLogger(name)


def get_logger_system() -> Optional[TapeReaderLogger]:
    """Retorna o sistema de logging global"""
    return _logger_system


def set_logger_system(logger_system: TapeReaderLogger):
    """Define o sistema de logging global"""
    global _logger_system
    _logger_system = logger_system