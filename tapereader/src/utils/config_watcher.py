"""
Config Watcher - Monitora mudanças em arquivos de configuração
Usa watchdog para detectar mudanças e executar callbacks
"""

import asyncio
import logging
from pathlib import Path
from typing import List, Callable, Optional, Dict, Any
from datetime import datetime, timedelta
import yaml

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileModifiedEvent
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    Observer = None
    FileSystemEventHandler = object


class ConfigFileHandler(FileSystemEventHandler):
    """Handler para eventos de mudança em arquivos de configuração"""
    
    def __init__(
        self,
        file_patterns: List[str],
        callback: Callable,
        validation_callback: Optional[Callable] = None,
        debounce_seconds: float = 1.0
    ):
        """
        Args:
            file_patterns: Padrões de arquivo para monitorar (ex: ['*.yaml'])
            callback: Função async a ser chamada quando houver mudança
            validation_callback: Função opcional para validar arquivo antes de notificar
            debounce_seconds: Tempo para aguardar múltiplas mudanças
        """
        super().__init__()
        self.file_patterns = file_patterns
        self.callback = callback
        self.validation_callback = validation_callback
        self.debounce_seconds = debounce_seconds
        self.logger = logging.getLogger(__name__)
        
        # Controle de debounce
        self._pending_changes: Dict[str, datetime] = {}
        self._processing = set()
        
        # Loop assíncrono
        self._loop = None
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            pass
    
    def on_modified(self, event):
        """Chamado quando um arquivo é modificado"""
        if event.is_directory:
            return
            
        filepath = Path(event.src_path)
        
        # Verifica se o arquivo corresponde aos padrões
        if not self._matches_pattern(filepath):
            return
        
        # Ignora arquivos temporários e backups
        if filepath.name.startswith('.') or filepath.name.endswith('~'):
            return
        
        # Registra mudança pendente
        self._pending_changes[str(filepath)] = datetime.now()
        
        # Agenda processamento
        if self._loop:
            asyncio.run_coroutine_threadsafe(
                self._process_change(filepath),
                self._loop
            )
    
    def _matches_pattern(self, filepath: Path) -> bool:
        """Verifica se arquivo corresponde aos padrões"""
        for pattern in self.file_patterns:
            if filepath.match(pattern):
                return True
        return False
    
    async def _process_change(self, filepath: Path):
        """Processa mudança com debounce"""
        filepath_str = str(filepath)
        
        # Evita processamento duplicado
        if filepath_str in self._processing:
            return
        
        self._processing.add(filepath_str)
        
        try:
            # Aguarda debounce
            await asyncio.sleep(self.debounce_seconds)
            
            # Verifica se ainda é a mudança mais recente
            if filepath_str in self._pending_changes:
                last_change = self._pending_changes[filepath_str]
                time_since_change = (datetime.now() - last_change).total_seconds()
                
                if time_since_change >= self.debounce_seconds:
                    # Valida arquivo se callback fornecido
                    if self.validation_callback:
                        try:
                            if not await self._validate_file(filepath):
                                self.logger.warning(f"Arquivo inválido: {filepath}")
                                return
                        except Exception as e:
                            self.logger.error(f"Erro na validação: {e}")
                            return
                    
                    # Executa callback
                    self.logger.info(f"Processando mudança em: {filepath}")
                    
                    if asyncio.iscoroutinefunction(self.callback):
                        await self.callback(filepath)
                    else:
                        self.callback(filepath)
                    
                    # Remove da lista de pendentes
                    del self._pending_changes[filepath_str]
                    
        except Exception as e:
            self.logger.error(f"Erro ao processar mudança em {filepath}: {e}")
        finally:
            self._processing.discard(filepath_str)
    
    async def _validate_file(self, filepath: Path) -> bool:
        """Valida arquivo antes de processar mudança"""
        if self.validation_callback:
            if asyncio.iscoroutinefunction(self.validation_callback):
                return await self.validation_callback(filepath)
            else:
                return self.validation_callback(filepath)
        return True


class ConfigWatcher:
    """Monitora mudanças em arquivos de configuração"""
    
    def __init__(
        self,
        watch_paths: List[str],
        file_patterns: List[str] = None,
        callback: Callable = None,
        validation_callback: Optional[Callable] = None,
        recursive: bool = True,
        debounce_seconds: float = 1.0
    ):
        """
        Args:
            watch_paths: Lista de diretórios para monitorar
            file_patterns: Padrões de arquivo (default: ['*.yaml', '*.yml'])
            callback: Função a ser chamada quando houver mudança
            validation_callback: Função para validar arquivo
            recursive: Se deve monitorar subdiretórios
            debounce_seconds: Tempo de debounce
        """
        if not WATCHDOG_AVAILABLE:
            raise ImportError("watchdog não está instalado. Execute: pip install watchdog>=4.0.0")
        
        self.watch_paths = [Path(p) for p in watch_paths]
        self.file_patterns = file_patterns or ['*.yaml', '*.yml']
        self.callback = callback
        self.validation_callback = validation_callback
        self.recursive = recursive
        self.debounce_seconds = debounce_seconds
        
        self.logger = logging.getLogger(__name__)
        self.observer = None
        self._running = False
        
        # Estatísticas
        self.stats = {
            'files_monitored': 0,
            'changes_detected': 0,
            'reloads_successful': 0,
            'reloads_failed': 0,
            'last_change': None
        }
    
    def start(self):
        """Inicia monitoramento"""
        if self._running:
            self.logger.warning("ConfigWatcher já está rodando")
            return
        
        self.observer = Observer()
        handler = ConfigFileHandler(
            self.file_patterns,
            self.callback,
            self.validation_callback,
            self.debounce_seconds
        )
        
        # Adiciona handlers para cada caminho
        for path in self.watch_paths:
            if path.exists() and path.is_dir():
                self.observer.schedule(
                    handler,
                    str(path),
                    recursive=self.recursive
                )
                self.logger.info(f"Monitorando: {path}")
                
                # Conta arquivos monitorados
                pattern_count = sum(
                    1 for p in path.rglob('*') if path.is_dir()
                    for pattern in self.file_patterns
                    if p.match(pattern)
                )
                self.stats['files_monitored'] += pattern_count
            else:
                self.logger.warning(f"Caminho não encontrado: {path}")
        
        self.observer.start()
        self._running = True
        self.logger.info(
            f"ConfigWatcher iniciado - "
            f"Monitorando {self.stats['files_monitored']} arquivos"
        )
    
    def stop(self):
        """Para monitoramento"""
        if not self._running:
            return
        
        self.observer.stop()
        self.observer.join(timeout=5)
        self._running = False
        self.logger.info("ConfigWatcher parado")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Retorna estatísticas do monitoramento"""
        return {
            **self.stats,
            'is_running': self._running,
            'watch_paths': [str(p) for p in self.watch_paths],
            'file_patterns': self.file_patterns
        }


def validate_yaml_file(filepath: Path) -> bool:
    """
    Valida se um arquivo YAML está bem formado
    
    Args:
        filepath: Caminho do arquivo YAML
        
    Returns:
        True se válido, False caso contrário
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            yaml.safe_load(f)
        return True
    except Exception as e:
        logging.error(f"YAML inválido em {filepath}: {e}")
        return False


def create_config_watcher(
    config_dir: str = "config",
    callback: Callable = None
) -> Optional[ConfigWatcher]:
    """
    Factory function para criar ConfigWatcher com configurações padrão
    
    Args:
        config_dir: Diretório de configuração
        callback: Função callback para mudanças
        
    Returns:
        ConfigWatcher configurado ou None se watchdog não disponível
    """
    if not WATCHDOG_AVAILABLE:
        logging.warning("watchdog não disponível - hot-reload desabilitado")
        return None
    
    return ConfigWatcher(
        watch_paths=[config_dir],
        file_patterns=['*.yaml', '*.yml'],
        callback=callback,
        validation_callback=validate_yaml_file,
        recursive=True,
        debounce_seconds=1.0
    )


# Exemplo de uso
if __name__ == "__main__":
    import asyncio
    
    async def on_config_change(filepath: Path):
        print(f"Configuração mudou: {filepath}")
        
    async def main():
        # Cria watcher
        watcher = create_config_watcher(
            config_dir="config",
            callback=on_config_change
        )
        
        if watcher:
            # Inicia monitoramento
            watcher.start()
            
            try:
                # Mantém rodando
                while True:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                print("\nParando...")
            finally:
                watcher.stop()
        else:
            print("watchdog não instalado")
    
    asyncio.run(main())