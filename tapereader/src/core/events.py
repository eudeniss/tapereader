"""
Sistema de eventos assíncrono
Permite comunicação entre componentes
"""

import asyncio
from typing import Dict, List, Callable, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field
import logging


@dataclass
class Event:
    """Evento base do sistema"""
    type: str
    data: Any
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = "system"


class EventBus:
    """Barramento de eventos assíncrono"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._handlers: Dict[str, List[Callable]] = {}
        self._queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        
    def subscribe(self, event_type: str, handler: Callable):
        """Inscreve um handler para um tipo de evento"""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
            
        self._handlers[event_type].append(handler)
        self.logger.debug(f"Handler inscrito para {event_type}")
        
    def unsubscribe(self, event_type: str, handler: Callable):
        """Remove inscrição de um handler"""
        if event_type in self._handlers:
            self._handlers[event_type].remove(handler)
            
    async def publish(self, event: Event):
        """Publica um evento"""
        await self._queue.put(event)
        self.logger.debug(f"Evento publicado: {event.type}")
        
    async def process_events(self):
        """Processa eventos na fila"""
        self._running = True
        self.logger.info("EventBus iniciado")
        
        while self._running:
            try:
                # Aguarda evento com timeout
                event = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=1.0
                )
                
                # Processa handlers
                handlers = self._handlers.get(event.type, [])
                
                for handler in handlers:
                    try:
                        if asyncio.iscoroutinefunction(handler):
                            await handler(event)
                        else:
                            handler(event)
                    except Exception as e:
                        self.logger.error(
                            f"Erro no handler para {event.type}: {e}",
                            exc_info=True
                        )
                        
            except asyncio.TimeoutError:
                # Timeout normal, continua
                continue
            except Exception as e:
                self.logger.error(f"Erro no EventBus: {e}", exc_info=True)
                
    def stop(self):
        """Para o processamento de eventos"""
        self._running = False
        self.logger.info("EventBus parado")
        
    async def wait_for_event(
        self, 
        event_type: str, 
        timeout: float = None
    ) -> Optional[Event]:
        """Aguarda um evento específico"""
        received_event = None
        event_received = asyncio.Event()
        
        def handler(event):
            nonlocal received_event
            received_event = event
            event_received.set()
            
        self.subscribe(event_type, handler)
        
        try:
            await asyncio.wait_for(event_received.wait(), timeout)
            return received_event
        except asyncio.TimeoutError:
            return None
        finally:
            self.unsubscribe(event_type, handler)