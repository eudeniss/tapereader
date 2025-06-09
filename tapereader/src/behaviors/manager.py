"""
Gerenciador de Comportamentos - Versão Final Otimizada
Coordena todos os detectores e consolida resultados
Otimização: ProcessPoolExecutor para detectores CPU-bound + as_completed
Versão: 3.1 - Com processamento de resultados conforme chegam
"""

import asyncio
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging
import functools

from .base import BehaviorDetector
from .absorption import AbsorptionDetector
from .exhaustion import ExhaustionDetector
from .institutional import InstitutionalFlowDetector
from .support_resistance import SupportResistanceEnhancedDetector
from .sweep import SweepDetector
from .stop_hunt import StopHuntDetector
from .iceberg import IcebergDetector
from .momentum import MomentumDetector
from .breakout import BreakoutDetector
from .divergence import DivergenceDetector
from .htf import HTFDetector
from .micro_aggression import MicroAggressionDetector
from .recurrence import RecurrenceDetector
from .renovation import RenovationDetector

from ..core.models import MarketData, BehaviorDetection
from ..core.events import Event


# Função helper para executar detector de forma síncrona (para ProcessPoolExecutor)
def run_detector_sync(detector_class, config, market_data_dict):
    """
    Função auxiliar para executar detector em processo separado
    Nota: Não pode usar objetos complexos, apenas dicts serializáveis
    """
    try:
        # Recria o detector no processo
        detector = detector_class(config)
        
        # Reconstrói MarketData do dict
        market_data = MarketData.parse_obj(market_data_dict)
        
        # Executa detecção de forma síncrona
        result = detector.detect_sync(market_data)
        
        # Retorna como dict para serialização
        return result.dict() if result and result.detected else None
    except Exception as e:
        return {'error': str(e)}


class BehaviorManager:
    """
    Gerencia todos os detectores de comportamento
    Executa análises em paralelo e consolida resultados
    OTIMIZADO: Usa ProcessPoolExecutor para detectores CPU-bound + as_completed
    """
    
    def __init__(self, config: Dict[str, Any], event_bus=None):
        self.config = config
        self.event_bus = event_bus
        self.logger = logging.getLogger(__name__)
        
        # OTIMIZAÇÃO: Configuração de workers
        self.max_workers = config.get('max_process_workers', 4)
        self.process_executor = ProcessPoolExecutor(max_workers=self.max_workers)
        
        # OTIMIZAÇÃO: Define detectores CPU-bound (computacionalmente intensivos)
        self.cpu_bound_detectors = config.get('cpu_bound_detectors', [
            'recurrence',      # Análise de padrões complexos
            'divergence',      # Cálculos estatísticos pesados
            'htf',            # Análise de algoritmos
            'micro_aggression' # Processamento de micro-estrutura
        ])
        
        # Inicializa detectores
        self.detectors = self._initialize_detectors()
        
        # Cache de resultados
        self.last_results: Dict[str, List[BehaviorDetection]] = {
            'DOLFUT': [],
            'WDOFUT': []
        }
        
        # Estatísticas
        self.stats = {
            'total_detections': 0,
            'detections_by_type': {},
            'detection_times': [],
            'process_pool_usage': 0,
            'async_usage': 0,
            'first_result_time': []  # Novo: tempo até primeiro resultado
        }
        
        self.logger.info(
            f"BehaviorManager inicializado (Otimizado) - "
            f"Max workers: {self.max_workers}, "
            f"CPU-bound detectors: {self.cpu_bound_detectors}"
        )
        
    def _initialize_detectors(self) -> Dict[str, BehaviorDetector]:
        """Inicializa todos os detectores configurados"""
        detectors = {}
        
        # Mapa de classes de detectores (14 behaviors)
        detector_classes = {
            'absorption': AbsorptionDetector,
            'exhaustion': ExhaustionDetector,
            'institutional': InstitutionalFlowDetector,
            'support_resistance': SupportResistanceEnhancedDetector,
            'sweep': SweepDetector,
            'stop_hunt': StopHuntDetector,
            'iceberg': IcebergDetector,
            'momentum': MomentumDetector,
            'breakout': BreakoutDetector,
            'divergence': DivergenceDetector,
            'htf': HTFDetector,
            'micro_aggression': MicroAggressionDetector,
            'recurrence': RecurrenceDetector,
            'renovation': RenovationDetector
        }
        
        # Carrega configurações de comportamentos
        behaviors_config = self.config.get('behaviors', {})
        
        for behavior_type, detector_class in detector_classes.items():
            behavior_config = behaviors_config.get(behavior_type, {})
            
            # Verifica se está habilitado
            if behavior_config.get('enabled', True):
                try:
                    detector = detector_class(behavior_config)
                    detectors[behavior_type] = detector
                    
                    # OTIMIZAÇÃO: Marca se é CPU-bound
                    detector._is_cpu_bound = behavior_type in self.cpu_bound_detectors
                    detector._class_ref = detector_class  # Referência para ProcessPool
                    
                    self.logger.info(
                        f"Detector {behavior_type} inicializado "
                        f"(CPU-bound: {detector._is_cpu_bound})"
                    )
                except Exception as e:
                    self.logger.error(f"Erro ao inicializar {behavior_type}: {e}")
                    
        self.logger.info(f"Total de detectores ativos: {len(detectors)}")
        return detectors
        
    async def analyze(self, market_data: MarketData) -> List[BehaviorDetection]:
        """
        Analisa dados de mercado com todos os detectores
        OTIMIZADO: Usa ProcessPool + as_completed para processar resultados conforme chegam
        
        Args:
            market_data: Dados atuais do mercado
            
        Returns:
            Lista de comportamentos detectados
        """
        start_time = datetime.now()
        first_result_time = None
        
        # Atualiza histórico em todos os detectores
        for detector in self.detectors.values():
            detector.update_history(market_data)
        
        # Lista para armazenar detecções conforme chegam
        detections = []
        
        # OTIMIZAÇÃO: Cria todas as tasks com identificação
        all_tasks: List[Tuple[asyncio.Task, str, str]] = []  # (task, detector_name, task_type)
        
        loop = asyncio.get_running_loop()
        
        for detector_name, detector in self.detectors.items():
            if getattr(detector, '_is_cpu_bound', False):
                # OTIMIZAÇÃO: Executa em processo separado
                self.stats['process_pool_usage'] += 1
                
                # Serializa MarketData para passar entre processos
                market_data_dict = market_data.dict()
                
                # Cria task para executar em processo
                task = asyncio.create_task(
                    loop.run_in_executor(
                        self.process_executor,
                        run_detector_sync,
                        detector._class_ref,
                        detector.config,
                        market_data_dict
                    )
                )
                all_tasks.append((task, detector_name, 'process'))
            else:
                # Executa como corrotina normal (I/O bound)
                self.stats['async_usage'] += 1
                task = asyncio.create_task(
                    self._run_detector_async(detector_name, detector, market_data)
                )
                all_tasks.append((task, detector_name, 'async'))
        
        # OTIMIZAÇÃO: Processa resultados conforme chegam usando as_completed
        for task in asyncio.as_completed([t[0] for t in all_tasks]):
            try:
                result = await task
                
                # Registra tempo do primeiro resultado
                if first_result_time is None:
                    first_result_time = datetime.now()
                    self.stats['first_result_time'].append(
                        (first_result_time - start_time).total_seconds()
                    )
                
                # Encontra informações da task
                task_info = next((t for t in all_tasks if t[0] == task), None)
                if not task_info:
                    continue
                    
                _, detector_name, task_type = task_info
                
                # Processa resultado baseado no tipo
                if task_type == 'async':
                    # Resultado direto de detector async
                    if isinstance(result, BehaviorDetection) and result.detected:
                        detections.append(result)
                        self.logger.debug(
                            f"Detecção processada: {detector_name} "
                            f"(confiança: {result.confidence:.2%})"
                        )
                else:  # process
                    # Resultado do ProcessPool (dict)
                    if isinstance(result, dict):
                        if 'error' in result:
                            self.logger.error(
                                f"Erro no processo {detector_name}: {result['error']}"
                            )
                        else:
                            # Reconstrói BehaviorDetection do dict
                            try:
                                detection = BehaviorDetection.parse_obj(result)
                                if detection.detected:
                                    detections.append(detection)
                                    self.logger.debug(
                                        f"Detecção processada: {detector_name} "
                                        f"(confiança: {detection.confidence:.2%})"
                                    )
                            except Exception as e:
                                self.logger.error(
                                    f"Erro ao reconstruir resultado de {detector_name}: {e}"
                                )
                                
            except Exception as e:
                # Encontra nome do detector para log
                task_info = next((t for t in all_tasks if t[0] == task), None)
                detector_name = task_info[1] if task_info else "unknown"
                self.logger.error(
                    f"Erro ao processar resultado do detector {detector_name}: {e}",
                    exc_info=True
                )
        
        # Atualiza cache e estatísticas
        self.last_results[market_data.asset] = detections
        self._update_statistics(detections, start_time)
        
        # Limita histórico de first_result_time
        if len(self.stats['first_result_time']) > 100:
            self.stats['first_result_time'].pop(0)
        
        # Emite evento se configurado
        if self.event_bus and detections:
            await self._emit_behavior_event(market_data.asset, detections)
            
        return detections
        
    async def _run_detector_async(
        self,
        name: str,
        detector: BehaviorDetector,
        market_data: MarketData
    ) -> Optional[BehaviorDetection]:
        """Executa detector assíncrono (I/O bound) com tratamento de erro"""
        try:
            # Timeout para evitar travamento
            result = await asyncio.wait_for(
                detector.detect(market_data),
                timeout=1.0  # 1 segundo máximo
            )
            
            if result.detected:
                self.logger.debug(
                    f"{name} detectado com confiança {result.confidence:.2%}"
                )
                
            return result
            
        except asyncio.TimeoutError:
            self.logger.warning(f"Timeout no detector {name}")
            return None
        except Exception as e:
            self.logger.error(f"Erro no detector {name}: {e}")
            raise
            
    def get_active_behaviors(self, asset: str) -> List[BehaviorDetection]:
        """Retorna comportamentos ativos para um ativo"""
        return self.last_results.get(asset, [])
        
    def get_behavior_summary(self, asset: str) -> Dict[str, Any]:
        """Retorna resumo dos comportamentos detectados"""
        behaviors = self.get_active_behaviors(asset)
        
        if not behaviors:
            return {
                'count': 0,
                'behaviors': [],
                'avg_confidence': 0.0,
                'strongest': None
            }
            
        # Resumo
        summary = {
            'count': len(behaviors),
            'behaviors': [b.behavior_type for b in behaviors],
            'avg_confidence': sum(b.confidence for b in behaviors) / len(behaviors),
            'strongest': max(behaviors, key=lambda b: b.confidence)
        }
        
        # Adiciona metadados importantes
        for behavior in behaviors:
            if behavior.behavior_type == 'absorption':
                summary['absorption_side'] = behavior.metadata.get('absorption_side')
            elif behavior.behavior_type == 'institutional':
                summary['institutional_direction'] = behavior.metadata.get('flow_direction')
            elif behavior.behavior_type == 'htf':
                summary['htf_algorithm'] = behavior.metadata.get('algorithm_type')
            elif behavior.behavior_type == 'micro_aggression':
                summary['micro_direction'] = behavior.metadata.get('direction')
                
        return summary
        
    def check_behavior_combinations(
        self, 
        behaviors: List[BehaviorDetection]
    ) -> List[Dict[str, Any]]:
        """
        Verifica combinações poderosas de comportamentos
        Versão 100% completa com todas as combinações e campos padronizados
        
        Returns:
            Lista de combinações detectadas
        """
        combinations = []
        behavior_types = {b.behavior_type: b for b in behaviors}
        
        # === COMBINAÇÕES DUPLAS EXISTENTES (COM PADRONIZAÇÃO) ===
        
        # 1. Absorção + Exaustão (Reversão forte)
        if 'absorption' in behavior_types and 'exhaustion' in behavior_types:
            combinations.append({
                'type': 'absorption_exhaustion',
                'strength': 'strong',
                'confidence': (
                    behavior_types['absorption'].confidence + 
                    behavior_types['exhaustion'].confidence
                ) / 2,
                'description': 'Reversão forte detectada',
                'priority': 'high',
                'risk_level': 'medium'
            })
            
        # 2. Institucional + Suporte/Resistência
        if 'institutional' in behavior_types and 'support_resistance' in behavior_types:
            combinations.append({
                'type': 'institutional_level',
                'strength': 'strong',
                'confidence': (
                    behavior_types['institutional'].confidence + 
                    behavior_types['support_resistance'].confidence
                ) / 2,
                'description': 'Institucional defendendo nível',
                'priority': 'high',
                'risk_level': 'low'
            })
            
        # 3. Stop Hunt + Exaustão (Fade)
        if 'stop_hunt' in behavior_types and 'exhaustion' in behavior_types:
            combinations.append({
                'type': 'stop_hunt_fade',
                'strength': 'medium',
                'confidence': (
                    behavior_types['stop_hunt'].confidence + 
                    behavior_types['exhaustion'].confidence
                ) / 2,
                'description': 'Oportunidade de fade após stop hunt',
                'priority': 'medium',
                'risk_level': 'medium'
            })
            
        # 4. Sweep + Momentum (Breakout)
        if 'sweep' in behavior_types and 'momentum' in behavior_types:
            combinations.append({
                'type': 'sweep_breakout',
                'strength': 'strong',
                'confidence': (
                    behavior_types['sweep'].confidence + 
                    behavior_types['momentum'].confidence
                ) / 2,
                'description': 'Breakout com força',
                'priority': 'high',
                'risk_level': 'medium'
            })
            
        # 5. HFT + Iceberg (Algoritmo complexo)
        if 'htf' in behavior_types and 'iceberg' in behavior_types:
            combinations.append({
                'type': 'complex_algorithm',
                'strength': 'strong',
                'confidence': (
                    behavior_types['htf'].confidence + 
                    behavior_types['iceberg'].confidence
                ) / 2,
                'description': 'Algoritmo complexo com ordens ocultas',
                'priority': 'high',
                'risk_level': 'low'
            })
            
        # 6. Micro Agressão + Divergência (Acumulação discreta)
        if 'micro_aggression' in behavior_types and 'divergence' in behavior_types:
            combinations.append({
                'type': 'stealth_accumulation',
                'strength': 'strong',
                'confidence': (
                    behavior_types['micro_aggression'].confidence + 
                    behavior_types['divergence'].confidence
                ) / 2,
                'description': 'Acumulação discreta com divergência',
                'priority': 'medium',
                'risk_level': 'low'
            })
            
        # 7. Recorrência + Breakout (Padrão confiável)
        if 'recurrence' in behavior_types and 'breakout' in behavior_types:
            combinations.append({
                'type': 'reliable_breakout',
                'strength': 'very_strong',
                'confidence': (
                    behavior_types['recurrence'].confidence + 
                    behavior_types['breakout'].confidence
                ) / 2,
                'description': 'Breakout com padrão recorrente confirmado',
                'priority': 'high',
                'risk_level': 'low'
            })
            
        # 8. Renovação + Suporte/Resistência (Defesa ativa)
        if 'renovation' in behavior_types and 'support_resistance' in behavior_types:
            combinations.append({
                'type': 'active_defense',
                'strength': 'strong',
                'confidence': (
                    behavior_types['renovation'].confidence + 
                    behavior_types['support_resistance'].confidence
                ) / 2,
                'description': 'Defesa ativa de nível importante',
                'priority': 'high',
                'risk_level': 'low'
            })
            
        # 9. Institucional + HFT (Big player automatizado)
        if 'institutional' in behavior_types and 'htf' in behavior_types:
            combinations.append({
                'type': 'automated_institutional',
                'strength': 'very_strong',
                'confidence': (
                    behavior_types['institutional'].confidence + 
                    behavior_types['htf'].confidence
                ) / 2,
                'description': 'Institucional operando via algoritmo',
                'priority': 'high',
                'risk_level': 'low'
            })
            
        # === NOVAS COMBINAÇÕES CRÍTICAS (COMPLETAS) ===
        
        # 10. Institutional + Iceberg
        if 'institutional' in behavior_types and 'iceberg' in behavior_types:
            combinations.append({
                'type': 'institutional_iceberg',
                'strength': 'very_strong',
                'confidence': min(1.0, (
                    behavior_types['institutional'].confidence + 
                    behavior_types['iceberg'].confidence
                ) / 2 * 1.10),  # 10% bonus
                'description': 'Grande player ocultando tamanho real - MUITO FORTE',
                'priority': 'high',
                'risk_level': 'low'
            })
            
        # 11. Absorption + Iceberg  
        if 'absorption' in behavior_types and 'iceberg' in behavior_types:
            combinations.append({
                'type': 'hidden_absorption',
                'strength': 'very_strong',
                'confidence': min(1.0, (
                    behavior_types['absorption'].confidence + 
                    behavior_types['iceberg'].confidence
                ) / 2 * 1.15),  # 15% bonus
                'description': 'Absorção oculta via iceberg - Sinal premium',
                'priority': 'high',
                'risk_level': 'low'
            })
            
        # 12. Sweep + Stop Hunt
        if 'sweep' in behavior_types and 'stop_hunt' in behavior_types:
            combinations.append({
                'type': 'liquidity_grab',
                'strength': 'strong',
                'confidence': (
                    behavior_types['sweep'].confidence + 
                    behavior_types['stop_hunt'].confidence
                ) / 2,
                'description': 'Limpeza agressiva de liquidez',
                'priority': 'high',
                'risk_level': 'medium'
            })
            
        # 13. Momentum + Institutional
        if 'momentum' in behavior_types and 'institutional' in behavior_types:
            combinations.append({
                'type': 'backed_momentum',
                'strength': 'very_strong',
                'confidence': min(1.0, (
                    behavior_types['momentum'].confidence + 
                    behavior_types['institutional'].confidence
                ) / 2 * 1.10),  # 10% bonus
                'description': 'Momentum com suporte de grande player',
                'priority': 'high',
                'risk_level': 'low'
            })
            
        # 14. Divergence + Support/Resistance
        if 'divergence' in behavior_types and 'support_resistance' in behavior_types:
            combinations.append({
                'type': 'level_divergence',
                'strength': 'strong',
                'confidence': (
                    behavior_types['divergence'].confidence + 
                    behavior_types['support_resistance'].confidence
                ) / 2,
                'description': 'Divergência em nível importante',
                'priority': 'medium',
                'risk_level': 'low'
            })
            
        # === COMBINAÇÕES TRIPLAS PREMIUM ===
        
        # 15. Tripla: Institutional + Absorption + Support
        if all(b in behavior_types for b in ['institutional', 'absorption', 'support_resistance']):
            combinations.append({
                'type': 'fortress_defense',
                'strength': 'premium',
                'confidence': min(1.0, sum(
                    behavior_types[b].confidence for b in 
                    ['institutional', 'absorption', 'support_resistance']
                ) / 3 * 1.25),  # 25% bonus
                'description': 'DEFESA INSTITUCIONAL COMPLETA - Sinal máximo',
                'confidence_boost': 0.20,
                'priority': 'critical',
                'risk_level': 'very_low'
            })
            
        # 16. Tripla: Momentum + Sweep + Breakout
        if all(b in behavior_types for b in ['momentum', 'sweep', 'breakout']):
            combinations.append({
                'type': 'explosive_breakout',
                'strength': 'premium',
                'confidence': min(1.0, sum(
                    behavior_types[b].confidence for b in 
                    ['momentum', 'sweep', 'breakout']
                ) / 3 * 1.25),  # 25% bonus
                'description': 'ROMPIMENTO EXPLOSIVO - Máxima força',
                'confidence_boost': 0.20,
                'priority': 'critical',
                'risk_level': 'medium'
            })
            
        # 17. Tripla: HFT + Iceberg + Institutional
        if all(b in behavior_types for b in ['htf', 'iceberg', 'institutional']):
            combinations.append({
                'type': 'algo_whale',
                'strength': 'premium',
                'confidence': min(1.0, sum(
                    behavior_types[b].confidence for b in 
                    ['htf', 'iceberg', 'institutional']
                ) / 3 * 1.30),  # 30% bonus
                'description': 'BALEIA ALGORÍTMICA - Manipulação profissional',
                'confidence_boost': 0.25,
                'priority': 'critical',
                'risk_level': 'very_low'
            })
            
        # 18. Tripla: Stop Hunt + Sweep + Exhaustion
        if all(b in behavior_types for b in ['stop_hunt', 'sweep', 'exhaustion']):
            combinations.append({
                'type': 'liquidity_trap',
                'strength': 'premium',
                'confidence': min(1.0, sum(
                    behavior_types[b].confidence for b in 
                    ['stop_hunt', 'sweep', 'exhaustion']
                ) / 3 * 1.20),  # 20% bonus
                'description': 'ARMADILHA COMPLETA - Reversão iminente',
                'confidence_boost': 0.18,
                'priority': 'critical',
                'risk_level': 'medium'
            })
            
        # === COMBINAÇÕES TRIPLAS ADICIONAIS DE ELITE ===
        
        # 19. Tripla: Absorption + Divergence + Support/Resistance
        if all(b in behavior_types for b in ['absorption', 'divergence', 'support_resistance']):
            combinations.append({
                'type': 'reversal_confluence',
                'strength': 'premium',
                'confidence': min(1.0, sum(
                    behavior_types[b].confidence for b in 
                    ['absorption', 'divergence', 'support_resistance']
                ) / 3 * 1.22),  # 22% bonus
                'description': 'CONFLUÊNCIA DE REVERSÃO - Setup completo',
                'confidence_boost': 0.18,
                'priority': 'critical',
                'risk_level': 'low'
            })
            
        # 20. Tripla: Micro Aggression + Renovation + Iceberg
        if all(b in behavior_types for b in ['micro_aggression', 'renovation', 'iceberg']):
            combinations.append({
                'type': 'stealth_whale',
                'strength': 'premium',
                'confidence': min(1.0, sum(
                    behavior_types[b].confidence for b in 
                    ['micro_aggression', 'renovation', 'iceberg']
                ) / 3 * 1.28),  # 28% bonus
                'description': 'BALEIA FURTIVA - Acumulação profissional oculta',
                'confidence_boost': 0.22,
                'priority': 'critical',
                'risk_level': 'very_low'
            })
            
        return combinations
        
    async def _emit_behavior_event(
        self,
        asset: str,
        detections: List[BehaviorDetection]
    ):
        """Emite evento com comportamentos detectados"""
        if not self.event_bus:
            return
            
        event = Event(
            type='behaviors_detected',
            data={
                'asset': asset,
                'detections': detections,
                'summary': self.get_behavior_summary(asset),
                'combinations': self.check_behavior_combinations(detections)
            },
            source='BehaviorManager'
        )
        
        await self.event_bus.publish(event)
        
    def _update_statistics(self, detections: List[BehaviorDetection], start_time: datetime):
        """Atualiza estatísticas de detecção"""
        # Tempo de processamento
        process_time = (datetime.now() - start_time).total_seconds()
        self.stats['detection_times'].append(process_time)
        
        # Mantém apenas últimas 100 medições
        if len(self.stats['detection_times']) > 100:
            self.stats['detection_times'].pop(0)
            
        # Contadores
        self.stats['total_detections'] += len(detections)
        
        for detection in detections:
            behavior_type = detection.behavior_type
            if behavior_type not in self.stats['detections_by_type']:
                self.stats['detections_by_type'][behavior_type] = 0
            self.stats['detections_by_type'][behavior_type] += 1
            
    def get_statistics(self) -> Dict[str, Any]:
        """Retorna estatísticas do manager"""
        avg_time = 0.0
        if self.stats['detection_times']:
            avg_time = sum(self.stats['detection_times']) / len(self.stats['detection_times'])
            
        avg_first_result = 0.0
        if self.stats['first_result_time']:
            avg_first_result = sum(self.stats['first_result_time']) / len(self.stats['first_result_time'])
            
        return {
            'total_detections': self.stats['total_detections'],
            'detections_by_type': self.stats['detections_by_type'],
            'average_process_time': avg_time,
            'average_first_result_time': avg_first_result,
            'active_detectors': len(self.detectors),
            'detector_list': list(self.detectors.keys()),
            'optimization_stats': {
                'process_pool_usage': self.stats['process_pool_usage'],
                'async_usage': self.stats['async_usage'],
                'cpu_bound_detectors': self.cpu_bound_detectors,
                'max_workers': self.max_workers
            }
        }
        
    def reset_statistics(self):
        """Reseta estatísticas"""
        self.stats = {
            'total_detections': 0,
            'detections_by_type': {},
            'detection_times': [],
            'process_pool_usage': 0,
            'async_usage': 0,
            'first_result_time': []
        }
        
    def get_detector_config(self, behavior_type: str) -> Optional[Dict[str, Any]]:
        """Retorna configuração de um detector específico"""
        if behavior_type in self.detectors:
            return self.detectors[behavior_type].config
        return None
        
    def update_detector_config(self, behavior_type: str, new_config: Dict[str, Any]):
        """Atualiza configuração de um detector"""
        if behavior_type in self.detectors:
            self.detectors[behavior_type].config.update(new_config)
            self.logger.info(f"Configuração do detector {behavior_type} atualizada")
            
    def reload_config(self, new_config: Dict[str, Any]):
        """Recarrega configurações do manager"""
        self.config.update(new_config)
        
        # Atualiza lista de detectores CPU-bound
        if 'cpu_bound_detectors' in new_config:
            self.cpu_bound_detectors = new_config['cpu_bound_detectors']
            
        # Atualiza max workers se especificado
        if 'max_process_workers' in new_config:
            new_max_workers = new_config['max_process_workers']
            if new_max_workers != self.max_workers:
                # Recria o executor com novo número de workers
                self.process_executor.shutdown(wait=False)
                self.max_workers = new_max_workers
                self.process_executor = ProcessPoolExecutor(max_workers=self.max_workers)
                
        self.logger.info("Configurações do BehaviorManager recarregadas")
        
    def shutdown(self):
        """Desliga o manager gracefully"""
        self.logger.info("Desligando BehaviorManager...")
        
        # Desliga o process pool
        self.process_executor.shutdown(wait=True)
        
        self.logger.info("BehaviorManager desligado")