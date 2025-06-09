"""
Behaviors Module - 14 Detectores de Comportamento
TapeReader Professional v2.0
"""

# Importa a classe base e o gerenciador
from .base import BehaviorDetector
from .manager import BehaviorManager  # <-- ADICIONADO: Importa o BehaviorManager

# Behaviors principais (10)
from .absorption import AbsorptionDetector
from .exhaustion import ExhaustionDetector
from .institutional import InstitutionalFlowDetector
from .support_resistance import SupportResistanceEnhancedDetector as SupportResistanceDetector
from .sweep import SweepDetector
from .stop_hunt import StopHuntDetector
from .iceberg import IcebergDetector
from .momentum import MomentumDetector
from .breakout import BreakoutDetector
from .divergence import DivergenceDetector

# Behaviors avançados (4)
from .htf import HTFDetector
from .micro_aggression import MicroAggressionDetector
from .recurrence import RecurrenceDetector
from .renovation import RenovationDetector

# Lista de todos os detectores
ALL_DETECTORS = [
    AbsorptionDetector,
    ExhaustionDetector,
    InstitutionalFlowDetector,
    SupportResistanceDetector,
    SweepDetector,
    StopHuntDetector,
    IcebergDetector,
    MomentumDetector,
    BreakoutDetector,
    DivergenceDetector,
    HTFDetector,
    MicroAggressionDetector,
    RecurrenceDetector,
    RenovationDetector
]

# Mapeamento por tipo
DETECTOR_MAP = {
    'absorption': AbsorptionDetector,
    'exhaustion': ExhaustionDetector,
    'institutional': InstitutionalFlowDetector,
    'support_resistance': SupportResistanceDetector,
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

# Define o que é exportado quando se faz "from src.behaviors import *"
__all__ = [
    'BehaviorDetector',
    'BehaviorManager',  # <-- ADICIONADO: Expõe o BehaviorManager
    'AbsorptionDetector',
    'ExhaustionDetector',
    'InstitutionalFlowDetector',
    'SupportResistanceDetector',
    'SweepDetector',
    'StopHuntDetector',
    'IcebergDetector',
    'MomentumDetector',
    'BreakoutDetector',
    'DivergenceDetector',
    'HTFDetector',
    'MicroAggressionDetector',
    'RecurrenceDetector',
    'RenovationDetector',
    'ALL_DETECTORS',
    'DETECTOR_MAP'
]