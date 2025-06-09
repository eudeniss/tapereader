"""
Testes para os detectores de comportamento
Execute da pasta tapereader:
python tests/test_behaviors.py
"""

import sys
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
from decimal import Decimal

# Adiciona src ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.models import MarketData, Trade, OrderBook, BookLevel, Side
from src.behaviors import (
    AbsorptionDetector,
    ExhaustionDetector,
    InstitutionalFlowDetector,
    SweepDetector,
    StopHuntDetector,
    MomentumDetector,
    BehaviorManager
)


def create_test_trade(price: float, volume: int, aggressor: Side, seconds_ago: int = 0) -> Trade:
    """Cria trade de teste"""
    return Trade(
        timestamp=datetime.now() - timedelta(seconds=seconds_ago),
        price=Decimal(str(price)),
        volume=volume,
        aggressor=aggressor
    )


def create_test_book(bid_price: float, ask_price: float, volume: int = 100) -> OrderBook:
    """Cria book de teste"""
    return OrderBook(
        timestamp=datetime.now(),
        bids=[BookLevel(price=Decimal(str(bid_price)), volume=volume)],
        asks=[BookLevel(price=Decimal(str(ask_price)), volume=volume)]
    )


async def test_absorption():
    """Testa detector de absorção"""
    print("\n🔍 Testando Absorção...")
    
    config = {
        'enabled': True,
        'min_confidence': 0.7,
        'min_volume_ratio': 2.0,
        'max_price_change': 1.0,
        'min_trades': 5
    }
    
    detector = AbsorptionDetector(config)
    
    # Simula absorção: muita agressão vendedora mas preço não cai
    trades = []
    for i in range(20):
        # Vendedores agressivos
        trades.append(create_test_trade(5750.0, 50, Side.SELL, 20-i))
        # Mas preço se mantém
        trades.append(create_test_trade(5750.0, 50, Side.BUY, 19-i))
    
    # Book mostra grande comprador
    book = OrderBook(
        timestamp=datetime.now(),
        bids=[BookLevel(price=Decimal('5750.0'), volume=500)],  # Grande ordem
        asks=[BookLevel(price=Decimal('5750.5'), volume=50)]
    )
    
    market_data = MarketData(
        asset='DOLFUT',
        timestamp=datetime.now(),
        trades=trades,
        book=book
    )
    
    # Adiciona histórico
    for _ in range(3):
        detector.update_history(market_data)
    
    detection = await detector.detect(market_data)
    
    if detection.detected:
        print(f"✅ Absorção detectada! Confiança: {detection.confidence:.2%}")
        print(f"   Lado absorvedor: {detection.metadata.get('absorption_side')}")
    else:
        print("❌ Absorção não detectada")


async def test_institutional():
    """Testa detector de fluxo institucional"""
    print("\n🏛️ Testando Fluxo Institucional...")
    
    config = {
        'enabled': True,
        'min_confidence': 0.7,
        'dolfut_institutional_size': 50,
        'min_institutional_trades': 3
    }
    
    detector = InstitutionalFlowDetector(config)
    
    # Mix de trades normais e institucionais
    trades = []
    
    # Trades normais
    for i in range(5):
        trades.append(create_test_trade(5750.0 + i*0.5, 20, Side.BUY, 30-i*2))
    
    # Trades institucionais
    for i in range(5):
        trades.append(create_test_trade(5751.0, 75, Side.BUY, 25-i*2))  # Grande
    
    market_data = MarketData(
        asset='DOLFUT',
        timestamp=datetime.now(),
        trades=trades,
        book=create_test_book(5751.0, 5751.5)
    )
    
    detector.update_history(market_data)
    detection = await detector.detect(market_data)
    
    if detection.detected:
        print(f"✅ Institucional detectado! Confiança: {detection.confidence:.2%}")
        print(f"   Direção: {detection.metadata.get('flow_direction')}")
        print(f"   Volume: {detection.metadata.get('institutional_volume')}")
    else:
        print("❌ Institucional não detectado")


async def test_sweep():
    """Testa detector de sweep"""
    print("\n💨 Testando Sweep...")
    
    config = {
        'enabled': True,
        'min_confidence': 0.7,
        'min_price_levels': 3,
        'max_time_seconds': 10
    }
    
    detector = SweepDetector(config)
    
    # Movimento rápido através de vários níveis
    trades = []
    start_price = 5750.0
    
    for i in range(20):
        # Varredura rápida para cima
        price = start_price + i * 0.5
        trades.append(create_test_trade(price, 40, Side.BUY, 10-i*0.5))
    
    market_data = MarketData(
        asset='DOLFUT',
        timestamp=datetime.now(),
        trades=trades,
        book=create_test_book(5759.5, 5760.0, 20)  # Pouca liquidez
    )
    
    detector.update_history(market_data)
    detection = await detector.detect(market_data)
    
    if detection.detected:
        print(f"✅ Sweep detectado! Confiança: {detection.confidence:.2%}")
        print(f"   Níveis varridos: {detection.metadata.get('levels_swept')}")
        print(f"   Direção: {detection.metadata.get('sweep_direction')}")
    else:
        print("❌ Sweep não detectado")


async def test_momentum():
    """Testa detector de momentum"""
    print("\n📈 Testando Momentum...")
    
    config = {
        'enabled': True,
        'min_confidence': 0.7,
        'min_price_move': 2.0,
        'min_directional_ratio': 0.7,
        'lookback_periods': 5
    }
    
    detector = MomentumDetector(config)
    
    # Cria série de dados com momentum crescente
    for period in range(6):
        trades = []
        base_price = 5750.0 + period * 2.0  # Movimento consistente
        
        for i in range(10):
            price = base_price + i * 0.1
            volume = 30 + period * 5  # Volume crescente
            trades.append(create_test_trade(price, volume, Side.BUY, 0))
        
        market_data = MarketData(
            asset='DOLFUT',
            timestamp=datetime.now() - timedelta(minutes=5-period),
            trades=trades,
            book=create_test_book(base_price + 0.5, base_price + 1.0)
        )
        
        detector.update_history(market_data)
    
    # Detecção final
    detection = await detector.detect(market_data)
    
    if detection.detected:
        print(f"✅ Momentum detectado! Confiança: {detection.confidence:.2%}")
        print(f"   Direção: {detection.metadata.get('momentum_direction')}")
        print(f"   Fase: {detection.metadata.get('current_phase')}")
    else:
        print("❌ Momentum não detectado")


async def test_behavior_manager():
    """Testa o gerenciador de comportamentos"""
    print("\n🎯 Testando Behavior Manager...")
    
    config = {
        'behaviors': {
            'absorption': {'enabled': True, 'min_confidence': 0.7},
            'institutional': {'enabled': True, 'min_confidence': 0.7},
            'momentum': {'enabled': True, 'min_confidence': 0.7}
        }
    }
    
    manager = BehaviorManager(config)
    
    # Cria dados que devem detectar múltiplos comportamentos
    trades = []
    
    # Padrão institucional com absorção
    for i in range(10):
        trades.append(create_test_trade(5750.0, 75, Side.BUY, 10-i))  # Institucional
        trades.append(create_test_trade(5750.0, 30, Side.SELL, 9-i))   # Absorvido
    
    market_data = MarketData(
        asset='DOLFUT',
        timestamp=datetime.now(),
        trades=trades,
        book=OrderBook(
            timestamp=datetime.now(),
            bids=[BookLevel(price=Decimal('5750.0'), volume=1000)],  # Grande ordem
            asks=[BookLevel(price=Decimal('5750.5'), volume=50)]
        )
    )
    
    # Analisa
    detections = await manager.analyze(market_data)
    
    print(f"\n📊 {len(detections)} comportamentos detectados:")
    for detection in detections:
        print(f"   - {detection.behavior_type}: {detection.confidence:.2%}")
    
    # Verifica combinações
    combinations = manager.check_behavior_combinations(detections)
    if combinations:
        print(f"\n🔗 Combinações encontradas:")
        for combo in combinations:
            print(f"   - {combo['type']}: {combo['description']} ({combo['confidence']:.2%})")
    
    # Estatísticas
    stats = manager.get_statistics()
    print(f"\n📈 Estatísticas:")
    print(f"   Total detecções: {stats['total_detections']}")
    print(f"   Detectores ativos: {stats['active_detectors']}")


async def main():
    """Executa todos os testes"""
    print("=" * 60)
    print("TESTE DOS DETECTORES DE COMPORTAMENTO")
    print("=" * 60)
    
    try:
        await test_absorption()
        await test_institutional()
        await test_sweep()
        await test_momentum()
        await test_behavior_manager()
        
        print("\n" + "=" * 60)
        print("✅ TODOS OS TESTES CONCLUÍDOS!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Erro nos testes: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())