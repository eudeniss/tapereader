"""
Mock Dinâmico Completo - Simula cenários de padrões de mercado
Testa automaticamente comportamentos e combinações sem contexto temporal
"""

import random
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional, List, Dict, Tuple
from enum import Enum
from dataclasses import dataclass
import logging

from .base import DataProvider
from ..core.models import MarketData, Trade, OrderBook, BookLevel, Side


class ScenarioType(str, Enum):
    """Cenários de padrões de mercado testáveis"""
    # COMPORTAMENTOS INDIVIDUAIS - COMPRA
    ABSORPTION_BUY = "absorption_buy"
    EXHAUSTION_BUY = "exhaustion_buy"
    INSTITUTIONAL_BUY = "institutional_buy"
    SWEEP_BUY = "sweep_buy"
    STOP_HUNT_BUY = "stop_hunt_buy"
    ICEBERG_BUY = "iceberg_buy"
    MOMENTUM_BUY = "momentum_buy"
    SUPPORT_LEVEL = "support_level"
    
    # COMPORTAMENTOS INDIVIDUAIS - VENDA
    ABSORPTION_SELL = "absorption_sell"
    EXHAUSTION_SELL = "exhaustion_sell"
    INSTITUTIONAL_SELL = "institutional_sell"
    SWEEP_SELL = "sweep_sell"
    STOP_HUNT_SELL = "stop_hunt_sell"
    ICEBERG_SELL = "iceberg_sell"
    MOMENTUM_SELL = "momentum_sell"
    RESISTANCE_LEVEL = "resistance_level"
    
    # COMBINAÇÕES PODEROSAS
    ABSORPTION_EXHAUSTION_BUY = "absorption_exhaustion_buy"
    ABSORPTION_EXHAUSTION_SELL = "absorption_exhaustion_sell"
    INSTITUTIONAL_SUPPORT = "institutional_support"
    INSTITUTIONAL_RESISTANCE = "institutional_resistance"
    STOP_HUNT_FADE_BUY = "stop_hunt_fade_buy"
    STOP_HUNT_FADE_SELL = "stop_hunt_fade_sell"
    
    # CENÁRIOS ESPECIAIS
    PERFECT_CONFLUENCE = "perfect_confluence"
    ASSET_DIVERGENCE = "asset_divergence"
    EXTREME_VOLATILITY = "extreme_volatility"
    
    # CENÁRIOS DE CONFLITO
    CONFLICTING_SIGNALS = "conflicting_signals"
    NO_CLEAR_SIGNAL = "no_clear_signal"


@dataclass
class ScenarioState:
    """Estado atual do cenário sendo simulado"""
    type: ScenarioType
    progress: int = 0
    max_duration: int = 50
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class MockDynamicProvider(DataProvider):
    """
    Gerador dinâmico que testa padrões de mercado
    Simula comportamentos realistas sem depender de contexto temporal
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        # --- CORREÇÃO 1: Acessar o objeto de configuração Pydantic corretamente ---
        # O objeto 'config' é um modelo Pydantic, não um dicionário.
        # Acessamos o atributo 'mock' e o convertemos para um dicionário para usar .get()
        mock_config_model = config.mock if hasattr(config, 'mock') and config.mock else None
        mock_config = mock_config_model.dict() if mock_config_model and hasattr(mock_config_model, 'dict') else {}
        # --- FIM DA CORREÇÃO 1 ---

        initial_prices = mock_config.get('initial_prices', {
            'DOLFUT': '5750.00', 'WDOFUT': '5750.00'
        })
        market_params = mock_config.get('market_params', {})
        self.volume_ranges = mock_config.get('volume_ranges', {})
        self.book_params = mock_config.get('book_params', {})
        self.scenarios_config = mock_config.get('scenarios', {})

        # Estado do mercado
        self.current_prices = {
            'DOLFUT': Decimal(str(initial_prices.get('DOLFUT'))),
            'WDOFUT': Decimal(str(initial_prices.get('WDOFUT')))
        }
        self.base_prices = self.current_prices.copy()
        
        # Cenários a testar
        self.all_scenarios = list(ScenarioType)
        self.tested_scenarios = []
        self.current_scenario: Optional[ScenarioState] = None
        
        # Estatísticas de teste
        self.scenario_stats = {
            'total': len(self.all_scenarios),
            'tested': 0,
            'signals_generated': {}
        }
        
        # Parâmetros de mercado
        self.market_params = {
            'volatility': market_params.get('volatility', 0.001),
            'spread': Decimal(str(market_params.get('spread', '0.5'))),
            'volume_multiplier': market_params.get('volume_multiplier', 1.0)
        }
        
        # Estado para confluência entre ativos
        self.asset_correlation = 0.0
        
        # Task de geração de dados
        self.generator_task = None
        self.generating = False
        
    async def connect(self) -> bool:
        """Inicia o mock dinâmico"""
        self.connected = True
        self.logger.info("=" * 60)
        self.logger.info("MOCK DINÂMICO - TESTE DE PADRÕES DE MERCADO")
        self.logger.info("=" * 60)
        self.logger.info(f"Total de cenários a testar: {len(self.all_scenarios)}")
        
        self._start_next_scenario()
        
        return True
        
    async def disconnect(self) -> bool:
        """Finaliza mock e mostra estatísticas"""
        self.generating = False
        if self.generator_task:
            self.generator_task.cancel()
            try:
                await self.generator_task
            except asyncio.CancelledError:
                pass
                
        self.connected = False
        
        self.logger.info("\n" + "=" * 60)
        self.logger.info("RELATÓRIO FINAL DE TESTES")
        self.logger.info("=" * 60)
        self.logger.info(f"Cenários testados: {self.scenario_stats['tested']}/{self.scenario_stats['total']}")
        
        if self.scenario_stats['signals_generated']:
            self.logger.info("\nSinais gerados por cenário:")
            for scenario, count in self.scenario_stats['signals_generated'].items():
                self.logger.info(f"  {scenario}: {count} sinais")
                
        self.logger.info("=" * 60)
        
        return True
        
    async def start(self, queue: asyncio.Queue):
        """Inicia geração contínua de dados para a fila"""
        self.data_queue = queue
        self.generating = True
        
        # Cria task de geração
        self.generator_task = asyncio.create_task(self._generate_data_continuously())
        self.logger.info("Geração de dados mock iniciada em modo event-driven")
        
    async def _generate_data_continuously(self):
        """Gera dados continuamente e envia para a fila"""
        mode_config = self.config.get_mode_config()
        generation_interval = mode_config.update_interval_ms / 1000.0
        
        while self.generating and self.connected:
            try:
                if not self.current_scenario:
                    break
                    
                # Gera dados para ambos os ativos
                for asset in ['DOLFUT', 'WDOFUT']:
                    market_data = await self._generate_scenario_data(asset)
                    if market_data:
                        await self.push_market_data(asset, market_data)
                        
                # Atualiza progresso do cenário
                self.current_scenario.progress += 1
                
                # Verifica se cenário terminou
                if self.current_scenario.progress >= self.current_scenario.max_duration:
                    self._complete_current_scenario()
                    self._start_next_scenario()
                    
                await asyncio.sleep(generation_interval)
                
            except Exception as e:
                self.logger.error(f"Erro na geração de dados mock: {e}")
                await asyncio.sleep(1)
                
    def _start_next_scenario(self):
        """Inicia próximo cenário não testado"""
        untested = [s for s in self.all_scenarios if s not in self.tested_scenarios]
        
        if not untested:
            self.logger.info("\n✅ TODOS OS CENÁRIOS FORAM TESTADOS!")
            self.tested_scenarios = []
            untested = self.all_scenarios
            
        next_scenario = untested[0]
        
        # --- CORREÇÃO 2: Acessar a configuração de duração aninhada corretamente ---
        duration_settings = self.scenarios_config.get('duration_settings', {})
        max_duration = duration_settings.get('default_duration', 40)
        if "long" in next_scenario.value or "confluence" in next_scenario.value or "divergence" in next_scenario.value:
            max_duration = duration_settings.get('long_duration', 80)
        elif "medium" in next_scenario.value or "exhaustion" in next_scenario.value:
            max_duration = duration_settings.get('medium_duration', 60)
        # --- FIM DA CORREÇÃO 2 ---

        self.current_scenario = ScenarioState(type=next_scenario, max_duration=max_duration)
        
        self.logger.info(f"\n🎯 INICIANDO CENÁRIO: {next_scenario.value} (Duração: {max_duration} updates)")
        
        self.current_prices = self.base_prices.copy()
        
    def _complete_current_scenario(self):
        """Marca cenário como completo"""
        if self.current_scenario:
            self.tested_scenarios.append(self.current_scenario.type)
            self.scenario_stats['tested'] = len(self.tested_scenarios)
            
            if self.current_scenario.type.value not in self.scenario_stats['signals_generated']:
                self.scenario_stats['signals_generated'][self.current_scenario.type.value] = 0
                
            self.logger.info(f"✓ Cenário {self.current_scenario.type.value} concluído")
            
    async def _generate_scenario_data(self, asset: str) -> MarketData:
        """Gera dados específicos para cada cenário usando getattr para dinamismo."""
        scenario_name = self.current_scenario.type.value
        
        # Mapeamento de nomes de cenários para métodos
        generator_method_name = f"_generate_{scenario_name}"
        
        # Para cenários de compra/venda (ex: "absorption_buy")
        if scenario_name.endswith(('_buy', '_sell')):
            base_name, side_str = scenario_name.rsplit('_', 1)
            side = Side.BUY if side_str == 'buy' else Side.SELL
            generator_method_name = f"_generate_{base_name}"
            generator_method = getattr(self, generator_method_name, self._generate_normal_market)
            return await generator_method(asset, side)

        # Para cenários que afetam o nível de suporte/resistência
        if scenario_name.endswith('_level') or scenario_name.endswith('_support') or scenario_name.endswith('_resistance'):
            base_name, level_type = scenario_name.rsplit('_', 1)
            if base_name == "support_resistance": base_name = level_type # Casos como "support_level"
            generator_method_name = f"_generate_{base_name}"
            generator_method = getattr(self, generator_method_name, self._generate_normal_market)
            return await generator_method(asset, level_type)
        
        # Para cenários sem direção explícita
        generator_method = getattr(self, generator_method_name, self._generate_normal_market)
        return await generator_method(asset)

    # ==================== GERADORES DE COMPORTAMENTOS ====================
    
    async def _generate_absorption(self, asset: str, side: Side) -> MarketData:
        """Gera padrão de absorção"""
        price = self.current_prices[asset]
        progress = self.current_scenario.progress
        trades = []
        aggressor_side = Side.SELL if side == Side.BUY else Side.BUY
        
        if progress < self.current_scenario.max_duration * 0.6:
            for i in range(20):
                trades.append(Trade(
                    timestamp=datetime.now() - timedelta(microseconds=(20-i)*100),
                    price=price,
                    volume=self._get_volume_for_asset(asset, "normal") * 2,
                    aggressor=aggressor_side
                ))
            book = self._create_absorption_book(asset, price, side)
        else:
            for i in range(15):
                new_price = price + (Decimal('0.1') * i if side == Side.BUY else Decimal('-0.1') * i)
                trades.append(Trade(
                    timestamp=datetime.now() - timedelta(microseconds=(15-i)*100),
                    price=new_price,
                    volume=self._get_volume_for_asset(asset, "large"),
                    aggressor=side
                ))
            self.current_prices[asset] = trades[-1].price
            book = self._create_normal_book(asset, self.current_prices[asset])
            
        return MarketData(asset=asset, timestamp=datetime.now(), trades=trades, book=book)
        
    async def _generate_exhaustion(self, asset: str, side: Side) -> MarketData:
        """Gera padrão de exaustão"""
        price = self.current_prices[asset]
        progress_ratio = self.current_scenario.progress / self.current_scenario.max_duration
        trades = []
        
        volume_multiplier = 1.0 - progress_ratio * 0.8
        
        for i in range(10):
            price_change = Decimal('0.5') * Decimal(str(volume_multiplier))
            self.current_prices[asset] += (price_change if side == Side.BUY else -price_change)
            
            trades.append(Trade(
                timestamp=datetime.now() - timedelta(microseconds=(10-i)*100),
                price=self.current_prices[asset],
                volume=int(self._get_volume_for_asset(asset, "normal") * volume_multiplier),
                aggressor=side
            ))
            
        book = self._create_normal_book(asset, self.current_prices[asset])
        return MarketData(asset=asset, timestamp=datetime.now(), trades=trades, book=book)
        
    async def _generate_institutional(self, asset: str, side: Side) -> MarketData:
        """Gera padrão de fluxo institucional"""
        price = self.current_prices[asset]
        trades = []
        
        for i in range(5):
            trades.append(Trade(
                timestamp=datetime.now() - timedelta(microseconds=(10-i*2)*100),
                price=price + Decimal(str(random.uniform(-0.5, 0.5))),
                volume=self._get_volume_for_asset(asset, "small"),
                aggressor=random.choice([Side.BUY, Side.SELL])
            ))
            
            self.current_prices[asset] += (Decimal('0.5') if side == Side.BUY else Decimal('-0.5'))
            trades.append(Trade(
                timestamp=datetime.now() - timedelta(microseconds=(9-i*2)*100),
                price=self.current_prices[asset],
                volume=self._get_volume_for_asset(asset, "institutional"),
                aggressor=side
            ))
            
        book = self._create_normal_book(asset, self.current_prices[asset])
        return MarketData(asset=asset, timestamp=datetime.now(), trades=trades, book=book)
        
    async def _generate_sweep(self, asset: str, side: Side) -> MarketData:
        """Gera padrão de sweep (varredura de liquidez)"""
        price = self.current_prices[asset]
        trades = []
        sweep_range = 5
        
        for i in range(20):
            tick_move = (i / 19) * sweep_range
            new_price = price + (Decimal(str(tick_move)) if side == Side.BUY else Decimal(str(-tick_move)))
            
            trades.append(Trade(
                timestamp=datetime.now() - timedelta(milliseconds=10*(20-i)),
                price=new_price,
                volume=self._get_volume_for_asset(asset, "large"),
                aggressor=side
            ))
            
        self.current_prices[asset] = trades[-1].price
        book = self._create_swept_book(asset, self.current_prices[asset], side)
        return MarketData(asset=asset, timestamp=datetime.now(), trades=trades, book=book)
        
    async def _generate_stop_hunt(self, asset: str, side: Side) -> MarketData:
        """Gera padrão de stop hunt"""
        base_price = self.base_prices[asset]
        progress = self.current_scenario.progress
        trades = []
        
        # O lado do stop hunt é o lado das vítimas. Se o "stop_hunt_buy", os stops dos VENDIDOS (shorts) são caçados.
        # O movimento é para CIMA para pegar os stops dos vendidos, e a reversão esperada é para BAIXO.
        hunt_direction = Side.BUY if side == Side.SELL else Side.SELL
        reversal_direction = side

        if progress < self.current_scenario.max_duration * 0.2:
            spike_price = base_price + (Decimal('5') if hunt_direction == Side.BUY else Decimal('-5'))
            for i in range(5):
                trades.append(Trade(
                    timestamp=datetime.now() - timedelta(microseconds=(5-i)*100),
                    price=spike_price,
                    volume=self._get_volume_for_asset(asset, "small"),
                    aggressor=hunt_direction
                ))
            self.current_prices[asset] = spike_price
        else:
            for i in range(10):
                trades.append(Trade(
                    timestamp=datetime.now() - timedelta(microseconds=(10-i)*100),
                    price=base_price,
                    volume=self._get_volume_for_asset(asset, "normal"),
                    aggressor=reversal_direction
                ))
            self.current_prices[asset] = base_price
            
        book = self._create_normal_book(asset, self.current_prices[asset])
        return MarketData(asset=asset, timestamp=datetime.now(), trades=trades, book=book)
        
    async def _generate_iceberg(self, asset: str, side: Side) -> MarketData:
        """Gera padrão de iceberg"""
        price = self.current_prices[asset]
        trades = []
        iceberg_clip = self._get_volume_for_asset(asset, "iceberg_clip")
        
        for i in range(15):
            trades.append(Trade(
                timestamp=datetime.now() - timedelta(microseconds=(15-i)*100),
                price=price,
                volume=iceberg_clip,
                aggressor=side
            ))
            
        book = self._create_iceberg_book(asset, price, side)
        return MarketData(asset=asset, timestamp=datetime.now(), trades=trades, book=book)
        
    async def _generate_momentum(self, asset: str, side: Side) -> MarketData:
        """Gera padrão de momentum forte"""
        trades = []
        
        for i in range(15):
            self.current_prices[asset] += (Decimal('0.5') if side == Side.BUY else Decimal('-0.5'))
            volume = self._get_volume_for_asset(asset, "normal") * (1 + i * 0.1)
            
            trades.append(Trade(
                timestamp=datetime.now() - timedelta(microseconds=(15-i)*100),
                price=self.current_prices[asset],
                volume=int(volume),
                aggressor=side
            ))
            
        book = self._create_momentum_book(asset, self.current_prices[asset], side)
        return MarketData(asset=asset, timestamp=datetime.now(), trades=trades, book=book)
        
    async def _generate_support(self, asset: str, side: Side) -> MarketData:
        return await self._generate_support_resistance(asset, 'support')

    async def _generate_resistance(self, asset: str, side: Side) -> MarketData:
        return await self._generate_support_resistance(asset, 'resistance')

    async def _generate_support_resistance(self, asset: str, level_type: str) -> MarketData:
        """Gera padrão de suporte ou resistência"""
        trades = []
        level_price = self.base_prices[asset]
        
        for i in range(20):
            if level_type == "support":
                test_price = level_price + Decimal(str(random.uniform(0, 2)))
                aggressor = Side.SELL
            else: # resistance
                test_price = level_price - Decimal(str(random.uniform(0, 2)))
                aggressor = Side.BUY
                
            trades.append(Trade(
                timestamp=datetime.now() - timedelta(microseconds=(20-i)*100),
                price=test_price,
                volume=self._get_volume_for_asset(asset, "normal"),
                aggressor=aggressor
            ))
        self.current_prices[asset] = trades[-1].price
        book = self._create_support_resistance_book(asset, level_price, level_type)
        return MarketData(asset=asset, timestamp=datetime.now(), trades=trades, book=book)
        
    # ==================== GERADORES DE CENÁRIOS ====================

    async def _generate_normal_market(self, asset: str) -> MarketData:
        """Mercado normal sem padrões especiais"""
        price = self.current_prices[asset]
        trades = []
        for i in range(10):
            trades.append(Trade(
                timestamp=datetime.now() - timedelta(microseconds=(10 - i) * 100),
                price=price + Decimal(str(random.uniform(-0.5, 0.5))),
                volume=self._get_volume_for_asset(asset, "normal"),
                aggressor=random.choice([Side.BUY, Side.SELL])
            ))
        book = self._create_normal_book(asset, price)
        return MarketData(asset=asset, timestamp=datetime.now(), trades=trades, book=book)

    # ==================== HELPERS PARA CRIAÇÃO DE BOOKS ====================
    
    def _create_normal_book(self, asset: str, current_price: Decimal) -> OrderBook:
        """Cria book normal balanceado"""
        bids, asks = [], []
        for i in range(self.book_params.get('levels', 10)):
            spread = self.market_params['spread']
            bid_price = current_price - spread * (i + 1)
            ask_price = current_price + spread * (i + 1)
            base_volume = self._get_book_volume_for_asset(asset)
            
            bids.append(BookLevel(price=bid_price, volume=base_volume + random.randint(-20, 20), orders=random.randint(1, 5)))
            asks.append(BookLevel(price=ask_price, volume=base_volume + random.randint(-20, 20), orders=random.randint(1, 5)))
            
        return OrderBook(timestamp=datetime.now(), bids=bids, asks=asks)
        
    def _create_absorption_book(self, asset: str, price: Decimal, absorption_side: Side) -> OrderBook:
        """Book com grande player absorvendo"""
        book = self._create_normal_book(asset, price)
        large_volume = self._get_volume_for_asset(asset, "institutional") * 5
        
        if absorption_side == Side.BUY:
            book.bids[0].volume = large_volume
            book.bids[0].orders = 1
        else:
            book.asks[0].volume = large_volume
            book.asks[0].orders = 1
        return book
        
    def _create_swept_book(self, asset: str, price: Decimal, swept_side: Side) -> OrderBook:
        """Book após sweep - lado varrido vazio"""
        book = self._create_normal_book(asset, price)
        
        if swept_side == Side.BUY:
            for i in range(3):
                book.asks[i].volume = self._get_volume_for_asset(asset, "small") // 2
        else:
            for i in range(3):
                book.bids[i].volume = self._get_volume_for_asset(asset, "small") // 2
        return book
        
    def _create_iceberg_book(self, asset: str, price: Decimal, iceberg_side: Side) -> OrderBook:
        """Book com iceberg - volume pequeno visível mas renovando"""
        book = self._create_normal_book(asset, price)
        iceberg_visible = self._get_volume_for_asset(asset, "iceberg_clip")
        
        if iceberg_side == Side.BUY:
            book.bids[0].price = price
            book.bids[0].volume = iceberg_visible
            book.bids[0].orders = 1
        else:
            book.asks[0].price = price
            book.asks[0].volume = iceberg_visible
            book.asks[0].orders = 1
        return book
        
    def _create_momentum_book(self, asset: str, price: Decimal, momentum_side: Side) -> OrderBook:
        """Book desequilibrado mostrando momentum"""
        book = self._create_normal_book(asset, price)
        
        if momentum_side == Side.BUY:
            for i in range(5):
                book.bids[i].volume *= 3
                book.asks[i].volume //= 2
        else:
            for i in range(5):
                book.asks[i].volume *= 3
                book.bids[i].volume //= 2
        return book
        
    def _create_support_resistance_book(self, asset: str, level_price: Decimal, level_type: str) -> OrderBook:
        """Book com nível forte de suporte/resistência"""
        book = self._create_normal_book(asset, level_price)
        large_order = self._get_volume_for_asset(asset, "institutional") * 10
        
        if level_type == "support":
            book.bids.insert(0, BookLevel(price=level_price, volume=large_order, orders=1))
        else:
            book.asks.insert(0, BookLevel(price=level_price, volume=large_order, orders=1))
        return book
        
    def _create_institutional_book(self, asset: str, price: Decimal, level_type: str) -> OrderBook:
        """Book com presença institucional"""
        book = self._create_normal_book(asset, price)
        inst_volume = self._get_volume_for_asset(asset, "institutional")
        
        if level_type == "support":
            for i in range(3):
                book.bids[i].volume = inst_volume
                book.bids[i].orders = 1
        else:
            for i in range(3):
                book.asks[i].volume = inst_volume
                book.asks[i].orders = 1
        return book
        
    # ==================== HELPERS PARA VOLUMES ====================
    
    def _get_volume_for_asset(self, asset: str, size_type: str) -> int:
        """Retorna volume apropriado para o ativo e tipo a partir da configuração."""
        asset_volumes = self.volume_ranges.get(asset, {})
        
        if size_type == 'iceberg_clip':
            return asset_volumes.get('iceberg_clip', 50)
        
        vol_range = asset_volumes.get(size_type, [20, 50])
        return random.randint(vol_range[0], vol_range[1])
        
    def _get_book_volume_for_asset(self, asset: str) -> int:
        """Volume base para book do ativo a partir da configuração."""
        base_vol_config = self.book_params.get('base_volume', {})
        vol_range = base_vol_config.get(asset, [50, 200])
        return random.randint(vol_range[0], vol_range[1])