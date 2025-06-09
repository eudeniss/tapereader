"""
Sistema de Cache de Memória de Preços - Versão Otimizada
Implementa cache de 3 camadas com batch operations e tuplas no L1
"""

import logging
import time
from typing import Dict, List, Optional, Tuple, Any, NamedTuple
from datetime import datetime, timedelta
from collections import defaultdict, deque
import threading
from contextlib import contextmanager

from ..core.models import Side, Trade, MarketData, VolumeProfile
from .database import DatabaseManager


# OTIMIZAÇÃO: Estrutura leve para L1 cache usando NamedTuple
class L1TradeData(NamedTuple):
    """Estrutura otimizada para trades no L1 cache"""
    timestamp: float  # Unix timestamp
    price: float
    volume: int
    aggressor: str  # 'BUY' ou 'SELL'
    added_at: float  # Quando foi adicionado ao cache


class L1MarketData(NamedTuple):
    """Estrutura otimizada para market data no L1 cache"""
    timestamp: float
    trades_count: int
    total_volume: int
    high: float
    low: float
    vwap: float
    buy_volume: int
    sell_volume: int
    added_at: float


class PriceMemoryCache:
    """
    Cache de memória em 3 camadas otimizado com batch operations
    L1: Hot cache (últimos segundos) - Alta frequência - TUPLAS LEVES
    L2: Warm cache (últimos minutos) - Média frequência  
    L3: Cold storage (SQLite) - Persistência
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Configurações de cache
        self.l1_ttl = config.get('l1_ttl', 10)  # segundos
        self.l2_ttl = config.get('l2_ttl', 300)  # 5 minutos
        self.max_l1_size = config.get('max_l1_size', 1000)
        self.max_l2_size = config.get('max_l2_size', 10000)
        
        # OTIMIZAÇÃO: Configurações de batch
        self.batch_size = config.get('batch_size', 100)
        self.batch_timeout = config.get('batch_timeout', 1.0)
        
        # OTIMIZAÇÃO: L1 usa tuplas leves, L2 mantém objetos completos
        self.l1_cache = defaultdict(lambda: deque(maxlen=self.max_l1_size))
        self.l1_trades = defaultdict(lambda: deque(maxlen=self.max_l1_size * 10))  # Trades individuais
        self.l2_cache = defaultdict(lambda: deque(maxlen=self.max_l2_size))
        
        # OTIMIZAÇÃO: Buffers para batch operations
        self.l1_to_l2_buffer = defaultdict(list)
        self.l2_to_l3_buffer = defaultdict(list)
        
        # Timestamps de última limpeza
        self.last_l1_cleanup = time.time()
        self.last_l2_cleanup = time.time()
        self.last_batch_flush = time.time()
        
        # Volume profile cache
        self.volume_profiles = defaultdict(lambda: VolumeProfileCache())
        
        # Database manager com batch otimizado
        db_path = config.get('db_path', '../data/price_history.db')
        self.db = DatabaseManager(
            db_path=db_path,
            batch_size=self.batch_size,
            batch_timeout=self.batch_timeout
        )
        
        # Locks para thread safety
        self.l1_lock = threading.RLock()
        self.l2_lock = threading.RLock()
        self.batch_lock = threading.Lock()
        
        # Thread de manutenção
        self.maintenance_thread = threading.Thread(
            target=self._maintenance_worker,
            daemon=True
        )
        self.maintenance_thread.start()
        
        # Estatísticas
        self.stats = {
            'l1_hits': 0,
            'l1_misses': 0,
            'l2_hits': 0,
            'l2_misses': 0,
            'l3_queries': 0,
            'batch_flushes': 0,
            'items_batched': 0,
            'memory_saved_mb': 0
        }
        
        self.logger.info(
            f"PriceMemoryCache inicializado (Otimizado com Tuplas) - "
            f"L1 TTL: {self.l1_ttl}s, L2 TTL: {self.l2_ttl}s, "
            f"Batch size: {self.batch_size}"
        )
        
    def add_market_data(self, asset: str, data: MarketData):
        """
        Adiciona dados de mercado ao cache L1
        OTIMIZADO: Usa tuplas leves no L1
        """
        current_time = time.time()
        
        with self.l1_lock:
            # OTIMIZAÇÃO: Cria tupla leve para L1
            if data.trades:
                # Calcula agregados
                prices = [float(t.price) for t in data.trades]
                volumes = [t.volume for t in data.trades]
                
                buy_volume = sum(
                    t.volume for t in data.trades 
                    if t.aggressor.value == 'BUY'
                )
                sell_volume = sum(
                    t.volume for t in data.trades 
                    if t.aggressor.value == 'SELL'
                )
                
                vwap = sum(p * v for p, v in zip(prices, volumes)) / sum(volumes) if volumes else prices[0]
                
                # Adiciona agregado ao L1
                l1_data = L1MarketData(
                    timestamp=data.timestamp.timestamp(),
                    trades_count=len(data.trades),
                    total_volume=sum(volumes),
                    high=max(prices),
                    low=min(prices),
                    vwap=vwap,
                    buy_volume=buy_volume,
                    sell_volume=sell_volume,
                    added_at=current_time
                )
                self.l1_cache[asset].append(l1_data)
                
                # OTIMIZAÇÃO: Armazena apenas trades mais importantes no L1
                # (maiores volumes ou preços extremos)
                for trade in data.trades:
                    if trade.volume > 10 or float(trade.price) in [max(prices), min(prices)]:
                        l1_trade = L1TradeData(
                            timestamp=trade.timestamp.timestamp(),
                            price=float(trade.price),
                            volume=trade.volume,
                            aggressor=trade.aggressor.value,
                            added_at=current_time
                        )
                        self.l1_trades[asset].append(l1_trade)
            
            # Atualiza volume profile
            self.volume_profiles[asset].update(data)
            
            # OTIMIZAÇÃO: Prepara para batch em vez de mover imediatamente
            if len(self.l1_cache[asset]) >= self.max_l1_size * 0.8:
                self._prepare_l1_to_l2_batch(asset)
                
    def get_recent_data(self, asset: str, seconds: int = 60) -> List[MarketData]:
        """
        Busca dados recentes (com cache hierárquico)
        OTIMIZADO: Reconstrói MarketData das tuplas quando necessário
        """
        cutoff = time.time() - seconds
        result = []
        
        # Busca em L1 (hot) - precisa reconstruir das tuplas
        with self.l1_lock:
            # Coleta dados agregados do L1
            l1_aggregates = [
                data for data in self.l1_cache[asset]
                if data.timestamp >= cutoff
            ]
            
            if l1_aggregates:
                self.stats['l1_hits'] += len(l1_aggregates)
                
                # Reconstrói MarketData simplificado das tuplas
                for agg in l1_aggregates:
                    # Busca trades correspondentes
                    trades_in_window = [
                        t for t in self.l1_trades[asset]
                        if abs(t.timestamp - agg.timestamp) < 1.0  # Trades no mesmo segundo
                    ]
                    
                    # Cria MarketData reconstruído
                    reconstructed = self._reconstruct_market_data(agg, trades_in_window)
                    if reconstructed:
                        result.append(reconstructed)
                    
        # Se não encontrou suficiente, busca em L2
        if len(result) < 10:  # Threshold arbitrário
            with self.l2_lock:
                for item in self.l2_cache[asset]:
                    if item['data'].timestamp >= datetime.fromtimestamp(cutoff):
                        result.append(item['data'])
                        self.stats['l2_hits'] += 1
                        
        # Se ainda não encontrou suficiente, busca em L3
        if len(result) < 10:
            self.stats['l3_queries'] += 1
            # Busca otimizada no banco
            db_data = self._fetch_from_database(asset, datetime.fromtimestamp(cutoff))
            result.extend(db_data)
            
        # Ordena por timestamp
        result.sort(key=lambda x: x.timestamp)
        
        return result
        
    def get_candles(self, asset: str, timeframe: str, count: int) -> List[Dict]:
        """
        Busca candles agregados
        OTIMIZADO: Agregação eficiente das tuplas em memória
        """
        # Tenta agregar dos caches primeiro
        candles = self._aggregate_candles_from_cache(asset, timeframe, count)
        
        if len(candles) < count:
            # Complementa com dados do banco
            db_candles = self._fetch_candles_from_database(
                asset, timeframe, count - len(candles)
            )
            candles = db_candles + candles
            
        return candles[:count]
        
    def get_volume_profile(self, asset: str, 
                          start: datetime, 
                          end: datetime) -> VolumeProfile:
        """
        Busca volume profile
        OTIMIZADO: Cache em memória + banco
        """
        # Combina profile em memória com dados históricos
        memory_profile = self.volume_profiles[asset].get_profile(start, end)
        
        # Busca dados históricos se necessário
        if start < datetime.now() - timedelta(minutes=self.l2_ttl/60):
            db_profile = self.db.get_volume_profile(
                asset, start.date(), end.date()
            )
            # Merge profiles
            memory_profile = self._merge_volume_profiles(memory_profile, db_profile)
            
        return memory_profile
        
    def _reconstruct_market_data(self, 
                                aggregate: L1MarketData, 
                                trades: List[L1TradeData]) -> Optional[MarketData]:
        """
        OTIMIZAÇÃO: Reconstrói MarketData a partir das tuplas leves
        """
        if not aggregate:
            return None
            
        # Reconstrói trades
        reconstructed_trades = []
        for t in trades:
            trade = Trade(
                timestamp=datetime.fromtimestamp(t.timestamp),
                price=str(t.price),
                volume=t.volume,
                aggressor=Side.BUY if t.aggressor == 'BUY' else Side.SELL
            )
            reconstructed_trades.append(trade)
            
        # Se não tem trades individuais, cria um representativo
        if not reconstructed_trades and aggregate.trades_count > 0:
            # Cria trade sintético com os agregados
            synthetic_trade = Trade(
                timestamp=datetime.fromtimestamp(aggregate.timestamp),
                price=str(aggregate.vwap),
                volume=aggregate.total_volume,
                aggressor=Side.BUY if aggregate.buy_volume > aggregate.sell_volume else Side.SELL
            )
            reconstructed_trades = [synthetic_trade]
            
        return MarketData(
            asset='UNKNOWN',  # Será preenchido pelo contexto
            timestamp=datetime.fromtimestamp(aggregate.timestamp),
            trades=reconstructed_trades,
            book=None,
            stats=None
        )
        
    def _prepare_l1_to_l2_batch(self, asset: str):
        """
        OTIMIZAÇÃO: Prepara dados para mover de L1 para L2 em batch
        Converte tuplas de volta para objetos completos
        """
        current_time = time.time()
        items_to_move = []
        
        # Identifica agregados expirados
        while self.l1_cache[asset]:
            item = self.l1_cache[asset][0]
            if current_time - item.added_at > self.l1_ttl:
                aggregate = self.l1_cache[asset].popleft()
                
                # Busca trades correspondentes
                trades_to_move = []
                remaining_trades = deque()
                
                for trade in self.l1_trades[asset]:
                    if current_time - trade.added_at > self.l1_ttl:
                        trades_to_move.append(trade)
                    else:
                        remaining_trades.append(trade)
                        
                self.l1_trades[asset] = remaining_trades
                
                # Reconstrói MarketData completo para L2
                market_data = self._reconstruct_market_data(aggregate, trades_to_move)
                if market_data:
                    items_to_move.append({
                        'timestamp': datetime.fromtimestamp(aggregate.timestamp),
                        'data': market_data,
                        'added_at': aggregate.added_at
                    })
            else:
                break
                
        if items_to_move:
            with self.batch_lock:
                self.l1_to_l2_buffer[asset].extend(items_to_move)
                
                # Trigger flush se buffer grande
                if len(self.l1_to_l2_buffer[asset]) >= self.batch_size:
                    self._flush_l1_to_l2_batch(asset)
                    
            # Estima memória economizada
            self._update_memory_stats()
                    
    def _flush_l1_to_l2_batch(self, asset: str):
        """
        OTIMIZAÇÃO: Move dados de L1 para L2 em batch
        """
        if asset not in self.l1_to_l2_buffer:
            return
            
        items = self.l1_to_l2_buffer[asset]
        if not items:
            return
            
        with self.l2_lock:
            # Move todos de uma vez
            self.l2_cache[asset].extend(items)
            
            # Prepara para L3 se L2 muito grande
            if len(self.l2_cache[asset]) >= self.max_l2_size * 0.8:
                self._prepare_l2_to_l3_batch(asset)
                
        # Limpa buffer
        self.l1_to_l2_buffer[asset].clear()
        self.stats['items_batched'] += len(items)
        
    def _prepare_l2_to_l3_batch(self, asset: str):
        """
        OTIMIZAÇÃO: Prepara dados para persistir de L2 para L3 em batch
        """
        current_time = time.time()
        items_to_persist = []
        
        # Identifica itens expirados
        while self.l2_cache[asset]:
            item = self.l2_cache[asset][0]
            if current_time - item['added_at'] > self.l2_ttl:
                items_to_persist.append(self.l2_cache[asset].popleft())
            else:
                break
                
        if items_to_persist:
            with self.batch_lock:
                self.l2_to_l3_buffer[asset].extend(items_to_persist)
                
                # Trigger flush se buffer grande
                if len(self.l2_to_l3_buffer[asset]) >= self.batch_size:
                    self._flush_l2_to_l3_batch(asset)
                    
    def _flush_l2_to_l3_batch(self, asset: str):
        """
        OTIMIZAÇÃO: Persiste dados de L2 para L3 (SQLite) em batch
        """
        if asset not in self.l2_to_l3_buffer:
            return
            
        items = self.l2_to_l3_buffer[asset]
        if not items:
            return
            
        # Prepara dados para batch insert
        candles_to_insert = []
        trades_to_insert = []
        
        for item in items:
            market_data = item['data']
            
            # Prepara candle
            if market_data.trades:
                candle = self._aggregate_to_candle(market_data)
                candles_to_insert.append((
                    asset,
                    market_data.timestamp,
                    candle
                ))
                
                # Prepara trades individuais
                for trade in market_data.trades[:10]:  # Limita trades por candle
                    trades_to_insert.append({
                        'asset': asset,
                        'timestamp': trade.timestamp,
                        'price': float(trade.price),
                        'volume': trade.volume,
                        'side': trade.side.value,
                        'aggressor': trade.aggressor.value,
                        'order_id': getattr(trade, 'order_id', None)
                    })
                    
        # OTIMIZAÇÃO: Insere tudo em batch usando transação única
        if candles_to_insert or trades_to_insert:
            with self.db.transaction():
                # Insere candles
                for asset, timestamp, candle in candles_to_insert:
                    self.db.insert_candle(asset, timestamp, candle)
                    
                # Insere trades
                if trades_to_insert:
                    self.db.insert_trades_batch(trades_to_insert)
                    
        # Limpa buffer
        self.l2_to_l3_buffer[asset].clear()
        self.stats['batch_flushes'] += 1
        self.stats['items_batched'] += len(items)
        
        self.logger.debug(
            f"Batch flush L2->L3: {len(candles_to_insert)} candles, "
            f"{len(trades_to_insert)} trades"
        )
        
    def _maintenance_worker(self):
        """
        Thread de manutenção que executa limpezas e flushes periódicos
        OTIMIZADO: Batch operations agendadas
        """
        while True:
            try:
                time.sleep(self.batch_timeout)
                current_time = time.time()
                
                # Flush de batches pendentes
                if current_time - self.last_batch_flush >= self.batch_timeout:
                    self._flush_all_batches()
                    self.last_batch_flush = current_time
                    
                # Limpeza L1 (a cada 10 segundos)
                if current_time - self.last_l1_cleanup >= 10:
                    self._cleanup_l1_cache()
                    self.last_l1_cleanup = current_time
                    
                # Limpeza L2 (a cada minuto)
                if current_time - self.last_l2_cleanup >= 60:
                    self._cleanup_l2_cache()
                    self.last_l2_cleanup = current_time
                    
            except Exception as e:
                self.logger.error(f"Erro no maintenance worker: {e}")
                
    def _flush_all_batches(self):
        """
        OTIMIZAÇÃO: Flush de todos os batches pendentes
        """
        with self.batch_lock:
            # Flush L1 -> L2
            for asset in list(self.l1_to_l2_buffer.keys()):
                if self.l1_to_l2_buffer[asset]:
                    self._flush_l1_to_l2_batch(asset)
                    
            # Flush L2 -> L3
            for asset in list(self.l2_to_l3_buffer.keys()):
                if self.l2_to_l3_buffer[asset]:
                    self._flush_l2_to_l3_batch(asset)
                    
        # Força commit no banco
        self.db.commit()
        
    def _cleanup_l1_cache(self):
        """Limpa itens expirados do L1 e move para L2"""
        with self.l1_lock:
            for asset in list(self.l1_cache.keys()):
                self._prepare_l1_to_l2_batch(asset)
                
    def _cleanup_l2_cache(self):
        """Limpa itens expirados do L2 e persiste em L3"""
        with self.l2_lock:
            for asset in list(self.l2_cache.keys()):
                self._prepare_l2_to_l3_batch(asset)
                
    def _aggregate_to_candle(self, market_data: MarketData) -> Dict:
        """Agrega MarketData em candle"""
        if not market_data.trades:
            return None
            
        prices = [float(t.price) for t in market_data.trades]
        volumes = [t.volume for t in market_data.trades]
        
        buy_volume = sum(
            t.volume for t in market_data.trades 
            if t.aggressor.value == 'BUY'
        )
        sell_volume = sum(
            t.volume for t in market_data.trades 
            if t.aggressor.value == 'SELL'
        )
        
        # VWAP
        if sum(volumes) > 0:
            vwap = sum(p * v for p, v in zip(prices, volumes)) / sum(volumes)
        else:
            vwap = sum(prices) / len(prices)
            
        return {
            'open': prices[0],
            'high': max(prices),
            'low': min(prices),
            'close': prices[-1],
            'volume': sum(volumes),
            'buy_volume': buy_volume,
            'sell_volume': sell_volume,
            'trades': len(market_data.trades),
            'vwap': vwap,
            'spread': getattr(market_data, 'spread', None)
        }
        
    def _aggregate_candles_from_cache(self, asset: str, 
                                     timeframe: str, 
                                     count: int) -> List[Dict]:
        """
        Agrega candles dos caches em memória
        OTIMIZADO: Usa dados já em memória quando possível
        """
        # Implementação simplificada
        # Em produção, implementaria agregação completa por timeframe
        candles = []
        
        # Coleta dados de L1 e L2
        all_data = []
        
        # L1: Reconstrói dos agregados
        with self.l1_lock:
            for agg in self.l1_cache[asset]:
                # Cria candle sintético do agregado
                candle = {
                    'timestamp': datetime.fromtimestamp(agg.timestamp),
                    'open': agg.vwap,
                    'high': agg.high,
                    'low': agg.low,
                    'close': agg.vwap,
                    'volume': agg.total_volume,
                    'buy_volume': agg.buy_volume,
                    'sell_volume': agg.sell_volume,
                    'trades': agg.trades_count
                }
                all_data.append(candle)
            
        # L2: Usa dados completos
        with self.l2_lock:
            for item in self.l2_cache[asset]:
                candle = self._aggregate_to_candle(item['data'])
                if candle:
                    candle['timestamp'] = item['data'].timestamp
                    all_data.append(candle)
            
        # Agrega por período (simplificado para 1min)
        if all_data and timeframe == '1m':
            current_candle = None
            current_minute = None
            
            for data in sorted(all_data, key=lambda x: x['timestamp']):
                minute = data['timestamp'].replace(second=0, microsecond=0)
                
                if minute != current_minute:
                    if current_candle:
                        candles.append(current_candle)
                    current_candle = data.copy()
                    current_minute = minute
                else:
                    # Merge com candle existente
                    current_candle['high'] = max(current_candle['high'], data['high'])
                    current_candle['low'] = min(current_candle['low'], data['low'])
                    current_candle['close'] = data['close']
                    current_candle['volume'] += data['volume']
                    current_candle['buy_volume'] += data.get('buy_volume', 0)
                    current_candle['sell_volume'] += data.get('sell_volume', 0)
                    current_candle['trades'] += data.get('trades', 1)
                    
            if current_candle:
                candles.append(current_candle)
                
        return candles[-count:] if candles else []
        
    def _fetch_from_database(self, asset: str, 
                           start: datetime) -> List[MarketData]:
        """Busca dados do banco (L3)"""
        # Implementação simplificada
        # Em produção, converteria registros do banco para MarketData
        return []
        
    def _fetch_candles_from_database(self, asset: str, 
                                   timeframe: str, 
                                   count: int) -> List[Dict]:
        """Busca candles do banco"""
        end = datetime.now()
        
        # Calcula início baseado no timeframe
        if timeframe == '1m':
            start = end - timedelta(minutes=count)
        elif timeframe == '5m':
            start = end - timedelta(minutes=count * 5)
        elif timeframe == '15m':
            start = end - timedelta(minutes=count * 15)
        else:
            start = end - timedelta(hours=count)
            
        return self.db.get_candles(asset, start, end)
        
    def _merge_volume_profiles(self, profile1: Any, profile2: Dict) -> Any:
        """Merge dois volume profiles"""
        # Implementação simplificada
        # Em produção, faria merge completo dos profiles
        return profile1
        
    def _update_memory_stats(self):
        """
        OTIMIZAÇÃO: Calcula economia de memória com tuplas
        """
        # Estima tamanho médio
        # Trade object: ~200 bytes
        # L1TradeData tuple: ~40 bytes
        # Economia: ~160 bytes por trade
        
        total_l1_items = sum(len(cache) for cache in self.l1_cache.values())
        total_l1_trades = sum(len(trades) for trades in self.l1_trades.values())
        
        # Economia estimada em MB
        saved_bytes = (total_l1_items * 100) + (total_l1_trades * 160)
        self.stats['memory_saved_mb'] = saved_bytes / (1024 * 1024)
        
    def get_statistics(self) -> Dict:
        """
        Retorna estatísticas do cache
        OTIMIZADO: Inclui métricas de economia de memória
        """
        total_queries = (
            self.stats['l1_hits'] + self.stats['l1_misses'] +
            self.stats['l2_hits'] + self.stats['l2_misses']
        )
        
        l1_hit_rate = (
            self.stats['l1_hits'] / max(1, self.stats['l1_hits'] + self.stats['l1_misses'])
        )
        l2_hit_rate = (
            self.stats['l2_hits'] / max(1, self.stats['l2_hits'] + self.stats['l2_misses'])
        )
        
        # Tamanhos dos caches
        l1_items = sum(len(cache) for cache in self.l1_cache.values())
        l1_trades = sum(len(trades) for trades in self.l1_trades.values())
        l2_items = sum(len(cache) for cache in self.l2_cache.values())
        
        # Tamanhos dos buffers
        l1_buffer_items = sum(len(buf) for buf in self.l1_to_l2_buffer.values())
        l2_buffer_items = sum(len(buf) for buf in self.l2_to_l3_buffer.values())
        
        # Atualiza estatísticas de memória
        self._update_memory_stats()
        
        return {
            'cache_sizes': {
                'l1_items': l1_items,
                'l1_trades': l1_trades,
                'l2_items': l2_items,
                'l1_buffer': l1_buffer_items,
                'l2_buffer': l2_buffer_items
            },
            'hit_rates': {
                'l1_hit_rate': round(l1_hit_rate * 100, 1),
                'l2_hit_rate': round(l2_hit_rate * 100, 1)
            },
            'operations': {
                'total_queries': total_queries,
                'l3_queries': self.stats['l3_queries'],
                'batch_flushes': self.stats['batch_flushes'],
                'items_batched': self.stats['items_batched']
            },
            'efficiency': {
                'avg_items_per_batch': (
                    self.stats['items_batched'] / max(1, self.stats['batch_flushes'])
                ),
                'memory_saved_mb': round(self.stats['memory_saved_mb'], 2)
            },
            'database_stats': self.db.get_statistics()
        }
        
    def shutdown(self):
        """Desliga o cache gracefully"""
        self.logger.info("Desligando PriceMemoryCache...")
        
        # Flush final de todos os batches
        self._flush_all_batches()
        
        # Fecha banco
        self.db.close()
        
        self.logger.info("PriceMemoryCache desligado")


class VolumeProfileCache:
    """Cache otimizado para volume profile"""
    
    def __init__(self):
        self.levels = defaultdict(lambda: {
            'volume': 0,
            'buy_volume': 0,
            'sell_volume': 0,
            'trades': 0,
            'time_at_price': 0
        })
        self.last_update = time.time()
        
    def update(self, market_data: MarketData):
        """Atualiza profile com novos dados"""
        for trade in market_data.trades:
            price_level = round(float(trade.price), 2)  # Arredonda para nível
            
            self.levels[price_level]['volume'] += trade.volume
            self.levels[price_level]['trades'] += 1
            
            if trade.aggressor.value == 'BUY':
                self.levels[price_level]['buy_volume'] += trade.volume
            else:
                self.levels[price_level]['sell_volume'] += trade.volume
                
        self.last_update = time.time()
        
    def get_profile(self, start: datetime, end: datetime) -> VolumeProfile:
        """Retorna profile para período"""
        # Implementação simplificada
        # Em produção, filtraria por período
        return VolumeProfile(
            levels=dict(self.levels),
            poc=self._calculate_poc(),
            vah=self._calculate_vah(),
            val=self._calculate_val()
        )
        
    def _calculate_poc(self) -> float:
        """Calcula Point of Control"""
        if not self.levels:
            return 0.0
            
        max_volume_level = max(
            self.levels.items(),
            key=lambda x: x[1]['volume']
        )
        return max_volume_level[0]
        
    def _calculate_vah(self) -> float:
        """Calcula Value Area High"""
        # Implementação simplificada
        prices = sorted(self.levels.keys())
        return prices[-1] if prices else 0.0
        
    def _calculate_val(self) -> float:
        """Calcula Value Area Low"""
        # Implementação simplificada
        prices = sorted(self.levels.keys())
        return prices[0] if prices else 0.0