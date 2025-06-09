"""
Gerenciador de Banco de Dados - Versão Otimizada com Batch Operations
Responsável pela persistência e queries do SQLite com performance melhorada
"""

import sqlite3
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from pathlib import Path
import threading
from contextlib import contextmanager
import time


class DatabaseManager:
    """Gerencia toda a persistência em SQLite com batch operations otimizadas"""
    
    def __init__(self, db_path: str = '../data/price_history.db', 
                 batch_size: int = 100, 
                 batch_timeout: float = 1.0):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Configurações de batch
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        
        # Thread lock para operações concurrent
        self.lock = threading.Lock()
        
        # Conexão com row factory
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        
        # OTIMIZAÇÃO: Configurações SQLite para performance
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA cache_size=10000")
        self.conn.execute("PRAGMA temp_store=MEMORY")
        
        # Inicializa schema
        self._init_schema()
        
        # Cache de prepared statements
        self.statements = {}
        self._prepare_statements()
        
        # OTIMIZAÇÃO: Buffers para batch operations
        self.batch_buffers = {
            'candles': [],
            'trades': [],
            'volume_profile': [],
            'market_events': []
        }
        
        # Timestamp do último flush
        self.last_flush_time = time.time()
        
        # Thread de flush automático
        self.flush_thread = threading.Thread(
            target=self._auto_flush_worker,
            daemon=True
        )
        self.flush_thread.start()
        
        self.logger.info(
            f"DatabaseManager inicializado (Otimizado) - "
            f"Batch size: {batch_size}, Timeout: {batch_timeout}s"
        )
        
    @contextmanager
    def transaction(self):
        """Context manager para transações com batch"""
        with self.lock:
            self.conn.execute("BEGIN TRANSACTION")
            try:
                yield
                self.conn.execute("COMMIT")
            except Exception:
                self.conn.execute("ROLLBACK")
                raise
    
    def _auto_flush_worker(self):
        """Worker thread que executa flush automático dos batches"""
        while True:
            try:
                time.sleep(self.batch_timeout)
                current_time = time.time()
                
                # Flush se timeout excedido
                if current_time - self.last_flush_time >= self.batch_timeout:
                    self.flush_all_batches()
                    
            except Exception as e:
                self.logger.error(f"Erro no auto flush worker: {e}")
    
    def _init_schema(self):
        """Cria todas as tabelas necessárias com índices otimizados"""
        with self.lock:
            # Tabela de candles
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS candles_1m (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    asset TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume INTEGER NOT NULL,
                    buy_volume INTEGER NOT NULL,
                    sell_volume INTEGER NOT NULL,
                    trades INTEGER NOT NULL,
                    vwap REAL,
                    spread REAL,
                    UNIQUE(asset, timestamp)
                )
            ''')
            
            # Tabela de trades (para análise detalhada)
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    asset TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    price REAL NOT NULL,
                    volume INTEGER NOT NULL,
                    side TEXT NOT NULL,
                    aggressor TEXT NOT NULL,
                    order_id TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Tabela de volume profile
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS volume_profile (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    asset TEXT NOT NULL,
                    date DATE NOT NULL,
                    price REAL NOT NULL,
                    volume INTEGER NOT NULL,
                    buy_volume INTEGER NOT NULL,
                    sell_volume INTEGER NOT NULL,
                    trades INTEGER NOT NULL,
                    time_at_price INTEGER DEFAULT 0,
                    UNIQUE(asset, date, price)
                )
            ''')
            
            # Tabela de market profile diário
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS market_profile_daily (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    asset TEXT NOT NULL,
                    date DATE NOT NULL,
                    session TEXT DEFAULT 'RTH',
                    poc REAL NOT NULL,
                    vah REAL NOT NULL,
                    val REAL NOT NULL,
                    volume INTEGER NOT NULL,
                    range_high REAL NOT NULL,
                    range_low REAL NOT NULL,
                    opening_price REAL,
                    closing_price REAL,
                    profile_data TEXT NOT NULL,
                    ib_high REAL,
                    ib_low REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(asset, date, session)
                )
            ''')
            
            # Tabela de níveis importantes detectados
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS important_levels (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    asset TEXT NOT NULL,
                    price REAL NOT NULL,
                    level_type TEXT NOT NULL,
                    strength REAL NOT NULL,
                    first_detected DATETIME NOT NULL,
                    last_updated DATETIME NOT NULL,
                    touches INTEGER DEFAULT 0,
                    rejections INTEGER DEFAULT 0,
                    volume INTEGER DEFAULT 0,
                    active BOOLEAN DEFAULT 1
                )
            ''')
            
            # Tabela de eventos de mercado
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS market_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    asset TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    event_type TEXT NOT NULL,
                    price REAL NOT NULL,
                    volume INTEGER,
                    metadata TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # OTIMIZAÇÃO: Índices compostos para queries frequentes
            indices = [
                # Índices compostos principais
                'CREATE INDEX IF NOT EXISTS idx_candles_asset_time_composite ON candles_1m(asset, timestamp DESC)',
                'CREATE INDEX IF NOT EXISTS idx_trades_asset_time_price ON trades(asset, timestamp DESC, price)',
                'CREATE INDEX IF NOT EXISTS idx_volume_profile_composite ON volume_profile(asset, date, price)',
                
                # Índices simples
                'CREATE INDEX IF NOT EXISTS idx_profile_asset_date ON market_profile_daily(asset, date)',
                'CREATE INDEX IF NOT EXISTS idx_levels_asset_price ON important_levels(asset, price)',
                'CREATE INDEX IF NOT EXISTS idx_levels_active ON important_levels(active)',
                'CREATE INDEX IF NOT EXISTS idx_events_asset_time ON market_events(asset, timestamp)',
                'CREATE INDEX IF NOT EXISTS idx_events_type ON market_events(event_type)'
            ]
            
            for idx in indices:
                self.conn.execute(idx)
                
            self.conn.commit()
    
    def insert_candle(self, asset: str, timestamp: datetime, candle_data: Dict):
        """
        Insere candle no buffer para batch insert
        OTIMIZAÇÃO: Não insere diretamente, adiciona ao buffer
        """
        self.batch_buffers['candles'].append({
            'asset': asset,
            'timestamp': timestamp,
            'data': candle_data
        })
        
        # Flush automático se buffer cheio
        if len(self.batch_buffers['candles']) >= self.batch_size:
            self._flush_candles_batch()
    
    def insert_trades_batch(self, trades: List[Dict]):
        """
        Adiciona trades ao buffer
        OTIMIZAÇÃO: Batch insert
        """
        self.batch_buffers['trades'].extend(trades)
        
        # Flush automático se buffer cheio
        if len(self.batch_buffers['trades']) >= self.batch_size:
            self._flush_trades_batch()
    
    def _flush_candles_batch(self):
        """
        OTIMIZAÇÃO: Flush batch de candles com transação única
        """
        if not self.batch_buffers['candles']:
            return
            
        with self.transaction():
            # Prepara dados para executemany
            candle_values = []
            for item in self.batch_buffers['candles']:
                data = item['data']
                candle_values.append((
                    item['asset'], item['timestamp'],
                    data['open'], data['high'], data['low'], data['close'],
                    data['volume'], data.get('buy_volume', 0),
                    data.get('sell_volume', 0), data.get('trades', 0),
                    data.get('vwap'), data.get('spread')
                ))
            
            # Batch insert
            self.conn.executemany(self.statements['insert_candle'], candle_values)
            
            self.logger.debug(f"Batch insert: {len(candle_values)} candles")
            
        # Limpa buffer
        self.batch_buffers['candles'].clear()
        self.last_flush_time = time.time()
    
    def _flush_trades_batch(self):
        """
        OTIMIZAÇÃO: Flush batch de trades com transação única
        """
        if not self.batch_buffers['trades']:
            return
            
        with self.transaction():
            # Prepara dados para executemany
            trade_values = []
            for trade in self.batch_buffers['trades']:
                trade_values.append((
                    trade['asset'], trade['timestamp'], 
                    trade['price'], trade['volume'],
                    trade['side'], trade['aggressor'], 
                    trade.get('order_id')
                ))
            
            # Batch insert
            self.conn.executemany(self.statements['insert_trade'], trade_values)
            
            self.logger.debug(f"Batch insert: {len(trade_values)} trades")
            
        # Limpa buffer
        self.batch_buffers['trades'].clear()
        self.last_flush_time = time.time()
    
    def flush_all_batches(self):
        """
        OTIMIZAÇÃO: Flush de todos os buffers pendentes
        """
        with self.lock:
            # Flush todos os tipos de dados
            self._flush_candles_batch()
            self._flush_trades_batch()
            self._flush_events_batch()
            self._flush_volume_profile_batch()
            
            # Força commit
            self.conn.commit()
            
            self.logger.debug("Todos os batches foram persistidos")
    
    def log_market_event(self, asset: str, event_type: str, price: float,
                        volume: int = None, metadata: Dict = None):
        """
        Registra evento de mercado no buffer
        OTIMIZAÇÃO: Batch insert
        """
        self.batch_buffers['market_events'].append({
            'asset': asset,
            'timestamp': datetime.now(),
            'event_type': event_type,
            'price': price,
            'volume': volume,
            'metadata': json.dumps(metadata) if metadata else None
        })
        
        # Flush se buffer cheio
        if len(self.batch_buffers['market_events']) >= self.batch_size:
            self._flush_events_batch()
    
    def _flush_events_batch(self):
        """OTIMIZAÇÃO: Flush batch de eventos"""
        if not self.batch_buffers['market_events']:
            return
            
        with self.transaction():
            event_values = [
                (e['asset'], e['timestamp'], e['event_type'], 
                 e['price'], e['volume'], e['metadata'])
                for e in self.batch_buffers['market_events']
            ]
            
            self.conn.executemany('''
                INSERT INTO market_events
                (asset, timestamp, event_type, price, volume, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', event_values)
            
        self.batch_buffers['market_events'].clear()
        
    def _flush_volume_profile_batch(self):
        """OTIMIZAÇÃO: Flush batch de volume profile"""
        if not self.batch_buffers['volume_profile']:
            return
            
        with self.transaction():
            profile_values = [
                (p['asset'], p['date'], p['price'], p['volume'],
                 p['buy_volume'], p['sell_volume'], p['trades'], p['time_at_price'])
                for p in self.batch_buffers['volume_profile']
            ]
            
            self.conn.executemany(self.statements['update_volume_profile'], profile_values)
            
        self.batch_buffers['volume_profile'].clear()
    
    def _prepare_statements(self):
        """Prepara statements frequentes"""
        self.statements = {
            'insert_candle': '''
                INSERT OR REPLACE INTO candles_1m 
                (asset, timestamp, open, high, low, close, volume, 
                 buy_volume, sell_volume, trades, vwap, spread)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''',
            
            'insert_trade': '''
                INSERT INTO trades 
                (asset, timestamp, price, volume, side, aggressor, order_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''',
            
            'update_volume_profile': '''
                INSERT OR REPLACE INTO volume_profile
                (asset, date, price, volume, buy_volume, sell_volume, trades, time_at_price)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''',
            
            'get_candles_range': '''
                SELECT * FROM candles_1m
                WHERE asset = ? AND timestamp BETWEEN ? AND ?
                ORDER BY timestamp
            ''',
            
            'get_recent_trades': '''
                SELECT * FROM trades
                WHERE asset = ? AND timestamp > ?
                ORDER BY timestamp DESC
                LIMIT ?
            ''',
            
            'get_volume_profile': '''
                SELECT price, SUM(volume) as volume, 
                       SUM(buy_volume) as buy_volume,
                       SUM(sell_volume) as sell_volume,
                       SUM(trades) as trades,
                       SUM(time_at_price) as time_at_price
                FROM volume_profile
                WHERE asset = ? AND date BETWEEN ? AND ?
                GROUP BY price
                ORDER BY volume DESC
            ''',
            
            'get_important_levels': '''
                SELECT * FROM important_levels
                WHERE asset = ? AND active = 1
                AND price BETWEEN ? AND ?
                ORDER BY strength DESC
            ''',
            
            'get_market_events': '''
                SELECT * FROM market_events
                WHERE asset = ? AND timestamp BETWEEN ? AND ?
                AND event_type = ?
                ORDER BY timestamp
            '''
        }
    
    def get_candles(self, asset: str, start: datetime, end: datetime) -> List[Dict]:
        """Busca candles do período (flush automático antes da query)"""
        # IMPORTANTE: Flush antes de queries para garantir dados atualizados
        self.flush_all_batches()
        
        with self.lock:
            cursor = self.conn.execute(
                self.statements['get_candles_range'],
                (asset, start, end)
            )
            
            return [dict(row) for row in cursor]
    
    def get_volume_profile(self, asset: str, start_date: datetime.date,
                          end_date: datetime.date) -> Dict[float, Dict]:
        """Busca volume profile agregado"""
        # Flush antes da query
        self.flush_all_batches()
        
        with self.lock:
            cursor = self.conn.execute(
                self.statements['get_volume_profile'],
                (asset, start_date, end_date)
            )
            
            profile = {}
            for row in cursor:
                profile[row['price']] = {
                    'volume': row['volume'],
                    'buy_volume': row['buy_volume'],
                    'sell_volume': row['sell_volume'],
                    'trades': row['trades'],
                    'time_at_price': row['time_at_price']
                }
                
            return profile
            
    def get_market_profile(self, asset: str, date: datetime.date,
                          session: str = 'RTH') -> Optional[Dict]:
        """Busca market profile de um dia"""
        self.flush_all_batches()
        
        with self.lock:
            row = self.conn.execute('''
                SELECT * FROM market_profile_daily
                WHERE asset = ? AND date = ? AND session = ?
            ''', (asset, date, session)).fetchone()
            
            if row:
                result = dict(row)
                result['profile'] = json.loads(result['profile_data'])
                del result['profile_data']
                return result
                
            return None
            
    def get_important_levels(self, asset: str, price_min: float,
                           price_max: float) -> List[Dict]:
        """Busca níveis importantes na faixa"""
        self.flush_all_batches()
        
        with self.lock:
            cursor = self.conn.execute(
                self.statements['get_important_levels'],
                (asset, price_min, price_max)
            )
            
            return [dict(row) for row in cursor]
            
    def get_market_events(self, asset: str, event_type: str,
                         start: datetime, end: datetime) -> List[Dict]:
        """Busca eventos de mercado"""
        self.flush_all_batches()
        
        with self.lock:
            cursor = self.conn.execute(
                self.statements['get_market_events'],
                (asset, start, end, event_type)
            )
            
            events = []
            for row in cursor:
                event = dict(row)
                if event['metadata']:
                    event['metadata'] = json.loads(event['metadata'])
                events.append(event)
                
            return events
    
    def insert_market_profile(self, asset: str, date: datetime.date, profile_data: Dict):
        """Insere market profile diário (direto, sem batch)"""
        with self.lock:
            self.conn.execute('''
                INSERT OR REPLACE INTO market_profile_daily
                (asset, date, session, poc, vah, val, volume, 
                 range_high, range_low, opening_price, closing_price,
                 profile_data, ib_high, ib_low)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                asset, date, profile_data.get('session', 'RTH'),
                profile_data['poc'], profile_data['vah'], profile_data['val'],
                profile_data['volume'], profile_data['high'], profile_data['low'],
                profile_data.get('open'), profile_data.get('close'),
                json.dumps(profile_data['profile']),
                profile_data.get('ib_high'), profile_data.get('ib_low')
            ))
            self.conn.commit()
            
    def update_important_level(self, asset: str, price: float, level_type: str,
                             strength: float, volume: int = 0):
        """Atualiza ou cria nível importante"""
        with self.lock:
            # Verifica se já existe
            existing = self.conn.execute('''
                SELECT id FROM important_levels
                WHERE asset = ? AND ABS(price - ?) < 0.01 AND level_type = ?
            ''', (asset, price, level_type)).fetchone()
            
            if existing:
                self.conn.execute('''
                    UPDATE important_levels
                    SET strength = ?, volume = volume + ?, 
                        last_updated = CURRENT_TIMESTAMP, touches = touches + 1
                    WHERE id = ?
                ''', (strength, volume, existing['id']))
            else:
                self.conn.execute('''
                    INSERT INTO important_levels
                    (asset, price, level_type, strength, first_detected, 
                     last_updated, volume)
                    VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, ?)
                ''', (asset, price, level_type, strength, volume))
            
            self.conn.commit()
            
    def cleanup_old_data(self, retention_days: int = 90):
        """Remove dados antigos"""
        cutoff = datetime.now() - timedelta(days=retention_days)
        
        with self.lock:
            tables = [
                ('candles_1m', 'timestamp'),
                ('trades', 'timestamp'),
                ('volume_profile', 'date'),
                ('market_profile_daily', 'date'),
                ('market_events', 'timestamp')
            ]
            
            for table, column in tables:
                self.conn.execute(f'DELETE FROM {table} WHERE {column} < ?', (cutoff,))
                
            # Desativa níveis muito antigos
            self.conn.execute('''
                UPDATE important_levels
                SET active = 0
                WHERE last_updated < ?
            ''', (cutoff,))
            
            self.conn.commit()
            self.logger.info(f"Dados anteriores a {cutoff} removidos")
            
    def optimize(self):
        """Otimiza banco de dados"""
        # Flush antes de otimizar
        self.flush_all_batches()
        
        with self.lock:
            self.conn.execute('VACUUM')
            self.conn.execute('ANALYZE')
            
    def get_statistics(self) -> Dict:
        """Retorna estatísticas do banco incluindo buffers"""
        with self.lock:
            stats = {}
            
            tables = ['candles_1m', 'trades', 'volume_profile', 
                     'market_profile_daily', 'important_levels', 'market_events']
            
            for table in tables:
                count = self.conn.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]
                stats[f'{table}_count'] = count
            
            # Tamanho do arquivo
            stats['db_size_mb'] = self.db_path.stat().st_size / (1024 * 1024)
            
            # Estatísticas de buffer
            stats['buffer_sizes'] = {
                name: len(buffer) for name, buffer in self.batch_buffers.items()
            }
            
            stats['total_buffered'] = sum(stats['buffer_sizes'].values())
            
            return stats
    
    def commit(self):
        """Força commit de transações pendentes"""
        self.flush_all_batches()
        
        with self.lock:
            self.conn.commit()
    
    def close(self):
        """Fecha conexão com banco após flush final"""
        self.logger.info("Fechando DatabaseManager...")
        
        # Flush final
        self.flush_all_batches()
        
        with self.lock:
            self.conn.close()
            
        self.logger.info("Conexão com banco fechada")