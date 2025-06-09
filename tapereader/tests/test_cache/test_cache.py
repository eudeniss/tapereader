"""
Testes unitários para o sistema de cache
"""

import unittest
import tempfile
import os
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

from tapereader.cache import PriceMemory, DatabaseManager, HistoricalAnalyzer
from tapereader.core.models import MarketData, Trade, OrderBook


class TestDatabaseManager(unittest.TestCase):
    """Testa o gerenciador de banco de dados"""
    
    def setUp(self):
        """Cria banco temporário para testes"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, 'test.db')
        self.db = DatabaseManager(self.db_path)
        
    def tearDown(self):
        """Limpa recursos"""
        self.db.close()
        # Remove arquivos temporários
        for file in Path(self.temp_dir).glob('*'):
            file.unlink()
        os.rmdir(self.temp_dir)
        
    def test_insert_and_retrieve_candle(self):
        """Testa inserção e recuperação de candles"""
        # Dados de teste
        asset = 'DOLFUT'
        timestamp = datetime.now().replace(second=0, microsecond=0)
        candle_data = {
            'open': 5750.0,
            'high': 5755.0,
            'low': 5748.0,
            'close': 5752.0,
            'volume': 100,
            'buy_volume': 60,
            'sell_volume': 40,
            'trades': 25
        }
        
        # Insere
        self.db.insert_candle(asset, timestamp, candle_data)
        self.db.commit()
        
        # Recupera
        candles = self.db.get_candles(
            asset, 
            timestamp - timedelta(minutes=1),
            timestamp + timedelta(minutes=1)
        )
        
        # Verifica
        self.assertEqual(len(candles), 1)
        self.assertEqual(candles[0]['open'], 5750.0)
        self.assertEqual(candles[0]['volume'], 100)
        
    def test_volume_profile(self):
        """Testa volume profile"""
        asset = 'DOLFUT'
        date = datetime.now().date()
        
        # Insere dados de volume
        for i in range(5):
            self.db.conn.execute('''
                INSERT INTO volume_profile
                (asset, date, price, volume, buy_volume, sell_volume, trades)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (asset, date, 5750.0 + i, 100 * (i+1), 60 * (i+1), 40 * (i+1), 10))
        
        self.db.commit()
        
        # Recupera
        profile = self.db.get_volume_profile(asset, date, date)
        
        # Verifica
        self.assertEqual(len(profile), 5)
        self.assertEqual(profile[5750.0]['volume'], 100)
        self.assertEqual(profile[5754.0]['volume'], 500)
        
    def test_important_levels(self):
        """Testa níveis importantes"""
        asset = 'DOLFUT'
        
        # Insere nível
        self.db.update_important_level(asset, 5750.0, 'support', 85.5, 1000)
        self.db.commit()
        
        # Recupera
        levels = self.db.get_important_levels(asset, 5740.0, 5760.0)
        
        # Verifica
        self.assertEqual(len(levels), 1)
        self.assertEqual(levels[0]['price'], 5750.0)
        self.assertEqual(levels[0]['level_type'], 'support')
        self.assertEqual(levels[0]['strength'], 85.5)
        
    def test_market_events(self):
        """Testa registro de eventos"""
        asset = 'DOLFUT'
        
        # Registra evento
        self.db.log_market_event(
            asset, 'breakout', 5755.0, 500,
            {'direction': 'up', 'strength': 'strong'}
        )
        self.db.commit()
        
        # Recupera
        events = self.db.get_market_events(
            asset, 'breakout',
            datetime.now() - timedelta(hours=1),
            datetime.now() + timedelta(hours=1)
        )
        
        # Verifica
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]['event_type'], 'breakout')
        self.assertEqual(events[0]['metadata']['direction'], 'up')


class TestPriceMemory(unittest.TestCase):
    """Testa o sistema de cache principal"""
    
    def setUp(self):
        """Configura cache temporário"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            'db_path': os.path.join(self.temp_dir, 'test.db'),
            'l1_duration': 30,
            'l2_duration': 24,
            'l3_retention': 90
        }
        self.cache = PriceMemory(self.config)
        
    def tearDown(self):
        """Limpa recursos"""
        self.cache.close()
        # Remove arquivos temporários
        for file in Path(self.temp_dir).glob('*'):
            file.unlink()
        os.rmdir(self.temp_dir)
        
    def test_add_market_data(self):
        """Testa adição de dados ao cache"""
        # Cria dados
        trades = [
            Trade(
                timestamp=datetime.now(),
                price=Decimal("5750.50"),
                volume=10,
                side="BUY",
                aggressor="BUY",
                order_id="123"
            )
        ]
        
        market_data = MarketData(
            asset='DOLFUT',
            timestamp=datetime.now(),
            trades=trades,
            order_book=OrderBook(
                timestamp=datetime.now(),
                bids=[],
                asks=[]
            )
        )
        
        # Adiciona
        self.cache.add_market_data(market_data)
        
        # Verifica L1
        self.assertEqual(len(self.cache.l1_cache['DOLFUT']), 1)
        self.assertEqual(self.cache.l1_cache['DOLFUT'][-1]['price'], 5750.50)
        
    def test_get_recent_trades(self):
        """Testa recuperação de trades recentes"""
        # Adiciona trades
        for i in range(5):
            trades = [
                Trade(
                    timestamp=datetime.now() - timedelta(minutes=4-i),
                    price=Decimal(f"{5750 + i}"),
                    volume=10,
                    side="BUY",
                    aggressor="BUY",
                    order_id=f"{i}"
                )
            ]
            
            market_data = MarketData(
                asset='DOLFUT',
                timestamp=trades[0].timestamp,
                trades=trades,
                order_book=OrderBook(
                    timestamp=trades[0].timestamp,
                    bids=[],
                    asks=[]
                )
            )
            
            self.cache.add_market_data(market_data)
        
        # Recupera últimos 3 minutos
        recent = self.cache.get_recent_trades('DOLFUT', minutes=3)
        
        # Verifica
        self.assertGreaterEqual(len(recent), 3)
        self.assertEqual(recent[-1]['price'], 5754.0)
        
    def test_candle_aggregation(self):
        """Testa agregação em candles"""
        # Adiciona vários trades no mesmo minuto
        minute = datetime.now().replace(second=0, microsecond=0)
        
        prices = [5750, 5755, 5748, 5752]
        for i, price in enumerate(prices):
            trades = [
                Trade(
                    timestamp=minute + timedelta(seconds=i*10),
                    price=Decimal(str(price)),
                    volume=25,
                    side="BUY" if i % 2 == 0 else "SELL",
                    aggressor="BUY" if i % 2 == 0 else "SELL",
                    order_id=str(i)
                )
            ]
            
            market_data = MarketData(
                asset='DOLFUT',
                timestamp=trades[0].timestamp,
                trades=trades,
                order_book=OrderBook(
                    timestamp=trades[0].timestamp,
                    bids=[],
                    asks=[]
                )
            )
            
            self.cache.add_market_data(market_data)
        
        # Verifica candle
        candle = self.cache.l2_cache['DOLFUT']['candles'].get(minute)
        self.assertIsNotNone(candle)
        self.assertEqual(candle['open'], 5750)
        self.assertEqual(candle['high'], 5755)
        self.assertEqual(candle['low'], 5748)
        self.assertEqual(candle['close'], 5752)
        self.assertEqual(candle['volume'], 100)
        
    def test_volume_profile_calculation(self):
        """Testa cálculo de volume profile"""
        # Adiciona trades em diferentes níveis
        levels = [5750, 5750, 5751, 5750, 5752]
        
        for i, price in enumerate(levels):
            trades = [
                Trade(
                    timestamp=datetime.now(),
                    price=Decimal(str(price)),
                    volume=10,
                    side="BUY",
                    aggressor="BUY",
                    order_id=str(i)
                )
            ]
            
            market_data = MarketData(
                asset='DOLFUT',
                timestamp=datetime.now(),
                trades=trades,
                order_book=OrderBook(
                    timestamp=datetime.now(),
                    bids=[],
                    asks=[]
                )
            )
            
            self.cache.add_market_data(market_data)
        
        # Verifica volume profile
        self.assertEqual(
            self.cache.l2_cache['DOLFUT']['volume_profile'][5750.0]['volume'], 
            30  # 3 trades de 10
        )
        
    def test_high_volume_nodes(self):
        """Testa identificação de HVNs"""
        # Adiciona volume concentrado em alguns níveis
        volume_distribution = {
            5750: 100,
            5751: 20,
            5752: 150,  # HVN
            5753: 30,
            5754: 120   # HVN
        }
        
        for price, volume in volume_distribution.items():
            # Simula volume no nível
            for _ in range(volume // 10):
                trades = [
                    Trade(
                        timestamp=datetime.now(),
                        price=Decimal(str(price)),
                        volume=10,
                        side="BUY",
                        aggressor="BUY",
                        order_id=f"{price}_{_}"
                    )
                ]
                
                market_data = MarketData(
                    asset='DOLFUT',
                    timestamp=datetime.now(),
                    trades=trades,
                    order_book=OrderBook(
                        timestamp=datetime.now(),
                        bids=[],
                        asks=[]
                    )
                )
                
                self.cache.add_market_data(market_data)
        
        # Busca HVNs (top 40%)
        hvns = self.cache.find_high_volume_nodes(
            'DOLFUT', 
            lookback_hours=1,
            min_volume_percentile=60
        )
        
        # Verifica
        hvn_prices = [hvn['price'] for hvn in hvns]
        self.assertIn(5752.0, hvn_prices)  # Maior volume
        self.assertIn(5754.0, hvn_prices)  # Segundo maior
        
    def test_statistics(self):
        """Testa estatísticas do sistema"""
        # Adiciona alguns dados
        for i in range(10):
            trades = [
                Trade(
                    timestamp=datetime.now(),
                    price=Decimal("5750"),
                    volume=5,
                    side="BUY",
                    aggressor="BUY",
                    order_id=str(i)
                )
            ]
            
            market_data = MarketData(
                asset='DOLFUT',
                timestamp=datetime.now(),
                trades=trades,
                order_book=OrderBook(
                    timestamp=datetime.now(),
                    bids=[],
                    asks=[]
                )
            )
            
            self.cache.add_market_data(market_data)
        
        # Obtém estatísticas
        stats = self.cache.get_statistics()
        
        # Verifica
        self.assertEqual(stats['DOLFUT_l1_size'], 10)
        self.assertEqual(stats['total_trades'], 10)
        self.assertIn('last_update', stats)


class TestHistoricalAnalyzer(unittest.TestCase):
    """Testa o analisador histórico"""
    
    def setUp(self):
        """Configura analisador com dados de teste"""
        self.temp_dir = tempfile.mkdtemp()
        self.db = DatabaseManager(os.path.join(self.temp_dir, 'test.db'))
        self.analyzer = HistoricalAnalyzer(self.db)
        
        # Adiciona dados de teste
        self._add_test_data()
        
    def tearDown(self):
        """Limpa recursos"""
        self.db.close()
        for file in Path(self.temp_dir).glob('*'):
            file.unlink()
        os.rmdir(self.temp_dir)
        
    def _add_test_data(self):
        """Adiciona dados de teste ao banco"""
        # Cria 100 candles de teste
        base_time = datetime.now() - timedelta(hours=2)
        
        for i in range(100):
            timestamp = base_time + timedelta(minutes=i)
            
            # Simula movimento de preço
            base_price = 5750
            price = base_price + (i % 10) - 5  # Oscila ±5
            
            candle = {
                'open': price,
                'high': price + 2,
                'low': price - 2,
                'close': price + 1,
                'volume': 100 + (i % 50),
                'buy_volume': 60,
                'sell_volume': 40,
                'trades': 10
            }
            
            self.db.insert_candle('DOLFUT', timestamp, candle)
            
        self.db.commit()
        
    def test_trend_analysis(self):
        """Testa análise de tendência"""
        context = self.analyzer.get_historical_context(
            'DOLFUT', 5750, lookback_hours=1
        )
        
        self.assertIn(context.trend, ['UP', 'DOWN', 'LATERAL', 
                                      'UP_STRONG', 'DOWN_STRONG'])
        
    def test_pattern_matching(self):
        """Testa busca de padrões similares"""
        # Padrão atual (últimos 10 preços)
        current_pattern = [5745, 5746, 5747, 5748, 5749, 
                          5750, 5749, 5748, 5747, 5746]
        
        # Busca similares
        similar = self.analyzer.find_similar_patterns(
            'DOLFUT', current_pattern, lookback_days=1
        )
        
        # Verifica
        if similar:  # Pode não encontrar dependendo dos dados
            self.assertLessEqual(len(similar), 10)  # Máximo 10
            self.assertGreaterEqual(similar[0]['correlation'], 0.8)


if __name__ == '__main__':
    unittest.main()