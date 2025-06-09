"""
Testes dos modelos de dados
"""

import pytest
from datetime import datetime
from decimal import Decimal

from src.core.models import Trade, OrderBook, BookLevel, MarketData, Side, TradingSignal


class TestTradeModel:
    """Testes do modelo Trade"""
    
    def test_trade_creation(self):
        """Testa criação de trade"""
        trade = Trade(
            timestamp=datetime.now(),
            price=Decimal('5750.00'),
            volume=50,
            aggressor=Side.BUY
        )
        
        assert trade.price == Decimal('5750.00')
        assert trade.volume == 50
        assert trade.aggressor == Side.BUY
        assert trade.quantity == 50  # Alias
        assert trade.side == 'buy'  # Alias
    
    def test_trade_serialization(self):
        """Testa serialização JSON"""
        trade = Trade(
            timestamp=datetime.now(),
            price=Decimal('5750.00'),
            volume=50,
            aggressor=Side.BUY
        )
        
        json_data = trade.json()
        assert isinstance(json_data, str)
        assert '5750.00' in json_data


class TestOrderBook:
    """Testes do OrderBook"""
    
    def test_order_book_validation(self):
        """Testa validação de ordenação"""
        # Bids devem estar em ordem decrescente
        with pytest.raises(ValueError, match="Bids devem estar ordenados"):
            OrderBook(
                timestamp=datetime.now(),
                bids=[
                    BookLevel(price=Decimal('100'), volume=10),
                    BookLevel(price=Decimal('101'), volume=10)  # Erro: maior preço depois
                ],
                asks=[]
            )
        
        # Asks devem estar em ordem crescente
        with pytest.raises(ValueError, match="Asks devem estar ordenados"):
            OrderBook(
                timestamp=datetime.now(),
                bids=[],
                asks=[
                    BookLevel(price=Decimal('101'), volume=10),
                    BookLevel(price=Decimal('100'), volume=10)  # Erro: menor preço depois
                ]
            )
    
    def test_valid_order_book(self, sample_order_book):
        """Testa book válido"""
        assert len(sample_order_book.bids) == 5
        assert len(sample_order_book.asks) == 5
        assert sample_order_book.bids[0].price > sample_order_book.bids[1].price
        assert sample_order_book.asks[0].price < sample_order_book.asks[1].price


class TestMarketData:
    """Testes do MarketData"""
    
    def test_market_data_validation(self):
        """Testa validação de asset"""
        with pytest.raises(ValueError, match="Asset deve ser um de"):
            MarketData(
                asset='INVALID',
                timestamp=datetime.now(),
                trades=[],
                book=OrderBook(timestamp=datetime.now(), bids=[], asks=[])
            )
    
    def test_market_data_aliases(self, sample_market_data):
        """Testa aliases de compatibilidade"""
        assert sample_market_data.symbol == 'DOLFUT'
        assert sample_market_data.order_book == sample_market_data.book