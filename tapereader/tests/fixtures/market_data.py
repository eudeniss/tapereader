"""
Fixtures de dados de mercado para testes
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from typing import List

from src.core.models import Trade, OrderBook, BookLevel, MarketData, Side


@pytest.fixture
def sample_trades():
    """Trades de exemplo"""
    base_time = datetime.now()
    
    return [
        Trade(
            timestamp=base_time - timedelta(seconds=30),
            price=Decimal('5750.00'),
            volume=25,
            aggressor=Side.BUY
        ),
        Trade(
            timestamp=base_time - timedelta(seconds=20),
            price=Decimal('5750.50'),
            volume=50,
            aggressor=Side.BUY
        ),
        Trade(
            timestamp=base_time - timedelta(seconds=10),
            price=Decimal('5751.00'),
            volume=100,
            aggressor=Side.SELL
        ),
        Trade(
            timestamp=base_time,
            price=Decimal('5750.50'),
            volume=30,
            aggressor=Side.BUY
        )
    ]


@pytest.fixture
def sample_order_book():
    """Order book de exemplo"""
    return OrderBook(
        timestamp=datetime.now(),
        bids=[
            BookLevel(price=Decimal('5750.00'), volume=100, orders=3),
            BookLevel(price=Decimal('5749.50'), volume=150, orders=5),
            BookLevel(price=Decimal('5749.00'), volume=200, orders=7),
            BookLevel(price=Decimal('5748.50'), volume=75, orders=2),
            BookLevel(price=Decimal('5748.00'), volume=125, orders=4)
        ],
        asks=[
            BookLevel(price=Decimal('5751.00'), volume=90, orders=3),
            BookLevel(price=Decimal('5751.50'), volume=120, orders=4),
            BookLevel(price=Decimal('5752.00'), volume=180, orders=6),
            BookLevel(price=Decimal('5752.50'), volume=60, orders=2),
            BookLevel(price=Decimal('5753.00'), volume=100, orders=3)
        ]
    )


@pytest.fixture
def sample_market_data(sample_trades, sample_order_book):
    """MarketData completo de exemplo"""
    return MarketData(
        asset='DOLFUT',
        timestamp=datetime.now(),
        trades=sample_trades,
        book=sample_order_book
    )


@pytest.fixture
def absorption_scenario_trades():
    """Trades simulando absorção"""
    base_time = datetime.now()
    trades = []
    
    # Agressão vendedora sem sucesso (preço não cai)
    for i in range(20):
        trades.append(Trade(
            timestamp=base_time - timedelta(seconds=20-i),
            price=Decimal('5750.00'),  # Preço não se move
            volume=50,  # Volume alto
            aggressor=Side.SELL
        ))
    
    return trades


@pytest.fixture
def institutional_trades():
    """Trades institucionais"""
    base_time = datetime.now()
    
    return [
        Trade(
            timestamp=base_time - timedelta(seconds=5),
            price=Decimal('5750.00'),
            volume=150,  # Grande para DOLFUT
            aggressor=Side.BUY
        ),
        Trade(
            timestamp=base_time - timedelta(seconds=3),
            price=Decimal('5750.50'),
            volume=200,
            aggressor=Side.BUY
        ),
        Trade(
            timestamp=base_time,
            price=Decimal('5751.00'),
            volume=175,
            aggressor=Side.BUY
        )
    ]