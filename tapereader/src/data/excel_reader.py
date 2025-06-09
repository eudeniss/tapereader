"""
Leitor de dados do Excel com RTD
Versão Final Robusta - Corrigida e Otimizada
"""

import asyncio
from datetime import datetime, time, timedelta
from decimal import Decimal, InvalidOperation
from typing import Optional, List, Dict, Any
import logging
from pathlib import Path

try:
    import xlwings as xw
except ImportError:
    xw = None
    logging.warning("xlwings não instalado - Excel RTD não disponível. Instale com: pip install xlwings")

from .base import DataProvider
from ..core.models import MarketData, Trade, OrderBook, BookLevel, Side


class ExcelRTDReader(DataProvider):
    """Leitor de dados RTD do Excel"""
    
    def __init__(self, config):
        super().__init__(config)
        self.excel_app = None
        self.workbook = None
        self.sheet = None
        self.excel_file = None
        # Otimização: Mantém um set dos hashes dos últimos trades para evitar processamento duplicado
        self.last_trade_hashes = {'DOLFUT': set(), 'WDOFUT': set()}
        self.ranges = config.excel.ranges
        self.column_mapping = config.excel.column_mapping
        self.monitor_task = None
        self.monitoring = False
        
    async def connect(self) -> bool:
        if xw is None:
            self.logger.error("A biblioteca 'xlwings' é necessária para a leitura do Excel, mas não está instalada.")
            self.logger.error("Instale com o comando: pip install xlwings")
            return False
            
        try:
            excel_file_path_str = self.config.excel.file_path
            excel_path = Path(excel_file_path_str).resolve()

            if not excel_path.exists():
                self.logger.critical(f"ARQUIVO EXCEL NÃO ENCONTRADO!")
                self.logger.critical(f"Caminho configurado: {excel_path}")
                self.logger.critical("Por favor, verifique o caminho 'file_path' no arquivo 'config/excel.yaml' e tente novamente.")
                return False

            self.excel_file = str(excel_path)
            self.logger.info(f"Conectando ao Excel: {self.excel_file}")

            # Tenta se conectar a uma instância ativa do Excel primeiro
            try:
                self.excel_app = xw.apps.active
                if self.excel_app is None: raise Exception("Nenhuma instância ativa do Excel.")
                for book in self.excel_app.books:
                    if Path(book.fullname).resolve() == excel_path:
                        self.workbook = book
                        break
            except Exception:
                self.logger.info("Nenhuma instância do Excel encontrada. Abrindo uma nova.")
                self.excel_app = xw.App(visible=True, add_book=False)

            # Abre o workbook se não foi encontrado em uma instância ativa
            if self.workbook is None:
                self.workbook = self.excel_app.books.open(self.excel_file)
            
            self.sheet = self.workbook.sheets[0]
            self.connected = True
            self.logger.info("Conectado ao Excel com sucesso.")
            
            await self._check_rtd_active()
            return True

        except Exception as e:
            self.logger.error(f"Erro fatal ao conectar ao Excel: {e}", exc_info=True)
            self.logger.error("Verifique se o Excel está instalado e se o caminho do arquivo está correto em 'config/excel.yaml'.")
            return False
            
    async def disconnect(self) -> bool:
        try:
            self.monitoring = False
            if self.monitor_task:
                self.monitor_task.cancel()
                try: await self.monitor_task
                except asyncio.CancelledError: pass
            
            # Não fechamos o workbook ou app para não interromper o trabalho do usuário
            self.workbook, self.excel_app = None, None
            self.connected = False
            self.logger.info("Desconectado do Excel.")
            return True
        except Exception as e:
            self.logger.error(f"Erro ao desconectar do Excel: {e}")
            return False
            
    async def start(self, queue: asyncio.Queue):
        self.data_queue = queue
        self.monitoring = True
        self.monitor_task = asyncio.create_task(self._monitor_excel())
        
    async def _monitor_excel(self):
        # Usa o intervalo de atualização definido no modo de operação (main.yaml)
        mode_config = self.config.get_mode_config()
        poll_interval = mode_config.update_interval_ms / 1000.0
        
        while self.monitoring and self.connected:
            try:
                for asset in ['DOLFUT', 'WDOFUT']:
                    market_data = await self._get_market_data(asset)
                    if market_data and (market_data.trades or market_data.book.bids):
                        await self.push_market_data(asset, market_data)
                await asyncio.sleep(poll_interval)
            except Exception as e:
                self.logger.error(f"Erro no loop de monitoramento do Excel: {e}", exc_info=True)
                await asyncio.sleep(5) # Pausa maior em caso de erro
                
    async def _get_market_data(self, asset: str) -> Optional[MarketData]:
        if not self.connected: return None
        try:
            # Executa a leitura de trades e book em paralelo para otimizar tempo
            trades_task = asyncio.create_task(self._read_time_trades(asset))
            book_task = asyncio.create_task(self._read_order_book(asset))
            
            trades, order_book = await asyncio.gather(trades_task, book_task)
            
            # Só retorna dados se houver alguma novidade
            if not trades and not order_book.bids: return None

            return MarketData(
                asset=asset,
                timestamp=datetime.now(),
                trades=trades,
                book=order_book
            )
        except Exception as e:
            self.logger.error(f"Erro ao obter dados de mercado para {asset}: {e}")
            return None

    def _parse_trade_row(self, row: List[Any], row_num: int) -> Optional[Trade]:
        """Valida e processa uma única linha de trade de forma segura."""
        try:
            # Mapeamento de colunas para flexibilidade
            mapping = self.column_mapping['time_trades']
            
            # Verifica se a linha tem o mínimo de colunas necessárias
            if not isinstance(row, list) or len(row) <= max(mapping.values()):
                return None

            time_val = row[mapping['time']]
            aggressor_val = row[mapping['aggressor']]
            price_val = row[mapping['price']]
            volume_val = row[mapping['volume']]
            
            # Validação: ignora linhas com valores essenciais faltando
            if any(v is None for v in [time_val, aggressor_val, price_val, volume_val]):
                return None
            
            # Limpeza e conversão de dados com tratamento de erro
            price_str = str(price_val).replace(',', '.').strip()
            volume_str = str(volume_val).strip()

            if not price_str or not volume_str: return None
            
            timestamp = self._parse_timestamp(time_val)
            if not timestamp: return None

            price = Decimal(price_str)
            volume = int(float(volume_str))
            
            # Validação extra para valores absurdos
            if volume <= 0 or price <= 0:
                self.logger.warning(f"Trade com valor inválido ignorado [linha {row_num}]: Preço={price}, Vol={volume}")
                return None

            aggressor_str = str(aggressor_val).upper().strip()
            aggressor = Side.BUY if aggressor_str in ['C', 'COMPRA', 'COMPRADOR', 'BUYER'] else Side.SELL
            
            return Trade(timestamp=timestamp, price=price, volume=volume, aggressor=aggressor)

        except (ValueError, TypeError, InvalidOperation) as e:
            self.logger.warning(f"Linha de trade inválida [planilha linha ~{row_num}] ignorada: {row}. Erro: {e}")
            return None

    async def _read_time_trades(self, asset: str) -> List[Trade]:
        try:
            range_str = self.ranges[asset.lower()]['time_trades']
            data = self.sheet.range(range_str).value
            if not data: return []
                
            new_trades = []
            for i, row in enumerate(data):
                trade = self._parse_trade_row(row, i + 4) # +4 para corresponder ao número da linha no Excel
                if trade:
                    # Otimização para evitar processar trades duplicados
                    trade_hash = hash((trade.timestamp.isoformat(), str(trade.price), trade.volume, trade.aggressor.value))
                    if trade_hash not in self.last_trade_hashes[asset]:
                        new_trades.append(trade)
                        self.last_trade_hashes[asset].add(trade_hash)
            
            # Limpeza do cache de hashes para evitar consumo de memória infinito
            if len(self.last_trade_hashes[asset]) > 2000:
                self.last_trade_hashes[asset] = set(list(self.last_trade_hashes[asset])[-1000:])

            return new_trades
        except Exception as e:
            self.logger.error(f"Erro fatal ao ler Time & Trades para {asset}: {e}", exc_info=True)
            return []

    async def _read_order_book(self, asset: str) -> OrderBook:
        """Lê e processa o book de ofertas de forma robusta."""
        try:
            range_str = self.ranges[asset.lower()]['order_book']
            data = self.sheet.range(range_str).value
            if not data: return OrderBook(timestamp=datetime.now(), bids=[], asks=[])

            bids, asks = [], []
            mapping = self.column_mapping['order_book']

            for row in data:
                try:
                    if len(row) <= max(mapping.values()): continue

                    bid_price = row[mapping['bid_price']]
                    bid_vol = row[mapping['bid_volume']]
                    ask_price = row[mapping['ask_price']]
                    ask_vol = row[mapping['ask_volume']]

                    if bid_price is not None and bid_vol is not None and float(bid_vol) > 0:
                        bid_price_str = str(bid_price).replace(',', '.').strip()
                        bids.append(BookLevel(price=Decimal(bid_price_str), volume=int(float(bid_vol)), orders=1))
                    
                    if ask_price is not None and ask_vol is not None and float(ask_vol) > 0:
                        ask_price_str = str(ask_price).replace(',', '.').strip()
                        asks.append(BookLevel(price=Decimal(ask_price_str), volume=int(float(ask_vol)), orders=1))
                except (ValueError, TypeError, InvalidOperation) as e:
                    self.logger.debug(f"Linha do book ignorada por erro de parsing: {row}. Erro: {e}")
                    continue

            bids.sort(key=lambda x: x.price, reverse=True)
            asks.sort(key=lambda x: x.price)
            return OrderBook(timestamp=datetime.now(), bids=bids[:10], asks=asks[:10])
        except Exception as e:
            self.logger.error(f"Erro ao ler Order Book para {asset}: {e}")
            return OrderBook(timestamp=datetime.now(), bids=[], asks=[])
            
    def _parse_timestamp(self, excel_time) -> Optional[datetime]:
        """Converte diferentes formatos de hora do Excel para datetime."""
        if isinstance(excel_time, datetime): return excel_time
        if isinstance(excel_time, time): return datetime.combine(datetime.now().date(), excel_time)
        
        # Converte número serial do Excel (float) para datetime
        if isinstance(excel_time, (int, float)):
            try: return datetime(1899, 12, 30) + timedelta(days=excel_time)
            except (TypeError, ValueError): return None
            
        # Tenta formatos de string comuns
        try:
            time_str = str(excel_time)
            # Tenta múltiplos formatos
            for fmt in ("%H:%M:%S", "%H:%M:%S.%f"):
                try:
                    return datetime.combine(datetime.now().date(), datetime.strptime(time_str, fmt).time())
                except ValueError:
                    continue
            return None
        except ValueError:
            return None

    async def _check_rtd_active(self):
        """Verifica se os dados RTD estão sendo atualizados na planilha."""
        self.logger.info("Verificando se o RTD está ativo (aguarde 2 segundos)...")
        try:
            cell_to_check = self.ranges['dolfut']['time_trades'].split(':')[0] # Pega a primeira célula do range
            test_range_1 = self.sheet.range(cell_to_check).value
            await asyncio.sleep(2)
            test_range_2 = self.sheet.range(cell_to_check).value
            
            if test_range_1 is None and test_range_2 is None:
                self.logger.warning(f"ATENÇÃO: Célula de teste ({cell_to_check}) está vazia. Não foi possível confirmar atividade do RTD.")
            elif test_range_1 == test_range_2:
                self.logger.warning(f"ATENÇÃO: Dados em '{cell_to_check}' não mudaram em 2 segundos. Verifique se o RTD está ativo na sua plataforma.")
            else:
                self.logger.info("✅ RTD parece estar ativo e funcionando.")
        except Exception as e:
            self.logger.error(f"Não foi possível verificar o status do RTD: {e}")