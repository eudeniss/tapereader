"""
Detector de Fluxo Institucional
Identifica atividade de grandes players (smart money)
"""

from typing import Dict, Any, List, Tuple
from decimal import Decimal
from datetime import datetime, timedelta
from collections import defaultdict

from .base import BehaviorDetector
from ..core.models import MarketData, BehaviorDetection, OrderBook, Trade, Side


class InstitutionalFlowDetector(BehaviorDetector):
    """
    Detecta fluxo institucional (big players)
    
    Características:
    - Trades de tamanho muito acima da média
    - Execução metódica e consistente
    - Movimento de preço proporcional ao volume
    - Padrões de acumulação/distribuição
    """
    
    @property
    def behavior_type(self) -> str:
        # CORREÇÃO: O tipo foi corrigido de "institutional" para "institutional"
        # para corresponder à configuração e às regras de estratégia.
        return "institutional" 
        
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Thresholds por ativo
        self.size_thresholds = {
            'DOLFUT': config.get('dolfut_institutional_size', 50),
            'WDOFUT': config.get('wdofut_institutional_size', 200)
        }
        
        # Parâmetros
        self.min_institutional_trades = config.get('min_institutional_trades', 3)
        self.clustering_window = config.get('clustering_window', 60)  # segundos
        self.price_impact_threshold = config.get('price_impact_threshold', 0.5)  # ticks por grande trade
        
    async def detect(self, market_data: MarketData) -> BehaviorDetection:
        """Detecta fluxo institucional"""
        # Atualiza histórico
        self.update_history(market_data)
        
        if not market_data.trades:
            return self.create_detection(False, 0.0)
            
        # Analisa fluxo institucional
        institutional_result = self._analyze_institutional(market_data)
        
        if institutional_result['detected']:
            metadata = {
                'flow_direction': institutional_result['direction'].value if institutional_result['direction'] != "mixed" else "mixed",
                'institutional_volume': institutional_result['total_volume'],
                'institutional_trades': institutional_result['trade_count'],
                'average_size': institutional_result['avg_size'],
                'flow_pattern': institutional_result['pattern'],
                'price_impact': str(institutional_result['price_impact']),
                'execution_quality': institutional_result['execution_quality'],
                'accumulation_score': institutional_result['accumulation_score']
            }

            # A direção é a do fluxo dominante.
            direction_result = institutional_result['direction']
            if direction_result == "mixed":
                direction_result = Side.NEUTRAL
            
            detection = self.create_detection(
                True,
                institutional_result['confidence'],
                metadata
            )
            detection.direction = direction_result
            return detection
            
        return self.create_detection(False, 0.0)
        
    def _analyze_institutional(self, market_data: MarketData) -> Dict[str, Any]:
        """Analisa presença de fluxo institucional"""
        asset = market_data.asset
        threshold = self.size_thresholds.get(asset, 50)
        
        # Identifica trades institucionais
        recent_trades = self.get_recent_trades(self.clustering_window)
        institutional_trades = [
            t for t in recent_trades 
            if t.volume >= threshold
        ]
        
        if len(institutional_trades) < self.min_institutional_trades:
            return {'detected': False}
            
        # Analisa características do fluxo
        flow_analysis = self._analyze_flow_characteristics(institutional_trades, recent_trades)
        
        # Analisa padrão de execução
        execution_pattern = self._analyze_execution_pattern(institutional_trades)
        
        # Analisa impacto no preço
        price_impact = self._analyze_price_impact(institutional_trades, recent_trades)
        
        # Detecta acumulação/distribuição
        accumulation_analysis = self._detect_accumulation_distribution(
            institutional_trades,
            market_data.book
        )
        
        # Calcula confiança
        confidence_signals = {
            'flow_consistency': flow_analysis['consistency'],
            'execution_quality': execution_pattern['quality_score'],
            'price_efficiency': price_impact['efficiency'],
            'accumulation_pattern': accumulation_analysis['score']
        }
        
        confidence = self.calculate_confidence(confidence_signals)
        
        if confidence < self.min_confidence:
            return {'detected': False}
            
        return {
            'detected': True,
            'confidence': confidence,
            'direction': flow_analysis['dominant_direction'],
            'total_volume': flow_analysis['total_volume'],
            'trade_count': len(institutional_trades),
            'avg_size': flow_analysis['avg_size'],
            'pattern': execution_pattern['pattern_type'],
            'price_impact': price_impact['total_impact'],
            'execution_quality': execution_pattern['quality_score'],
            'accumulation_score': accumulation_analysis['score']
        }
        
    def _analyze_flow_characteristics(
        self, 
        institutional_trades: List[Trade],
        all_trades: List[Trade]
    ) -> Dict[str, Any]:
        """Analisa características do fluxo institucional"""
        # Volume e direção
        buy_volume = sum(t.volume for t in institutional_trades if t.aggressor == Side.BUY)
        sell_volume = sum(t.volume for t in institutional_trades if t.aggressor == Side.SELL)
        total_volume = buy_volume + sell_volume
        
        # Direção dominante
        if buy_volume > sell_volume * 1.5:
            dominant_direction = Side.BUY
        elif sell_volume > buy_volume * 1.5:
            dominant_direction = Side.SELL
        else:
            dominant_direction = "mixed"
            
        # Consistência do fluxo (todos na mesma direção?)
        if dominant_direction != "mixed":
            same_direction = sum(
                1 for t in institutional_trades 
                if t.aggressor == dominant_direction
            )
            consistency = same_direction / len(institutional_trades)
        else:
            consistency = 0.5
            
        # Comparação com volume total
        total_market_volume = sum(t.volume for t in all_trades)
        institutional_ratio = total_volume / total_market_volume if total_market_volume > 0 else 0
        
        return {
            'dominant_direction': dominant_direction,
            'total_volume': total_volume,
            'buy_volume': buy_volume,
            'sell_volume': sell_volume,
            'avg_size': total_volume / len(institutional_trades) if institutional_trades else 0,
            'consistency': consistency,
            'market_share': institutional_ratio
        }
        
    def _analyze_execution_pattern(self, institutional_trades: List[Trade]) -> Dict[str, Any]:
        """Analisa padrão de execução dos trades institucionais"""
        if len(institutional_trades) < 2:
            return {
                'pattern_type': 'single',
                'quality_score': 0.5,
                'timing_consistency': 0.0
            }
            
        # Analisa intervalos entre trades
        intervals = []
        for i in range(1, len(institutional_trades)):
            interval = (
                institutional_trades[i].timestamp - 
                institutional_trades[i-1].timestamp
            ).total_seconds()
            intervals.append(interval)
            
        # Identifica padrão
        if not intervals:
            pattern_type = 'single'
            timing_consistency = 0.0
            avg_interval = 0
        else:
            avg_interval = sum(intervals) / len(intervals)
            
            # Verifica consistência dos intervalos
            if all(abs(i - avg_interval) < avg_interval * 0.5 for i in intervals):
                pattern_type = 'algorithmic'  # Execução algorítmica regular
                timing_consistency = 0.9
            elif all(i < 5 for i in intervals):
                pattern_type = 'aggressive'  # Execução agressiva rápida
                timing_consistency = 0.7
            elif max(intervals) > 30:
                pattern_type = 'patient'  # Execução paciente
                timing_consistency = 0.6
            else:
                pattern_type = 'mixed'
                timing_consistency = 0.5
                
        # Analisa tamanhos dos trades
        sizes = [t.volume for t in institutional_trades]
        avg_size = sum(sizes) / len(sizes)
        
        # Verifica se usa clips consistentes (iceberg)
        size_consistency = 0.0
        if len(set(sizes)) == 1:
            # Todos mesmo tamanho = iceberg
            size_consistency = 1.0
        else:
            # Calcula desvio padrão relativo
            if avg_size > 0 and len(sizes) > 1:
                std_dev = (sum((s - avg_size) ** 2 for s in sizes) / len(sizes)) ** 0.5
                size_consistency = max(0, 1 - (std_dev / avg_size))
                
        # Score de qualidade de execução
        quality_score = (timing_consistency + size_consistency) / 2
        
        return {
            'pattern_type': pattern_type,
            'quality_score': quality_score,
            'timing_consistency': timing_consistency,
            'size_consistency': size_consistency,
            'avg_interval': avg_interval if intervals else 0
        }
        
    def _analyze_price_impact(
        self,
        institutional_trades: List[Trade],
        all_trades: List[Trade]
    ) -> Dict[str, Any]:
        """Analisa impacto no preço dos trades institucionais"""
        if not institutional_trades or not all_trades:
            return {
                'total_impact': Decimal('0'),
                'efficiency': 0.5,
                'impact_per_trade': Decimal('0')
            }
            
        # Ordena todos os trades por timestamp
        all_sorted = sorted(all_trades, key=lambda t: t.timestamp)
        
        impacts = []
        
        for inst_trade in institutional_trades:
            # Encontra preço antes e depois do trade institucional
            before_price = None
            after_price = None
            
            for i, trade in enumerate(all_sorted):
                if trade.timestamp >= inst_trade.timestamp:
                    # Este é o trade institucional ou posterior
                    if i > 0:
                        before_price = all_sorted[i-1].price
                    if i < len(all_sorted) - 1:
                        after_price = all_sorted[i+1].price
                    break
                    
            if before_price and after_price:
                # Calcula impacto
                if inst_trade.aggressor == Side.BUY:
                    impact = after_price - before_price
                else:
                    impact = before_price - after_price
                    
                impacts.append(impact)
                
        if not impacts:
            return {
                'total_impact': Decimal('0'),
                'efficiency': 0.5,
                'impact_per_trade': Decimal('0')
            }
            
        # Calcula métricas
        total_impact = sum(impacts)
        avg_impact = total_impact / len(impacts)
        
        # Eficiência: impacto positivo consistente com a direção
        positive_impacts = sum(1 for i in impacts if i > 0)
        efficiency = positive_impacts / len(impacts) if impacts else 0.5
        
        return {
            'total_impact': total_impact,
            'efficiency': efficiency,
            'impact_per_trade': avg_impact,
            'favorable_impacts': positive_impacts
        }
        
    def _detect_accumulation_distribution(
        self,
        institutional_trades: List[Trade],
        order_book: OrderBook
    ) -> Dict[str, Any]:
        """Detecta padrões de acumulação ou distribuição"""
        # Agrupa trades por nível de preço
        price_levels = defaultdict(lambda: {'volume': 0, 'count': 0, 'side': None})
        
        for trade in institutional_trades:
            # Arredonda preço para nível
            price_level = round(float(trade.price), 1)
            price_levels[price_level]['volume'] += trade.volume
            price_levels[price_level]['count'] += 1
            price_levels[price_level]['side'] = trade.aggressor
            
        if not price_levels:
            return {'score': 0.0, 'pattern': 'none'}
            
        # Analisa distribuição dos níveis
        sorted_levels = sorted(price_levels.items())
        
        # Detecta padrão
        if len(sorted_levels) >= 3:
            # Verifica se está acumulando em range estreito
            price_range = sorted_levels[-1][0] - sorted_levels[0][0]
            
            if price_range < 2.0:  # Range estreito
                # Acumulação: compras em range estreito
                buy_volume = sum(
                    data['volume'] for _, data in sorted_levels
                    if data['side'] == Side.BUY
                )
                total_volume = sum(data['volume'] for _, data in sorted_levels)
                
                if total_volume > 0 and buy_volume / total_volume > 0.7:
                    pattern = 'accumulation'
                    score = 0.8
                elif total_volume > 0 and buy_volume / total_volume < 0.3:
                    pattern = 'distribution'
                    score = 0.8
                else:
                    pattern = 'mixed'
                    score = 0.5
            else:
                # Range largo - possível markup/markdown
                pattern = 'trending'
                score = 0.6
        else:
            pattern = 'insufficient_data'
            score = 0.3
            
        # Verifica alinhamento com book
        book_support = self._check_book_alignment(order_book, pattern) if order_book else 1.0
        
        return {
            'score': score * book_support,
            'pattern': pattern,
            'price_levels': len(price_levels),
            'concentration': self._calculate_concentration(price_levels)
        }
        
    def _check_book_alignment(self, book: OrderBook, pattern: str) -> float:
        """Verifica se book suporta o padrão detectado"""
        if not book.bids or not book.asks:
            return 1.0  # Neutro
            
        book_imbalance = self.analyze_book_imbalance(book)
        
        # Alinhamento esperado
        if pattern == 'accumulation' and book_imbalance['pressure'] == 'buy':
            return 1.2  # Bonus
        elif pattern == 'distribution' and book_imbalance['pressure'] == 'sell':
            return 1.2  # Bonus
        elif pattern in ['accumulation', 'distribution'] and book_imbalance['pressure'] == 'neutral':
            return 1.0  # Neutro
        else:
            return 0.8  # Penalidade por desalinhamento
            
    def _calculate_concentration(self, price_levels: Dict) -> float:
        """Calcula concentração de volume em poucos níveis"""
        if not price_levels:
            return 0.0
            
        volumes = sorted(
            [data['volume'] for data in price_levels.values()],
            reverse=True
        )
        
        total_volume = sum(volumes)
        if total_volume == 0:
            return 0.0
            
        # Verifica se top 3 níveis concentram maior parte do volume
        top_3_volume = sum(volumes[:3])
        concentration = top_3_volume / total_volume
        
        return concentration