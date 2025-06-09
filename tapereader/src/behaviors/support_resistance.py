"""
Detector de Suporte e Resistência Aprimorado
Com persistência histórica e análise avançada
"""

import json
import sqlite3
from typing import Dict, Any, List, Tuple, Optional
from decimal import Decimal
from datetime import datetime, timedelta
from collections import defaultdict
from pathlib import Path
import threading

from .base import BehaviorDetector
from ..core.models import MarketData, BehaviorDetection, Trade, OrderBook, Side


class SupportResistanceEnhancedDetector(BehaviorDetector):
    """
    Detector avançado de S/R com persistência histórica
    
    Melhorias:
    - Persistência em SQLite
    - Análise de força histórica
    - Decay temporal (níveis antigos perdem força)
    - Clustering de níveis próximos
    - Estatísticas de eficácia
    """
    
    @property
    def behavior_type(self) -> str:
        return "support_resistance"
        
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Parâmetros
        self.level_tolerance = Decimal(str(config.get('level_tolerance', 0.5)))
        self.min_touches = config.get('min_touches', 3)
        self.lookback_minutes = config.get('lookback_minutes', 15)
        self.rejection_threshold = config.get('rejection_threshold', 2.0)
        self.clustering_threshold = Decimal(str(config.get('clustering_threshold', 1.0)))
        
        # Decay temporal
        self.decay_enabled = config.get('enable_decay', True)
        self.decay_half_life_hours = config.get('decay_half_life_hours', 24)
        
        # Persistência
        self.persistence_enabled = config.get('enable_persistence', True)
        self.db_path = Path(config.get('db_path', '../data/levels.db'))
        
        # Cache em memória
        self.level_cache = {
            'DOLFUT': {},
            'WDOFUT': {}
        }
        
        # Thread lock para SQLite
        self.db_lock = threading.Lock()
        
        # Inicializa banco se habilitado
        if self.persistence_enabled:
            self._initialize_database()
            self._load_historical_levels()
            
    def _initialize_database(self):
        """Cria estrutura do banco de dados"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        with self.db_lock:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Tabela de níveis
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS price_levels (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    asset TEXT NOT NULL,
                    price REAL NOT NULL,
                    first_touch TIMESTAMP,
                    last_touch TIMESTAMP,
                    touches INTEGER DEFAULT 0,
                    rejections INTEGER DEFAULT 0,
                    successful_breaks INTEGER DEFAULT 0,
                    total_volume INTEGER DEFAULT 0,
                    avg_rejection_size REAL DEFAULT 0,
                    max_rejection_size REAL DEFAULT 0,
                    strength_score REAL DEFAULT 0,
                    is_active BOOLEAN DEFAULT 1,
                    UNIQUE(asset, price)
                )
            ''')
            
            # Tabela de eventos (toques, rejeições, rompimentos)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS level_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    level_id INTEGER,
                    event_type TEXT,
                    timestamp TIMESTAMP,
                    price REAL,
                    volume INTEGER,
                    price_before REAL,
                    price_after REAL,
                    metadata TEXT,
                    FOREIGN KEY(level_id) REFERENCES price_levels(id)
                )
            ''')
            
            # Índices para performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_asset_price ON price_levels(asset, price)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_level_events ON level_events(level_id, timestamp)')
            
            conn.commit()
            conn.close()
            
        self.logger.info("Banco de dados de níveis inicializado")
        
    def _load_historical_levels(self):
        """Carrega níveis históricos do banco para memória"""
        with self.db_lock:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Carrega níveis ativos dos últimos 7 dias
            cutoff = datetime.now() - timedelta(days=7)
            
            cursor.execute('''
                SELECT asset, price, touches, rejections, successful_breaks,
                       total_volume, strength_score, last_touch
                FROM price_levels
                WHERE is_active = 1 AND last_touch > ?
                ORDER BY strength_score DESC
            ''', (cutoff,))
            
            for row in cursor.fetchall():
                asset, price, touches, rejections, breaks, volume, strength, last_touch = row
                price = Decimal(str(price))
                
                if asset not in self.level_cache:
                    self.level_cache[asset] = {}
                    
                self.level_cache[asset][price] = {
                    'touches': touches,
                    'rejections': rejections,
                    'breaks': breaks,
                    'volume': volume,
                    'strength': strength,
                    'last_touch': datetime.fromisoformat(last_touch),
                    'events': []
                }
                
            conn.close()
            
        self.logger.info(f"Carregados {sum(len(levels) for levels in self.level_cache.values())} níveis históricos")
        
    async def detect(self, market_data: MarketData) -> BehaviorDetection:
        """Detecta níveis de suporte/resistência"""
        # Atualiza histórico
        self.update_history(market_data)
        
        if len(self.historical_data) < 5:
            return self.create_detection(False, 0.0)
            
        # Processa novos toques/rejeições
        self._process_market_data(market_data)
        
        # Aplica decay temporal se habilitado
        if self.decay_enabled:
            self._apply_temporal_decay(market_data.asset)
            
        # Clusteriza níveis próximos
        clustered_levels = self._cluster_levels(market_data.asset)
        
        # Analisa níveis
        level_result = self._analyze_levels_advanced(market_data, clustered_levels)
        
        if level_result['detected']:
            # Persiste evento se habilitado
            if self.persistence_enabled:
                self._persist_level_event(market_data.asset, level_result)
                
            metadata = {
                'level_type': level_result['type'],
                'level_price': str(level_result['price']),
                'touches': level_result['touches'],
                'rejections': level_result['rejections'],
                'successful_breaks': level_result.get('breaks', 0),
                'strength': level_result['strength'],
                'historical_strength': level_result.get('historical_strength', 0),
                'book_support': level_result['book_support'],
                'volume_profile': level_result.get('volume_profile', {}),
                'cluster_size': level_result.get('cluster_size', 1),
                'effectiveness': level_result.get('effectiveness', 0)
            }
            
            # --- CORREÇÃO APLICADA AQUI ---
            # A direção é baseada no tipo de nível:
            # Suporte -> Espera-se uma alta (BUY)
            # Resistência -> Espera-se uma baixa (SELL)
            direction_result = Side.BUY if level_result['type'] == 'support' else Side.SELL
            
            detection = self.create_detection(
                True,
                level_result['confidence'],
                metadata
            )
            detection.direction = direction_result
            return detection
            
        return self.create_detection(False, 0.0)
        
    def _process_market_data(self, market_data: MarketData):
        """Processa dados de mercado para atualizar níveis"""
        asset = market_data.asset
        
        if asset not in self.level_cache:
            self.level_cache[asset] = {}
            
        # Processa cada trade
        for trade in market_data.trades:
            level_price = self._round_to_level(trade.price)
            
            # Cria nível se não existe
            if level_price not in self.level_cache[asset]:
                self.level_cache[asset][level_price] = {
                    'touches': 0,
                    'rejections': 0,
                    'breaks': 0,
                    'volume': 0,
                    'strength': 0.0,
                    'first_touch': trade.timestamp,
                    'last_touch': trade.timestamp,
                    'events': []
                }
                
            # Atualiza dados
            level = self.level_cache[asset][level_price]
            level['touches'] += 1
            level['volume'] += trade.volume
            level['last_touch'] = trade.timestamp
            
            # Adiciona evento
            level['events'].append({
                'type': 'touch',
                'timestamp': trade.timestamp,
                'price': trade.price,
                'volume': trade.volume
            })
            
            # Mantém apenas últimos 100 eventos
            if len(level['events']) > 100:
                level['events'] = level['events'][-100:]
                
        # Detecta rejeições e rompimentos
        self._detect_rejections_and_breaks(asset)
        
    def _detect_rejections_and_breaks(self, asset: str):
        """Detecta rejeições e rompimentos de níveis"""
        recent_trades = self.get_recent_trades(120)  # 2 minutos
        
        if len(recent_trades) < 20:
            return
            
        # Analisa sequências de preços
        for i in range(10, len(recent_trades) - 10):
            current_price = recent_trades[i].price
            
            # Preços antes e depois
            before_prices = [t.price for t in recent_trades[i-10:i]]
            after_prices = [t.price for t in recent_trades[i:i+10]]
            
            if not before_prices or not after_prices:
                continue
                
            # Verifica reversão de alta (resistência)
            max_before = max(before_prices)
            max_after = max(after_prices)
            
            if max_before > max_after + self.rejection_threshold:
                # Rejeição em resistência
                level_price = self._round_to_level(max_before)
                if level_price in self.level_cache[asset]:
                    level = self.level_cache[asset][level_price]
                    level['rejections'] += 1
                    level['events'].append({
                        'type': 'rejection',
                        'timestamp': recent_trades[i].timestamp,
                        'price': max_before,
                        'rejection_size': float(max_before - max_after)
                    })
                    
            # Verifica reversão de baixa (suporte)
            min_before = min(before_prices)
            min_after = min(after_prices)
            
            if min_before < min_after - self.rejection_threshold:
                # Rejeição em suporte
                level_price = self._round_to_level(min_before)
                if level_price in self.level_cache[asset]:
                    level = self.level_cache[asset][level_price]
                    level['rejections'] += 1
                    level['events'].append({
                        'type': 'rejection',
                        'timestamp': recent_trades[i].timestamp,
                        'price': min_before,
                        'rejection_size': float(min_after - min_before)
                    })
                    
            # Verifica rompimentos
            for level_price, level_data in self.level_cache[asset].items():
                # Rompimento de resistência
                if current_price > level_price + self.rejection_threshold:
                    if any(p < level_price for p in before_prices[-5:]):
                        # Cruzou de baixo para cima
                        level_data['breaks'] += 1
                        level_data['events'].append({
                            'type': 'breakout',
                            'timestamp': recent_trades[i].timestamp,
                            'price': current_price,
                            'direction': 'up'
                        })
                        
                # Rompimento de suporte
                elif current_price < level_price - self.rejection_threshold:
                    if any(p > level_price for p in before_prices[-5:]):
                        # Cruzou de cima para baixo
                        level_data['breaks'] += 1
                        level_data['events'].append({
                            'type': 'breakout',
                            'timestamp': recent_trades[i].timestamp,
                            'price': current_price,
                            'direction': 'down'
                        })
                        
    def _apply_temporal_decay(self, asset: str):
        """Aplica decay temporal aos níveis antigos"""
        now = datetime.now()
        half_life_seconds = self.decay_half_life_hours * 3600
        
        for level_price, level_data in list(self.level_cache[asset].items()):
            # Calcula idade do nível
            age_seconds = (now - level_data['last_touch']).total_seconds()
            
            # Fator de decay exponencial
            decay_factor = 0.5 ** (age_seconds / half_life_seconds)
            
            # Aplica decay à força
            level_data['strength'] *= decay_factor
            
            # Remove níveis muito fracos
            if level_data['strength'] < 0.1 and age_seconds > 86400:  # 24h
                del self.level_cache[asset][level_price]
                
    def _cluster_levels(self, asset: str) -> List[Dict[str, Any]]:
        """Agrupa níveis próximos em clusters"""
        if asset not in self.level_cache:
            return []
            
        # Ordena níveis por preço
        sorted_levels = sorted(
            self.level_cache[asset].items(),
            key=lambda x: x[0]
        )
        
        if not sorted_levels:
            return []
            
        # Clusteriza níveis próximos
        clusters = []
        current_cluster = {
            'levels': [sorted_levels[0]],
            'center_price': sorted_levels[0][0],
            'total_touches': sorted_levels[0][1]['touches'],
            'total_rejections': sorted_levels[0][1]['rejections'],
            'total_volume': sorted_levels[0][1]['volume']
        }
        
        for i in range(1, len(sorted_levels)):
            price, data = sorted_levels[i]
            
            # Verifica se está próximo do cluster atual
            if abs(price - current_cluster['center_price']) <= self.clustering_threshold:
                # Adiciona ao cluster
                current_cluster['levels'].append((price, data))
                current_cluster['total_touches'] += data['touches']
                current_cluster['total_rejections'] += data['rejections']
                current_cluster['total_volume'] += data['volume']
                
                # Recalcula centro
                all_prices = [p for p, _ in current_cluster['levels']]
                current_cluster['center_price'] = sum(all_prices) / len(all_prices)
            else:
                # Finaliza cluster atual e inicia novo
                clusters.append(self._finalize_cluster(current_cluster))
                current_cluster = {
                    'levels': [(price, data)],
                    'center_price': price,
                    'total_touches': data['touches'],
                    'total_rejections': data['rejections'],
                    'total_volume': data['volume']
                }
                
        # Adiciona último cluster
        if current_cluster['levels']:
            clusters.append(self._finalize_cluster(current_cluster))
            
        return clusters
        
    def _finalize_cluster(self, cluster: Dict) -> Dict[str, Any]:
        """Finaliza e calcula métricas do cluster"""
        # Calcula força do cluster
        strength = self._calculate_cluster_strength(cluster)
        
        # Calcula eficácia histórica
        effectiveness = self._calculate_effectiveness(cluster)
        
        return {
            'price': Decimal(str(cluster['center_price'])),
            'touches': cluster['total_touches'],
            'rejections': cluster['total_rejections'],
            'volume': cluster['total_volume'],
            'strength': strength,
            'effectiveness': effectiveness,
            'size': len(cluster['levels']),
            'range': max(p for p, _ in cluster['levels']) - min(p for p, _ in cluster['levels'])
        }
        
    def _calculate_cluster_strength(self, cluster: Dict) -> float:
        """Calcula força do cluster de níveis"""
        touches = cluster['total_touches']
        rejections = cluster['total_rejections']
        volume = cluster['total_volume']
        
        # Normaliza valores
        touch_score = min(1.0, touches / 20)
        rejection_score = min(1.0, rejections / 5)
        volume_score = min(1.0, volume / 5000)
        
        # Peso maior para rejeições
        strength = (
            touch_score * 0.3 +
            rejection_score * 0.5 +
            volume_score * 0.2
        )
        
        # Bonus por múltiplos níveis no cluster
        if len(cluster['levels']) > 1:
            strength *= (1 + 0.1 * min(3, len(cluster['levels']) - 1))
            
        return min(1.0, strength)
        
    def _calculate_effectiveness(self, cluster: Dict) -> float:
        """Calcula eficácia histórica do nível"""
        total_tests = cluster['total_touches']
        successful_defenses = cluster['total_rejections']
        
        # Pega dados de rompimentos
        total_breaks = sum(
            data.get('breaks', 0) 
            for _, data in cluster['levels']
        )
        
        if total_tests == 0:
            return 0.0
            
        # Taxa de defesa bem-sucedida
        defense_rate = successful_defenses / total_tests
        
        # Penaliza por rompimentos
        break_penalty = min(0.5, total_breaks * 0.1)
        
        effectiveness = max(0.0, defense_rate - break_penalty)
        
        return effectiveness
        
    def _analyze_levels_advanced(
        self,
        market_data: MarketData,
        clusters: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Análise avançada de níveis com dados históricos"""
        if not clusters:
            return {'detected': False}
            
        current_price = market_data.trades[-1].price if market_data.trades else Decimal('0')
        
        # Filtra clusters relevantes (até 2% de distância)
        max_distance = current_price * Decimal('0.02')
        relevant_clusters = [
            c for c in clusters
            if abs(c['price'] - current_price) <= max_distance
        ]
        
        if not relevant_clusters:
            return {'detected': False}
            
        # Analisa cluster mais próximo
        closest = min(relevant_clusters, key=lambda c: abs(c['price'] - current_price))
        
        # Determina tipo
        level_type = 'support' if closest['price'] < current_price else 'resistance'
        
        # Verifica atividade recente
        recent_activity = self._check_recent_cluster_activity(
            closest,
            market_data,
            level_type
        )
        
        # Análise do book
        book_analysis = self._analyze_book_support_advanced(
            market_data.book,
            closest['price'],
            level_type
        )
        
        # Análise de volume profile
        volume_profile = self._analyze_volume_profile_at_level(
            closest['price'],
            market_data.asset
        )
        
        # Calcula confiança com dados históricos
        confidence_signals = {
            'cluster_strength': closest['strength'],
            'historical_effectiveness': closest['effectiveness'],
            'recent_activity': recent_activity['score'],
            'book_support': book_analysis['support_score'],
            'volume_concentration': volume_profile['concentration']
        }
        
        confidence = self.calculate_confidence(confidence_signals)
        
        if confidence < self.min_confidence:
            return {'detected': False}
            
        return {
            'detected': True,
            'confidence': confidence,
            'type': level_type,
            'price': closest['price'],
            'touches': closest['touches'],
            'rejections': closest['rejections'],
            'breaks': closest.get('breaks', 0),
            'strength': closest['strength'],
            'historical_strength': closest['effectiveness'],
            'book_support': book_analysis['support_score'],
            'volume_profile': volume_profile,
            'cluster_size': closest['size'],
            'effectiveness': closest['effectiveness']
        }
        
    def _check_recent_cluster_activity(
        self,
        cluster: Dict,
        market_data: MarketData,
        level_type: str
    ) -> Dict[str, Any]:
        """Verifica atividade recente no cluster"""
        recent_trades = market_data.trades
        cluster_range = cluster.get('range', self.clustering_threshold)
        
        # Trades próximos ao cluster
        near_cluster = []
        for trade in recent_trades:
            if abs(trade.price - cluster['price']) <= cluster_range:
                near_cluster.append(trade)
                
        if not near_cluster:
            return {'score': 0.0}
            
        # Analisa defesa do nível
        defending_trades = 0
        for trade in near_cluster:
            if level_type == 'support' and trade.aggressor == Side.BUY:
                defending_trades += 1
            elif level_type == 'resistance' and trade.aggressor == Side.SELL:
                defending_trades += 1
                
        defense_ratio = defending_trades / len(near_cluster) if near_cluster else 0
        
        # Score baseado em atividade e defesa
        activity_score = min(1.0, len(near_cluster) / 10) * defense_ratio
        
        return {
            'score': activity_score,
            'trades': len(near_cluster),
            'defense_ratio': defense_ratio
        }
        
    def _analyze_book_support_advanced(
        self,
        book: OrderBook,
        level_price: Decimal,
        level_type: str
    ) -> Dict[str, Any]:
        """Análise avançada do suporte no book"""
        if not book or not book.bids or not book.asks:
            return {'support_score': 0.5}
            
        # Seleciona lado relevante
        if level_type == 'support':
            relevant_levels = book.bids[:10]
        else:
            relevant_levels = book.asks[:10]
            
        # Análise de profundidade
        depth_analysis = self._analyze_order_depth(
            relevant_levels,
            level_price
        )
        
        # Detecta "escoras" (ordens muito grandes)
        walls = self._detect_order_walls(relevant_levels)
        
        # Calcula score final
        support_score = (
            depth_analysis['concentration'] * 0.5 +
            depth_analysis['proximity'] * 0.3 +
            (0.2 if walls else 0.0)
        )
        
        return {
            'support_score': support_score,
            'depth': depth_analysis,
            'has_walls': bool(walls),
            'wall_positions': walls
        }
        
    def _analyze_order_depth(
        self,
        levels: List,
        target_price: Decimal
    ) -> Dict[str, Any]:
        """Analisa profundidade de ordens"""
        if not levels:
            return {'concentration': 0.0, 'proximity': 0.0}
            
        total_volume = sum(level.volume for level in levels)
        
        # Volume próximo ao nível
        near_volume = sum(
            level.volume for level in levels
            if abs(level.price - target_price) <= self.level_tolerance
        )
        
        concentration = near_volume / total_volume if total_volume > 0 else 0
        
        # Proximidade média ponderada
        weighted_distance = sum(
            abs(level.price - target_price) * level.volume
            for level in levels
        )
        
        avg_distance = weighted_distance / total_volume if total_volume > 0 else float('inf')
        proximity = max(0, 1 - (float(avg_distance) / 10))
        
        return {
            'concentration': concentration,
            'proximity': proximity,
            'total_volume': total_volume,
            'near_volume': near_volume
        }
        
    def _detect_order_walls(self, levels: List) -> List[Decimal]:
        """Detecta "muralhas" de ordens"""
        if not levels:
            return []
            
        avg_volume = sum(l.volume for l in levels) / len(levels)
        
        # Ordens 3x maiores que a média
        walls = [
            level.price for level in levels
            if avg_volume > 0 and level.volume > avg_volume * 3
        ]
        
        return walls
        
    def _analyze_volume_profile_at_level(
        self,
        level_price: Decimal,
        asset: str
    ) -> Dict[str, Any]:
        """Analisa perfil de volume no nível"""
        if asset not in self.level_cache:
            return {'concentration': 0.0}
            
        # Coleta volume em range ao redor do nível
        range_size = self.clustering_threshold
        volume_in_range = 0
        total_market_volume = 0
        
        for price, data in self.level_cache[asset].items():
            total_market_volume += data['volume']
            
            if abs(price - level_price) <= range_size:
                volume_in_range += data['volume']
                
        concentration = volume_in_range / total_market_volume if total_market_volume > 0 else 0
        
        return {
            'concentration': concentration,
            'volume_in_range': volume_in_range,
            'total_volume': total_market_volume
        }
        
    def _persist_level_event(self, asset: str, level_result: Dict):
        """Persiste evento de nível no banco de dados"""
        if not self.persistence_enabled:
            return
            
        with self.db_lock:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            try:
                # Verifica se nível existe
                cursor.execute('''
                    SELECT id FROM price_levels
                    WHERE asset = ? AND ABS(price - ?) < ?
                ''', (asset, float(level_result['price']), float(self.level_tolerance)))
                
                row = cursor.fetchone()
                
                if row:
                    level_id = row[0]
                    # Atualiza nível existente
                    cursor.execute('''
                        UPDATE price_levels
                        SET touches = ?, rejections = ?, successful_breaks = ?,
                            total_volume = ?, strength_score = ?, last_touch = ?
                        WHERE id = ?
                    ''', (
                        level_result['touches'],
                        level_result['rejections'],
                        level_result.get('breaks', 0),
                        level_result.get('volume', 0),
                        level_result['strength'],
                        datetime.now().isoformat(),
                        level_id
                    ))
                else:
                    # Cria novo nível
                    cursor.execute('''
                        INSERT INTO price_levels
                        (asset, price, first_touch, last_touch, touches, rejections,
                         successful_breaks, total_volume, strength_score)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        asset,
                        float(level_result['price']),
                        datetime.now().isoformat(),
                        datetime.now().isoformat(),
                        level_result['touches'],
                        level_result['rejections'],
                        level_result.get('breaks', 0),
                        level_result.get('volume', 0),
                        level_result['strength']
                    ))
                    level_id = cursor.lastrowid
                    
                # Registra evento
                cursor.execute('''
                    INSERT INTO level_events
                    (level_id, event_type, timestamp, price, metadata)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    level_id,
                    'detection',
                    datetime.now().isoformat(),
                    float(level_result['price']),
                    json.dumps({
                        'confidence': level_result['confidence'],
                        'type': level_result['type'],
                        'book_support': level_result['book_support']
                    })
                ))
                
                conn.commit()
                
            except Exception as e:
                self.logger.error(f"Erro ao persistir evento: {e}")
                conn.rollback()
            finally:
                conn.close()
                
    def get_historical_statistics(self, asset: str, days: int = 30) -> Dict[str, Any]:
        """Retorna estatísticas históricas dos níveis"""
        if not self.persistence_enabled:
            return {}
            
        with self.db_lock:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cutoff = datetime.now() - timedelta(days=days)
            
            # Estatísticas gerais
            cursor.execute('''
                SELECT 
                    COUNT(DISTINCT price) as total_levels,
                    SUM(touches) as total_touches,
                    SUM(rejections) as total_rejections,
                    SUM(successful_breaks) as total_breaks,
                    AVG(strength_score) as avg_strength
                FROM price_levels
                WHERE asset = ? AND last_touch > ?
            ''', (asset, cutoff.isoformat()))
            
            stats = cursor.fetchone()
            
            # Níveis mais fortes
            cursor.execute('''
                SELECT price, strength_score, touches, rejections
                FROM price_levels
                WHERE asset = ? AND last_touch > ?
                ORDER BY strength_score DESC
                LIMIT 10
            ''', (asset, cutoff.isoformat()))
            
            strongest_levels = [
                {
                    'price': row[0],
                    'strength': row[1],
                    'touches': row[2],
                    'rejections': row[3]
                }
                for row in cursor.fetchall()
            ]
            
            conn.close()
            
            return {
                'total_levels': stats[0] or 0,
                'total_touches': stats[1] or 0,
                'total_rejections': stats[2] or 0,
                'total_breaks': stats[3] or 0,
                'average_strength': stats[4] or 0,
                'strongest_levels': strongest_levels
            }
            
    def _round_to_level(self, price: Decimal) -> Decimal:
        """Arredonda preço para nível"""
        return (price / self.level_tolerance).quantize(Decimal('1')) * self.level_tolerance