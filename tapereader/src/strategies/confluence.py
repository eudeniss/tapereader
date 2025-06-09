"""
Sistema de Análise de Confluência - Versão Final Otimizada
Analisa correlação e confirmação entre DOLFUT e WDOFUT
Versão 3.2 - Usando módulo centralizado de estatísticas
Performance melhorada com funções unificadas
REATORADO: A direção do comportamento é lida diretamente para maior clareza.
"""

from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
import statistics
import logging
import time
import numpy as np

from ..core.models import BehaviorDetection, MarketData, Side

# OTIMIZAÇÃO: Importa funções do módulo centralizado de estatísticas
from ..utils.statistics import (
    calculate_correlation,
    calculate_returns,
    calculate_trend,
    calculate_volatility,
    calculate_sma,
    calculate_ema
)


class ConfluenceLevel(str, Enum):
    """Níveis de confluência entre ativos"""
    PREMIUM = "PREMIUM"      # 90%+ - Ambos confirmam fortemente
    STRONG = "STRONG"        # 85-89% - Líder forte + seguidor
    STANDARD = "STANDARD"    # 80-84% - Confirmação básica
    WEAK = "WEAK"           # <80% - Sem confluência clara


class ConfluenceAnalyzer:
    """
    Analisa confluência entre DOLFUT e WDOFUT
    Aumenta confiança quando ambos ativos confirmam o sinal
    Versão 3.2 - Otimizada com módulo centralizado
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Parâmetros do config
        self.correlation_window = config.get('correlation_window', 30)  # minutos
        self.min_correlation = config.get('min_correlation', 0.6)
        self.leader_threshold = config.get('leader_threshold', 0.7)
        self.confluence_boost = config.get('confluence_boost', 0.15)  # Boost máximo
        
        # Parâmetros adicionais
        self.price_sync_tolerance = config.get('price_sync_tolerance', 60)  # segundos
        self.min_data_points = config.get('min_data_points', 10)
        self.leadership_window = config.get('leadership_window', 10)  # últimos N padrões
        self.enable_volume_profile = config.get('enable_volume_profile', True)
        
        # Thresholds de confluência customizáveis
        self.confluence_thresholds = config.get('confluence_thresholds', {
            'premium': 0.90,
            'strong': 0.85,
            'standard': 0.80
        })
        
        # Cache de dados históricos
        self.price_history = {
            'DOLFUT': [],
            'WDOFUT': []
        }
        
        # Cache de comportamentos
        self.behavior_history = {
            'DOLFUT': [],
            'WDOFUT': []
        }
        
        # Análise de liderança
        self.leadership_stats = {
            'DOLFUT_leads': 0,
            'WDOFUT_leads': 0,
            'simultaneous': 0
        }
        
        # Estatísticas adicionais
        self.correlation_history = []
        self.confluence_results = []
        
        # OTIMIZAÇÃO: Cache de correlações com TTL
        self.correlation_cache = {}
        self.cache_ttl = config.get('correlation_cache_ttl', 5)  # segundos
        
        # OTIMIZAÇÃO: Pré-alocação de arrays NumPy
        self.max_history_size = config.get('max_history_size', 1000)
        
        # Volume Profile cache
        self.volume_profile_cache = {}
        self.volume_profile_ttl = config.get('volume_profile_ttl', 30)  # segundos
        
        # Estatísticas de performance
        self._cache_stats = {'hits': 0, 'misses': 0}
        
        self.logger.info(
            f"ConfluenceAnalyzer v3.2 inicializado - "
            f"Janela: {self.correlation_window}min, "
            f"Correlação mínima: {self.min_correlation:.1%}, "
            f"Cache TTL: {self.cache_ttl}s, "
            f"Volume Profile: {'Habilitado' if self.enable_volume_profile else 'Desabilitado'}"
        )
        
    def update_market_data(self, asset: str, market_data: MarketData):
        """Atualiza histórico de preços"""
        if asset not in self.price_history:
            self.logger.warning(f"Ativo desconhecido: {asset}")
            return
            
        # OTIMIZAÇÃO: Cálculo vetorizado de preço médio
        if market_data.trades:
            prices = np.array([float(t.price) for t in market_data.trades])
            volumes = np.array([t.volume for t in market_data.trades])
            
            avg_price = np.average(prices, weights=volumes) if volumes.sum() > 0 else prices.mean()
            volume = volumes.sum()
            
            # Adiciona ao histórico
            self.price_history[asset].append({
                'timestamp': market_data.timestamp,
                'price': float(avg_price),
                'volume': float(volume),
                'high': float(prices.max()),
                'low': float(prices.min()),
                'trades': len(market_data.trades)
            })
            
            # Mantém janela de tempo
            cutoff = datetime.now() - timedelta(minutes=self.correlation_window)
            self.price_history[asset] = [
                p for p in self.price_history[asset] 
                if p['timestamp'] > cutoff
            ]
            
            # OTIMIZAÇÃO: Invalida caches quando dados mudam
            self._invalidate_caches()
            
            # Log periódico do tamanho do histórico
            if len(self.price_history[asset]) % 50 == 0:
                self.logger.debug(
                    f"Histórico {asset}: {len(self.price_history[asset])} pontos"
                )
            
    def update_behaviors(self, asset: str, behaviors: List[BehaviorDetection]):
        """Atualiza histórico de comportamentos"""
        if asset not in self.behavior_history:
            return
            
        for behavior in behaviors:
            self.behavior_history[asset].append({
                'timestamp': behavior.timestamp,
                'type': behavior.behavior_type,
                'confidence': behavior.confidence,
                'metadata': behavior.metadata,
                'direction': getattr(behavior, 'direction', None) # Adiciona a direção
            })
            
        # Mantém janela de tempo (5 minutos para comportamentos)
        behavior_window = self.config.get('behavior_window', 5)
        cutoff = datetime.now() - timedelta(minutes=behavior_window)
        
        self.behavior_history[asset] = [
            b for b in self.behavior_history[asset]
            if b['timestamp'] > cutoff
        ]
        
    def analyze_confluence(
        self,
        primary_asset: str,
        primary_behaviors: List[BehaviorDetection],
        secondary_behaviors: List[BehaviorDetection]
    ) -> Dict[str, Any]:
        """
        Analisa confluência entre os ativos
        Versão completa com Volume Profile
        
        Returns:
            Dict com análise de confluência e boost de confiança
        """
        # Determina ativo secundário
        secondary_asset = 'WDOFUT' if primary_asset == 'DOLFUT' else 'DOLFUT'
        
        # Calcula correlação de preços (com cache)
        correlation = self._calculate_price_correlation()
        
        # Adiciona à histórico
        self.correlation_history.append({
            'timestamp': datetime.now(),
            'correlation': correlation
        })
        
        # Mantém apenas últimas 100 correlações
        if len(self.correlation_history) > 100:
            self.correlation_history = self.correlation_history[-100:]
        
        # Analisa comportamentos coincidentes
        behavior_confluence = self._analyze_behavior_confluence(
            primary_behaviors,
            secondary_behaviors
        )
        
        # Determina líder/seguidor
        leadership = self._analyze_leadership(primary_asset, secondary_asset)
        
        # NOVO: Análise de Volume Profile
        volume_profile_analysis = {}
        if self.enable_volume_profile:
            volume_profile_analysis = self._analyze_volume_profile_confluence(
                primary_asset,
                secondary_asset
            )
        
        # Calcula nível de confluência (agora inclui Volume Profile)
        confluence_level = self._determine_confluence_level(
            correlation,
            behavior_confluence,
            leadership,
            volume_profile_analysis
        )
        
        # Calcula boost de confiança
        confidence_boost = self._calculate_confidence_boost(
            confluence_level,
            behavior_confluence,
            volume_profile_analysis
        )
        
        # Análise de divergência
        divergence = self._check_divergence(
            primary_behaviors,
            secondary_behaviors,
            correlation
        )
        
        # Análise adicional de qualidade
        quality_metrics = self._calculate_quality_metrics()
        
        result = {
            'level': confluence_level,
            'correlation': correlation,
            'behavior_match': behavior_confluence['match_score'],
            'matching_behaviors': behavior_confluence['matching'],
            'conflicting_behaviors': behavior_confluence['conflicting'],
            'leader': leadership['leader'],
            'leader_confidence': leadership['confidence'],
            'confidence_boost': confidence_boost,
            'has_divergence': divergence['detected'],
            'divergence_type': divergence.get('type'),
            'volume_profile': volume_profile_analysis,
            'quality_metrics': quality_metrics,
            'recommendation': self._generate_recommendation(
                confluence_level,
                divergence,
                leadership,
                volume_profile_analysis
            )
        }
        
        # Armazena resultado
        self.confluence_results.append({
            'timestamp': datetime.now(),
            'result': result
        })
        
        # Mantém apenas últimos 50 resultados
        if len(self.confluence_results) > 50:
            self.confluence_results = self.confluence_results[-50:]
        
        return result
        
    def _calculate_price_correlation(self) -> float:
        """
        Calcula correlação entre os preços dos ativos
        OTIMIZAÇÃO: Usa função centralizada
        """
        # OTIMIZAÇÃO: Verifica cache primeiro
        cache_key = f"{len(self.price_history['DOLFUT'])}_{len(self.price_history['WDOFUT'])}"
        
        if cache_key in self.correlation_cache:
            cached_time, cached_value = self.correlation_cache[cache_key]
            if time.time() - cached_time < self.cache_ttl:
                self._cache_stats['hits'] += 1
                self.logger.debug(f"Correlação do cache: {cached_value:.3f}")
                return cached_value
        
        self._cache_stats['misses'] += 1
        
        # Verifica dados mínimos
        dol_len = len(self.price_history['DOLFUT'])
        wdo_len = len(self.price_history['WDOFUT'])
        
        if dol_len < self.min_data_points or wdo_len < self.min_data_points:
            self.logger.debug(
                f"Dados insuficientes para correlação: "
                f"DOL={dol_len}, WDO={wdo_len}"
            )
            return 0.0
            
        # Sincroniza séries por timestamp
        synced_data = self._sync_price_series_optimized()
        
        if len(synced_data) < self.min_data_points:
            return 0.0
            
        try:
            # OTIMIZAÇÃO: Extrai arrays NumPy diretamente
            dol_prices = np.array([d['dol'] for d in synced_data], dtype=np.float64)
            wdo_prices = np.array([d['wdo'] for d in synced_data], dtype=np.float64)
            
            # OTIMIZAÇÃO: Calcula retornos usando função centralizada
            dol_returns = calculate_returns(dol_prices, method='simple')
            wdo_returns = calculate_returns(wdo_prices, method='simple')
            
            # OTIMIZAÇÃO: Usa função centralizada de correlação
            if len(dol_returns) > 1:
                correlation = calculate_correlation(dol_returns, wdo_returns, method='pearson')
                
                # Armazena no cache
                self.correlation_cache[cache_key] = (time.time(), float(correlation))
                
                return float(correlation)
                
        except Exception as e:
            self.logger.error(f"Erro ao calcular correlação: {e}")
            
        return 0.0
        
    def _sync_price_series_optimized(self) -> List[Dict[str, float]]:
        """
        Sincroniza séries de preços por timestamp
        OTIMIZAÇÃO: Usa busca binária para matching mais rápido
        """
        synced = []
        
        # OTIMIZAÇÃO: Converte para arrays numpy para operações mais rápidas
        dol_data = self.price_history['DOLFUT']
        wdo_data = self.price_history['WDOFUT']
        
        if not dol_data or not wdo_data:
            return synced
            
        # OTIMIZAÇÃO: Cria índice de timestamps do WDO para busca rápida
        wdo_timestamps = np.array([
            w['timestamp'].timestamp() for w in wdo_data
        ])
        
        for dol_point in dol_data:
            dol_ts = dol_point['timestamp'].timestamp()
            
            # OTIMIZAÇÃO: Busca binária do timestamp mais próximo
            idx = np.searchsorted(wdo_timestamps, dol_ts)
            
            # Verifica vizinhos
            candidates = []
            
            if idx > 0:
                time_diff = abs(dol_ts - wdo_timestamps[idx - 1])
                if time_diff < self.price_sync_tolerance:
                    candidates.append((idx - 1, time_diff))
                    
            if idx < len(wdo_timestamps):
                time_diff = abs(dol_ts - wdo_timestamps[idx])
                if time_diff < self.price_sync_tolerance:
                    candidates.append((idx, time_diff))
                    
            # Escolhe o mais próximo
            if candidates:
                best_idx = min(candidates, key=lambda x: x[1])[0]
                wdo_point = wdo_data[best_idx]
                
                synced.append({
                    'timestamp': dol_point['timestamp'],
                    'dol': dol_point['price'],
                    'wdo': wdo_point['price'],
                    'dol_volume': dol_point['volume'],
                    'wdo_volume': wdo_point['volume']
                })
                
        return synced
        
    def _analyze_volume_profile_confluence(self, primary_asset: str, secondary_asset: str) -> Dict[str, Any]:
        """
        Analisa confluência de níveis de Volume Profile entre ativos
        POC (Point of Control), VAH (Value Area High), VAL (Value Area Low)
        """
        # Verifica cache primeiro
        cache_key = f"vp_{len(self.price_history[primary_asset])}_{len(self.price_history[secondary_asset])}"
        
        if cache_key in self.volume_profile_cache:
            cached_time, cached_value = self.volume_profile_cache[cache_key]
            if time.time() - cached_time < self.volume_profile_ttl:
                return cached_value
        
        # Obter dados sincronizados
        synced_data = self._sync_price_series_optimized()
        
        if len(synced_data) < 20:  # Mínimo de dados para volume profile
            return {
                'aligned': False,
                'poc_alignment': 0.0,
                'value_area_overlap': 0.0,
                'volume_concentration_similarity': 0.0
            }
        
        # Extrair arrays numpy para cálculo eficiente
        dol_prices = np.array([d['dol'] for d in synced_data])
        dol_volumes = np.array([d['dol_volume'] for d in synced_data])
        wdo_prices = np.array([d['wdo'] for d in synced_data])
        wdo_volumes = np.array([d['wdo_volume'] for d in synced_data])
        
        # Calcular Volume Profile para cada ativo
        dol_profile = self._calculate_volume_profile(dol_prices, dol_volumes)
        wdo_profile = self._calculate_volume_profile(wdo_prices, wdo_volumes)
        
        # Comparar POCs (Point of Control)
        poc_diff_pct = abs(dol_profile['poc'] - wdo_profile['poc']) / dol_profile['poc'] if dol_profile['poc'] != 0 else 1.0
        poc_aligned = poc_diff_pct < 0.001  # 0.1% de diferença
        
        # Calcular overlap das Value Areas
        va_overlap = self._calculate_value_area_overlap(
            dol_profile['val'], dol_profile['vah'],
            wdo_profile['val'], wdo_profile['vah']
        )
        
        # Analisar concentração de volume
        volume_concentration_similarity = 1.0 - abs(
            dol_profile['concentration'] - wdo_profile['concentration']
        )
        
        result = {
            'aligned': poc_aligned and va_overlap > 0.7,
            'poc_alignment': 1.0 - poc_diff_pct,
            'value_area_overlap': va_overlap,
            'volume_concentration_similarity': volume_concentration_similarity,
            'profiles': {
                primary_asset: dol_profile,
                secondary_asset: wdo_profile
            }
        }
        
        # Armazena no cache
        self.volume_profile_cache[cache_key] = (time.time(), result)
        
        return result

    def _calculate_volume_profile(self, prices: np.ndarray, volumes: np.ndarray, bins: int = 20) -> Dict[str, float]:
        """
        Calcula o Volume Profile usando NumPy para performance
        """
        # Criar bins de preço
        price_min, price_max = prices.min(), prices.max()
        price_bins = np.linspace(price_min, price_max, bins + 1)
        
        # Acumular volume por bin
        volume_by_bin = np.zeros(bins)
        
        for i in range(len(prices)):
            bin_idx = np.searchsorted(price_bins[1:], prices[i])
            if bin_idx < bins:
                volume_by_bin[bin_idx] += volumes[i]
        
        # Encontrar POC (bin com maior volume)
        poc_idx = np.argmax(volume_by_bin)
        poc_price = (price_bins[poc_idx] + price_bins[poc_idx + 1]) / 2
        
        # Calcular Value Area (70% do volume)
        total_volume = volume_by_bin.sum()
        target_volume = total_volume * 0.7
        
        # Expandir do POC até atingir 70% do volume
        accumulated_volume = volume_by_bin[poc_idx]
        lower_idx, upper_idx = poc_idx, poc_idx
        
        while accumulated_volume < target_volume:
            # Expandir para o lado com mais volume
            expand_lower = lower_idx > 0
            expand_upper = upper_idx < bins - 1
            
            if expand_lower and expand_upper:
                if volume_by_bin[lower_idx - 1] > volume_by_bin[upper_idx + 1]:
                    lower_idx -= 1
                    accumulated_volume += volume_by_bin[lower_idx]
                else:
                    upper_idx += 1
                    accumulated_volume += volume_by_bin[upper_idx]
            elif expand_lower:
                lower_idx -= 1
                accumulated_volume += volume_by_bin[lower_idx]
            elif expand_upper:
                upper_idx += 1
                accumulated_volume += volume_by_bin[upper_idx]
            else:
                break
        
        # Value Area Low e High
        val = (price_bins[lower_idx] + price_bins[lower_idx + 1]) / 2
        vah = (price_bins[upper_idx] + price_bins[upper_idx + 1]) / 2
        
        # Concentração de volume (quanto do volume está na value area)
        concentration = accumulated_volume / total_volume if total_volume > 0 else 0
        
        return {
            'poc': poc_price,
            'val': val,
            'vah': vah,
            'concentration': concentration,
            'total_volume': total_volume
        }

    def _calculate_value_area_overlap(self, val1: float, vah1: float, val2: float, vah2: float) -> float:
        """
        Calcula o percentual de overlap entre duas Value Areas
        """
        # Encontrar a interseção
        overlap_start = max(val1, val2)
        overlap_end = min(vah1, vah2)
        
        if overlap_start >= overlap_end:
            return 0.0  # Sem overlap
        
        # Calcular tamanhos
        size1 = vah1 - val1
        size2 = vah2 - val2
        overlap_size = overlap_end - overlap_start
        
        # Percentual de overlap (média do overlap relativo a cada value area)
        overlap_pct1 = overlap_size / size1 if size1 > 0 else 0
        overlap_pct2 = overlap_size / size2 if size2 > 0 else 0
        
        return (overlap_pct1 + overlap_pct2) / 2
        
    def _invalidate_caches(self):
        """Invalida todos os caches quando dados mudam"""
        self.correlation_cache.clear()
        self.volume_profile_cache.clear()
        
    def _analyze_behavior_confluence(
        self,
        primary: List[BehaviorDetection],
        secondary: List[BehaviorDetection]
    ) -> Dict[str, Any]:
        """Analisa confluência de comportamentos lendo a direção explícita."""
        primary_types = {b.behavior_type: b for b in primary}
        secondary_types = {b.behavior_type: b for b in secondary}
        
        matching = []
        conflicting = []
        
        # Verifica comportamentos coincidentes
        for behavior_type, primary_detection in primary_types.items():
            if behavior_type in secondary_types:
                secondary_detection = secondary_types[behavior_type]
                
                # Verifica se apontam na mesma direção lendo o atributo 'direction'
                primary_dir = getattr(primary_detection, 'direction', None)
                secondary_dir = getattr(secondary_detection, 'direction', None)
                
                if primary_dir and secondary_dir and primary_dir != Side.NEUTRAL:
                    if primary_dir == secondary_dir:
                        matching.append({
                            'type': behavior_type,
                            'confidence': (
                                primary_detection.confidence + 
                                secondary_detection.confidence
                            ) / 2,
                            'primary_confidence': primary_detection.confidence,
                            'secondary_confidence': secondary_detection.confidence
                        })
                    else:
                        conflicting.append({
                            'type': behavior_type,
                            'primary_direction': primary_dir.value,
                            'secondary_direction': secondary_dir.value
                        })
                        
        # Calcula score de match
        total_behaviors = len(set(list(primary_types.keys()) + list(secondary_types.keys())))
        match_score = len(matching) / total_behaviors if total_behaviors > 0 else 0
        
        # Penaliza por conflitos
        conflict_penalty = len(conflicting) * 0.1
        adjusted_score = max(0, match_score - conflict_penalty)
        
        return {
            'matching': matching,
            'conflicting': conflicting,
            'match_score': adjusted_score,
            'total_primary': len(primary),
            'total_secondary': len(secondary),
            'overlap_ratio': len(matching) / min(len(primary), len(secondary)) if min(len(primary), len(secondary)) > 0 else 0
        }
        
    def _analyze_leadership(self, primary_asset: str, secondary_asset: str) -> Dict[str, Any]:
        """Analisa qual ativo está liderando o movimento"""
        # Analisa movimentos recentes
        recent_moves = self._detect_leadership_patterns_optimized()
        
        if not recent_moves:
            return {
                'leader': 'unclear',
                'confidence': 0.0,
                'lag_seconds': 0,
                'pattern_count': 0
            }
            
        # Usa apenas padrões recentes
        window_moves = recent_moves[-self.leadership_window:]
        
        # Conta liderança
        primary_leads = sum(1 for m in window_moves if m['leader'] == primary_asset)
        secondary_leads = sum(1 for m in window_moves if m['leader'] == secondary_asset)
        
        # Atualiza estatísticas globais
        self.leadership_stats[f'{primary_asset}_leads'] += primary_leads
        self.leadership_stats[f'{secondary_asset}_leads'] += secondary_leads
        
        # Determina líder atual
        total_moves = len(window_moves)
        
        if primary_leads > secondary_leads * 1.5:
            leader = primary_asset
            confidence = primary_leads / total_moves
        elif secondary_leads > primary_leads * 1.5:
            leader = secondary_asset
            confidence = secondary_leads / total_moves
        else:
            leader = 'simultaneous'
            confidence = 0.5
            self.leadership_stats['simultaneous'] += 1
            
        # Calcula lag médio
        lags = [m['lag'] for m in window_moves if m['leader'] in [primary_asset, secondary_asset]]
        avg_lag = sum(lags) / len(lags) if lags else 0
        
        # NOVO: Análise de qualidade de liderança
        leadership_quality = self._analyze_leadership_quality(window_moves)
        
        return {
            'leader': leader,
            'confidence': confidence,
            'lag_seconds': avg_lag,
            'pattern_count': len(window_moves),
            'primary_leads': primary_leads,
            'secondary_leads': secondary_leads,
            'quality_metrics': leadership_quality
        }
        
    def _detect_leadership_patterns_optimized(self) -> List[Dict[str, Any]]:
        """
        Detecta padrões de liderança de forma mais eficiente.
        OTIMIZAÇÃO FINAL: Une e ordena os movimentos para evitar loops aninhados.
        Complexidade: O(K log K) ao invés de O(N * M)
        """
        patterns = []
        
        # Precisa de dados suficientes
        min_points = self.config.get('min_leadership_points', 5)
        
        if (len(self.price_history['DOLFUT']) < min_points or 
            len(self.price_history['WDOFUT']) < min_points):
            return patterns
        
        # 1. Encontra movimentos significativos para ambos os ativos
        dol_moves = self._find_significant_moves_optimized('DOLFUT')
        wdo_moves = self._find_significant_moves_optimized('WDOFUT')
        
        if not dol_moves or not wdo_moves:
            return patterns
        
        # 2. Une as duas listas de movimentos
        all_moves = dol_moves + wdo_moves
        
        # 3. Ordena a lista combinada por timestamp (O(K log K))
        all_moves.sort(key=lambda x: x['timestamp'])
        
        # Parâmetros de configuração
        max_lag = self.config.get('max_leadership_lag', 120)
        
        # 4. Itera uma única vez na lista ordenada (O(K))
        i = 0
        while i < len(all_moves):
            current_move = all_moves[i]
            
            # Procura por movimentos correlacionados dentro da janela de tempo
            j = i + 1
            while j < len(all_moves):
                next_move = all_moves[j]
                
                # Calcula diferença de tempo
                time_diff = (next_move['timestamp'] - current_move['timestamp']).total_seconds()
                
                # Se passou do lag máximo, não precisa verificar mais
                if time_diff > max_lag:
                    break
                
                # Verifica se são de ativos diferentes e mesma direção
                if (current_move['asset'] != next_move['asset'] and 
                    current_move['direction'] == next_move['direction']):
                    
                    # Encontrou um par líder/seguidor
                    patterns.append({
                        'leader': current_move['asset'],
                        'follower': next_move['asset'],
                        'lag': time_diff,
                        'direction': current_move['direction'],
                        'leader_magnitude': current_move['magnitude'],
                        'follower_magnitude': next_move['magnitude'],
                        'combined_magnitude': (current_move['magnitude'] + next_move['magnitude']) / 2,
                        'leader_volume': current_move['volume'],
                        'follower_volume': next_move['volume']
                    })
                    
                    # OTIMIZAÇÃO: Pula o movimento seguidor para evitar duplicatas
                    if j == i + 1:
                        i = j
                        
                j += 1
                
            i += 1
        
        # ANÁLISE ADICIONAL: Calcula estatísticas de qualidade dos padrões
        if patterns:
            # Agrupa por direção para análise
            buy_patterns = [p for p in patterns if p['direction'] == Side.BUY]
            sell_patterns = [p for p in patterns if p['direction'] == Side.SELL]
            
            # Calcula lag médio por direção
            avg_lag_buy = np.mean([p['lag'] for p in buy_patterns]) if buy_patterns else 0
            avg_lag_sell = np.mean([p['lag'] for p in sell_patterns]) if sell_patterns else 0
            
            # Log de estatísticas (apenas em debug)
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(
                    f"Padrões de liderança detectados: {len(patterns)} "
                    f"(BUY: {len(buy_patterns)}, SELL: {len(sell_patterns)}) "
                    f"Lag médio - BUY: {avg_lag_buy:.1f}s, SELL: {avg_lag_sell:.1f}s"
                )
        
        # Retorna os últimos 50 padrões (mais recentes são mais relevantes)
        return patterns[-50:]
        
    def _find_significant_moves_optimized(self, asset: str) -> List[Dict[str, Any]]:
        """
        Encontra movimentos significativos no preço
        OTIMIZAÇÃO: Usa NumPy para detecção vetorizada
        ATUALIZADO: Inclui campo 'asset' para otimização de liderança
        """
        moves = []
        history = self.price_history[asset]
        
        if len(history) < 3:
            return moves
            
        # Threshold de movimento significativo
        thresholds = self.config.get('significant_move_thresholds', {
            'DOLFUT': 0.5,
            'WDOFUT': 1.0
        })
        
        threshold = thresholds.get(asset, 0.5)
        
        # OTIMIZAÇÃO: Extrai arrays numpy
        prices = np.array([h['price'] for h in history])
        volumes = np.array([h['volume'] for h in history])
        
        # Calcula mudanças de preço (janela de 2 períodos)
        if len(prices) >= 3:
            price_changes = prices[2:] - prices[:-2]
            significant_mask = np.abs(price_changes) >= threshold
            
            # Encontra índices de movimentos significativos
            significant_indices = np.where(significant_mask)[0] + 2
            
            for idx in significant_indices:
                moves.append({
                    'timestamp': history[idx]['timestamp'],
                    'asset': asset,  # NOVO: Inclui o ativo
                    'direction': Side.BUY if price_changes[idx-2] > 0 else Side.SELL,
                    'magnitude': abs(price_changes[idx-2]),
                    'volume': volumes[idx]
                })
                
        return moves
        
    def _analyze_leadership_quality(self, patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analisa a qualidade e consistência dos padrões de liderança
        """
        if not patterns:
            return {
                'quality_score': 0.0,
                'consistency': 0.0,
                'dominant_leader': None
            }
        
        # Conta liderança por ativo
        leader_counts = {'DOLFUT': 0, 'WDOFUT': 0}
        lags = []
        
        for pattern in patterns:
            leader_counts[pattern['leader']] += 1
            lags.append(pattern['lag'])
        
        # Calcula métricas
        total_patterns = len(patterns)
        dol_ratio = leader_counts['DOLFUT'] / total_patterns
        wdo_ratio = leader_counts['WDOFUT'] / total_patterns
        
        # Consistência: quão claro é o líder (0-1, onde 1 é sempre o mesmo líder)
        consistency = abs(dol_ratio - wdo_ratio)
        
        # Qualidade: baseada na consistência do lag
        lag_std = np.std(lags) if len(lags) > 1 else 0
        lag_quality = 1.0 / (1.0 + lag_std / 30)  # Normaliza por 30 segundos
        
        # Score geral de qualidade
        quality_score = (consistency + lag_quality) / 2
        
        # Determina líder dominante
        if dol_ratio > 0.6:
            dominant_leader = 'DOLFUT'
        elif wdo_ratio > 0.6:
            dominant_leader = 'WDOFUT'
        else:
            dominant_leader = None
        
        return {
            'quality_score': round(quality_score, 3),
            'consistency': round(consistency, 3),
            'dominant_leader': dominant_leader,
            'lag_stability': round(lag_quality, 3),
            'avg_lag': round(np.mean(lags), 1) if lags else 0,
            'lag_std': round(lag_std, 1)
        }
        
    def _determine_confluence_level(
        self,
        correlation: float,
        behavior_confluence: Dict[str, Any],
        leadership: Dict[str, Any],
        volume_profile: Dict[str, Any]
    ) -> ConfluenceLevel:
        """Determina nível de confluência incluindo Volume Profile e qualidade de liderança"""
        # Score base na correlação
        if correlation > 0.8:
            base_score = 0.9
        elif correlation > self.min_correlation:
            base_score = 0.8
        elif correlation > 0.4:
            base_score = 0.7
        else:
            base_score = 0.5
            
        # Ajusta por comportamentos
        behavior_score = behavior_confluence['match_score']
        
        # Ajusta por liderança clara
        if leadership['confidence'] > self.leader_threshold:
            leadership_bonus = 0.05
        else:
            leadership_bonus = 0.0
            
        # NOVO: Bonus por qualidade de liderança
        if 'quality_metrics' in leadership:
            quality_score = leadership['quality_metrics'].get('quality_score', 0)
            leadership_bonus += quality_score * 0.03
            
        # Bonus por overlap alto
        overlap_bonus = 0.0
        if behavior_confluence['overlap_ratio'] > 0.8:
            overlap_bonus = 0.05
            
        # NOVO: Bonus por Volume Profile alinhado
        volume_profile_bonus = 0.0
        if volume_profile and self.enable_volume_profile:
            if volume_profile.get('aligned', False):
                volume_profile_bonus = 0.10
            elif volume_profile.get('value_area_overlap', 0) > 0.7:
                volume_profile_bonus = 0.05
            
        # Score final
        final_score = (
            base_score * 0.4 + 
            behavior_score * 0.3 + 
            leadership_bonus + 
            overlap_bonus +
            volume_profile_bonus * 0.3  # Volume Profile tem peso significativo
        )
        
        # Usa thresholds customizados
        thresholds = self.confluence_thresholds
        
        # Determina nível
        if final_score >= thresholds.get('premium', 0.90):
            return ConfluenceLevel.PREMIUM
        elif final_score >= thresholds.get('strong', 0.85):
            return ConfluenceLevel.STRONG
        elif final_score >= thresholds.get('standard', 0.80):
            return ConfluenceLevel.STANDARD
        else:
            return ConfluenceLevel.WEAK
            
    def _calculate_confidence_boost(
        self,
        level: ConfluenceLevel,
        behavior_confluence: Dict[str, Any],
        volume_profile: Dict[str, Any]
    ) -> float:
        """Calcula boost de confiança baseado na confluência incluindo Volume Profile"""
        # Boost base por nível (pode vir do config)
        boost_config = self.config.get('confluence_boost_levels', {
            'PREMIUM': 0.15,
            'STRONG': 0.10,
            'STANDARD': 0.05,
            'WEAK': 0.0
        })
        
        base_boost = boost_config.get(level.value, 0.0)
        
        # Adiciona boost extra por Volume Profile
        if volume_profile and self.enable_volume_profile:
            if volume_profile.get('aligned', False):
                base_boost += 0.03
            if volume_profile.get('poc_alignment', 0) > 0.95:
                base_boost += 0.02
        
        # Limita ao máximo configurado
        boost = min(base_boost, self.confluence_boost)
        
        # Penaliza por comportamentos conflitantes
        if behavior_confluence['conflicting']:
            penalty = len(behavior_confluence['conflicting']) * 0.02
            boost = max(0, boost - penalty)
            
        return boost
        
    def _check_divergence(
        self,
        primary: List[BehaviorDetection],
        secondary: List[BehaviorDetection],
        correlation: float
    ) -> Dict[str, Any]:
        """Verifica se há divergência entre ativos"""
        # Threshold de divergência customizável
        divergence_threshold = self.config.get('divergence_threshold', 0.3)
        
        # Baixa correlação pode indicar divergência
        if correlation < divergence_threshold:
            # Verifica se há comportamentos opostos
            primary_bullish = sum(
                1 for b in primary 
                if getattr(b, 'direction', None) == Side.BUY
            )
            
            secondary_bullish = sum(
                1 for b in secondary 
                if getattr(b, 'direction', None) == Side.BUY
            )
            
            primary_bearish = len(primary) - primary_bullish
            secondary_bearish = len(secondary) - secondary_bullish
            
            # Divergência clara
            if (primary_bullish > primary_bearish and 
                secondary_bearish > secondary_bullish):
                return {
                    'detected': True,
                    'type': 'directional',
                    'description': 'Ativos movendo em direções opostas',
                    'severity': 'high' if correlation < 0.1 else 'medium'
                }
                
            # Divergência de força
            elif abs(primary_bullish - secondary_bullish) > 3:
                return {
                    'detected': True,
                    'type': 'strength',
                    'description': 'Diferença significativa na força do movimento',
                    'severity': 'low'
                }
                
        return {'detected': False}
        
    def _calculate_quality_metrics(self) -> Dict[str, Any]:
        """
        Calcula métricas de qualidade da análise
        OTIMIZAÇÃO: Usa NumPy para cálculos estatísticos
        """
        # Quantidade de dados
        dol_count = len(self.price_history['DOLFUT'])
        wdo_count = len(self.price_history['WDOFUT'])
        
        # Qualidade dos dados
        data_quality = min(dol_count, wdo_count) / max(
            self.min_data_points * 2, 1
        )
        data_quality = min(1.0, data_quality)
        
        # Estabilidade da correlação
        if len(self.correlation_history) > 5:
            recent_correlations = np.array([
                h['correlation'] for h in self.correlation_history[-10:]
            ])
            # OTIMIZAÇÃO: NumPy std é mais rápido
            correlation_std = np.std(recent_correlations)
            correlation_stability = max(0, 1 - correlation_std)
        else:
            correlation_stability = 0.5
            
        return {
            'data_quality': round(data_quality, 2),
            'correlation_stability': round(correlation_stability, 2),
            'data_points': {
                'DOLFUT': dol_count,
                'WDOFUT': wdo_count
            },
            'cache_hit_rate': self._calculate_cache_hit_rate()
        }
        
    def _calculate_cache_hit_rate(self) -> float:
        """Calcula taxa de acerto do cache"""
        total = self._cache_stats['hits'] + self._cache_stats['misses']
        if total == 0:
            return 0.0
            
        return self._cache_stats['hits'] / total
        
    def _generate_recommendation(
        self,
        level: ConfluenceLevel,
        divergence: Dict[str, Any],
        leadership: Dict[str, Any],
        volume_profile: Dict[str, Any]
    ) -> str:
        """Gera recomendação baseada na análise incluindo Volume Profile e qualidade de liderança"""
        if divergence['detected']:
            severity = divergence.get('severity', 'medium')
            if severity == 'high':
                return "⚠️ DIVERGÊNCIA ALTA - Evitar operações até convergência"
            else:
                return "⚠️ Divergência detectada - Aguardar confirmação"
            
        # Volume Profile tem prioridade se habilitado
        if volume_profile and self.enable_volume_profile:
            if volume_profile.get('aligned', False):
                vp_msg = " + Volume Profile ALINHADO"
            else:
                vp_msg = ""
        else:
            vp_msg = ""
            
        # Informação sobre qualidade de liderança
        if 'quality_metrics' in leadership:
            quality = leadership['quality_metrics'].get('quality_score', 0)
            if quality > 0.8:
                quality_msg = " (liderança muito clara)"
            elif quality > 0.6:
                quality_msg = " (liderança clara)"
            else:
                quality_msg = ""
        else:
            quality_msg = ""
            
        if level == ConfluenceLevel.PREMIUM:
            return f"✅ Confluência PREMIUM - Sinal muito forte, máxima confiança{vp_msg}{quality_msg}"
            
        elif level == ConfluenceLevel.STRONG:
            leader = leadership.get('leader', 'unclear')
            if leader != 'unclear' and leader != 'simultaneous':
                return f"✅ Confluência FORTE - Seguir {leader} (líder identificado){vp_msg}{quality_msg}"
            else:
                return f"✅ Confluência FORTE - Sinal confiável{vp_msg}"
                
        elif level == ConfluenceLevel.STANDARD:
            return f"✓ Confluência PADRÃO - Sinal válido com confirmação{vp_msg}"
            
        else:
            return "⚠️ Confluência FRACA - Cautela recomendada, buscar confirmações"
            
    def get_statistics(self) -> Dict[str, Any]:
        """Retorna estatísticas de confluência com métricas de performance"""
        total_leads = sum(self.leadership_stats.values())
        
        if total_leads > 0:
            dol_lead_pct = self.leadership_stats['DOLFUT_leads'] / total_leads
            wdo_lead_pct = self.leadership_stats['WDOFUT_leads'] / total_leads
            simul_pct = self.leadership_stats['simultaneous'] / total_leads
        else:
            dol_lead_pct = wdo_lead_pct = simul_pct = 0.0
            
        # Correlação média recente
        if self.correlation_history:
            recent_correlations = [
                h['correlation'] for h in self.correlation_history[-20:]
            ]
            avg_correlation = sum(recent_correlations) / len(recent_correlations)
        else:
            avg_correlation = 0.0
            
        return {
            'correlation_current': self._calculate_price_correlation(),
            'correlation_average': round(avg_correlation, 3),
            'leadership_stats': {
                'DOLFUT_leads_%': round(dol_lead_pct * 100, 1),
                'WDOFUT_leads_%': round(wdo_lead_pct * 100, 1),
                'simultaneous_%': round(simul_pct * 100, 1)
            },
            'total_observations': total_leads,
            'price_history_size': {
                'DOLFUT': len(self.price_history['DOLFUT']),
                'WDOFUT': len(self.price_history['WDOFUT'])
            },
            'confluence_results': len(self.confluence_results),
            'performance_metrics': {
                'correlation_cache_size': len(self.correlation_cache),
                'volume_profile_cache_size': len(self.volume_profile_cache),
                'cache_hit_rate': round(self._calculate_cache_hit_rate() * 100, 1),
                'complexity': 'O(K log K) para liderança'
            },
            'config': {
                'correlation_window': self.correlation_window,
                'min_correlation': self.min_correlation,
                'confluence_boost': self.confluence_boost,
                'cache_ttl': self.cache_ttl,
                'volume_profile_enabled': self.enable_volume_profile
            }
        }
        
    def reload_config(self, new_config: Dict[str, Any]):
        """Recarrega configurações"""
        self.config.update(new_config)
        
        # Atualiza parâmetros
        self.correlation_window = new_config.get('correlation_window', self.correlation_window)
        self.min_correlation = new_config.get('min_correlation', self.min_correlation)
        self.leader_threshold = new_config.get('leader_threshold', self.leader_threshold)
        self.confluence_boost = new_config.get('confluence_boost', self.confluence_boost)
        self.cache_ttl = new_config.get('correlation_cache_ttl', self.cache_ttl)
        self.enable_volume_profile = new_config.get('enable_volume_profile', self.enable_volume_profile)
        
        # Atualiza thresholds se presentes
        if 'confluence_thresholds' in new_config:
            self.confluence_thresholds.update(new_config['confluence_thresholds'])
            
        # Invalida caches ao recarregar config
        self._invalidate_caches()
            
        self.logger.info("Configurações do ConfluenceAnalyzer v3.2 recarregadas")