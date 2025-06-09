"""
Visualizador de Dados Históricos do TapeReader
Ferramenta para acessar e analisar dados salvos no banco
"""

import sqlite3
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
from decimal import Decimal

# Tenta importar pandas para melhor visualização
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("⚠️  Pandas não instalado. Usando visualização básica.")


class DatabaseViewer:
    """Visualizador de dados históricos do TapeReader"""
    
    def __init__(self, db_path: str = "../data"):
        self.db_path = Path(db_path)
        self.levels_db = self.db_path / "levels.db"
        self.price_db = self.db_path / "price_history.db"
        
    def check_databases(self) -> Dict[str, bool]:
        """Verifica quais bancos existem"""
        return {
            'levels.db': self.levels_db.exists(),
            'price_history.db': self.price_db.exists()
        }
        
    def list_tables(self, db_file: str) -> List[str]:
        """Lista todas as tabelas em um banco"""
        if not Path(db_file).exists():
            return []
            
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        return tables
        
    # ==================== NÍVEIS DE SUPORTE/RESISTÊNCIA ====================
    
    def get_support_resistance_levels(
        self, 
        asset: str = "DOLFUT",
        min_touches: int = 3,
        days_back: int = 7
    ) -> List[Dict]:
        """Obtém níveis de suporte/resistência históricos"""
        if not self.levels_db.exists():
            print("❌ Banco de níveis não encontrado. Execute o TapeReader primeiro.")
            return []
            
        conn = sqlite3.connect(str(self.levels_db))
        
        query = """
        SELECT 
            price,
            touches,
            rejections,
            successful_breaks,
            total_volume,
            strength_score,
            last_touch,
            CASE 
                WHEN rejections > successful_breaks * 2 THEN 'Strong'
                WHEN rejections > successful_breaks THEN 'Moderate'
                ELSE 'Weak'
            END as level_quality
        FROM price_levels
        WHERE asset = ? 
          AND touches >= ?
          AND last_touch > datetime('now', '-' || ? || ' days')
        ORDER BY strength_score DESC
        """
        
        if HAS_PANDAS:
            df = pd.read_sql_query(query, conn, params=(asset, min_touches, days_back))
            conn.close()
            return df
        else:
            cursor = conn.cursor()
            cursor.execute(query, (asset, min_touches, days_back))
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            conn.close()
            
            return [dict(zip(columns, row)) for row in rows]
            
    def get_level_events(self, asset: str, price: float, tolerance: float = 0.5) -> List[Dict]:
        """Obtém histórico de eventos em um nível específico"""
        if not self.levels_db.exists():
            return []
            
        conn = sqlite3.connect(str(self.levels_db))
        
        query = """
        SELECT 
            e.event_type,
            e.timestamp,
            e.price,
            e.volume,
            e.metadata
        FROM level_events e
        JOIN price_levels l ON e.level_id = l.id
        WHERE l.asset = ? 
          AND ABS(l.price - ?) <= ?
        ORDER BY e.timestamp DESC
        LIMIT 50
        """
        
        if HAS_PANDAS:
            df = pd.read_sql_query(query, conn, params=(asset, price, tolerance))
            conn.close()
            return df
        else:
            cursor = conn.cursor()
            cursor.execute(query, (asset, price, tolerance))
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            conn.close()
            
            return [dict(zip(columns, row)) for row in rows]
            
    # ==================== HISTÓRICO DE PREÇOS ====================
    
    def get_ohlc_data(
        self,
        asset: str = "DOLFUT",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ):
        """Obtém dados OHLC históricos"""
        if not self.price_db.exists():
            print("❌ Banco de preços não encontrado. Execute o TapeReader primeiro.")
            return []
            
        if not start_time:
            start_time = datetime.now() - timedelta(hours=24)
        if not end_time:
            end_time = datetime.now()
            
        conn = sqlite3.connect(str(self.price_db))
        
        query = """
        SELECT 
            timestamp,
            open,
            high,
            low,
            close,
            volume,
            buy_volume,
            sell_volume,
            vwap,
            trades_count
        FROM candles_1m
        WHERE asset = ?
          AND timestamp >= ?
          AND timestamp <= ?
        ORDER BY timestamp DESC
        LIMIT ?
        """
        
        if HAS_PANDAS:
            df = pd.read_sql_query(
                query, 
                conn, 
                params=(asset, start_time, end_time, limit),
                parse_dates=['timestamp']
            )
            conn.close()
            return df
        else:
            cursor = conn.cursor()
            cursor.execute(query, (asset, start_time, end_time, limit))
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            conn.close()
            
            return [dict(zip(columns, row)) for row in rows]
            
    def get_price_contexts(
        self,
        asset: str = "DOLFUT",
        min_visits: int = 5
    ):
        """Obtém contextos de preço (níveis visitados frequentemente)"""
        if not self.price_db.exists():
            return []
            
        conn = sqlite3.connect(str(self.price_db))
        
        query = """
        SELECT 
            price,
            visits_count,
            total_volume,
            buy_volume,
            sell_volume,
            rejections,
            breakouts,
            support_strength,
            resistance_strength,
            big_player_interest,
            last_update,
            ROUND(CAST(buy_volume AS FLOAT) / NULLIF(total_volume, 0) * 100, 2) as buy_percentage
        FROM price_contexts
        WHERE asset = ?
          AND visits_count >= ?
        ORDER BY visits_count DESC
        """
        
        if HAS_PANDAS:
            df = pd.read_sql_query(query, conn, params=(asset, min_visits))
            conn.close()
            return df
        else:
            cursor = conn.cursor()
            cursor.execute(query, (asset, min_visits))
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            conn.close()
            
            return [dict(zip(columns, row)) for row in rows]
            
    def get_market_profile(self, asset: str = "DOLFUT", date: Optional[str] = None):
        """Obtém Market Profile de um dia específico"""
        if not self.price_db.exists():
            return []
            
        if not date:
            date = datetime.now().strftime("%Y-%m-%d")
            
        conn = sqlite3.connect(str(self.price_db))
        
        query = """
        SELECT 
            price_level,
            volume,
            time_at_level,
            is_poc,
            is_val,
            is_vah
        FROM market_profile_daily
        WHERE asset = ?
          AND date = ?
        ORDER BY volume DESC
        """
        
        if HAS_PANDAS:
            df = pd.read_sql_query(query, conn, params=(asset, date))
            conn.close()
            return df
        else:
            cursor = conn.cursor()
            cursor.execute(query, (asset, date))
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            conn.close()
            
            return [dict(zip(columns, row)) for row in rows]
            
    # ==================== ESTATÍSTICAS ====================
    
    def get_statistics(self, asset: str = "DOLFUT") -> Dict[str, Any]:
        """Obtém estatísticas gerais dos dados históricos"""
        stats = {}
        
        # Estatísticas de níveis
        if self.levels_db.exists():
            conn = sqlite3.connect(str(self.levels_db))
            cursor = conn.cursor()
            
            # Total de níveis
            cursor.execute(
                "SELECT COUNT(*) FROM price_levels WHERE asset = ?",
                (asset,)
            )
            stats['total_levels'] = cursor.fetchone()[0]
            
            # Nível mais forte
            cursor.execute("""
                SELECT price, strength_score, touches, rejections
                FROM price_levels
                WHERE asset = ?
                ORDER BY strength_score DESC
                LIMIT 1
            """, (asset,))
            
            strongest = cursor.fetchone()
            if strongest:
                stats['strongest_level'] = {
                    'price': strongest[0],
                    'strength': strongest[1],
                    'touches': strongest[2],
                    'rejections': strongest[3]
                }
                
            conn.close()
            
        # Estatísticas de preços
        if self.price_db.exists():
            conn = sqlite3.connect(str(self.price_db))
            cursor = conn.cursor()
            
            # Range de datas
            cursor.execute("""
                SELECT 
                    MIN(timestamp) as first_candle,
                    MAX(timestamp) as last_candle,
                    COUNT(*) as total_candles
                FROM candles_1m
                WHERE asset = ?
            """, (asset,))
            
            candle_stats = cursor.fetchone()
            if candle_stats[0]:
                stats['price_history'] = {
                    'first_candle': candle_stats[0],
                    'last_candle': candle_stats[1],
                    'total_candles': candle_stats[2]
                }
                
            # Volume total
            cursor.execute("""
                SELECT SUM(volume), SUM(buy_volume), SUM(sell_volume)
                FROM candles_1m
                WHERE asset = ?
            """, (asset,))
            
            volume_stats = cursor.fetchone()
            if volume_stats[0]:
                stats['volume'] = {
                    'total': volume_stats[0],
                    'buy': volume_stats[1],
                    'sell': volume_stats[2]
                }
                
            conn.close()
            
        return stats


# ==================== FUNÇÕES DE EXEMPLO ====================

def print_section(title: str):
    """Imprime seção formatada"""
    print(f"\n{'=' * 60}")
    print(f"{title:^60}")
    print('=' * 60)


def example_usage():
    """Exemplo de uso do visualizador"""
    viewer = DatabaseViewer()
    
    # Verifica bancos
    print_section("VERIFICAÇÃO DE BANCOS DE DADOS")
    databases = viewer.check_databases()
    for db, exists in databases.items():
        status = "✅ Encontrado" if exists else "❌ Não encontrado"
        print(f"{db}: {status}")
    
    # Se não houver bancos, avisa
    if not any(databases.values()):
        print("\n⚠️  Nenhum banco encontrado!")
        print("Execute o TapeReader primeiro para gerar dados.")
        return
        
    # Lista tabelas
    if databases['levels.db']:
        print_section("TABELAS EM levels.db")
        tables = viewer.list_tables(str(viewer.levels_db))
        for table in tables:
            print(f"  - {table}")
            
    # Mostra níveis de S/R
    print_section("NÍVEIS DE SUPORTE/RESISTÊNCIA - DOLFUT")
    levels = viewer.get_support_resistance_levels("DOLFUT", min_touches=3)
    
    if HAS_PANDAS and isinstance(levels, pd.DataFrame):
        if not levels.empty:
            print(levels.head(10).to_string())
        else:
            print("Nenhum nível encontrado.")
    else:
        for i, level in enumerate(levels[:10]):
            print(f"\nNível {i+1}:")
            for key, value in level.items():
                print(f"  {key}: {value}")
                
    # Estatísticas
    print_section("ESTATÍSTICAS GERAIS")
    stats = viewer.get_statistics("DOLFUT")
    
    if 'total_levels' in stats:
        print(f"\nNíveis de Preço:")
        print(f"  Total de níveis: {stats['total_levels']}")
        
        if 'strongest_level' in stats:
            sl = stats['strongest_level']
            print(f"  Nível mais forte: {sl['price']} (força: {sl['strength']:.2f})")
            
    if 'price_history' in stats:
        ph = stats['price_history']
        print(f"\nHistórico de Preços:")
        print(f"  Primeiro candle: {ph['first_candle']}")
        print(f"  Último candle: {ph['last_candle']}")
        print(f"  Total de candles: {ph['total_candles']:,}")
        
    if 'volume' in stats:
        v = stats['volume']
        print(f"\nVolume Total:")
        print(f"  Total: {v['total']:,}")
        print(f"  Compra: {v['buy']:,}")
        print(f"  Venda: {v['sell']:,}")


if __name__ == "__main__":
    print("\n🔍 VISUALIZADOR DE DADOS HISTÓRICOS - TAPEREADER\n")
    
    # Executa exemplo
    example_usage()
    
    print("\n" + "=" * 60)
    print("💡 Dicas:")
    print("  - Use pandas para melhor visualização: pip install pandas")
    print("  - Os bancos são criados em: ../data/")
    print("  - Execute o TapeReader para gerar dados")
    print("=" * 60)