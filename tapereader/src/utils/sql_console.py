"""
Console SQL Interativo para TapeReader
Execute queries SQL diretamente nos bancos de dados
"""

import sqlite3
import os
from pathlib import Path
from datetime import datetime

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


class SQLConsole:
    """Console interativo para consultas SQL"""
    
    def __init__(self):
        self.db_path = Path("../data")
        self.current_db = None
        self.conn = None
        
        # Queries de exemplo
        self.example_queries = {
            "1": {
                "desc": "Top 10 níveis mais fortes",
                "db": "levels.db",
                "query": """
                    SELECT price, touches, rejections, strength_score
                    FROM price_levels
                    WHERE asset = 'DOLFUT'
                    ORDER BY strength_score DESC
                    LIMIT 10
                """
            },
            "2": {
                "desc": "Volume dos últimos 60 minutos",
                "db": "price_history.db",
                "query": """
                    SELECT 
                        strftime('%H:%M', timestamp) as time,
                        SUM(volume) as total_volume,
                        SUM(buy_volume) as buy_vol,
                        SUM(sell_volume) as sell_vol
                    FROM candles_1m
                    WHERE asset = 'DOLFUT'
                      AND timestamp > datetime('now', '-1 hour')
                    GROUP BY strftime('%H:%M', timestamp)
                    ORDER BY timestamp
                """
            },
            "3": {
                "desc": "Níveis com interesse institucional",
                "db": "price_history.db",
                "query": """
                    SELECT price, visits_count, total_volume, 
                           buy_volume, sell_volume
                    FROM price_contexts
                    WHERE asset = 'DOLFUT'
                      AND big_player_interest = 1
                    ORDER BY total_volume DESC
                """
            },
            "4": {
                "desc": "Eventos de rejeição recentes",
                "db": "levels.db",
                "query": """
                    SELECT 
                        l.price,
                        e.event_type,
                        e.timestamp,
                        e.volume
                    FROM level_events e
                    JOIN price_levels l ON e.level_id = l.id
                    WHERE l.asset = 'DOLFUT'
                      AND e.event_type = 'rejection'
                      AND e.timestamp > datetime('now', '-1 day')
                    ORDER BY e.timestamp DESC
                """
            },
            "5": {
                "desc": "OHLC da última hora",
                "db": "price_history.db",
                "query": """
                    SELECT 
                        timestamp,
                        open,
                        high,
                        low,
                        close,
                        volume
                    FROM candles_1m
                    WHERE asset = 'DOLFUT'
                      AND timestamp > datetime('now', '-1 hour')
                    ORDER BY timestamp
                """
            }
        }
        
    def connect(self, db_name: str) -> bool:
        """Conecta a um banco de dados"""
        db_file = self.db_path / db_name
        
        if not db_file.exists():
            print(f"❌ Banco {db_name} não encontrado em {self.db_path}")
            return False
            
        try:
            if self.conn:
                self.conn.close()
                
            self.conn = sqlite3.connect(str(db_file))
            self.current_db = db_name
            print(f"✅ Conectado a {db_name}")
            return True
            
        except Exception as e:
            print(f"❌ Erro ao conectar: {e}")
            return False
            
    def execute_query(self, query: str):
        """Executa uma query SQL"""
        if not self.conn:
            print("❌ Nenhum banco conectado. Use 'connect' primeiro.")
            return
            
        try:
            if HAS_PANDAS:
                # Usa pandas para melhor visualização
                df = pd.read_sql_query(query, self.conn)
                
                if df.empty:
                    print("Nenhum resultado encontrado.")
                else:
                    print(f"\n{len(df)} resultados encontrados:\n")
                    print(df.to_string())
            else:
                # Visualização básica
                cursor = self.conn.cursor()
                cursor.execute(query)
                
                # Pega nomes das colunas
                columns = [desc[0] for desc in cursor.description]
                rows = cursor.fetchall()
                
                if not rows:
                    print("Nenhum resultado encontrado.")
                else:
                    print(f"\n{len(rows)} resultados encontrados:\n")
                    
                    # Cabeçalho
                    print(" | ".join(f"{col:15}" for col in columns))
                    print("-" * (len(columns) * 17))
                    
                    # Dados
                    for row in rows[:50]:  # Limita a 50 linhas
                        print(" | ".join(f"{str(val):15}" for val in row))
                        
                    if len(rows) > 50:
                        print(f"\n... e mais {len(rows) - 50} linhas")
                        
        except Exception as e:
            print(f"❌ Erro na query: {e}")
            
    def list_tables(self):
        """Lista todas as tabelas do banco atual"""
        if not self.conn:
            print("❌ Nenhum banco conectado.")
            return
            
        query = "SELECT name FROM sqlite_master WHERE type='table'"
        cursor = self.conn.cursor()
        cursor.execute(query)
        
        tables = cursor.fetchall()
        print(f"\nTabelas em {self.current_db}:")
        for table in tables:
            print(f"  - {table[0]}")
            
            # Mostra estrutura da tabela
            cursor.execute(f"PRAGMA table_info({table[0]})")
            columns = cursor.fetchall()
            for col in columns:
                print(f"      {col[1]} ({col[2]})")
                
    def show_examples(self):
        """Mostra queries de exemplo"""
        print("\nQUERIES DE EXEMPLO:")
        print("=" * 60)
        
        for key, example in self.example_queries.items():
            print(f"\n[{key}] {example['desc']}")
            print(f"    Banco: {example['db']}")
            print(f"    Query: {example['query'].strip()}")
            
    def interactive_mode(self):
        """Modo interativo do console"""
        print("\n" + "=" * 60)
        print("CONSOLE SQL INTERATIVO - TAPEREADER")
        print("=" * 60)
        print("\nComandos disponíveis:")
        print("  connect <banco.db>  - Conecta a um banco")
        print("  tables             - Lista tabelas")
        print("  examples           - Mostra queries de exemplo")
        print("  run <N>            - Executa exemplo N")
        print("  quit               - Sair")
        print("\nOu digite uma query SQL diretamente")
        print("=" * 60)
        
        # Tenta conectar automaticamente
        for db in ['levels.db', 'price_history.db']:
            if (self.db_path / db).exists():
                self.connect(db)
                break
                
        while True:
            try:
                if self.current_db:
                    prompt = f"\n[{self.current_db}]> "
                else:
                    prompt = "\n[sem conexão]> "
                    
                command = input(prompt).strip()
                
                if not command:
                    continue
                    
                # Comandos especiais
                if command.lower() == 'quit':
                    break
                    
                elif command.lower() == 'tables':
                    self.list_tables()
                    
                elif command.lower() == 'examples':
                    self.show_examples()
                    
                elif command.lower().startswith('connect '):
                    db_name = command[8:].strip()
                    self.connect(db_name)
                    
                elif command.lower().startswith('run '):
                    example_num = command[4:].strip()
                    if example_num in self.example_queries:
                        example = self.example_queries[example_num]
                        # Conecta ao banco correto
                        if self.connect(example['db']):
                            print(f"\nExecutando: {example['desc']}")
                            self.execute_query(example['query'])
                    else:
                        print(f"❌ Exemplo {example_num} não encontrado")
                        
                else:
                    # Assume que é uma query SQL
                    self.execute_query(command)
                    
            except KeyboardInterrupt:
                print("\n\nSaindo...")
                break
            except Exception as e:
                print(f"❌ Erro: {e}")
                
        if self.conn:
            self.conn.close()
            

if __name__ == "__main__":
    console = SQLConsole()
    console.interactive_mode()