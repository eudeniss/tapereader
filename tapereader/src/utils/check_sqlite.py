"""
Verificador de SQLite e visualizador de dados
"""

import sys
import os
from pathlib import Path

def check_sqlite():
    """Verifica se SQLite está disponível"""
    print("=" * 60)
    print("VERIFICAÇÃO DO SQLITE")
    print("=" * 60)
    
    try:
        import sqlite3
        print("✅ SQLite está instalado!")
        print(f"   Versão do módulo Python: {sqlite3.version}")
        print(f"   Versão do SQLite: {sqlite3.sqlite_version}")
        
        # Testa criação de banco
        test_db = "test_sqlite.db"
        conn = sqlite3.connect(test_db)
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE IF NOT EXISTS test (id INTEGER PRIMARY KEY)")
        cursor.execute("INSERT INTO test VALUES (1)")
        cursor.execute("SELECT * FROM test")
        result = cursor.fetchone()
        conn.close()
        
        if result:
            print("✅ SQLite funcionando corretamente!")
            os.remove(test_db)  # Remove arquivo de teste
        
        return True
        
    except ImportError:
        print("❌ SQLite NÃO está instalado!")
        print("   Isso é muito raro, pois vem com Python.")
        print("   Tente reinstalar Python.")
        return False
    except Exception as e:
        print(f"❌ Erro ao testar SQLite: {e}")
        return False


def check_pandas():
    """Verifica se pandas está instalado para visualização"""
    try:
        import pandas as pd
        print("\n✅ Pandas está instalado (ótimo para visualizar dados)")
        return True
    except ImportError:
        print("\n⚠️  Pandas não está instalado")
        print("   Instale com: pip install pandas")
        print("   Mas não é obrigatório!")
        return False


if __name__ == "__main__":
    print("\n🔍 Verificando ambiente...\n")
    
    sqlite_ok = check_sqlite()
    pandas_ok = check_pandas()
    
    print("\n" + "=" * 60)
    if sqlite_ok:
        print("✅ TUDO PRONTO! SQLite está funcionando.")
        print("   Os bancos de dados serão criados automaticamente")
        print("   quando o TapeReader for executado.")
    else:
        print("❌ PROBLEMA DETECTADO! Verifique a instalação do Python.")
        
    print("=" * 60)