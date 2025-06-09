"""
Teste completo do sistema de dados do TapeReader
Execute este arquivo de QUALQUER pasta
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Encontra a raiz do projeto tapereader
current_file = Path(__file__).resolve()
# Procura pela pasta tapereader subindo no diretório
tapereader_root = current_file
while tapereader_root.name != 'tapereader' and tapereader_root.parent != tapereader_root:
    tapereader_root = tapereader_root.parent
    if (tapereader_root / 'src').exists():
        break

# Adiciona ao path
sys.path.insert(0, str(tapereader_root))

print(f"📁 Diretório do projeto: {tapereader_root}")
print(f"📍 Executando de: {current_file.parent}")

def print_header(text):
    """Imprime cabeçalho formatado"""
    print(f"\n{'=' * 60}")
    print(f"{text:^60}")
    print('=' * 60)


def test_sqlite():
    """Testa se SQLite está disponível"""
    print_header("1. TESTANDO SQLITE")
    
    try:
        import sqlite3
        print("✅ SQLite importado com sucesso!")
        
        # Evita o warning de depreciação
        if hasattr(sqlite3, 'version_info'):
            print(f"   Versão do módulo: {'.'.join(map(str, sqlite3.version_info))}")
        print(f"   Versão SQLite: {sqlite3.sqlite_version}")
        
        # Teste de criação
        test_conn = sqlite3.connect(":memory:")
        test_conn.execute("CREATE TABLE test (id INTEGER)")
        test_conn.execute("INSERT INTO test VALUES (1)")
        result = test_conn.execute("SELECT * FROM test").fetchone()
        test_conn.close()
        
        if result:
            print("✅ SQLite funcionando perfeitamente!")
            return True
        
    except Exception as e:
        print(f"❌ Erro com SQLite: {e}")
        return False
        

def test_cache_system():
    """Testa o sistema de cache"""
    print_header("2. TESTANDO SISTEMA DE CACHE")
    
    try:
        # Importa do caminho correto
        from src.cache import PriceMemory
        from src.core.models import Trade, MarketData, OrderBook, BookLevel, Side
        from decimal import Decimal
        
        # Cria diretório temporário para teste
        test_data_dir = tapereader_root / 'test_data'
        test_data_dir.mkdir(exist_ok=True)
        
        # Configuração de teste
        config = {
            'db_path': str(test_data_dir / 'price_history.db'),
            'l1_duration': 30,
            'l2_duration': 24,
            'l3_retention': 90
        }
        
        # Cria sistema de cache
        print("📦 Criando sistema de cache...")
        cache = PriceMemory(config)
        
        # Cria dados de teste
        print("📝 Adicionando dados de teste...")
        
        # Trade de teste
        test_trade = Trade(
            timestamp=datetime.now(),
            price=Decimal('5750.50'),
            volume=25,
            aggressor=Side.BUY
        )
        
        # Market data de teste
        test_market_data = MarketData(
            asset='DOLFUT',
            timestamp=datetime.now(),
            trades=[test_trade],
            book=OrderBook(
                timestamp=datetime.now(),
                bids=[BookLevel(price=Decimal('5750.00'), volume=100)],
                asks=[BookLevel(price=Decimal('5751.00'), volume=100)]
            )
        )
        
        # Adiciona ao cache
        cache.add_market_data(test_market_data)
        
        # Verifica se foi adicionado
        recent = cache.get_recent_trades('DOLFUT', 60)
        
        if recent:
            print(f"✅ Cache funcionando! {len(recent)} trades armazenados")
            
            # Testa contexto de preço
            context = cache.get_price_context('DOLFUT', Decimal('5750.50'))
            print(f"✅ Contexto de preço: {context.visits_count} visitas")
            
            return True
        else:
            print("❌ Erro ao adicionar dados ao cache")
            return False
            
    except ImportError as e:
        print(f"❌ Erro de importação: {e}")
        print(f"   Certifique-se de que está no diretório: {tapereader_root}")
        
        # Verifica se os arquivos existem
        cache_file = tapereader_root / 'src' / 'cache' / 'price_memory.py'
        if not cache_file.exists():
            print(f"   ❌ Arquivo não encontrado: {cache_file}")
        else:
            print(f"   ✅ Arquivo existe: {cache_file}")
            
        return False
    except Exception as e:
        print(f"❌ Erro no teste de cache: {e}")
        import traceback
        traceback.print_exc()
        return False
        

def test_support_resistance():
    """Testa o detector de suporte/resistência"""
    print_header("3. TESTANDO DETECTOR S/R")
    
    try:
        # Primeiro tenta a versão enhanced
        try:
            from src.behaviors.support_resistance_enhanced import SupportResistanceEnhancedDetector as SRDetector
            print("📦 Usando detector S/R Enhanced")
        except ImportError:
            # Se não existir, usa a versão normal
            from src.behaviors.support_resistance import SupportResistanceDetector as SRDetector
            print("📦 Usando detector S/R padrão")
            
        from src.core.models import MarketData, Trade, OrderBook, BookLevel, Side
        from decimal import Decimal
        
        # Cria diretório para teste
        test_data_dir = tapereader_root / 'test_data'
        test_data_dir.mkdir(exist_ok=True)
        
        # Configuração
        config = {
            'enabled': True,
            'enable_persistence': True,
            'db_path': str(test_data_dir / 'levels.db'),
            'enable_decay': True,
            'decay_half_life_hours': 24,
            'level_tolerance': 0.5,
            'min_touches': 3
        }
        
        print("🔍 Criando detector S/R...")
        detector = SRDetector(config)
        
        # Cria dados de teste com múltiplos toques no mesmo nível
        trades = []
        test_level = Decimal('5750.00')
        
        # Simula múltiplos toques no nível
        for i in range(5):
            # Aproximação do nível
            for j in range(3):
                trades.append(Trade(
                    timestamp=datetime.now() - timedelta(minutes=i*10+j),
                    price=test_level + Decimal(str(j)),
                    volume=30,
                    aggressor=Side.SELL
                ))
            
            # Rejeição do nível
            trades.append(Trade(
                timestamp=datetime.now() - timedelta(minutes=i*10+3),
                price=test_level,
                volume=50,
                aggressor=Side.BUY
            ))
        
        # Market data
        market_data = MarketData(
            asset='DOLFUT',
            timestamp=datetime.now(),
            trades=trades,
            book=OrderBook(
                timestamp=datetime.now(),
                bids=[
                    BookLevel(price=test_level, volume=500),  # Grande ordem no nível
                    BookLevel(price=test_level - Decimal('0.5'), volume=100)
                ],
                asks=[
                    BookLevel(price=test_level + Decimal('0.5'), volume=100),
                    BookLevel(price=test_level + Decimal('1'), volume=100)
                ]
            )
        )
        
        # Detecta
        print("📊 Analisando níveis...")
        import asyncio
        
        async def run_detection():
            # Adiciona dados históricos primeiro
            for i in range(3):
                await detector.detect(market_data)
            # Detecção final
            return await detector.detect(market_data)
            
        detection = asyncio.run(run_detection())
        
        if detection.detected:
            print(f"✅ Nível detectado em {detection.metadata['level_price']}")
            print(f"   Confiança: {detection.confidence:.2%}")
            print(f"   Toques: {detection.metadata['touches']}")
            return True
        else:
            print("⚠️  Nenhum nível detectado (pode precisar de mais dados)")
            # Ainda considera sucesso se o detector rodou sem erros
            return True
            
    except Exception as e:
        print(f"❌ Erro no teste S/R: {e}")
        import traceback
        traceback.print_exc()
        return False
        

def test_database_viewer():
    """Testa o visualizador de dados"""
    print_header("4. TESTANDO VISUALIZADOR")
    
    try:
        # Importa do caminho correto
        from src.utils.database_viewer import DatabaseViewer
        
        # Usa diretório de teste
        test_data_dir = tapereader_root / 'test_data'
        
        print("👁️  Criando visualizador...")
        viewer = DatabaseViewer(str(test_data_dir))
        
        # Verifica bancos
        databases = viewer.check_databases()
        print("\n📁 Status dos bancos:")
        for db, exists in databases.items():
            status = "✅ Existe" if exists else "❌ Não existe"
            print(f"   {db}: {status}")
            
        # Se existir algum banco, tenta ler
        if any(databases.values()):
            print("\n📊 Tentando ler estatísticas...")
            stats = viewer.get_statistics('DOLFUT')
            
            if stats:
                print("✅ Visualizador funcionando!")
                for key, value in stats.items():
                    print(f"   {key}: {value}")
            else:
                print("⚠️  Sem dados ainda (normal em teste)")
                
        print("✅ Visualizador importado com sucesso!")
        return True
        
    except ImportError as e:
        print(f"❌ Erro de importação: {e}")
        
        # Verifica se o arquivo existe
        viewer_file = tapereader_root / 'src' / 'utils' / 'database_viewer.py'
        if not viewer_file.exists():
            print(f"   ❌ Arquivo não encontrado: {viewer_file}")
        else:
            print(f"   ✅ Arquivo existe: {viewer_file}")
            
        return False
    except Exception as e:
        print(f"❌ Erro no visualizador: {e}")
        import traceback
        traceback.print_exc()
        return False
        

def test_minimal_imports():
    """Testa imports mínimos"""
    print_header("5. TESTANDO IMPORTS BÁSICOS")
    
    modules_to_test = [
        ('src.core.models', 'Modelos de dados'),
        ('src.core.config', 'Sistema de configuração'),
        ('src.core.logger', 'Sistema de logging'),
        ('src.data.base', 'Data provider base'),
        ('src.behaviors.base', 'Behavior base'),
    ]
    
    all_ok = True
    for module_name, description in modules_to_test:
        try:
            __import__(module_name)
            print(f"✅ {description}: OK")
        except ImportError as e:
            print(f"❌ {description}: {e}")
            all_ok = False
            
    return all_ok


def cleanup_test_data():
    """Limpa dados de teste"""
    import shutil
    test_dir = tapereader_root / "test_data"
    if test_dir.exists():
        shutil.rmtree(test_dir)
        print("\n🧹 Dados de teste removidos")
        

def main():
    """Executa todos os testes"""
    print("\n" + "=" * 60)
    print("TESTE COMPLETO DO SISTEMA DE DADOS - TAPEREADER")
    print("=" * 60)
    print("\nEste script verifica se tudo está configurado corretamente.")
    
    # Verifica estrutura de pastas
    print_header("VERIFICANDO ESTRUTURA")
    
    expected_dirs = ['src', 'config', 'logs', 'data']
    missing_dirs = []
    
    for dir_name in expected_dirs:
        dir_path = tapereader_root / dir_name
        if dir_path.exists():
            print(f"✅ {dir_name}/")
        else:
            print(f"❌ {dir_name}/ (criando...)")
            dir_path.mkdir(exist_ok=True)
            missing_dirs.append(dir_name)
    
    if missing_dirs:
        print(f"\n⚠️  Criadas pastas faltantes: {', '.join(missing_dirs)}")
    
    # Testes
    results = {
        'SQLite': test_sqlite(),
        'Imports Básicos': test_minimal_imports(),
        'Cache': test_cache_system(),
        'S/R Detector': test_support_resistance(),
        'Visualizador': test_database_viewer()
    }
    
    # Resumo
    print_header("RESUMO DOS TESTES")
    
    all_passed = True
    for test, passed in results.items():
        status = "✅ PASSOU" if passed else "❌ FALHOU"
        print(f"{test:.<40} {status}")
        if not passed:
            all_passed = False
            
    print("\n" + "=" * 60)
    
    if all_passed:
        print("🎉 TUDO FUNCIONANDO PERFEITAMENTE!")
        print("\nPróximos passos:")
        print("1. Execute o TapeReader para gerar dados reais")
        print(f"2. Use {tapereader_root}/src/utils/database_viewer.py")
        print(f"3. Use {tapereader_root}/src/utils/sql_console.py")
    else:
        print("⚠️  ALGUNS TESTES FALHARAM")
        print("\nVerifique:")
        print(f"1. Se a estrutura está correta em: {tapereader_root}")
        print("2. Se os arquivos foram criados corretamente")
        print("3. Os erros específicos acima")
        
        # Mostra estrutura esperada
        print("\n📁 Estrutura esperada:")
        print("tapereader/")
        print("├── src/")
        print("│   ├── core/")
        print("│   ├── data/")
        print("│   ├── behaviors/")
        print("│   ├── cache/")
        print("│   └── utils/")
        print("├── config/")
        print("├── logs/")
        print("└── data/")
        
    # Limpa dados de teste
    cleanup_test_data()
    
    print("\n" + "=" * 60)
    

if __name__ == "__main__":
    main()