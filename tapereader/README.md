TapeReader Professional v2.0
Um sistema avançado de análise de fluxo de ordens (tape reading) projetado para identificar padrões complexos de mercado em tempo real e gerar sinais de trading de alta probabilidade para os contratos de Dólar (DOLFUT) e Mini Dólar (WDOFUT).

📋 Índice
Sobre o Projeto
Principais Funcionalidades
Arquitetura do Sistema
Requisitos
Instalação
Configuração
Executando o Sistema
Solução de Problemas Comuns
Estrutura do Projeto
Comportamentos e Estratégias
🎯 Sobre o Projeto
O TapeReader Professional é uma ferramenta de análise que monitora em tempo real os dados de Time & Sales e Book de Ofertas. O sistema processa esse fluxo de dados para detectar 14 padrões de comportamento de mercado, que são então avaliados por uma matriz de decisão para gerar sinais de trading com alta confiança.

Principais Objetivos:

Alta Acurácia: Geração de sinais com alta taxa de acerto.
Análise em Tempo Real: Processamento de dados de baixa latência para reações rápidas ao mercado.
Detecção de Comportamentos: Identificação de padrões complexos, desde absorção até atividades algorítmicas de HFT.
Confluência Inteligente: Análise correlacionada entre DOLFUT e WDOFUT para confirmação de sinais.
Gestão de Risco Integrada: Controles de risco para proteger o capital e otimizar o tamanho das posições.
✨ Principais Funcionalidades
✅ Leitura de Dados via Excel RTD: Integração direta com planilhas que recebem dados do mercado em tempo real.
✅ 14 Comportamentos de Mercado: Detecta uma vasta gama de padrões, incluindo institucionais, algoritmos e exaustão de movimento.
✅ Sistema de Confluência: Análise correlacionada entre ativos.
✅ Matriz de Decisão Dinâmica: Utiliza um conjunto de regras configuráveis para combinar comportamentos e gerar sinais de trading.
✅ Seleção Inteligente de Sinais: Avalia múltiplos sinais candidatos e seleciona o melhor com base em um score ponderado (confiança, risco/retorno, etc.).
✅ Gerenciador de Risco: Controla perdas diárias, número de posições e ajusta o tamanho da operação com base no risco do mercado.
✅ Classificador de Regime de Mercado: Identifica se o mercado está em tendência, lateralizado ou volátil, e ajusta as estratégias dinamicamente.
✅ Console Visual Detalhado: Interface de terminal rica para acompanhar o status do mercado, sinais ativos e performance.
✅ Persistência de Estado: Salva e restaura o estado do sistema (sinais ativos, P&amp;L, etc.) para continuar a operação após uma interrupção.
🏗️ Arquitetura do Sistema
O sistema segue um pipeline claro de processamento de dados:

┌──────────────────────────────────┐
│        Fonte de Dados (Excel RTD)        │
└──────────────────┬─────────────────┘
                   ▼
┌──────────────────────────────────┐
│    DataProvider (src/data)           │
└──────────────────┬─────────────────┘
                   ▼
┌──────────────────────────────────┐
│   BehaviorManager (src/behaviors)    │
└──────────────────┬─────────────────┘
                   ▼
┌──────────────────────────────────┐
│  Strategy Engine (src/strategies)    │
└──────────────────┬─────────────────┘
                   ▼
┌──────────────────────────────────┐
│      Risk & Output (RiskManager)     │
└──────────────────┬─────────────────┘
                   ▼
┌──────────────────────────────────┐
│         Saída (Console, Logs)          │
└──────────────────────────────────┘
💻 Requisitos
Sistema
Sistema Operacional: Windows 10 ou 11 (64-bit)
Software: Microsoft Excel com suporte a RTD
Python: Versão 3.10 ou superior
Dependências Python
Crie um arquivo chamado requirements.txt na pasta tapereader com o seguinte conteúdo:

Plaintext

# Core
xlwings>=0.30.0
pydantic>=2.0.0
pyyaml>=6.0.0
numpy>=1.26.0

# Console & Logging
rich>=13.0.0
orjson>=3.9.0
python-json-logger>=2.0.0

# Opcional para desenvolvimento
# pytest
# pytest-asyncio
📦 Instalação
Clone o Repositório:

Bash

git clone <url_do_repositorio>
cd tape_reader_final
Crie e Ative um Ambiente Virtual (Recomendado):

Bash

python -m venv venv
.\venv\Scripts\activate
Instale as Dependências:
Navegue até a pasta tapereader e instale os pacotes.

Bash

cd tapereader
pip install -r requirements.txt
cd ..
Configure a Planilha Excel:

Certifique-se de que o arquivo rtd_tapeReading.xlsx está na pasta raiz do projeto (tape_reader_final).
Abra a planilha e garanta que os dados da sua plataforma de trading estão sendo recebidos em tempo real via RTD.
⚙️ Configuração
A configuração do sistema é modular e baseada em arquivos YAML na pasta config/.

config/main.yaml: Arquivo principal. Define os modos de operação e configurações globais.
config/excel.yaml: MUITO IMPORTANTE! Define o caminho para a planilha e os ranges das células de dados. Verifique se o file_path está correto para o seu computador.
config/strategies.yaml: O coração do sistema. Define todas as regras, parâmetros de risco/retorno e lógicas de combinação de comportamentos.
config/behaviors.yaml: Parâmetros detalhados para cada um dos 14 detectores de comportamento.
🚀 Executando o Sistema
Para iniciar o sistema, utilize o script de inicialização na pasta raiz:

Bash

tapereader.bat
O script apresentará um menu para escolher o modo de operação:

[1] PRODUÇÃO: Usa os dados em tempo real do Excel para operar.
[2] TESTE: Roda a suíte de testes pytest para validar a integridade do código.
[3] STATUS: Verifica as dependências e a estrutura de arquivos.
🆘 Solução de Problemas Comuns
Erro: "Arquivo Excel não encontrado"

Causa: O caminho para o arquivo .xlsx no arquivo de configuração está incorreto.
Solução: Abra o arquivo tapereader/config/excel.yaml. No campo file_path, coloque o caminho absoluto e completo para a sua planilha. Use barras normais (/) para compatibilidade.
YAML

file_path: "C:/Caminho/Completo/Para/O/Seu/Projeto/tape_reader_final/rtd_tapeReading.xlsx"
Erro: ConversionSyntax ou TypeError nos Logs

Causa: O formato dos dados na planilha (preço ou volume) não é um número válido (pode conter texto, estar vazio, ou usar vírgula como decimal).
Solução: O código foi ajustado para ser mais robusto, mas verifique as colunas na sua planilha para garantir que os dados de preço e volume sejam sempre numéricos. O código já trata a conversão de vírgula para ponto.
Avisos: "Sem trades" ou "Dados não mudaram"

Causa: A conexão RTD entre sua plataforma de trading (ProfitChart, etc.) e o Excel não está enviando novos dados.
Solução: Com o mercado aberto, olhe para sua planilha. As células com dados de mercado devem "piscar" (atualizar em tempo real). Se estiverem estáticas, o problema está na sua plataforma ou no link RTD. Reinicie a plataforma e a planilha para tentar reestabelecer a conexão.
📂 Estrutura do Projeto
tape_reader_final/
│
├── tapereader.bat
├── rtd_tapeReading.xlsx
│
├── logs/
│   ├── app.log
│   └── analysis/
│
└── tapereader/
    ├── config/         # Arquivos de configuração YAML
    ├── src/            # Código fonte principal
    │   ├── core/       # Modelos, config, tracking
    │   ├── data/       # Provedores de dados
    │   ├── behaviors/  # Detectores de comportamento
    │   ├── strategies/ # Lógica de decisão
    │   └── utils/      # Utilitários
    └── tests/          # Testes automatizados
🎯 Comportamentos e Estratégias
O sistema é capaz de detectar 14 comportamentos distintos no fluxo de ordens:

Absorção: Grande volume de agressão sem mover o preço.
Exaustão: Enfraquecimento de movimento direcional.
Fluxo Institucional: Atividade de grandes players.
Sweep: Varredura rápida de múltiplos níveis de liquidez.
Stop Hunt: Movimento abrupto para acionar ordens de stop.
Iceberg: Ordens grandes executadas em lotes menores e ocultos.
Momentum: Força direcional consistente e sustentada.
Suporte/Resistência: Níveis de preço defendidos no book.
Renovação: Reposição constante de liquidez em um nível.
Divergência: Desalinhamento entre preço e indicadores (ex: volume).
HTF (High-Frequency Trading): Padrões de algoritmos de alta frequência.
Micro Agressão: Acumulação ou distribuição discreta com lotes pequenos.
Breakout: Rompimento de zonas de consolidação.
Recorrência: Padrões de negociação que se repetem de forma cíclica.
Estes comportamentos são combinados em estratégias definidas no arquivo config/strategies.yaml para gerar os sinais finais.