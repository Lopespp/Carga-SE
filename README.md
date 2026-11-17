# Previsão de Carga de Energia Diária — ONS (Subsistema Sudeste/Centro-Oeste)

Pipeline completo de Engenharia e Ciência de Dados para prever a **carga diária de energia elétrica (em MWmed)** do subsistema **Sudeste/Centro-Oeste (`SE`)** para um horizonte de **7 dias à frente**, utilizando os dados abertos do Operador Nacional do Sistema Elétrico (ONS) de 2018 até o presente.

---

## Resposta à Pergunta do Desafio (em até 5 linhas)

**Sim.** No backtesting temporal de 26 janelas sucessivas de 7 dias (182 dias, sem vazamento de dados e usando apenas $\text{lags} \ge 7$), o modelo **LightGBM** superou o baseline (*Seasonal Naive* de 7 dias), reduzindo o **MAE em 24,68%** (de **2.235,22 MWmed** para **1.683,57 MWmed**, ganho de **551,65 MWmed**) e reduzindo o **MAPE em 1,30 ponto percentual** (de **5,16%** para **3,86%**, queda relativa de **25,19%** no erro percentual), vencendo o baseline em **18 das 26 semanas** avaliadas.

---

## Página Web Unificada (pipeline + modelo + MLflow + resultados)

Todos os resultados ficam numa única página interativa, `docs/index.html`, gerada pela última etapa do pipeline (`src/build_report.py`):

| Seção | O que mostra |
| :--- | :--- |
| **Resposta** | Veredito, MAE/MAPE do LightGBM vs. baseline, semanas vencidas e pico previsto |
| **Pipeline** | Fluxo Bronze → Silver → Gold → Baseline → LightGBM → Backtest → Previsão com o status de cada etapa, os 10 testes do `dbt` e os arquivos Parquet brutos |
| **Dados** | Série histórica com zoom, média móvel, perfil semanal e o degrau da MMGD (mai/2023) |
| **Previsão 7 dias** | Últimas 5 semanas + previsões (LightGBM, AutoETS, baseline) com feriados destacados |
| **Avaliação** | Tabela comparativa, placar semanal clicável, real × previsto, MAE por semana e por dia do horizonte |
| **Modelo** | Linha do tempo interativa da regra anti-vazamento, importância das variáveis e hiperparâmetros |
| **MLflow** | Runs de treino, backtest e modelo global, com links diretos para a interface do MLflow |
| **Todos subsistemas** | Modelo global único para SE, S, NE e N |

```bash
uv run python -m src.serve
```

Esse comando sobe a página em `http://127.0.0.1:8000` e a interface do MLflow em `http://127.0.0.1:5000`; os links das runs abrem direto no MLflow. A página também abre direto do disco (`docs/index.html`) e funciona offline, com tema claro/escuro e layout para celular.

---

## Tabela Comparativa no Backtesting Temporal (26 Janelas de 7 Dias = 182 Dias)

Validação *walk-forward* (*expanding window*) re-treinando os modelos a cada janela de 7 dias apenas com dados passados ($\le t_0$) e prevendo simultaneamente os dias $t_0+1$ a $t_0+7$. Essa é a avaliação oficial, registrada no MLflow como a run `backtest_walk_forward`:

| Modelo | MAE (MWmed) | MAPE (%) | RMSE (MWmed) | Redução de MAE vs. Baseline (MWmed) | Redução Relativa de MAE (%) | Redução de MAPE (p.p.) | Janelas de 7d Vencidas (em 26) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Seasonal Naive — Lag 7)** | 2.235,22 | 5,16% | 2.784,43 | 0,00 | 0,00% | 0,00 p.p. | 8 / 26 |
| **AutoETS (`statsforecast` — Sazonal 7d)** *(Bônus)* | 1.826,02 | 4,17% | 2.378,38 | -409,20 | **-18,31%** | -0,99 p.p. | 20 / 26 |
| **LightGBM (Lags $\ge$ 7 + Calendário/Feriados/Regime)** | **1.683,57** | **3,86%** | **2.147,96** | **-551,65** | **-24,68%** | **-1,30 p.p.** | **18 / 26** |

> A run `lightgbm_lag7_calendar` (Etapa 5) registra o modelo treinado com um único corte (holdout dos mesmos 182 dias, sem re-treino semanal): MAE 1.719,41 MWmed, MAPE 3,95%. O backtest walk-forward é um pouco melhor porque re-treina o modelo a cada semana.

---

## Previsão Operacional para os Próximos 7 Dias (Subsistema `SE`)

Última data observada no dataset da ONS: **2026-11-14**. Previsão gerada para **2026-11-15 a 2026-11-21**:

| Data Prevista | Dia da Semana | Feriado Nacional? | Baseline Lag 7 (MWmed) | AutoETS (MWmed) | LightGBM (MWmed) | Diferença LightGBM vs. Baseline (MWmed) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **2026-11-15** | Domingo | Sim (Proclamação da República) | 42.726,56 | 41.673,03 | **39.268,62** | -3.457,94 |
| **2026-11-16** | Segunda-feira | Não | 44.313,35 | 47.367,78 | **45.676,83** | +1.363,48 |
| **2026-11-17** | Terça-feira | Não | 44.512,00 | 48.466,72 | **47.312,66** | +2.800,66 |
| **2026-11-18** | Quarta-feira | Não | 46.071,53 | 48.586,56 | **48.062,63** | +1.991,10 |
| **2026-11-19** | Quinta-feira | Não | 47.496,08 | 48.317,10 | **47.664,32** | +168,23 |
| **2026-11-20** | Sexta-feira | Sim (Consciência Negra) | 48.216,85 | 48.532,84 | **43.212,10** | -5.004,76 |
| **2026-11-21** | Sábado | Não | 45.241,43 | 45.241,30 | **44.995,19** | -246,24 |

> **Destaque de Negócio:** Na sexta-feira **20/11/2026** (*Dia Nacional de Zumbi e da Consciência Negra*), o baseline ingênuo repete a sexta-feira útil anterior (48.216,85 MWmed), enquanto o **LightGBM** identifica o feriado nacional e reduz a carga prevista em **5.004,76 MWmed** (para 43.212,10 MWmed).

---

## Mudança de Medição da Carga (Dicionário de Dados do ONS)

O dicionário de dados (`DicionarioDados_Carga_Energia_Diaria.pdf`, v1.2 de 01/05/2023) é baixado junto com a camada Bronze em `data/docs/`. Ele define os tipos e proíbe carga nula, zerada ou negativa (regras aplicadas na Silver). A versão coincide com a inclusão da **Micro e Minigeração Distribuída (MMGD)** na carga pelo ONS, que aparece nos dados SE como um degrau: a variação anual passa de **-3,5% (abr/2023)** para **+4,0% (mai/2023)** e chega a **+19% (nov–dez/2023)**.

Tratamento no modelo: duas variáveis conhecidas *ex-ante* em `src/features.py`:

* `regime_mmgd`: 1 a partir de 01/05/2023.
* `lag364_cruza_regime`: 1 enquanto o `lag_364` ainda aponta para o regime antigo (mai/2023 a abr/2024).

O efeito no backtest é pequeno (MAE de 1.684,74 → 1.683,57 MWmed), porque todo o período de teste já está no regime novo e os lags de 7 a 28 dias se ajustam ao nível atual. Mesmo assim, as variáveis evitam que o modelo leia o degrau como crescimento de consumo.

---

## Bônus: Modelo Global Único para Todos os Subsistemas (`SE`, `S`, `NE`, `N`)

Normalizando as séries pela média móvel defasada em 7 dias (`rolling_mean_28_lag7`), um único modelo global `LightGBM` supera o baseline em todos os 4 subsistemas do Sistema Interligado Nacional (SIN). Registrado no MLflow como `global_model_4_subsistemas`:

| Subsistema | Dias de Teste | MAE Baseline (MWmed) | MAPE Baseline (%) | MAE LightGBM Global (MWmed) | MAPE LightGBM Global (%) | Redução Relativa de MAE (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SE** (Sudeste/CO) | 182 | 2.235,22 | 5,16% | **1.686,32** | **3,89%** | **-24,56%** |
| **S** (Sul) | 182 | 606,12 | 4,61% | **463,41** | **3,46%** | **-23,55%** |
| **NE** (Nordeste) | 182 | 425,88 | 3,06% | **319,91** | **2,29%** | **-24,88%** |
| **N** (Norte) | 182 | 236,35 | 2,60% | **212,03** | **2,33%** | **-10,29%** |

---

## Arquitetura do Projeto

```text
Previsao_Carga_Energia/
├── data/
│   ├── bronze/                  # Parquets brutos do ONS (2018–presente)
│   ├── docs/                    # Dicionário de dados oficial do ONS (PDF + JSON)
│   ├── warehouse.duckdb         # Banco DuckDB com tabelas silver e gold
│   └── outputs/                 # CSVs de backtesting, previsão 7d e resultados do dbt
├── dbt_ons/                     # Projeto dbt (Etapa 3)
│   ├── models/
│   │   ├── silver/silver_carga_energia.sql
│   │   ├── gold/gold_carga_se.sql
│   │   └── schema.yml           # Testes de schema (unique, not_null, accepted_values)
│   └── tests/assert_carga_positiva.sql
├── docs/
│   ├── index.html               # Página web unificada (gerada pelo pipeline)
│   └── assets/plotly.min.js     # Plotly.js local (página funciona offline)
├── src/
│   ├── ingest_bronze.py         # Etapa 2: Parquets do S3 da ONS + dicionário de dados
│   ├── dbt_runner.py            # Etapa 3: dbt run + dbt test (guarda os resultados)
│   ├── baseline.py              # Etapa 4: Modelo Seasonal Naive (lag 7)
│   ├── features.py              # Etapa 5: Atributos (lags >= 7, feriados, regime MMGD)
│   ├── train_model.py           # Etapa 5: Treinamento LightGBM + tracking no MLflow
│   ├── evaluate.py              # Etapa 6: Backtesting walk-forward + previsão 7d (MLflow)
│   ├── global_model.py          # Bônus: Modelo global único para SE, S, NE e N (MLflow)
│   ├── build_report.py          # Página unificada: junta pipeline, dbt, MLflow e resultados
│   ├── report_template.html     # Template da página (HTML + CSS + Plotly.js)
│   ├── serve.py                 # Sobe a página (8000) + MLflow UI (5000)
│   └── run_pipeline.py          # Orquestrador de comando único
├── tests/test_pipeline.py       # Testes (anti-vazamento, baseline, métricas, regime, página)
├── Dockerfile                   # Containerização reprodutível
├── Makefile                     # Atalhos de automação
└── pyproject.toml               # Dependências e configuração do ruff/pytest
```

---

## Como Rodar o Projeto

### 1. Instalar dependências (via `uv`)
```bash
uv sync --python 3.12
```

### 2. Executar o pipeline completo com um único comando
```bash
uv run python -m src.run_pipeline
```
Ou executando cada etapa individualmente na ordem (atalhos no `Makefile`):
```bash
uv run python -m src.ingest_bronze   # Etapa 2: Bronze (Parquet + dicionário de dados)
uv run python -m src.dbt_runner      # Etapa 3: Silver/Gold + 10 testes do dbt
uv run python -m src.baseline        # Etapa 4: Baseline (lag 7)
uv run python -m src.train_model     # Etapa 5: LightGBM + MLflow
uv run python -m src.evaluate        # Etapa 6: Backtesting + previsão 7 dias (MLflow)
uv run python -m src.global_model    # Bônus: modelo global (SE, S, NE, N)
uv run python -m src.build_report    # Página unificada docs/index.html
```

### 3. Abrir a página unificada (com o MLflow ligado)
```bash
uv run python -m src.serve
```

### 4. Testes e lint
```bash
uv run pytest -v
uv run ruff check src tests
```

### Docker
```bash
docker build -t carga-ons .
docker run -p 8000:8000 -p 5000:5000 carga-ons
```

---

## O Que Eu Faria com Mais Tempo

1. **Integração de Temperatura Exógena (INMET / Open-Meteo):** A carga do subsistema `SE` é fortemente sensível a ondas de calor (acionamento de ar-condicionado comercial e residencial). Incorporar a previsão meteorológica de temperatura máxima/mínima ponderada por população (São Paulo, Rio de Janeiro, Belo Horizonte, Goiânia) reduziria substancialmente os erros de pico no verão.
2. **Modelos Diretos por Horizonte ($h = 1, \dots, 7$) e Previsão Probabilística (Quantis):** O gráfico de MAE por dia do horizonte mostra que o AutoETS, que usa o último dado, acerta bem melhor em D+1. Regressores específicos por passo $h$ (o passo $h=1$ pode usar até $y_{t-1}$) capturariam esse ganho, e `objective="quantile"` daria intervalos ($P_{10}, P_{50}, P_{90}$) para o dimensionamento de reserva operativa.
3. **Reconstituição da Série Pré-MMGD:** Hoje o degrau de maio/2023 é tratado com variáveis de regime; com a série histórica de MMGD do ONS seria possível corrigir o histórico anterior para a mesma base de medição.

---

## Roteiro para a Apresentação de 5 Minutos

* **Resultado:** O pipeline vai do Parquet bruto no S3 da ONS até o backtesting de 26 semanas (182 dias), passando por 10 testes no `dbt`. O **LightGBM** reduziu o erro absoluto médio (**MAE**) em **24,68%** frente ao baseline de repetir a semana anterior (**1.683,57 MWmed** vs **2.235,22 MWmed**) e reduziu o **MAPE** de **5,16%** para **3,86%**, além de superar também o concorrente estatístico `AutoETS` (**1.826,02 MWmed**). Tudo é apresentado numa página única (`docs/index.html`) ligada ao MLflow.
* **Uma Decisão Técnica Importante:** Além de criar indicadores de feriado e dia-ponte para a data-alvo $t$, criamos variáveis indicando **se 7 ou 14 dias atrás ($t-7$, $t-14$) foram feriados (`is_feriado_lag_7`, `is_ponte_lag_7`)**, combinadas com a média ponderada das 4 últimas semanas do mesmo dia da semana (`weighted_same_dow_4w`). Assim, o modelo aprende tanto a derrubar a carga quando $t$ é feriado quanto a **não** repetir o valor de $t-7$ quando a semana anterior teve feriado e a semana atual é normal.
* **Uma Dificuldade Encontrada e Como Foi Resolvida:** Ao treinar o modelo global único para os 4 subsistemas (`SE`, `S`, `NE`, `N`), inicialmente o modelo melhorava `SE` e `S`, mas piorava `NE` e `N` devido à diferença de escala de 5,5x entre o Sudeste (~45.000 MWmed) e o Norte (~8.000 MWmed) na função perda L1. Resolvemos normalizando o alvo e os *lags* de cada subsistema pela sua própria média móvel de 28 dias defasada em 7 dias (`rolling_mean_28_lag7`), fazendo o modelo global bater o baseline nos 4 subsistemas simultaneamente.
