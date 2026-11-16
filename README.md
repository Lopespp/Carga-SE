# Previsão de Carga de Energia Diária — ONS (Subsistema Sudeste/Centro-Oeste)

Pipeline completo de Engenharia e Ciência de Dados para prever a **carga diária de energia elétrica (em MWmed)** do subsistema **Sudeste/Centro-Oeste (`SE`)** para um horizonte de **7 dias à frente**, utilizando os dados abertos do Operador Nacional do Sistema Elétrico (ONS) de 2018 até o presente.

---

## Resposta à Pergunta do Desafio (em até 5 linhas)

**Sim.** No backtesting temporal de 26 janelas sucessivas de 7 dias (182 dias, sem vazamento de dados e usando apenas $\text{lags} \ge 7$), o modelo **LightGBM** superou o baseline (*Seasonal Naive* de 7 dias), reduzindo o **MAE em 24,63%** (de **2.235,22 MWmed** para **1.684,74 MWmed**, ganho de **550,48 MWmed**) e reduzindo o **MAPE em 1,30 ponto percentual** (de **5,16%** para **3,86%**, queda relativa de **25,19%** no erro percentual), vencendo o baseline em **18 das 26 semanas** avaliadas.

---

## Tabela Comparativa no Backtesting Temporal (26 Janelas de 7 Dias = 182 Dias)

Validação *walk-forward* (*expanding window*) re-treinando os modelos a cada janela de 7 dias apenas com dados passados ($\le t_0$) e prevendo simultaneamente os dias $t_0+1$ a $t_0+7$:

| Modelo | MAE (MWmed) | MAPE (%) | RMSE (MWmed) | Redução de MAE vs. Baseline (MWmed) | Redução Relativa de MAE (%) | Redução de MAPE (p.p.) | Janelas de 7d Vencidas (em 26) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Seasonal Naive — Lag 7)** | 2.235,22 | 5,16% | 2.784,43 | 0,00 | 0,00% | 0,00 p.p. | 8 / 26 |
| **AutoETS (`statsforecast` — Sazonal 7d)** *(Bônus)* | 1.826,02 | 4,17% | 2.378,38 | -409,20 | **-18,31%** | -0,99 p.p. | 20 / 26 |
| **LightGBM (Lags $\ge$ 7 + Calendário/Feriados)** | **1.684,74** | **3,86%** | **2.132,94** | **-550,48** | **-24,63%** | **-1,30 p.p.** | **18 / 26** |

---

## Previsão Operacional para os Próximos 7 Dias (Subsistema `SE`)

Última data observada no dataset da ONS: **2026-11-14**. Previsão gerada para **2026-11-15 a 2026-11-21**:

| Data Prevista | Dia da Semana | Feriado Nacional? | Baseline Lag 7 (MWmed) | AutoETS (MWmed) | LightGBM (MWmed) | Diferença LightGBM vs. Baseline (MWmed) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **2026-11-15** | Domingo | Sim (15/Nov) | 42.726,56 | 41.673,03 | **39.393,70** | -3.332,86 |
| **2026-11-16** | Segunda-feira | Não | 44.313,35 | 47.367,78 | **45.799,17** | +1.485,82 |
| **2026-11-17** | Terça-feira | Não | 44.512,00 | 48.466,72 | **47.698,21** | +3.186,21 |
| **2026-11-18** | Quarta-feira | Não | 46.071,53 | 48.586,56 | **48.402,75** | +2.331,22 |
| **2026-11-19** | Quinta-feira | Não | 47.496,08 | 48.317,10 | **47.625,35** | +129,26 |
| **2026-11-20** | Sexta-feira | Sim (20/Nov) | 48.216,85 | 48.532,84 | **43.391,14** | -4.825,72 |
| **2026-11-21** | Sábado | Não | 45.241,43 | 45.241,30 | **44.817,84** | -423,59 |

> **Destaque de Negócio:** Na sexta-feira **20/11/2026** (*Dia Nacional de Zumbi e da Consciência Negra*), o baseline ingênuo repete a sexta-feira útil anterior (48.216,85 MWmed), enquanto o **LightGBM** identifica o feriado nacional e reduz a carga prevista em **4.825,72 MWmed** (para 43.391,14 MWmed).

---

## Bônus: Modelo Global Único para Todos os Subsistemas (`SE`, `S`, `NE`, `N`)

Normalizando as séries pela média móvel defasada em 7 dias (`rolling_mean_28_lag7`), um único modelo global `LightGBM` supera o baseline em todos os 4 subsistemas do Sistema Interligado Nacional (SIN):

| Subsistema | Dias de Teste | MAE Baseline (MWmed) | MAPE Baseline (%) | MAE LightGBM Global (MWmed) | MAPE LightGBM Global (%) | Redução Relativa de MAE (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SE** (Sudeste/CO) | 182 | 2.235,22 | 5,16% | **1.681,04** | **3,89%** | **-24,79%** |
| **S** (Sul) | 182 | 606,12 | 4,61% | **465,17** | **3,49%** | **-23,25%** |
| **NE** (Nordeste) | 182 | 425,88 | 3,06% | **318,56** | **2,28%** | **-25,20%** |
| **N** (Norte) | 182 | 236,35 | 2,60% | **212,43** | **2,33%** | **-10,12%** |

---

## Arquitetura do Projeto

```text
Previsao_Carga_Energia/
├── data/
│   ├── bronze/                  # Parquets brutos do ONS (2018–presente)
│   ├── warehouse.duckdb         # Banco DuckDB com tabelas silver e gold
│   └── outputs/                 # CSVs de backtesting, feature importance e previsão 7d
├── dbt_ons/                     # Projeto dbt (Etapa 3)
│   ├── dbt_project.yml
│   ├── profiles.yml
│   ├── models/
│   │   ├── silver/silver_carga_energia.sql
│   │   ├── gold/gold_carga_se.sql
│   │   └── schema.yml           # Testes de schema (unique, not_null, accepted_values)
│   └── tests/
│       └── assert_carga_positiva.sql
├── src/
│   ├── ingest_bronze.py         # Etapa 2: Download dos Parquets do S3 da ONS
│   ├── baseline.py              # Etapa 4: Modelo Seasonal Naive (lag 7)
│   ├── features.py              # Etapa 5: Engenharia de atributos (lags >= 7 + feriados)
│   ├── train_model.py           # Etapa 5: Treinamento LightGBM + tracking no MLflow
│   ├── evaluate.py              # Etapa 6: Backtesting temporal (LightGBM vs AutoETS vs Baseline)
│   ├── global_model.py          # Bônus: Modelo global único para SE, S, NE e N
│   └── run_pipeline.py          # Bônus: Orquestrador de comando único
├── tests/
│   └── test_pipeline.py         # Bônus: Testes automatizados (incluindo prova contra vazamento)
├── app.py                       # Bônus: Dashboard interativo em Streamlit
├── Dockerfile                   # Bônus: Containerização reprodutível
├── Makefile                     # Bônus: Atalhos de automação
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
Ou executando cada etapa individualmente na ordem:
```bash
# Etapa 2: Ingestão Bronze (S3 ONS -> data/bronze/*.parquet)
uv run python -m src.ingest_bronze

# Etapa 3: Transformação Silver e Gold + 10 testes de qualidade no dbt
cd dbt_ons && uv run dbt run --profiles-dir . && uv run dbt test --profiles-dir . && cd ..

# Etapa 4: Baseline Seasonal Naive (Lag 7)
uv run python -m src.baseline

# Etapa 5: Treinamento LightGBM + Registro no MLflow
uv run python -m src.train_model

# Etapa 6: Backtesting em janela temporal (26 janelas de 7 dias) + Previsão 7 dias
uv run python -m src.evaluate

# Bônus: Modelo Global (SE, S, NE, N), Testes e Lint
uv run python -m src.global_model
uv run pytest -v
uv run ruff check src tests app.py
```

### 3. Abrir o Dashboard Interativo (`Streamlit`) ou a UI do `MLflow`
```bash
uv run streamlit run app.py
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

---

## O Que Eu Faria com Mais Tempo

1. **Integração de Temperatura Exógena (INMET / Open-Meteo):** A carga do subsistema `SE` é fortemente sensível a ondas de calor (acionamento de ar-condicionado comercial e residencial). Incorporar a previsão meteorológica de temperatura máxima/mínima ponderada por população (São Paulo, Rio de Janeiro, Belo Horizonte, Goiânia) reduziria substancialmente os erros de pico no verão.
2. **Modelos Diretos por Horizonte ($h = 1, \dots, 7$) e Previsão Probabilística (Quantis):** Treinar regressores específicos para cada passo $h \in \{1, \dots, 7\}$ (onde o passo $h=1$ pode usar até $y_{t-1}$ e o passo $h=7$ usa até $y_{t-7}$) e estimar intervalos de confiança ($P_{10}, P_{50}, P_{90}$) com `objective="quantile"` para apoiar o dimensionamento de reserva operativa.
3. **Tratamento Explícito da Mudança Metodológica de MMGD (Maio/2023):** Modelar separadamente a reconstituição da Micro e Minigeração Distribuída (MMGD) incorporada pelo ONS a partir de maio de 2023.

---

## Roteiro para a Apresentação de 5 Minutos

* **Resultado:** O pipeline vai do Parquet bruto no S3 da ONS até o backtesting de 26 semanas (182 dias) passando por 10 testes no `dbt`. O **LightGBM** reduziu o erro absoluto médio (**MAE**) em **24,63%** frente ao baseline de repetir a semana anterior (**1.684,74 MWmed** vs **2.235,22 MWmed**) e reduziu o **MAPE** de **5,16%** para **3,86%**, além de superar também o concorrente estatístico `AutoETS` (**1.826,02 MWmed**).
* **Uma Decisão Técnica Importante:** Além de criar indicadores de feriado e dia-ponte para a data-alvo $t$, criamos variáveis indicando **se 7 ou 14 dias atrás ($t-7$, $t-14$) foram feriados (`is_feriado_lag_7`, `is_ponte_lag_7`)**, combinadas com a média ponderada das 4 últimas semanas do mesmo dia da semana (`weighted_same_dow_4w`). Assim, o modelo aprende tanto a derrubar a carga quando $t$ é feriado quanto a **não** repetir o valor de $t-7$ quando a semana anterior teve feriado e a semana atual é normal.
* **Uma Dificuldade Encontrada e Como Foi Resolvida:** Ao treinar o modelo global único para os 4 subsistemas (`SE`, `S`, `NE`, `N`), inicialmente o modelo melhorava `SE` e `S`, mas piorava `NE` e `N` devido à diferença de escala de 5,5x entre o Sudeste (~45.000 MWmed) e o Norte (~8.000 MWmed) na função perda L1. Resolvemos normalizando o alvo e os *lags* de cada subsistema pela sua própria média móvel de 28 dias defasada em 7 dias (`rolling_mean_28_lag7`), fazendo o modelo global bater o baseline nos 4 subsistemas simultaneamente.
