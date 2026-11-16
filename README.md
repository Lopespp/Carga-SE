# Previsão de Carga de Energia Diária — ONS (Subsistema SE/CO)

Pipeline de Engenharia e Ciência de Dados para prever a **carga diária de energia elétrica (em MWmed)** do subsistema **Sudeste/Centro-Oeste (`SE`)** para os **próximos 7 dias**, utilizando dados abertos do Operador Nacional do Sistema Elétrico (ONS).

## Resposta à Pergunta do Desafio

*(Será preenchida na Etapa 6 após a execução do backtesting comparando LightGBM vs. Baseline Seasonal Naive).*

## Arquitetura do Pipeline

1. **Ambiente:** Python (`uv`), Git, Docker.
2. **Ingestão (Bronze):** Extração dos arquivos Parquet brutos (2018–presente) do bucket público da ONS na AWS (`s3://ons-aws-prod-opendata/dataset/carga_energia_di/`).
3. **Transformação (Silver & Gold):** Modelagem e testes de qualidade de dados com `dbt` + `DuckDB`.
4. **Baseline:** Previsor ingênuo sazonal semanal ($\hat{y}_t = y_{t-7}$).
5. **Modelo de Machine Learning:** `LightGBM` com variáveis de calendário, feriados nacionais e *lags* $\ge 7$ dias (sem vazamento temporal), registrado no `MLflow`.
6. **Avaliação:** Backtesting em janela temporal comparando MAE (MWmed) e MAPE (%).
