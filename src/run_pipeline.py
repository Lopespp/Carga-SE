"""Script orquestrador para executar todas as etapas do pipeline com um único comando."""

from __future__ import annotations

import logging

from src.baseline import add_baseline_predictions, load_gold_se
from src.build_report import build_report
from src.dbt_runner import run_dbt
from src.evaluate import main as run_evaluation
from src.global_model import run_global_model_evaluation
from src.ingest_bronze import download_data_dictionary, ingest_bronze
from src.train_model import train_and_log_mlflow

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("=== [1/7] Ingestão Bronze (S3 ONS -> Parquet + Dicionário de Dados) ===")
    download_data_dictionary()
    ingest_bronze()

    logger.info("=== [2/7] Transformação Silver & Gold + Testes (dbt + DuckDB) ===")
    run_dbt()

    logger.info("=== [3/7] Cálculo do Baseline (Lag 7) ===")
    df_gold = load_gold_se()
    df_base = add_baseline_predictions(df_gold, horizon=7).dropna(subset=["pred_baseline"])
    mae_hist = (df_base["carga_mwmed"] - df_base["pred_baseline"]).abs().mean()
    logger.info("Baseline histórico validado: MAE = %.2f MWmed", mae_hist)

    logger.info("=== [4/7] Treinamento LightGBM + Registro no MLflow ===")
    train_and_log_mlflow()

    logger.info("=== [5/7] Avaliação em Backtesting Temporal + Previsão 7 Dias ===")
    run_evaluation()

    logger.info("=== [6/7] Bônus: Modelo Global (SE, S, NE, N) ===")
    run_global_model_evaluation()

    logger.info("=== [7/7] Página Web Unificada de Resultados (docs/index.html) ===")
    build_report()

    logger.info("Pipeline completo executado com sucesso!")


if __name__ == "__main__":
    main()
