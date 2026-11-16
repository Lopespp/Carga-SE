"""Script orquestrador para executar todas as etapas do pipeline com um único comando."""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

from src.baseline import add_baseline_predictions, load_gold_se
from src.evaluate import main as run_evaluation
from src.ingest_bronze import ingest_bronze
from src.train_model import train_and_log_mlflow

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def run_dbt() -> None:
    """Executa `dbt run` e `dbt test` no diretório `dbt_ons`."""
    dbt_dir = Path("dbt_ons")
    for subcmd in ["run", "test"]:
        logger.info("Executando dbt %s...", subcmd)
        subprocess.run(
            [sys.executable, "-m", "dbt.cli.main", subcmd, "--profiles-dir", "."],
            cwd=dbt_dir,
            check=True,
        )


def main() -> None:
    logger.info("=== [1/5] Ingestão Bronze (S3 ONS -> Parquet) ===")
    ingest_bronze()

    logger.info("=== [2/5] Transformação Silver & Gold + Testes (dbt + DuckDB) ===")
    run_dbt()

    logger.info("=== [3/5] Cálculo do Baseline (Lag 7) ===")
    df_gold = load_gold_se()
    df_base = add_baseline_predictions(df_gold, horizon=7).dropna(subset=["pred_baseline"])
    mae_hist = (df_base["carga_mwmed"] - df_base["pred_baseline"]).abs().mean()
    logger.info("Baseline histórico validado: MAE = %.2f MWmed", mae_hist)

    logger.info("=== [4/5] Treinamento LightGBM + Registro no MLflow ===")
    train_and_log_mlflow()

    logger.info("=== [5/5] Avaliação em Backtesting Temporal + Previsão 7 Dias ===")
    run_evaluation()

    logger.info("Pipeline completo executado com sucesso!")


if __name__ == "__main__":
    main()
