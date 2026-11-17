"""Etapa 3: Executa `dbt run` e `dbt test` e preserva os resultados para o relatório."""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

DBT_DIR = Path("dbt_ons")
OUTPUT_DIR = Path("data/outputs")


def run_dbt(dbt_dir: Path = DBT_DIR, output_dir: Path = OUTPUT_DIR) -> None:
    """Roda `dbt run` e `dbt test`, copiando cada `run_results.json` para `data/outputs/`.

    O dbt sobrescreve `target/run_results.json` a cada comando; guardar as duas cópias
    permite que a página de resultados mostre o status de modelos e testes.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for subcmd in ["run", "test"]:
        logger.info("Executando dbt %s...", subcmd)
        subprocess.run(
            [sys.executable, "-m", "dbt.cli.main", subcmd, "--profiles-dir", "."],
            cwd=dbt_dir,
            check=True,
        )
        shutil.copyfile(
            dbt_dir / "target" / "run_results.json",
            output_dir / f"dbt_{subcmd}_results.json",
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    run_dbt()
