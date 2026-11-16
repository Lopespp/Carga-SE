"""Etapa 2: Ingestão (Camada Bronze).

Baixa os arquivos Parquet de Carga de Energia Diária do ONS (de 2018 até o ano atual)
diretamente do bucket público da AWS sem nenhuma alteração e salva em `data/bronze/`.
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import duckdb
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

S3_BUCKET_URL = "https://ons-aws-prod-opendata.s3.amazonaws.com"
DATASET_PREFIX = "dataset/carga_energia_di/"
START_YEAR = 2018
BRONZE_DIR = Path("data/bronze")


def list_parquet_keys(start_year: int = START_YEAR) -> list[str]:
    """Lista todas as chaves Parquet do dataset a partir de `start_year`."""
    resp = requests.get(
        S3_BUCKET_URL,
        params={"list-type": "2", "prefix": DATASET_PREFIX},
        timeout=30,
    )
    resp.raise_for_status()
    root = ET.fromstring(resp.text)
    ns = "{http://s3.amazonaws.com/doc/2006-03-01/}"

    keys: list[str] = []
    pattern = re.compile(r"CARGA_ENERGIA_(\d{4})\.parquet$")
    for elem in root.iter(f"{ns}Key"):
        if not elem.text:
            continue
        match = pattern.search(elem.text)
        if match and int(match.group(1)) >= start_year:
            keys.append(elem.text)

    return sorted(keys)


def ingest_bronze(bronze_dir: Path = BRONZE_DIR, start_year: int = START_YEAR) -> list[Path]:
    """Baixa os arquivos Parquet brutos de `start_year` até o presente sem alterações."""
    bronze_dir.mkdir(parents=True, exist_ok=True)
    keys = list_parquet_keys(start_year=start_year)
    if not keys:
        raise RuntimeError("Nenhum arquivo Parquet encontrado no bucket da ONS.")

    downloaded_files: list[Path] = []
    for key in keys:
        filename = Path(key).name
        target_path = bronze_dir / filename
        url = f"{S3_BUCKET_URL}/{key}"
        logger.info("Baixando %s -> %s", url, target_path)
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        target_path.write_bytes(resp.content)
        downloaded_files.append(target_path)

    # Validação de leitura com DuckDB
    con = duckdb.connect()
    glob_path = str(bronze_dir / "CARGA_ENERGIA_*.parquet").replace("\\", "/")
    summary = con.execute(
        f"""
        SELECT
            COUNT(*) AS total_rows,
            MIN(din_instante) AS min_date,
            MAX(din_instante) AS max_date,
            COUNT(DISTINCT id_subsistema) AS n_subsistemas
        FROM read_parquet('{glob_path}')
        """
    ).fetchone()
    logger.info(
        "Camada Bronze concluída: %d arquivos | %d linhas | período: %s a %s | %d subsistemas",
        len(downloaded_files),
        summary[0],
        summary[1],
        summary[2],
        summary[3],
    )
    return downloaded_files


if __name__ == "__main__":
    ingest_bronze()
