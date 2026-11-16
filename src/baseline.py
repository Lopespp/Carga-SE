"""Etapa 4: Modelo Baseline (Seasonal Naive de 7 dias).

A carga prevista para o dia `t` (7 dias à frente) é igual à carga observada
no mesmo dia da semana anterior (`t - 7`).
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

WAREHOUSE_PATH = Path("data/warehouse.duckdb")


def load_gold_se(warehouse_path: Path = WAREHOUSE_PATH) -> pd.DataFrame:
    """Carrega a série diária do subsistema SE da camada Gold no DuckDB."""
    if not warehouse_path.exists():
        raise FileNotFoundError(
            f"Banco DuckDB não encontrado em {warehouse_path}. Execute o dbt primeiro."
        )
    con = duckdb.connect(str(warehouse_path), read_only=True)
    df = con.execute(
        """
        SELECT
            data_referencia,
            id_subsistema,
            nom_subsistema,
            carga_mwmed
        FROM gold_carga_se
        ORDER BY data_referencia ASC
        """
    ).df()
    con.close()
    df["data_referencia"] = pd.to_datetime(df["data_referencia"])
    return df.sort_values("data_referencia").reset_index(drop=True)


def add_baseline_predictions(df: pd.DataFrame, horizon: int = 7) -> pd.DataFrame:
    """Adiciona a coluna `pred_baseline` repetindo o valor de `horizon` (7) dias atrás."""
    out = df.copy().sort_values("data_referencia").reset_index(drop=True)
    out["pred_baseline"] = out["carga_mwmed"].shift(horizon)
    return out


def forecast_next_7_days_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """Gera a previsão baseline para os próximos 7 dias após a última data observada."""
    ordered = df.sort_values("data_referencia").reset_index(drop=True)
    last_7 = ordered.tail(7).copy()
    last_7["data_previsao"] = last_7["data_referencia"] + pd.Timedelta(days=7)
    last_7["pred_baseline"] = last_7["carga_mwmed"]
    return last_7[["data_previsao", "id_subsistema", "pred_baseline"]].reset_index(drop=True)


if __name__ == "__main__":
    df_gold = load_gold_se()
    df_base = add_baseline_predictions(df_gold)
    valid = df_base.dropna(subset=["pred_baseline"])
    err = (valid["carga_mwmed"] - valid["pred_baseline"]).abs()
    mae = err.mean()
    mape = (err / valid["carga_mwmed"]).mean() * 100
    print(
        f"Baseline (Lag 7) calculado em {len(valid)} dias | "
        f"MAE histórico global: {mae:.2f} MWmed | MAPE: {mape:.2f}%"
    )
