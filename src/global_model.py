"""Desafio Bônus: Modelo Global LightGBM para todos os subsistemas (SE, S, NE, N).

Treina um único modelo LightGBM sobre a camada Silver (`silver_carga_energia`)
prevendo simultaneamente os 4 subsistemas do SIN com horizonte de 7 dias.
Usa normalização por escala local (`rolling_mean_28_lag7`, estritamente <= t-7)
para unificar séries de portes distintos (SE ~45 GW vs N ~8 GW) num único modelo.
"""

from __future__ import annotations

import logging
from pathlib import Path

import duckdb
import lightgbm as lgb
import mlflow
import pandas as pd

from src.baseline import WAREHOUSE_PATH, add_baseline_predictions
from src.features import FEATURE_COLUMNS, build_features
from src.train_model import (
    DEFAULT_LGBM_PARAMS,
    EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    compute_metrics,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("data/outputs")
SUBSYSTEM_MAP = {"SE": 0, "S": 1, "NE": 2, "N": 3}

SCALE_COLS = [
    "lag_7",
    "lag_14",
    "lag_21",
    "lag_28",
    "lag_35",
    "lag_56",
    "lag_364",
    "lag_8",
    "lag_9",
    "lag_10",
    "lag_11",
    "lag_12",
    "lag_13",
    "mean_same_dow_4w",
    "weighted_same_dow_4w",
    "std_same_dow_4w",
    "rolling_mean_7_lag7",
    "rolling_std_7_lag7",
    "rolling_min_7_lag7",
    "rolling_max_7_lag7",
    "rolling_mean_14_lag7",
    "diff_lag7_lag14",
    "diff_lag7_mean_dow4w",
]


def load_silver_all(warehouse_path: Path = WAREHOUSE_PATH) -> pd.DataFrame:
    """Carrega a série diária de todos os 4 subsistemas da camada Silver."""
    con = duckdb.connect(str(warehouse_path), read_only=True)
    df = con.execute(
        """
        SELECT
            data_referencia,
            id_subsistema,
            nom_subsistema,
            carga_mwmed
        FROM silver_carga_energia
        ORDER BY id_subsistema, data_referencia
        """
    ).df()
    con.close()
    df["data_referencia"] = pd.to_datetime(df["data_referencia"])
    return df


def run_global_model_evaluation(test_days: int = 182) -> pd.DataFrame:
    """Constroi features normalizadas por subsistema e avalia modelo global único."""
    df_all = load_silver_all()

    parts: list[pd.DataFrame] = []
    for sub_id, group in df_all.groupby("id_subsistema"):
        feat_sub = add_baseline_predictions(build_features(group), horizon=7)
        feat_sub["subsistema_code"] = SUBSYSTEM_MAP.get(str(sub_id), -1)
        scale = feat_sub["rolling_mean_28_lag7"]
        for col in SCALE_COLS:
            feat_sub[col] = feat_sub[col] / scale
        feat_sub["target_norm"] = feat_sub["carga_mwmed"] / scale
        parts.append(feat_sub)

    full_df = pd.concat(parts, ignore_index=True)
    global_features = ["subsistema_code"] + FEATURE_COLUMNS
    req_cols = global_features + ["target_norm", "carga_mwmed", "pred_baseline"]
    valid_df = full_df.dropna(subset=req_cols).copy()

    cutoff_date = valid_df["data_referencia"].max() - pd.Timedelta(days=test_days)
    train_df = valid_df[valid_df["data_referencia"] <= cutoff_date].copy()
    test_df = valid_df[valid_df["data_referencia"] > cutoff_date].copy()

    params = {**DEFAULT_LGBM_PARAMS, "n_estimators": 450}
    model = lgb.LGBMRegressor(**params)
    model.fit(
        train_df[global_features],
        train_df["target_norm"],
        categorical_feature=["subsistema_code"],
    )

    test_df["pred_global_lgbm"] = (
        model.predict(test_df[global_features]) * test_df["rolling_mean_28_lag7"]
    )

    rows: list[dict] = []
    for sub_id in ["SE", "S", "NE", "N"]:
        sub_test = test_df[test_df["id_subsistema"] == sub_id]
        m_base = compute_metrics(sub_test["carga_mwmed"], sub_test["pred_baseline"])
        m_glob = compute_metrics(sub_test["carga_mwmed"], sub_test["pred_global_lgbm"])
        rows.append(
            {
                "id_subsistema": sub_id,
                "dias_teste": len(sub_test),
                "mae_baseline_mwmed": round(m_base["mae"], 2),
                "mape_baseline_pct": round(m_base["mape"], 2),
                "mae_global_lgbm_mwmed": round(m_glob["mae"], 2),
                "mape_global_lgbm_pct": round(m_glob["mape"], 2),
                "reducao_mae_pct": round(
                    (m_base["mae"] - m_glob["mae"]) / m_base["mae"] * 100.0, 2
                ),
            }
        )

    res_df = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "global_model_summary.csv"
    res_df.to_csv(out_path, index=False)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="global_model_4_subsistemas"):
        mlflow.set_tag("etapa", "bonus_modelo_global")
        mlflow.log_params(
            {
                **params,
                "test_days": test_days,
                "subsistemas": "SE,S,NE,N",
                "normalizacao": "rolling_mean_28_lag7",
            }
        )
        for row in rows:
            sub = row["id_subsistema"]
            mlflow.log_metric(f"mae_baseline_{sub}", row["mae_baseline_mwmed"])
            mlflow.log_metric(f"mae_global_{sub}", row["mae_global_lgbm_mwmed"])
            mlflow.log_metric(f"mape_global_{sub}", row["mape_global_lgbm_pct"])
            mlflow.log_metric(f"reducao_mae_pct_{sub}", row["reducao_mae_pct"])
        mlflow.log_artifact(str(out_path))
    print("\n=== MODELO GLOBAL LIGHTGBM (TODOS OS SUBSISTEMAS: SE, S, NE, N) ===")
    print(res_df.to_string(index=False))
    return res_df


if __name__ == "__main__":
    run_global_model_evaluation()
