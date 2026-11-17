"""Etapa 5: Treinamento do Modelo LightGBM e Registro no MLflow.

Treina um regressor LightGBM usando exclusivamente lags >= 7 dias e variáveis
de calendário/feriados, validando em corte temporal estrito (passado -> futuro)
e registrando parâmetros, métricas e modelo no MLflow.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd

from src.baseline import add_baseline_predictions, load_gold_se
from src.features import FEATURE_COLUMNS, build_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "ONS_Carga_Energia_SE_7d"
MODEL_DIR = Path("data/outputs")

DEFAULT_LGBM_PARAMS: dict[str, Any] = {
    "objective": "regression_l1",
    "metric": "mae",
    "boosting_type": "gbdt",
    "n_estimators": 600,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": 6,
    "min_child_samples": 25,
    "subsample": 0.85,
    "subsample_freq": 1,
    "colsample_bytree": 0.85,
    "reg_alpha": 1.0,
    "reg_lambda": 2.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}


def compute_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> dict[str, float]:
    """Calcula MAE (MWmed) e MAPE (%)."""
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    abs_err = np.abs(y_t - y_p)
    mae = float(np.mean(abs_err))
    mape = float(np.mean(abs_err / y_t) * 100.0)
    rmse = float(np.sqrt(np.mean((y_t - y_p) ** 2)))
    return {"mae": mae, "mape": mape, "rmse": rmse}


def fit_lgbm_pipeline(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None = None,
    params: dict[str, Any] | None = None,
) -> lgb.LGBMRegressor:
    """Treina o modelo LightGBM com early stopping opcional na janela temporal de validação."""
    model_params = {**DEFAULT_LGBM_PARAMS, **(params or {})}
    model = lgb.LGBMRegressor(**model_params)

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["carga_mwmed"]

    if val_df is not None and len(val_df) > 0:
        X_val = val_df[FEATURE_COLUMNS]
        y_val = val_df["carga_mwmed"]
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        )
    else:
        model.fit(X_train, y_train)

    return model


def train_and_log_mlflow(test_days: int = 182) -> dict[str, Any]:
    """Executa o treinamento temporal e registra experimento completo no MLflow."""
    df_gold = load_gold_se()
    df_feat = add_baseline_predictions(build_features(df_gold), horizon=7)
    required_cols = FEATURE_COLUMNS + ["carga_mwmed", "pred_baseline"]
    df_valid = df_feat.dropna(subset=required_cols).reset_index(drop=True)

    # Split estritamente temporal: passado (treino + validação interna) vs futuro (teste)
    split_idx = len(df_valid) - test_days
    history_df = df_valid.iloc[:split_idx].copy()
    test_df = df_valid.iloc[split_idx:].copy()

    val_size = min(90, max(30, int(len(history_df) * 0.1)))
    train_df = history_df.iloc[:-val_size].copy()
    val_df = history_df.iloc[-val_size:].copy()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="lightgbm_lag7_calendar") as run:
        mlflow.set_tag("etapa", "5_modelo")
        model = fit_lgbm_pipeline(train_df, val_df)
        best_iter = getattr(model, "best_iteration_", None) or DEFAULT_LGBM_PARAMS["n_estimators"]

        # Re-treina no histórico completo pré-teste com o número ótimo de árvores
        final_params = {**DEFAULT_LGBM_PARAMS, "n_estimators": max(50, int(best_iter))}
        final_model = fit_lgbm_pipeline(history_df, val_df=None, params=final_params)

        preds_test = final_model.predict(test_df[FEATURE_COLUMNS])
        lgbm_metrics = compute_metrics(test_df["carga_mwmed"], preds_test)
        base_metrics = compute_metrics(test_df["carga_mwmed"], test_df["pred_baseline"])

        mlflow.log_params(final_params)
        mlflow.log_param("n_features", len(FEATURE_COLUMNS))
        mlflow.log_param("min_lag_days", 7)
        mlflow.log_param("train_start", str(history_df["data_referencia"].min().date()))
        mlflow.log_param("train_end", str(history_df["data_referencia"].max().date()))
        mlflow.log_param("test_start", str(test_df["data_referencia"].min().date()))
        mlflow.log_param("test_end", str(test_df["data_referencia"].max().date()))
        mlflow.log_param("test_days", len(test_df))

        mlflow.log_metric("test_mae_lgbm", lgbm_metrics["mae"])
        mlflow.log_metric("test_mape_lgbm", lgbm_metrics["mape"])
        mlflow.log_metric("test_rmse_lgbm", lgbm_metrics["rmse"])
        mlflow.log_metric("test_mae_baseline", base_metrics["mae"])
        mlflow.log_metric("test_mape_baseline", base_metrics["mape"])
        mlflow.log_metric(
            "mae_improvement_pct",
            (base_metrics["mae"] - lgbm_metrics["mae"]) / base_metrics["mae"] * 100.0,
        )

        # Salva importância das variáveis
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        imp_df = pd.DataFrame(
            {
                "feature": FEATURE_COLUMNS,
                "importance": final_model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)
        imp_path = MODEL_DIR / "feature_importance.csv"
        imp_df.to_csv(imp_path, index=False)
        mlflow.log_artifact(str(imp_path))

        mlflow.lightgbm.log_model(
            lgb_model=final_model,
            name="lightgbm_carga_se_7d",
            input_example=test_df[FEATURE_COLUMNS].head(5),
        )

        logger.info("Run registrada no MLflow (ID: %s)", run.info.run_id)
        logger.info(
            "Holdout Temporal (%d dias) | Baseline MAE: %.2f MWmed (MAPE: %.2f%%) | "
            "LightGBM MAE: %.2f MWmed (MAPE: %.2f%%)",
            len(test_df),
            base_metrics["mae"],
            base_metrics["mape"],
            lgbm_metrics["mae"],
            lgbm_metrics["mape"],
        )

        return {
            "run_id": run.info.run_id,
            "lgbm_metrics": lgbm_metrics,
            "baseline_metrics": base_metrics,
            "top_features": imp_df.head(10).to_dict(orient="records"),
        }


if __name__ == "__main__":
    train_and_log_mlflow()
