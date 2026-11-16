"""Etapa 6: Avaliação por Backtesting em Janela Temporal e Previsão de 7 Dias.

Executa walk-forward backtesting com horizonte de 7 dias (expanding window),
comparando o Baseline (Seasonal Naive - Lag 7), modelo estatístico AutoETS
(`statsforecast`) e o LightGBM em MAE (MWmed) e MAPE (%).
Também gera a previsão operacional para os próximos 7 dias do subsistema SE.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoETS

from src.baseline import add_baseline_predictions, load_gold_se
from src.features import FEATURE_COLUMNS, build_features
from src.train_model import DEFAULT_LGBM_PARAMS, compute_metrics, fit_lgbm_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("data/outputs")


def _predict_auto_ets_fold(train_series_df: pd.DataFrame, horizon: int = 7) -> np.ndarray:
    """Ajusta AutoETS com sazonalidade semanal (season_length=7) nos últimos 365 dias."""
    sf_df = (
        train_series_df.tail(365)[["data_referencia", "carga_mwmed"]]
        .rename(columns={"data_referencia": "ds", "carga_mwmed": "y"})
        .copy()
    )
    sf_df["unique_id"] = "SE"
    sf = StatsForecast(
        models=[AutoETS(season_length=7, model="ZZZ")],
        freq="D",
        n_jobs=1,
    )
    forecast_df = sf.forecast(df=sf_df, h=horizon)
    return forecast_df["AutoETS"].to_numpy(dtype=float)


def run_temporal_backtest(
    df_gold: pd.DataFrame,
    n_windows: int = 26,
    horizon: int = 7,
    include_ets: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Executa backtesting com janela expansiva de `n_windows` janelas de `horizon` (7) dias.

    Em cada janela `k`:
    - Treina apenas no passado estrito (`data_referencia <= cutoff_date`)
    - Prevê os próximos `horizon` (7) dias (`cutoff_date + 1` até `cutoff_date + 7`)
    - Usa apenas variáveis defasadas em >= 7 dias (sem vazamento temporal).
    """
    df_feat = add_baseline_predictions(build_features(df_gold), horizon=horizon)
    required_cols = FEATURE_COLUMNS + ["carga_mwmed", "pred_baseline"]
    df_valid = df_feat.dropna(subset=required_cols).reset_index(drop=True)

    total_test_days = n_windows * horizon
    start_test_idx = len(df_valid) - total_test_days

    all_preds: list[pd.DataFrame] = []
    fold_records: list[dict] = []

    fast_params = {**DEFAULT_LGBM_PARAMS, "n_estimators": 350}

    for w in range(n_windows):
        fold_start = start_test_idx + w * horizon
        fold_end = fold_start + horizon

        train_fold = df_valid.iloc[:fold_start].copy()
        test_fold = df_valid.iloc[fold_start:fold_end].copy()

        model = fit_lgbm_pipeline(train_fold, val_df=None, params=fast_params)
        test_fold["pred_lightgbm"] = model.predict(test_fold[FEATURE_COLUMNS])

        if include_ets:
            try:
                test_fold["pred_auto_ets"] = _predict_auto_ets_fold(train_fold, horizon=horizon)
            except Exception as exc:  # pragma: no cover
                logger.warning("Falha no AutoETS na janela %d: %s", w + 1, exc)
                test_fold["pred_auto_ets"] = test_fold["pred_baseline"]

        test_fold["fold"] = w + 1
        test_fold["horizonte_dia"] = np.arange(1, horizon + 1)
        all_preds.append(test_fold)

        m_base = compute_metrics(test_fold["carga_mwmed"], test_fold["pred_baseline"])
        m_lgbm = compute_metrics(test_fold["carga_mwmed"], test_fold["pred_lightgbm"])
        rec = {
            "fold": w + 1,
            "inicio_janela": str(test_fold["data_referencia"].min().date()),
            "fim_janela": str(test_fold["data_referencia"].max().date()),
            "mae_baseline": m_base["mae"],
            "mape_baseline": m_base["mape"],
            "mae_lightgbm": m_lgbm["mae"],
            "mape_lightgbm": m_lgbm["mape"],
        }
        if include_ets and "pred_auto_ets" in test_fold:
            m_ets = compute_metrics(test_fold["carga_mwmed"], test_fold["pred_auto_ets"])
            rec["mae_auto_ets"] = m_ets["mae"]
            rec["mape_auto_ets"] = m_ets["mape"]
        fold_records.append(rec)

    preds_df = pd.concat(all_preds, ignore_index=True)
    folds_df = pd.DataFrame(fold_records)

    # Tabela resumo consolidada
    base_overall = compute_metrics(preds_df["carga_mwmed"], preds_df["pred_baseline"])
    lgbm_overall = compute_metrics(preds_df["carga_mwmed"], preds_df["pred_lightgbm"])

    rows = [
        {
            "modelo": "Baseline (Seasonal Naive - Lag 7)",
            "mae_mwmed": round(base_overall["mae"], 2),
            "mape_pct": round(base_overall["mape"], 2),
            "rmse_mwmed": round(base_overall["rmse"], 2),
            "ganho_mae_vs_baseline_mwmed": 0.0,
            "ganho_mae_vs_baseline_pct": 0.0,
            "ganho_mape_vs_baseline_pp": 0.0,
            "vitorias_janelas_7d": int(
                (folds_df["mae_baseline"] < folds_df["mae_lightgbm"]).sum()
            ),
        },
    ]

    if include_ets and "pred_auto_ets" in preds_df:
        ets_overall = compute_metrics(preds_df["carga_mwmed"], preds_df["pred_auto_ets"])
        rows.append(
            {
                "modelo": "AutoETS (statsforecast - Sazonal 7d)",
                "mae_mwmed": round(ets_overall["mae"], 2),
                "mape_pct": round(ets_overall["mape"], 2),
                "rmse_mwmed": round(ets_overall["rmse"], 2),
                "ganho_mae_vs_baseline_mwmed": round(
                    base_overall["mae"] - ets_overall["mae"], 2
                ),
                "ganho_mae_vs_baseline_pct": round(
                    (base_overall["mae"] - ets_overall["mae"]) / base_overall["mae"] * 100.0,
                    2,
                ),
                "ganho_mape_vs_baseline_pp": round(
                    base_overall["mape"] - ets_overall["mape"], 2
                ),
                "vitorias_janelas_7d": int(
                    (folds_df["mae_auto_ets"] < folds_df["mae_baseline"]).sum()
                ),
            }
        )

    rows.append(
        {
            "modelo": "LightGBM (Lags >= 7 + Calendário/Feriados)",
            "mae_mwmed": round(lgbm_overall["mae"], 2),
            "mape_pct": round(lgbm_overall["mape"], 2),
            "rmse_mwmed": round(lgbm_overall["rmse"], 2),
            "ganho_mae_vs_baseline_mwmed": round(base_overall["mae"] - lgbm_overall["mae"], 2),
            "ganho_mae_vs_baseline_pct": round(
                (base_overall["mae"] - lgbm_overall["mae"]) / base_overall["mae"] * 100.0, 2
            ),
            "ganho_mape_vs_baseline_pp": round(base_overall["mape"] - lgbm_overall["mape"], 2),
            "vitorias_janelas_7d": int(
                (folds_df["mae_lightgbm"] < folds_df["mae_baseline"]).sum()
            ),
        }
    )

    summary_df = pd.DataFrame(rows)
    return summary_df, preds_df, folds_df


def forecast_next_7_days(df_gold: pd.DataFrame) -> pd.DataFrame:
    """Treina o LightGBM em toda a série histórica e prevê os próximos 7 dias."""
    ordered = df_gold.sort_values("data_referencia").reset_index(drop=True)
    last_date = pd.to_datetime(ordered["data_referencia"].max())

    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1), periods=7, freq="D"
    )
    future_shell = pd.DataFrame(
        {
            "data_referencia": future_dates,
            "id_subsistema": "SE",
            "nom_subsistema": "SUDESTE",
            "carga_mwmed": np.nan,
        }
    )

    combined = pd.concat([ordered, future_shell], ignore_index=True)
    combined_feat = add_baseline_predictions(build_features(combined), horizon=7)

    train_full = combined_feat.iloc[:-7].dropna(subset=FEATURE_COLUMNS + ["carga_mwmed"]).copy()
    future_feat = combined_feat.iloc[-7:].copy()

    model = fit_lgbm_pipeline(
        train_full,
        val_df=None,
        params={**DEFAULT_LGBM_PARAMS, "n_estimators": 350},
    )
    future_feat["pred_lightgbm"] = model.predict(future_feat[FEATURE_COLUMNS])
    try:
        future_feat["pred_auto_ets"] = _predict_auto_ets_fold(train_full, horizon=7)
    except Exception:
        future_feat["pred_auto_ets"] = future_feat["pred_baseline"]

    out = future_feat[
        [
            "data_referencia",
            "id_subsistema",
            "dia_semana",
            "is_feriado",
            "pred_baseline",
            "pred_auto_ets",
            "pred_lightgbm",
        ]
    ].copy()
    out["diferenca_lgbm_vs_baseline_mwmed"] = out["pred_lightgbm"] - out["pred_baseline"]
    return out.reset_index(drop=True)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df_gold = load_gold_se()

    logger.info("Iniciando Backtesting em Janela Temporal (26 janelas de 7 dias = 182 dias)...")
    summary_df, preds_df, folds_df = run_temporal_backtest(
        df_gold, n_windows=26, horizon=7, include_ets=True
    )

    summary_df.to_csv(OUTPUT_DIR / "backtest_summary.csv", index=False)
    preds_df.to_csv(OUTPUT_DIR / "backtest_predictions.csv", index=False)
    folds_df.to_csv(OUTPUT_DIR / "backtest_by_fold.csv", index=False)

    next_7_df = forecast_next_7_days(df_gold)
    next_7_df.to_csv(OUTPUT_DIR / "forecast_next_7_days.csv", index=False)

    print("\n=== RESULTADO DO BACKTESTING TEMPORAL (26 JANELAS DE 7 DIAS / 182 DIAS) ===")
    print(summary_df.to_string(index=False))
    print("\n=== PREVISÃO PARA OS PRÓXIMOS 7 DIAS (SUBSISTEMA SE) ===")
    print(next_7_df.to_string(index=False))


if __name__ == "__main__":
    main()
