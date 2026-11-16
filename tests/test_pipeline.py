"""Testes automatizados de rigor metodológico, prevenção de data leakage e qualidade."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.baseline import add_baseline_predictions, forecast_next_7_days_baseline
from src.features import FEATURE_COLUMNS, build_features
from src.train_model import compute_metrics


def _make_synthetic_series(n_days: int = 420) -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=n_days, freq="D")
    rng = np.random.default_rng(42)
    seasonal = 3000.0 * np.sin(2 * np.pi * np.arange(n_days) / 7.0)
    carga = 40000.0 + seasonal + rng.normal(0, 500, n_days)
    return pd.DataFrame(
        {
            "data_referencia": dates,
            "id_subsistema": "SE",
            "nom_subsistema": "SUDESTE",
            "carga_mwmed": carga,
        }
    )


def test_zero_data_leakage_within_7_day_horizon() -> None:
    """Garante que alterar os valores de carga em [t-6, ..., t] NÃO altera nenhuma feature em t."""
    df_orig = _make_synthetic_series(420)
    feat_orig = build_features(df_orig)

    # Perturba drasticamente os últimos 7 dias (t-6 até t)
    df_perturbed = df_orig.copy()
    df_perturbed.loc[df_perturbed.index[-7:], "carga_mwmed"] = 999999.0
    feat_perturbed = build_features(df_perturbed)

    # Na última data t (e em todas as 7 datas do horizonte [t-6..t] se previstas de t-7),
    # especificamente na data t-6..t, a linha `t` só pode depender de <= t-7!
    target_row_orig = feat_orig.iloc[-1][FEATURE_COLUMNS]
    target_row_pert = feat_perturbed.iloc[-1][FEATURE_COLUMNS]

    pd.testing.assert_series_equal(target_row_orig, target_row_pert, check_exact=False, rtol=1e-12)


def test_baseline_is_exact_7_day_lag() -> None:
    """Verifica se o baseline repete exatamente a carga de 7 dias atrás."""
    df = _make_synthetic_series(30)
    out = add_baseline_predictions(df, horizon=7)
    assert out["pred_baseline"].iloc[:7].isna().all()
    np.testing.assert_allclose(
        out["pred_baseline"].iloc[7:].to_numpy(),
        df["carga_mwmed"].iloc[:-7].to_numpy(),
    )


def test_forecast_next_7_days_baseline_shape_and_dates() -> None:
    """Verifica se a previsão futura de 7 dias do baseline gera 7 datas consecutivas."""
    df = _make_synthetic_series(30)
    fc = forecast_next_7_days_baseline(df)
    assert len(fc) == 7
    assert fc["data_previsao"].min() == df["data_referencia"].max() + pd.Timedelta(days=1)
    assert fc["data_previsao"].max() == df["data_referencia"].max() + pd.Timedelta(days=7)


def test_compute_metrics_mae_and_mape() -> None:
    """Testa o cálculo exato de MAE e MAPE."""
    y_true = np.array([100.0, 200.0])
    y_pred = np.array([110.0, 180.0])
    metrics = compute_metrics(y_true, y_pred)
    assert np.isclose(metrics["mae"], 15.0)
    assert np.isclose(metrics["mape"], 10.0)
