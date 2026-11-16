"""Etapa 5: Engenharia de Atributos sem Vazamento Temporal (Zero Data Leakage).

Regra estrita do desafio:
Para prever um horizonte de 7 dias à frente, todas as variáveis derivadas do alvo
(`carga_mwmed`) usam exclusivamente informações defasadas em 7 dias ou mais (`lag >= 7`).
"""

from __future__ import annotations

import holidays
import numpy as np
import pandas as pd

FEATURE_COLUMNS: list[str] = [
    # Lags de mesmo dia da semana (>= 7 dias)
    "lag_7",
    "lag_14",
    "lag_21",
    "lag_28",
    "lag_35",
    "lag_56",
    "lag_364",
    # Lags da última semana observada completa (t-8 a t-13)
    "lag_8",
    "lag_9",
    "lag_10",
    "lag_11",
    "lag_12",
    "lag_13",
    # Estatísticas móveis de mesmo dia da semana e janelas deslocadas em >= 7 dias
    "mean_same_dow_4w",
    "weighted_same_dow_4w",
    "std_same_dow_4w",
    "rolling_mean_7_lag7",
    "rolling_std_7_lag7",
    "rolling_min_7_lag7",
    "rolling_max_7_lag7",
    "rolling_mean_14_lag7",
    "rolling_mean_28_lag7",
    "diff_lag7_lag14",
    "diff_lag7_mean_dow4w",
    "ratio_lag7_rolling7",
    # Variáveis de calendário e feriados nacionais (conhecidas ex-ante)
    "dia_semana",
    "fim_de_semana",
    "mes",
    "dia_mes",
    "dia_ano",
    "semana_ano",
    "sin_dia_ano",
    "cos_dia_ano",
    "sin_dia_semana",
    "cos_dia_semana",
    "is_feriado",
    "is_vespera_feriado",
    "is_pos_feriado",
    "is_ponte_feriado",
    "is_feriado_lag_7",
    "is_feriado_lag_14",
    "is_ponte_lag_7",
]


def _get_br_holidays(years: list[int]) -> set:
    """Retorna conjunto de datas de feriados nacionais do Brasil."""
    br_holidays = holidays.Brazil(years=years)
    return set(pd.to_datetime(list(br_holidays.keys())))


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Constrói matriz de atributos garantindo que nenhum dado de `t - 6` a `t` seja usado.

    Aceita um DataFrame com `data_referencia` e `carga_mwmed` (pode conter `NaN` em
    `carga_mwmed` para os 7 dias futuros que desejamos prever).
    """
    out = df.copy().sort_values("data_referencia").reset_index(drop=True)
    out["data_referencia"] = pd.to_datetime(out["data_referencia"])

    y = out["carga_mwmed"]

    # 1. Lags seguros (todos >= 7 dias)
    for lag in [7, 8, 9, 10, 11, 12, 13, 14, 21, 28, 35, 56, 364]:
        out[f"lag_{lag}"] = y.shift(lag)

    # 2. Combinações do mesmo dia da semana nas 4 semanas anteriores (t-7, t-14, t-21, t-28)
    dow_lags = out[["lag_7", "lag_14", "lag_21", "lag_28"]]
    out["mean_same_dow_4w"] = dow_lags.mean(axis=1)
    out["weighted_same_dow_4w"] = (
        0.45 * out["lag_7"]
        + 0.25 * out["lag_14"]
        + 0.18 * out["lag_21"]
        + 0.12 * out["lag_28"]
    )
    out["std_same_dow_4w"] = dow_lags.std(axis=1)

    # 3. Estatísticas móveis calculadas sobre `y.shift(7)` (usa apenas t-7 para trás)
    y_lag7 = y.shift(7)
    out["rolling_mean_7_lag7"] = y_lag7.rolling(window=7, min_periods=7).mean()
    out["rolling_std_7_lag7"] = y_lag7.rolling(window=7, min_periods=7).std()
    out["rolling_min_7_lag7"] = y_lag7.rolling(window=7, min_periods=7).min()
    out["rolling_max_7_lag7"] = y_lag7.rolling(window=7, min_periods=7).max()
    out["rolling_mean_14_lag7"] = y_lag7.rolling(window=14, min_periods=14).mean()
    out["rolling_mean_28_lag7"] = y_lag7.rolling(window=28, min_periods=28).mean()

    # 4. Diferenças e razões seguras (todas baseadas em lags >= 7)
    out["diff_lag7_lag14"] = out["lag_7"] - out["lag_14"]
    out["diff_lag7_mean_dow4w"] = out["lag_7"] - out["mean_same_dow_4w"]
    out["ratio_lag7_rolling7"] = out["lag_7"] / out["rolling_mean_7_lag7"]

    # 5. Variáveis de calendário e feriados (conhecidas para qualquer data futura)
    dt = out["data_referencia"].dt
    out["dia_semana"] = dt.dayofweek.astype(int)  # 0=Seg .. 6=Dom
    out["fim_de_semana"] = (out["dia_semana"] >= 5).astype(int)
    out["mes"] = dt.month.astype(int)
    out["dia_mes"] = dt.day.astype(int)
    out["dia_ano"] = dt.dayofyear.astype(int)
    out["semana_ano"] = dt.isocalendar().week.astype(int)

    out["sin_dia_ano"] = np.sin(2 * np.pi * out["dia_ano"] / 365.25)
    out["cos_dia_ano"] = np.cos(2 * np.pi * out["dia_ano"] / 365.25)
    out["sin_dia_semana"] = np.sin(2 * np.pi * out["dia_semana"] / 7.0)
    out["cos_dia_semana"] = np.cos(2 * np.pi * out["dia_semana"] / 7.0)

    years = sorted(out["data_referencia"].dt.year.unique().tolist())
    holiday_dates = _get_br_holidays(years + [min(years) - 1, max(years) + 1])

    out["is_feriado"] = out["data_referencia"].isin(holiday_dates).astype(int)
    out["is_vespera_feriado"] = (
        (out["data_referencia"] + pd.Timedelta(days=1)).isin(holiday_dates).astype(int)
    )
    out["is_pos_feriado"] = (
        (out["data_referencia"] - pd.Timedelta(days=1)).isin(holiday_dates).astype(int)
    )
    # Dia-ponte: Segunda-feira antes de feriado na Terça, ou Sexta-feira após feriado na Quinta
    out["is_ponte_feriado"] = (
        ((out["dia_semana"] == 0) & (out["is_vespera_feriado"] == 1))
        | ((out["dia_semana"] == 4) & (out["is_pos_feriado"] == 1))
    ).astype(int)

    # Indicadores de se o dia usado pelo baseline (t-7 ou t-14) foi feriado/ponte!
    # Isso permite ao modelo corrigir distorções quando a semana anterior teve feriado.
    out["is_feriado_lag_7"] = out["is_feriado"].shift(7).fillna(0).astype(int)
    out["is_feriado_lag_14"] = out["is_feriado"].shift(14).fillna(0).astype(int)
    out["is_ponte_lag_7"] = out["is_ponte_feriado"].shift(7).fillna(0).astype(int)

    return out
