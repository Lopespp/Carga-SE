"""Desafio Bônus: Dashboard Interativo em Streamlit para Operação e Negócio."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.baseline import load_gold_se

OUTPUT_DIR = Path("data/outputs")

st.set_page_config(
    page_title="Previsão de Carga ONS — Subsistema SE/CO",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ Previsão de Carga Diária de Energia — ONS (Subsistema SE/CO)")
st.markdown(
    r"""
    **Objetivo:** Prever a carga diária de energia elétrica (em **MWmed**) do subsistema
    **Sudeste/Centro-Oeste (`SE`)** para os **próximos 7 dias**, respeitando rigorosamente a
    restrição de horizonte ($\text{lags} \ge 7$ dias, sem vazamento temporal).
    """
)

summary_path = OUTPUT_DIR / "backtest_summary.csv"
preds_path = OUTPUT_DIR / "backtest_predictions.csv"
folds_path = OUTPUT_DIR / "backtest_by_fold.csv"
forecast_path = OUTPUT_DIR / "forecast_next_7_days.csv"
imp_path = OUTPUT_DIR / "feature_importance.csv"
global_path = OUTPUT_DIR / "global_model_summary.csv"

if not summary_path.exists() or not preds_path.exists():
    st.warning("Execute `uv run python -m src.run_pipeline` primeiro para gerar os artefatos.")
    st.stop()

summary_df = pd.read_csv(summary_path)
preds_df = pd.read_csv(preds_path, parse_dates=["data_referencia"])
folds_df = pd.read_csv(folds_path)
forecast_df = pd.read_csv(forecast_path, parse_dates=["data_referencia"])
df_gold = load_gold_se()

base_row = summary_df[summary_df["modelo"].str.contains("Baseline")].iloc[0]
lgbm_row = summary_df[summary_df["modelo"].str.contains("LightGBM")].iloc[0]

col1, col2, col3, col4 = st.columns(4)
col1.metric(
    "LightGBM — MAE (Backtesting)",
    f"{lgbm_row['mae_mwmed']:,.2f} MWmed",
    delta=f"-{lgbm_row['ganho_mae_vs_baseline_mwmed']:,.2f} MWmed vs Baseline",
    delta_color="inverse",
)
col2.metric(
    "LightGBM — MAPE (Backtesting)",
    f"{lgbm_row['mape_pct']:.2f}%",
    delta=f"-{lgbm_row['ganho_mape_vs_baseline_pp']:.2f} p.p. vs Baseline",
    delta_color="inverse",
)
col3.metric(
    "Redução Relativa de Erro (MAE)",
    f"{lgbm_row['ganho_mae_vs_baseline_pct']:.2f}%",
    help="Ganho percentual de MAE do LightGBM sobre o Baseline Seasonal Naive (Lag 7).",
)
col4.metric(
    "Vitórias em Janelas de 7 Dias",
    f"{int(lgbm_row['vitorias_janelas_7d'])} / {len(folds_df)} janelas",
)

st.divider()

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📅 Previsão Próximos 7 Dias",
        "📊 Backtesting Temporal (Modelo vs Baseline)",
        "📈 Série Histórica (2018–Presente)",
        "🌎 Bônus: Modelo Global (SE, S, NE, N) & Features",
    ]
)

with tab1:
    st.subheader("Previsão Operacional para os Próximos 7 Dias — Subsistema SE")
    recent_hist = df_gold.tail(28).copy()

    fig_fc = go.Figure()
    fig_fc.add_trace(
        go.Scatter(
            x=recent_hist["data_referencia"],
            y=recent_hist["carga_mwmed"],
            mode="lines+markers",
            name="Carga Observada (Últimos 28 dias)",
            line={"color": "#1f77b4", "width": 2.5},
        )
    )
    fig_fc.add_trace(
        go.Scatter(
            x=forecast_df["data_referencia"],
            y=forecast_df["pred_baseline"],
            mode="lines+markers",
            name="Baseline (Mesmo Dia Semana Anterior - Lag 7)",
            line={"color": "#ff7f0e", "dash": "dash", "width": 2},
        )
    )
    fig_fc.add_trace(
        go.Scatter(
            x=forecast_df["data_referencia"],
            y=forecast_df["pred_lightgbm"],
            mode="lines+markers",
            name="Modelo LightGBM (Lags >= 7 + Calendário/Feriados)",
            line={"color": "#2ca02c", "width": 3},
        )
    )
    fig_fc.update_layout(
        xaxis_title="Data",
        yaxis_title="Carga Diária (MWmed)",
        hovermode="x unified",
        height=440,
    )
    st.plotly_chart(fig_fc, use_container_width=True)
    st.dataframe(forecast_df, use_container_width=True)

with tab2:
    st.subheader("Tabela Comparativa no Backtesting Temporal (26 Janelas de 7 Dias = 182 Dias)")
    st.dataframe(summary_df, use_container_width=True)

    fig_bt = go.Figure()
    fig_bt.add_trace(
        go.Scatter(
            x=preds_df["data_referencia"],
            y=preds_df["carga_mwmed"],
            mode="lines",
            name="Real (ONS)",
            line={"color": "#111827", "width": 2},
        )
    )
    fig_bt.add_trace(
        go.Scatter(
            x=preds_df["data_referencia"],
            y=preds_df["pred_baseline"],
            mode="lines",
            name="Baseline (Lag 7)",
            line={"color": "#ef4444", "dash": "dot", "width": 1.6},
        )
    )
    fig_bt.add_trace(
        go.Scatter(
            x=preds_df["data_referencia"],
            y=preds_df["pred_lightgbm"],
            mode="lines",
            name="LightGBM",
            line={"color": "#10b981", "width": 2},
        )
    )
    fig_bt.update_layout(
        title="Série Real vs. Previsões no Período de Backtesting (182 dias)",
        xaxis_title="Data",
        yaxis_title="Carga (MWmed)",
        hovermode="x unified",
        height=430,
    )
    st.plotly_chart(fig_bt, use_container_width=True)

    st.subheader("Erro Absoluto Médio (MAE) por Janela de 7 Dias (Fold)")
    fig_folds = go.Figure()
    fig_folds.add_trace(
        go.Bar(
            x=folds_df["inicio_janela"],
            y=folds_df["mae_baseline"],
            name="MAE Baseline (Lag 7)",
            marker_color="#f87171",
        )
    )
    fig_folds.add_trace(
        go.Bar(
            x=folds_df["inicio_janela"],
            y=folds_df["mae_lightgbm"],
            name="MAE LightGBM",
            marker_color="#34d399",
        )
    )
    fig_folds.update_layout(
        barmode="group",
        xaxis_title="Início da Janela de 7 Dias",
        yaxis_title="MAE (MWmed)",
        height=380,
    )
    st.plotly_chart(fig_folds, use_container_width=True)

with tab3:
    st.subheader("Série Histórica Completa — Subsistema SE/CO (2018–Presente)")
    fig_hist = px.line(
        df_gold,
        x="data_referencia",
        y="carga_mwmed",
        title="Carga Diária de Energia (MWmed) — Subsistema SE/CO",
        labels={"data_referencia": "Data", "carga_mwmed": "Carga (MWmed)"},
    )
    st.plotly_chart(fig_hist, use_container_width=True)

with tab4:
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Top 15 Variáveis Mais Importantes (LightGBM)")
        if imp_path.exists():
            imp_df = pd.read_csv(imp_path).head(15).sort_values("importance", ascending=True)
            fig_imp = px.bar(
                imp_df,
                x="importance",
                y="feature",
                orientation="h",
                labels={"importance": "Splits (Importância)", "feature": "Atributo"},
            )
            st.plotly_chart(fig_imp, use_container_width=True)
    with col_b:
        st.subheader("Modelo Global Único para Todos os Subsistemas (SE, S, NE, N)")
        if global_path.exists():
            glob_df = pd.read_csv(global_path)
            st.dataframe(glob_df, use_container_width=True)
