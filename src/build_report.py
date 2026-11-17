"""Página web unificada: pipeline, dados, modelo, MLflow e resultados num único HTML.

Lê os artefatos produzidos pelas etapas anteriores (Parquet bronze, DuckDB silver/gold,
resultados do dbt, CSVs de avaliação e o banco do MLflow) e gera `docs/index.html`,
uma página estática e interativa (Plotly.js) que pode ser aberta direto no navegador,
servida com `python -m src.serve` ou publicada no GitHub Pages.
"""

from __future__ import annotations

import json
import logging
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb
import holidays
import pandas as pd

from src.features import MMGD_CUTOFF

logger = logging.getLogger(__name__)

ROOT = Path(".")
BRONZE_DIR = ROOT / "data" / "bronze"
WAREHOUSE_PATH = ROOT / "data" / "warehouse.duckdb"
OUTPUT_DIR = ROOT / "data" / "outputs"
MLFLOW_DB = ROOT / "mlflow.db"
TEMPLATE_PATH = Path(__file__).with_name("report_template.html")
DOCS_DIR = ROOT / "docs"
MLFLOW_UI_URL = "http://127.0.0.1:5000"

FEATURE_GROUPS = {
    "lag_": "Lags (≥ 7 dias)",
    "mean_same_dow": "Mesmo dia da semana",
    "weighted_same_dow": "Mesmo dia da semana",
    "std_same_dow": "Mesmo dia da semana",
    "rolling_": "Janelas móveis (≥ 7 dias)",
    "diff_": "Diferenças/razões",
    "ratio_": "Diferenças/razões",
    "is_": "Feriados",
    "regime_mmgd": "Regime de medição (MMGD)",
    "lag364_cruza": "Regime de medição (MMGD)",
}


def _feature_group(name: str) -> str:
    for prefix, group in sorted(FEATURE_GROUPS.items(), key=lambda kv: -len(kv[0])):
        if name.startswith(prefix):
            return group
    return "Calendário"


def _read_csv(name: str) -> pd.DataFrame | None:
    path = OUTPUT_DIR / name
    return pd.read_csv(path) if path.exists() else None


def _records(df: pd.DataFrame | None) -> list[dict[str, Any]]:
    if df is None:
        return []
    return json.loads(df.to_json(orient="records", date_format="iso", force_ascii=False))


def collect_bronze() -> dict[str, Any]:
    files = sorted(BRONZE_DIR.glob("CARGA_ENERGIA_*.parquet"))
    if not files:
        return {"files": [], "total_rows": 0}
    con = duckdb.connect()
    rows = []
    for f in files:
        n, dmin, dmax = con.execute(
            f"SELECT COUNT(*), MIN(din_instante), MAX(din_instante) "
            f"FROM read_parquet('{f.as_posix()}')"
        ).fetchone()
        rows.append(
            {
                "arquivo": f.name,
                "linhas": int(n),
                "tamanho_kb": round(f.stat().st_size / 1024, 1),
                "inicio": str(pd.Timestamp(dmin).date()),
                "fim": str(pd.Timestamp(dmax).date()),
            }
        )
    con.close()
    return {
        "files": rows,
        "total_rows": sum(r["linhas"] for r in rows),
        "total_kb": round(sum(r["tamanho_kb"] for r in rows), 1),
        "inicio": rows[0]["inicio"],
        "fim": rows[-1]["fim"],
    }


def collect_warehouse() -> dict[str, Any]:
    con = duckdb.connect(str(WAREHOUSE_PATH), read_only=True)
    silver_rows, n_sub = con.execute(
        "SELECT COUNT(*), COUNT(DISTINCT id_subsistema) FROM silver_carga_energia"
    ).fetchone()
    gold = con.execute(
        "SELECT data_referencia, carga_mwmed FROM gold_carga_se ORDER BY data_referencia"
    ).df()
    con.close()
    gold["data_referencia"] = pd.to_datetime(gold["data_referencia"])
    span = (gold["data_referencia"].max() - gold["data_referencia"].min()).days + 1

    recent = gold[gold["data_referencia"] > gold["data_referencia"].max() - pd.Timedelta(days=364)]
    dow_names = ["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"]
    profile = recent.groupby(recent["data_referencia"].dt.day_of_week)["carga_mwmed"].mean()

    yearly = gold.groupby(gold["data_referencia"].dt.year)["carga_mwmed"].mean().round(1)

    return {
        "silver_rows": int(silver_rows),
        "silver_subsistemas": int(n_sub),
        "gold_rows": len(gold),
        "gold_span_days": int(span),
        "gold_inicio": str(gold["data_referencia"].min().date()),
        "gold_fim": str(gold["data_referencia"].max().date()),
        "series": {
            "x": gold["data_referencia"].dt.strftime("%Y-%m-%d").tolist(),
            "y": gold["carga_mwmed"].round(1).tolist(),
        },
        "dow_profile": [
            {"dia": dow_names[int(k)], "carga": round(float(v), 1)} for k, v in profile.items()
        ],
        "yearly_mean": [{"ano": int(k), "carga": float(v)} for k, v in yearly.items()],
    }


DBT_MANIFEST = ROOT / "dbt_ons" / "target" / "manifest.json"


def _load_manifest_nodes() -> dict[str, Any]:
    if not DBT_MANIFEST.exists():
        return {}
    return json.loads(DBT_MANIFEST.read_text(encoding="utf-8")).get("nodes", {})


def _describe_dbt_node(unique_id: str, nodes: dict[str, Any]) -> tuple[str, str, str]:
    """Retorna (tipo, modelo, regra) legíveis para um teste ou modelo do dbt."""
    parts = unique_id.split(".")
    name = parts[2] if len(parts) > 2 else unique_id
    if parts[0] != "test":
        return "model", name, "materializado como tabela"

    node = nodes.get(unique_id, {})
    meta = node.get("test_metadata") or {}
    model = (node.get("attached_node") or "").split(".")[-1]
    if not model:
        deps = node.get("depends_on", {}).get("nodes", [])
        model = deps[0].split(".")[-1] if deps else ""
    if meta:
        kwargs = meta.get("kwargs", {})
        column = node.get("column_name") or kwargs.get("column_name", "")
        values = kwargs.get("values") or kwargs.get("arguments", {}).get("values")
        rule = f"{column} ∈ {{{', '.join(map(str, values))}}}" if values else column
        return meta.get("name", "generic"), model, rule
    return "singular", model, name


def collect_dbt() -> dict[str, Any]:
    out: dict[str, Any] = {"models": [], "tests": []}
    nodes = _load_manifest_nodes()
    for kind in ["run", "test"]:
        path = OUTPUT_DIR / f"dbt_{kind}_results.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for res in payload.get("results", []):
            test_type, model, rule = _describe_dbt_node(res["unique_id"], nodes)
            item = {
                "tipo": test_type,
                "modelo": model,
                "alvo": rule,
                "status": res["status"],
                "tempo_s": round(res.get("execution_time", 0.0), 2),
            }
            out["models" if kind == "run" else "tests"].append(item)
        out[f"{kind}_generated_at"] = payload.get("metadata", {}).get("generated_at")
    return out


def collect_mlflow() -> dict[str, Any]:
    if not MLFLOW_DB.exists():
        return {"runs": [], "ui_url": MLFLOW_UI_URL}
    con = sqlite3.connect(MLFLOW_DB)
    exp = con.execute(
        "SELECT experiment_id, name FROM experiments WHERE name = 'ONS_Carga_Energia_SE_7d'"
    ).fetchone()
    if exp is None:
        return {"runs": [], "ui_url": MLFLOW_UI_URL}
    exp_id, exp_name = exp
    runs = pd.read_sql_query(
        "SELECT run_uuid, name, status, start_time, end_time FROM runs "
        "WHERE experiment_id = ? AND lifecycle_stage = 'active' ORDER BY start_time DESC",
        con,
        params=(exp_id,),
    )
    total_runs = len(runs)
    latest = runs.drop_duplicates(subset="name", keep="first")

    result = []
    for _, r in latest.iterrows():
        rid = r["run_uuid"]
        metrics = dict(
            con.execute(
                "SELECT key, value FROM latest_metrics WHERE run_uuid = ?", (rid,)
            ).fetchall()
        )
        params = dict(con.execute("SELECT key, value FROM params WHERE run_uuid = ?", (rid,)))
        tags = dict(con.execute("SELECT key, value FROM tags WHERE run_uuid = ?", (rid,)))
        result.append(
            {
                "run_id": rid,
                "nome": r["name"],
                "status": r["status"],
                "etapa": tags.get("etapa", ""),
                "inicio": datetime.fromtimestamp(r["start_time"] / 1000).strftime(
                    "%Y-%m-%d %H:%M"
                ),
                "duracao_s": round(((r["end_time"] or r["start_time"]) - r["start_time"]) / 1e3, 1),
                "metrics": {
                    k: round(v, 4) for k, v in metrics.items() if not k.startswith("fold_")
                },
                "params": params,
                "url": f"{MLFLOW_UI_URL}/#/experiments/{exp_id}/runs/{rid}",
            }
        )
    con.close()
    return {
        "experiment": exp_name,
        "experiment_id": exp_id,
        "experiment_url": f"{MLFLOW_UI_URL}/#/experiments/{exp_id}",
        "ui_url": MLFLOW_UI_URL,
        "total_runs": total_runs,
        "runs": result,
    }


def collect_results() -> dict[str, Any]:
    preds = _read_csv("backtest_predictions.csv")
    backtest_series: dict[str, Any] = {}
    by_horizon: list[dict[str, Any]] = []
    if preds is not None:
        cols = ["data_referencia", "carga_mwmed", "pred_baseline", "pred_lightgbm", "fold"]
        if "pred_auto_ets" in preds:
            cols.append("pred_auto_ets")
        sub = preds[cols].copy()
        backtest_series = {c: sub[c].round(1).tolist() for c in cols if c != "data_referencia"}
        backtest_series["x"] = sub["data_referencia"].tolist()
        for h, g in preds.groupby("horizonte_dia"):
            row = {"h": int(h)}
            for key in ["baseline", "lightgbm", "auto_ets"]:
                col = f"pred_{key}"
                if col in g:
                    row[key] = round(float((g["carga_mwmed"] - g[col]).abs().mean()), 1)
            by_horizon.append(row)

    imp = _read_csv("feature_importance.csv")
    if imp is not None:
        total = imp["importance"].sum()
        imp["share_pct"] = (imp["importance"] / total * 100).round(2)
        imp["grupo"] = imp["feature"].map(_feature_group)
        groups = (
            imp.groupby("grupo")["share_pct"].sum().sort_values(ascending=False).round(1)
        )
        group_records = [{"grupo": k, "share_pct": float(v)} for k, v in groups.items()]
    else:
        group_records = []

    forecast = _read_csv("forecast_next_7_days.csv")
    if forecast is not None:
        dates = pd.to_datetime(forecast["data_referencia"])
        years = sorted(dates.dt.year.unique().tolist())
        br_holidays = holidays.Brazil(years=years, language="pt_BR")
        forecast["nome_feriado"] = [br_holidays.get(d.date(), "") for d in dates]

    return {
        "summary": _records(_read_csv("backtest_summary.csv")),
        "folds": _records(_read_csv("backtest_by_fold.csv")),
        "forecast": _records(forecast),
        "importance": _records(imp),
        "importance_groups": group_records,
        "global": _records(_read_csv("global_model_summary.csv")),
        "backtest": backtest_series,
        "by_horizon": by_horizon,
    }


def collect_all() -> dict[str, Any]:
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "mmgd_cutoff": str(MMGD_CUTOFF.date()),
        "bronze": collect_bronze(),
        "warehouse": collect_warehouse(),
        "dbt": collect_dbt(),
        "mlflow": collect_mlflow(),
        "results": collect_results(),
    }


def _copy_plotly(docs_dir: Path) -> None:
    """Copia o Plotly.js do pacote Python para `docs/assets/` (página funciona offline)."""
    import plotly

    src = Path(plotly.__file__).parent / "package_data" / "plotly.min.js"
    target = docs_dir / "assets" / "plotly.min.js"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists() or target.stat().st_size != src.stat().st_size:
        shutil.copyfile(src, target)


def build_report(docs_dir: Path = DOCS_DIR) -> Path:
    data = collect_all()
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    # Evita que "</script>" dentro de algum texto feche o bloco de dados.
    payload = payload.replace("</", "<\\/")
    html = TEMPLATE_PATH.read_text(encoding="utf-8").replace("/*__REPORT_DATA__*/null", payload)
    docs_dir.mkdir(parents=True, exist_ok=True)
    _copy_plotly(docs_dir)
    out = docs_dir / "index.html"
    out.write_text(html, encoding="utf-8")
    logger.info("Página unificada gerada em %s", out.resolve())
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    build_report()
