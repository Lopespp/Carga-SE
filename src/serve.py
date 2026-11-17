"""Sobe a página unificada (porta 8000) e a interface do MLflow (porta 5000) juntas.

Uso: `uv run python -m src.serve` e abra http://127.0.0.1:8000 (Ctrl+C encerra os dois).
"""

from __future__ import annotations

import argparse
import functools
import http.server
import logging
import subprocess
import sys
import webbrowser
from pathlib import Path

from src.build_report import DOCS_DIR, build_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Interface de rede (0.0.0.0 no Docker)")
    parser.add_argument("--port", type=int, default=8000, help="Porta da página")
    parser.add_argument("--mlflow-port", type=int, default=5000, help="Porta do MLflow UI")
    parser.add_argument("--no-mlflow", action="store_true", help="Não sobe o MLflow UI")
    parser.add_argument("--no-browser", action="store_true", help="Não abre o navegador")
    parser.add_argument("--rebuild", action="store_true", help="Regera docs/index.html antes")
    args = parser.parse_args()

    if args.rebuild or not (DOCS_DIR / "index.html").exists():
        build_report()

    mlflow_proc = None
    if not args.no_mlflow:
        mlflow_proc = subprocess.Popen(
            [
                sys.executable, "-m", "mlflow", "ui",
                "--backend-store-uri", "sqlite:///mlflow.db",
                "--host", args.host, "--port", str(args.mlflow_port),
            ]
        )
        logger.info("MLflow UI em http://127.0.0.1:%d", args.mlflow_port)

    handler = functools.partial(
        http.server.SimpleHTTPRequestHandler, directory=str(Path(DOCS_DIR).resolve())
    )
    server = http.server.ThreadingHTTPServer((args.host, args.port), handler)
    url = f"http://127.0.0.1:{args.port}"
    logger.info("Página unificada em %s (Ctrl+C para sair)", url)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        if mlflow_proc is not None:
            mlflow_proc.terminate()


if __name__ == "__main__":
    main()
