FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock .python-version README.md ./
COPY src/ ./src/
COPY dbt_ons/ ./dbt_ons/
COPY tests/ ./tests/

RUN uv sync --frozen --python 3.12

EXPOSE 8000 5000

# Roda o pipeline completo (gera docs/index.html) e serve a página + MLflow UI
CMD ["sh", "-c", "uv run python -m src.run_pipeline && uv run python -m src.serve --no-browser --host 0.0.0.0"]
