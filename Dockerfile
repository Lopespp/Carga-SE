FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock .python-version README.md ./
COPY src/ ./src/
COPY dbt_ons/ ./dbt_ons/
COPY tests/ ./tests/
COPY app.py ./

RUN uv sync --frozen --python 3.12

CMD ["uv", "run", "python", "-m", "src.run_pipeline"]
