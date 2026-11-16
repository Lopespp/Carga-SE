.PHONY: setup bronze dbt baseline train evaluate pipeline global test lint dashboard

setup:
	uv sync --python 3.12

bronze:
	uv run python -m src.ingest_bronze

dbt:
	cd dbt_ons && uv run dbt run --profiles-dir . && uv run dbt test --profiles-dir .

baseline:
	uv run python -m src.baseline

train:
	uv run python -m src.train_model

evaluate:
	uv run python -m src.evaluate

global:
	uv run python -m src.global_model

pipeline:
	uv run python -m src.run_pipeline

test:
	uv run pytest -v

lint:
	uv run ruff check src tests app.py

dashboard:
	uv run streamlit run app.py
