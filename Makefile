.PHONY: setup bronze dbt baseline train evaluate global report serve pipeline test lint

setup:
	uv sync --python 3.12

bronze:
	uv run python -m src.ingest_bronze

dbt:
	uv run python -m src.dbt_runner

baseline:
	uv run python -m src.baseline

train:
	uv run python -m src.train_model

evaluate:
	uv run python -m src.evaluate

global:
	uv run python -m src.global_model

report:
	uv run python -m src.build_report

serve:
	uv run python -m src.serve

pipeline:
	uv run python -m src.run_pipeline

test:
	uv run pytest -v

lint:
	uv run ruff check src tests
