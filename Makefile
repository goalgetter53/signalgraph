PY ?= python3.11
VENV ?= .venv
BIN := $(VENV)/bin
DEMO_DB ?= demo.duckdb

.PHONY: install lint format typecheck test test-live check train eval bench real-data real-eval demo serve docker clean

install:  ## create venv and install with dev extras
	$(PY) -m venv $(VENV)
	$(BIN)/pip install -U pip
	$(BIN)/pip install -e ".[dev]"

lint:
	$(BIN)/ruff check .
	$(BIN)/ruff format --check .

format:
	$(BIN)/ruff format .
	$(BIN)/ruff check --fix .

typecheck:
	$(BIN)/mypy

test:
	$(BIN)/pytest

test-live:  ## hits real EDGAR + Greenhouse
	SIGNALGRAPH_LIVE=1 $(BIN)/pytest tests/live

check: lint typecheck test

train:  ## retrain the pair scorer on synthetic data -> src/signalgraph/resolution/er_model.json
	$(BIN)/python scripts/train_er.py

eval:  ## evaluate ER on the gold set -> reports/er_eval.json
	$(BIN)/python scripts/eval_er.py

bench:  ## ER throughput at 10k / 50k synthetic records -> reports/benchmark.json
	$(BIN)/python scripts/bench_er.py

real-data:  ## re-pull the Wikidata gold set (network; cached in data/real/raw/)
	$(BIN)/python benchmarks/real/build_wikidata_gold.py

real-eval:  ## ER on the committed REAL Wikidata gold set -> reports/real_eval.{json,md}
	$(BIN)/python benchmarks/real/run_real_eval.py

demo:  ## offline end-to-end run on recorded fixtures (fictional companies)
	rm -f $(DEMO_DB)
	$(BIN)/signalgraph --db $(DEMO_DB) ingest --fixtures fixtures --today 2026-11-19
	$(BIN)/signalgraph --db $(DEMO_DB) ingest --fixtures fixtures --today 2026-11-19
	$(BIN)/signalgraph --db $(DEMO_DB) resolve
	$(BIN)/signalgraph --db $(DEMO_DB) score --as-of 2026-11-19T00:00:00+00:00
	$(BIN)/signalgraph --db $(DEMO_DB) export --format csv

serve:
	$(BIN)/signalgraph --db $(DEMO_DB) serve

docker:
	docker build -t signalgraph:dev .

clean:
	rm -rf .mypy_cache .ruff_cache .pytest_cache .hypothesis $(DEMO_DB)
