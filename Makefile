.DEFAULT_GOAL := help
PY := python

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	 awk 'BEGIN{FS=":.*?## "}{printf "\033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Install package + all extras (editable)
	$(PY) -m pip install -e ".[api,dashboard,ml,dev]"

fmt: ## Auto-format (black + ruff --fix)
	black src tests && ruff check --fix src tests

lint: ## Lint + type-check
	ruff check src tests && mypy src

test: ## Run tests with coverage (offline; APIs are mocked)
	pytest

coverage: ## Run tests and write an HTML coverage report to htmlcov/
	pytest --cov-report=html --cov-report=term-missing
	@echo "Open htmlcov/index.html"

demo: ## End-to-end optimisation on live data
	zephyrus demo

data: ## Fetch a live data snapshot into data/raw
	$(PY) scripts/fetch_sample_data.py

api: ## Serve the FastAPI app
	uvicorn zephyrus.api.main:app --reload

dashboard: ## Launch the Streamlit dashboard
	streamlit run src/zephyrus/dashboard/app.py

retrain: ## Backtest, register + regression-check the demand model (M7)
	$(PY) scripts/retrain.py

.PHONY: help install fmt lint test coverage demo data api dashboard retrain
