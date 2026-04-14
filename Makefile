.PHONY: help install dev db ingest train serve test lint typecheck clean docker-up docker-down streamlit all

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install production dependencies
	pip install -e .

dev: ## Install dev dependencies
	pip install -e ".[dev]"

db: ## Start Postgres in Docker
	docker compose up -d postgres
	@echo "Waiting for Postgres to be healthy..."
	@until docker compose exec postgres pg_isready -U ec_user -d ec_prices > /dev/null 2>&1; do sleep 1; done
	@echo "Postgres is ready."

ingest: ## Ingest URA transactions into Postgres
	python scripts/ingest_data.py

train: ## Train the ML model
	python scripts/train_model.py

serve: ## Start the FastAPI server locally
	uvicorn src.api.app:app --reload --port 8000

test: ## Run unit tests
	pytest -v --tb=short -m "not integration"

test-all: ## Run all tests including integration
	pytest -v --tb=short

lint: ## Lint with ruff
	ruff check src/ tests/

typecheck: ## Run mypy type checking
	mypy src/

clean: ## Remove artifacts and caches
	rm -rf artifacts/ __pycache__ .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

docker-up: ## Start full stack (Postgres + API) in Docker
	docker compose up -d --build

docker-down: ## Stop all Docker services
	docker compose down

streamlit: ## Launch Streamlit frontend
	streamlit run frontend/app.py --server.port 8501

all: db ingest train serve ## Run full pipeline: DB -> Ingest -> Train -> Serve
