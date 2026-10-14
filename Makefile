.PHONY: help install train run test lint typecheck fmt audit check docker clean

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Install runtime + dev dependencies
	python -m pip install -r requirements-dev.txt

train: ## Train the model and regenerate docs/MODEL.md
	FLASK_ENV=development python -m loanapp.ml.train

run: ## Run the development server
	FLASK_ENV=development python app.py

test: ## Run the test suite with coverage
	python -m pytest --cov=loanapp --cov-report=term-missing

lint: ## Check formatting and lint rules
	python -m ruff check loanapp tests app.py
	python -m ruff format --check loanapp tests app.py

typecheck: ## Strict static type check
	python -m mypy

fmt: ## Auto-format
	python -m ruff format loanapp tests app.py
	python -m ruff check --fix loanapp tests app.py

audit: ## Dependency vulnerability scan + static security scan
	python -m pip_audit -r requirements.txt -r requirements-dev.txt --strict
	python -m bandit -r loanapp app.py -ll

check: lint typecheck test audit ## Everything CI runs

docker: ## Build the production image
	docker build -t loanwise:latest .

clean:
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
