.PHONY: help install css train run test lint typecheck fmt audit check docker clean

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Install runtime + dev dependencies
	python -m pip install -r requirements-dev.txt

# Pinned so the committed CSS is reproducible. v3.4 is Tailwind's official
# v3-lts line; the templates were written against v3. See docs/ARCHITECTURE.md.
TAILWINDCSS_VERSION ?= v3.4.19
export TAILWINDCSS_VERSION

css: ## Rebuild static/css/app.css after changing Tailwind classes in templates/
	python -m pytailwindcss -i static/src/tailwind.css -o static/css/app.css --content "./templates/**/*.html" --minify

train: ## Train the model and regenerate docs/MODEL.md
	FLASK_ENV=development python -m loanapp.ml.train

run: ## Run the development server
	FLASK_ENV=development python app.py

test: ## Run the test suite with coverage
	python -m pytest --cov=loanapp --cov-report=term-missing

lint: ## Check formatting and lint rules
	python -m ruff check loanapp tests scripts app.py
	python -m ruff format --check loanapp tests scripts app.py

typecheck: ## Strict static type check
	python -m mypy

fmt: ## Auto-format
	python -m ruff format loanapp tests scripts app.py
	python -m ruff check --fix loanapp tests scripts app.py

audit: ## Dependency vulnerability scan + static security scan
	python -m pip_audit -r requirements.txt -r requirements-dev.txt --strict
	python -m bandit -r loanapp app.py -ll

check: lint typecheck test audit ## Core checks; CI also requires browser, CSS and retraining gates

docker: ## Build the production image
	docker build -t loanwise:latest .

clean: ## Preview ignored root tooling output; inspect paths before removing it
	git clean -ndX -- .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
