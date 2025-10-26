.PHONY: help install dev-install format lint typecheck test all clean

help:  ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install package
	pip install -e .

dev-install:  ## Install package with dev dependencies
	pip install -e ".[dev]"

format:  ## Format code with black and ruff
	@echo "Running black..."
	black neural_net/ experiments/ scripts/
	@echo "Running ruff --fix..."
	ruff check --fix neural_net/ experiments/ scripts/
	@echo "✓ Formatting complete!"

lint:  ## Run ruff linter
	@echo "Running ruff..."
	ruff check neural_net/ experiments/ scripts/

typecheck:  ## Run mypy type checker
	@echo "Running mypy..."
	mypy neural_net/ experiments/ scripts/

test:  ## Run tests with pytest
	@echo "Running pytest..."
	pytest tests/ -v

test-cov:  ## Run tests with coverage report
	@echo "Running pytest with coverage..."
	pytest tests/ -v --cov=neural_net --cov-report=term-missing --cov-report=html
	@echo "✓ Coverage report generated in htmlcov/"

all: format lint typecheck test  ## Run all checks (format, lint, typecheck, test)
	@echo "✓ All checks passed!"

check: lint typecheck  ## Quick check (lint + typecheck, no formatting)
	@echo "✓ Checks complete!"

clean:  ## Clean up cache files
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov/ .coverage
	@echo "✓ Cleaned up cache files!"

structure:  ## Test package structure
	python test_structure.py

train:  ## Run training script
	python scripts/train.py

# Example experiment shortcuts
m0:  ## Run M0 experiment
	python scripts/train.py --experiment M0

m1:  ## Run M1 experiment
	python scripts/train.py --experiment M1

m2:  ## Run M2 experiment
	python scripts/train.py --experiment M2

m3:  ## Run M3 experiment
	python scripts/train.py --experiment M3
