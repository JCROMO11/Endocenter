.PHONY: help install install-dev clean test lint format run gpu-test

# Variables
PYTHON := python3
PIP := pip
VENV := venv
SRC_DIR := src
TEST_DIR := tests

help: ## Show this help message
	@echo "EndoCenter MLOps - GPU Optimized Commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install production dependencies
	$(PIP) install -r requirements/base.txt

install-dev: ## Install development dependencies
	$(PIP) install -r requirements/dev.txt

install-gpu: ## Install GPU monitoring tools
	$(PIP) install -r requirements/dev.txt nvidia-ml-py gpustat

clean: ## Clean cache and build artifacts
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build dist

test: ## Run tests
	pytest $(TEST_DIR) -v --cov=$(SRC_DIR) --cov-report=html

lint: ## Run linting
	flake8 $(SRC_DIR) $(TEST_DIR)
	mypy $(SRC_DIR)

format: ## Format code
	black $(SRC_DIR) $(TEST_DIR)
	isort $(SRC_DIR) $(TEST_DIR)

run: ## Run the API server
	uvicorn src.endocenter.api.main:app --host 0.0.0.0 --port 8000 --reload

gpu-test: ## Test GPU setup
	python -c "import torch; print('CUDA:', torch.cuda.is_available(), 'GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

gpu-monitor: ## Monitor GPU usage
	watch -n 1 nvidia-smi

setup-env: ## Setup development environment
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -r requirements/dev.txt

env: ## Create .env file from template
	cp .env.example .env
	@echo "Created .env file. Please update with your configuration."
