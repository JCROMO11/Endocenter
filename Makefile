.PHONY: help install install-dev clean test lint format run gpu-test db-init db-migrate db-upgrade db-downgrade db-reset

# Variables
PYTHON := python3
PIP := pip
VENV := venv
SRC_DIR := src
TEST_DIR := tests

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[1;33m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(BLUE)EndoCenter MLOps - Available Commands:$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""

# =============================================================================
# INSTALLATION
# =============================================================================

install: ## Install production dependencies
	$(PIP) install -r requirements/base.txt

install-dev: ## Install development dependencies
	$(PIP) install -r requirements/dev.txt

install-gpu: ## Install GPU monitoring tools
	$(PIP) install nvidia-ml-py gpustat

setup-env: ## Setup development environment
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -r requirements/dev.txt

# =============================================================================
# CODE QUALITY
# =============================================================================

clean: ## Clean cache and build artifacts
	@echo "$(YELLOW)🧹 Cleaning cache and artifacts...$(NC)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build dist htmlcov .pytest_cache .coverage
	@echo "$(GREEN)✅ Cleaned!$(NC)"

format: ## Format code with black and isort
	@echo "$(YELLOW)🎨 Formatting code...$(NC)"
	black $(SRC_DIR) $(TEST_DIR)
	isort $(SRC_DIR) $(TEST_DIR)
	@echo "$(GREEN)✅ Code formatted!$(NC)"

lint: ## Run linting
	@echo "$(YELLOW)🔍 Running linters...$(NC)"
	flake8 $(SRC_DIR) $(TEST_DIR)
	mypy $(SRC_DIR)
	@echo "$(GREEN)✅ Linting complete!$(NC)"

test: ## Run tests
	@echo "$(YELLOW)🧪 Running tests...$(NC)"
	pytest $(TEST_DIR) -v --cov=$(SRC_DIR) --cov-report=html --cov-report=term
	@echo "$(GREEN)✅ Tests complete!$(NC)"

# =============================================================================
# GPU
# =============================================================================

gpu-test: ## Test GPU setup
	@echo "$(YELLOW)🎮 Testing GPU setup...$(NC)"
	@python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

gpu-monitor: ## Monitor GPU usage
	@echo "$(YELLOW)📊 Monitoring GPU...$(NC)"
	watch -n 1 nvidia-smi

gpu-info: ## Show detailed GPU information
	@echo "$(YELLOW)ℹ️ GPU Information:$(NC)"
	@python -c "from endocenter.config import gpu_manager; import json; print(json.dumps(gpu_manager.get_device_info(), indent=2))"

# =============================================================================
# DATABASE - PostgreSQL
# =============================================================================

db-start: ## Start PostgreSQL with Docker
	@echo "$(YELLOW)🐳 Starting PostgreSQL...$(NC)"
	docker-compose -f infrastructure/docker/docker-compose.dev.yml up -d
	@echo "$(GREEN)✅ PostgreSQL started!$(NC)"

db-stop: ## Stop PostgreSQL
	@echo "$(YELLOW)🛑 Stopping PostgreSQL...$(NC)"
	docker-compose -f infrastructure/docker/docker-compose.dev.yml down
	@echo "$(GREEN)✅ PostgreSQL stopped!$(NC)"

db-logs: ## Show PostgreSQL logs
	docker-compose -f infrastructure/docker/docker-compose.dev.yml logs -f postgres

db-shell: ## Open PostgreSQL shell
	@echo "$(YELLOW)🐚 Opening PostgreSQL shell...$(NC)"
	docker exec -it endocenter_postgres psql -U endocenter_user -d endocenter

db-test: ## Test database connection
	@echo "$(YELLOW)🔌 Testing database connection...$(NC)"
	@python scripts/test_postgres.py

# =============================================================================
# ALEMBIC - Database Migrations
# =============================================================================

db-init: ## Initialize Alembic (first time only)
	@echo "$(YELLOW)🚀 Initializing Alembic...$(NC)"
	alembic init alembic
	@echo "$(GREEN)✅ Alembic initialized!$(NC)"
	@echo "$(YELLOW)⚠️ Remember to configure alembic.ini and alembic/env.py$(NC)"

db-migrate: ## Create new migration (usage: make db-migrate MSG="description")
	@echo "$(YELLOW)📝 Creating migration: $(MSG)$(NC)"
	@if [ -z "$(MSG)" ]; then \
		echo "$(YELLOW)⚠️ No message provided, using default...$(NC)"; \
		alembic revision --autogenerate -m "auto_migration"; \
	else \
		alembic revision --autogenerate -m "$(MSG)"; \
	fi
	@echo "$(GREEN)✅ Migration created!$(NC)"

db-upgrade: ## Apply all pending migrations
	@echo "$(YELLOW)⬆️ Applying migrations...$(NC)"
	alembic upgrade head
	@echo "$(GREEN)✅ Migrations applied!$(NC)"

db-downgrade: ## Revert last migration
	@echo "$(YELLOW)⬇️ Reverting last migration...$(NC)"
	alembic downgrade -1
	@echo "$(GREEN)✅ Migration reverted!$(NC)"

db-current: ## Show current migration
	@echo "$(YELLOW)📍 Current migration:$(NC)"
	@alembic current

db-history: ## Show migration history
	@echo "$(YELLOW)📜 Migration history:$(NC)"
	@alembic history --verbose

db-heads: ## Show migration heads
	@echo "$(YELLOW)🎯 Migration heads:$(NC)"
	@alembic heads

db-reset: ## ⚠️ DANGER: Reset database (drop all tables and recreate)
	@echo "$(YELLOW)⚠️ WARNING: This will delete ALL data!$(NC)"
	@read -p "Are you sure? (type 'yes' to confirm): " confirm; \
	if [ "$$confirm" = "yes" ]; then \
		echo "$(YELLOW)🗑️ Resetting database...$(NC)"; \
		python -c "from endocenter.db.engine import reset_db; reset_db()"; \
		alembic stamp head; \
		echo "$(GREEN)✅ Database reset!$(NC)"; \
	else \
		echo "$(GREEN)❌ Reset cancelled$(NC)"; \
	fi

db-seed: ## Load example data
	@echo "$(YELLOW)🌱 Loading example data...$(NC)"
	@python -c "\
from endocenter.db.engine import get_db_context; \
from endocenter.db.models import Patient, Doctor, Appointment; \
from datetime import datetime, timedelta; \
with get_db_context() as db: \
    doctor = Doctor(first_name='María', last_name='González', email='maria@endocenter.com', phone='+57 300 123 4567', license_number='MED-12345', specialty='Endocrinología', years_experience=10); \
    db.add(doctor); \
    patient = Patient(first_name='Juan', last_name='Pérez', email='juan@email.com', phone='+57 300 765 4321', date_of_birth=datetime(1985, 5, 15), gender='Masculino', blood_type='O+'); \
    db.add(patient); \
    db.commit(); \
    appointment = Appointment(patient_id=patient.id, doctor_id=doctor.id, scheduled_at=datetime.now() + timedelta(days=7), duration_minutes=30, appointment_type='consulta', reason='Control de tiroides', status='scheduled'); \
    db.add(appointment); \
    db.commit(); \
    print('✅ Example data loaded: 1 Doctor, 1 Patient, 1 Appointment')"
	@echo "$(GREEN)✅ Data loaded!$(NC)"

db-setup: ## Complete database setup (start, migrate, seed)
	@echo "$(BLUE)🗄️ Complete Database Setup$(NC)"
	@echo ""
	@$(MAKE) db-start
	@sleep 3
	@$(MAKE) db-test
	@$(MAKE) db-upgrade
	@$(MAKE) db-seed
	@echo ""
	@echo "$(GREEN)✅ Database setup complete!$(NC)"

# =============================================================================
# APPLICATION
# =============================================================================

run: ## Run the API server
	@echo "$(YELLOW)🚀 Starting API server...$(NC)"
	uvicorn src.endocenter.api.main:app --host 0.0.0.0 --port 8000 --reload

env: ## Create .env file from template
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "$(GREEN)✅ Created .env file$(NC)"; \
		echo "$(YELLOW)⚠️ Please update with your configuration$(NC)"; \
	else \
		echo "$(YELLOW)⚠️ .env file already exists$(NC)"; \
	fi

config-check: ## Check configuration
	@echo "$(YELLOW)🔧 Checking configuration...$(NC)"
	@python -c "from endocenter.config import setup_gpu_optimizations; setup_gpu_optimizations()"

# =============================================================================
# DOCKER
# =============================================================================

docker-build: ## Build Docker image
	@echo "$(YELLOW)🐳 Building Docker image...$(NC)"
	docker build -t endocenter-mlops .
	@echo "$(GREEN)✅ Image built!$(NC)"

docker-run: ## Run Docker container
	@echo "$(YELLOW)🐳 Running Docker container...$(NC)"
	docker run -p 8000:8000 --gpus all endocenter-mlops

# =============================================================================
# DEVELOPMENT WORKFLOW
# =============================================================================

dev-setup: ## Complete development setup
	@echo "$(BLUE)🚀 Complete Development Setup$(NC)"
	@echo ""
	@$(MAKE) setup-env
	@$(MAKE) install-dev
	@$(MAKE) env
	@$(MAKE) gpu-test
	@$(MAKE) db-setup
	@$(MAKE) config-check
	@echo ""
	@echo "$(GREEN)✅ Development environment ready!$(NC)"
	@echo "$(YELLOW)💡 Next steps:$(NC)"
	@echo "  1. Update .env with your settings"
	@echo "  2. Run: make run"
	@echo "  3. Visit: http://localhost:8000/docs"

dev-check: ## Check development environment
	@echo "$(BLUE)🔍 Development Environment Check$(NC)"
	@echo ""
	@echo "$(YELLOW)Python & Dependencies:$(NC)"
	@$(PYTHON) --version
	@echo ""
	@echo "$(YELLOW)GPU:$(NC)"
	@$(MAKE) gpu-test
	@echo ""
	@echo "$(YELLOW)Database:$(NC)"
	@$(MAKE) db-test
	@echo ""
	@echo "$(YELLOW)Configuration:$(NC)"
	@$(MAKE) config-check

# =============================================================================
# QUICK COMMANDS
# =============================================================================

quick-start: db-start run ## Quick start: Start DB and run server

quick-reset: db-stop db-start db-upgrade ## Quick reset: Restart DB and apply migrations

quick-test: test lint ## Quick test: Run tests and linting