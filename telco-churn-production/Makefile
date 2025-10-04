SHELL := /usr/bin/env bash
.ONESHELL:
.PHONY: help install setup-dirs clean test download-data

# ============================================================================
# CONFIGURATION
# ============================================================================
PYTHON := python
VENV := .venv/bin/activate
MLFLOW_PORT ?= 5000
AIRFLOW_PORT ?= 8080
PROJECT_ROOT := $(shell pwd)

# Colors for terminal output
GREEN := \033[0;32m
YELLOW := \033[1;33m
RED := \033[0;31m
NC := \033[0m # No Color

# ============================================================================
# HELP & DOCUMENTATION
# ============================================================================

.DEFAULT_GOAL := help

help: ## 📋 Show this help message
	@echo "=========================================="
	@echo "Telco Churn Production ML Pipeline"
	@echo "=========================================="
	@echo ""
	@echo "Available targets:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-30s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "Quick Start:"
	@echo "  1. make install          - Set up environment"
	@echo "  2. make download-data    - Get dataset"
	@echo "  3. make run-all          - Run complete pipeline"
	@echo "  4. make mlflow-ui        - View experiments"
	@echo "  5. make airflow-up       - Start orchestration"
	@echo ""

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

install: ## 🔧 Install dependencies and set up virtual environment
	@echo "$(YELLOW)Installing project dependencies...$(NC)"
	@echo "Creating virtual environment..."
	@$(PYTHON) -m venv .venv
	@echo "$(GREEN)✓ Virtual environment created$(NC)"
	@echo "Activating environment and upgrading pip..."
	@source $(VENV) && pip install --upgrade pip
	@echo "$(GREEN)✓ Pip upgraded$(NC)"
	@echo "Installing requirements..."
	@source $(VENV) && pip install -r requirements.txt
	@echo "$(GREEN)✓ Dependencies installed successfully!$(NC)"
	@echo ""
	@echo "To activate the virtual environment:"
	@echo "  source .venv/bin/activate    (Linux/Mac)"
	@echo "  .venv\\Scripts\\activate      (Windows)"

setup-dirs: ## 📁 Create necessary project directories
	@echo "$(YELLOW)Creating project directories...$(NC)"
	@mkdir -p src/{pipelines,models,utils}
	@mkdir -p data/{raw,processed,predictions}
	@mkdir -p config
	@mkdir -p reports/figures
	@mkdir -p mlruns
	@mkdir -p notebooks
	@mkdir -p tests
	@mkdir -p airflow-docker/{dags,logs,plugins,data}
	@echo "$(GREEN)✓ Directories created successfully!$(NC)"

verify-env: ## ✅ Verify environment and dependencies
	@echo "$(YELLOW)Verifying environment...$(NC)"
	@source $(VENV) && python --version
	@source $(VENV) && python -c "import sklearn; print('✓ scikit-learn:', sklearn.__version__)"
	@source $(VENV) && python -c "import mlflow; print('✓ MLflow:', mlflow.__version__)"
	@source $(VENV) && python -c "import pyspark; print('✓ PySpark:', pyspark.__version__)"
	@source $(VENV) && python -c "import xgboost; print('✓ XGBoost:', xgboost.__version__)"
	@which java > /dev/null && echo "✓ Java: $$(java -version 2>&1 | head -n 1)" || echo "⚠️  Java not found (required for PySpark)"
	@which docker > /dev/null && echo "✓ Docker: $$(docker --version)" || echo "⚠️  Docker not found (required for Airflow)"
	@echo "$(GREEN)Environment verification complete!$(NC)"

download-data: ## 📥 Download Telco Customer Churn dataset
	@echo "$(YELLOW)Downloading Telco Customer Churn dataset...$(NC)"
	@mkdir -p data/raw
	@if [ -f data/raw/Telco-Customer-Churn.csv ]; then \
		echo "$(GREEN)✓ Dataset already exists$(NC)"; \
	else \
		curl -o data/raw/Telco-Customer-Churn.csv \
			https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv && \
		echo "$(GREEN)✓ Dataset downloaded successfully$(NC)" || \
		echo "$(RED)✗ Download failed. Please download manually from Kaggle$(NC)"; \
	fi
	@ls -lh data/raw/Telco-Customer-Churn.csv 2>/dev/null || true

# ============================================================================
# PART 1: SCIKIT-LEARN PIPELINES (25 MARKS)
# ============================================================================

train-sklearn: setup-dirs ## 🤖 Train Scikit-Learn pipeline (Part 1)
	@echo "$(YELLOW)Training Scikit-Learn pipeline...$(NC)"
	@source $(VENV) && $(PYTHON) src/pipelines/sklearn_pipeline.py
	@echo "$(GREEN)✓ Training complete! Model saved to src/models/sklearn_pipeline.pkl$(NC)"

test-inference: ## 🔮 Test inference pipeline with sample data (Part 1)
	@echo "$(YELLOW)Testing inference pipeline...$(NC)"
	@source $(VENV) && $(PYTHON) src/pipelines/inference_pipeline.py
	@echo "$(GREEN)✓ Inference test complete$(NC)"

# ============================================================================
# PART 2: MLFLOW TRACKING (25 MARKS)
# ============================================================================

train-mlflow: setup-dirs ## 📊 Train models with MLflow tracking (Part 2)
	@echo "$(YELLOW)Training models with MLflow tracking...$(NC)"
	@mkdir -p reports/figures
	@source $(VENV) && $(PYTHON) src/pipelines/sklearn_mlflow_pipeline.py
	@echo "$(GREEN)✓ MLflow tracking complete! View results with: make mlflow-ui$(NC)"

mlflow-ui: ## 🌐 Start MLflow UI (Part 2)
	@echo "$(YELLOW)Starting MLflow UI...$(NC)"
	@echo "MLflow UI will be available at: http://localhost:$(MLFLOW_PORT)"
	@echo "Press Ctrl+C to stop"
	@source $(VENV) && mlflow ui --host 0.0.0.0 --port $(MLFLOW_PORT)

mlflow-ui-bg: ## 🌐 Start MLflow UI in background (Part 2)
	@echo "$(YELLOW)Starting MLflow UI in background...$(NC)"
	@source $(VENV) && mlflow ui --host 0.0.0.0 --port $(MLFLOW_PORT) > /dev/null 2>&1 &
	@sleep 2
	@echo "$(GREEN)✓ MLflow UI running at: http://localhost:$(MLFLOW_PORT)$(NC)"
	@echo "Stop with: make mlflow-stop"

mlflow-stop: ## 🛑 Stop MLflow UI
	@echo "$(YELLOW)Stopping MLflow UI...$(NC)"
	@-lsof -ti:$(MLFLOW_PORT) | xargs kill -9 2>/dev/null || true
	@-ps aux | grep '[m]lflow ui' | awk '{print $$2}' | xargs kill -9 2>/dev/null || true
	@echo "$(GREEN)✓ MLflow UI stopped$(NC)"

# ============================================================================
# PART 3: PYSPARK MLLIB (30 MARKS)
# ============================================================================

check-java: ## ☕ Check Java installation (required for PySpark)
	@echo "$(YELLOW)Checking Java installation...$(NC)"
	@if which java > /dev/null 2>&1; then \
		echo "$(GREEN)✓ Java found: $$(java -version 2>&1 | head -n 1)$(NC)"; \
		java -version; \
	else \
		echo "$(RED)✗ Java not found!$(NC)"; \
		echo "PySpark requires Java JDK 8, 11, or 17"; \
		echo ""; \
		echo "Installation:"; \
		echo "  Ubuntu/Debian: sudo apt install openjdk-17-jdk"; \
		echo "  Mac: brew install openjdk@17"; \
		echo "  Windows: Download from https://adoptium.net/"; \
		echo ""; \
		echo "Set JAVA_HOME:"; \
		echo "  export JAVA_HOME=/path/to/jdk"; \
		exit 1; \
	fi

train-pyspark: check-java setup-dirs ## 🔥 Train PySpark MLlib pipeline (Part 3)
	@echo "$(YELLOW)Training PySpark MLlib pipeline...$(NC)"
	@source $(VENV) && $(PYTHON) src/pipelines/pyspark_pipeline.py
	@echo "$(GREEN)✓ PySpark training complete! Model saved to src/models/pyspark_model/$(NC)"

compare-frameworks: ## 📊 Compare Scikit-Learn vs PySpark performance (Part 3)
	@echo "$(YELLOW)Comparing framework performance...$(NC)"
	@source $(VENV) && $(PYTHON) src/pipelines/compare_frameworks.py
	@echo "$(GREEN)✓ Comparison saved to reports/framework_comparison.csv$(NC)"

# ============================================================================
# PART 4: AIRFLOW ORCHESTRATION (20 MARKS)
# ============================================================================

airflow-check: ## 🐳 Check Docker installation (required for Airflow)
	@echo "$(YELLOW)Checking Docker installation...$(NC)"
	@if which docker > /dev/null && which docker-compose > /dev/null; then \
		echo "$(GREEN)✓ Docker found: $$(docker --version)$(NC)"; \
		echo "$(GREEN)✓ Docker Compose found: $$(docker-compose --version)$(NC)"; \
	else \
		echo "$(RED)✗ Docker or Docker Compose not found!$(NC)"; \
		echo "Please install Docker Desktop from https://www.docker.com/products/docker-desktop/"; \
		exit 1; \
	fi

airflow-up: airflow-check ## 🚀 Start Airflow with Docker (Part 4)
	@echo "$(YELLOW)Starting Airflow with Docker...$(NC)"
	@cd airflow-docker && docker-compose up -d
	@echo "$(GREEN)✓ Airflow started!$(NC)"
	@echo ""
	@echo "Waiting for Airflow to initialize (60 seconds)..."
	@sleep 60
	@echo ""
	@echo "$(GREEN)Airflow Web UI: http://localhost:$(AIRFLOW_PORT)$(NC)"
	@echo "Username: admin"
	@echo "Password: admin"
	@echo ""
	@echo "View logs: make airflow-logs"
	@echo "Stop Airflow: make airflow-down"

airflow-down: ## 🛑 Stop Airflow
	@echo "$(YELLOW)Stopping Airflow...$(NC)"
	@cd airflow-docker && docker-compose down
	@echo "$(GREEN)✓ Airflow stopped$(NC)"

airflow-restart: ## 🔄 Restart Airflow
	@echo "$(YELLOW)Restarting Airflow...$(NC)"
	@make airflow-down
	@sleep 2
	@make airflow-up

airflow-logs: ## 📜 View Airflow logs
	@echo "$(YELLOW)Viewing Airflow logs (Ctrl+C to exit)...$(NC)"
	@cd airflow-docker && docker-compose logs -f airflow

airflow-status: ## 📊 Check Airflow container status
	@echo "$(YELLOW)Airflow container status:$(NC)"
	@cd airflow-docker && docker-compose ps

airflow-trigger-dag: ## ▶️  Trigger DAG manually (Part 4)
	@echo "$(YELLOW)Available DAGs:$(NC)"
	@echo "  1. telco_churn_ml_pipeline"
	@echo ""
	@echo "Triggering DAG: telco_churn_ml_pipeline"
	@cd airflow-docker && docker-compose exec airflow airflow dags trigger telco_churn_ml_pipeline || \
		echo "$(RED)Failed to trigger DAG. Is Airflow running? (make airflow-up)$(NC)"

airflow-list-dags: ## 📋 List all DAGs
	@cd airflow-docker && docker-compose exec airflow airflow dags list || \
		echo "$(RED)Airflow not running. Start with: make airflow-up$(NC)"

airflow-shell: ## 🐚 Access Airflow container shell
	@echo "$(YELLOW)Opening Airflow container shell...$(NC)"
	@cd airflow-docker && docker-compose exec airflow bash

# ============================================================================
# COMPLETE PIPELINE EXECUTION
# ============================================================================

run-all: setup-dirs ## 🏃 Run complete ML pipeline (all parts)
	@echo "$(YELLOW)========================================"
	@echo "Running Complete ML Pipeline"
	@echo "========================================$(NC)"
	@echo ""
	@echo "$(YELLOW)Step 1/5: Training Scikit-Learn Pipeline (Part 1)$(NC)"
	@make train-sklearn
	@echo ""
	@echo "$(YELLOW)Step 2/5: Training with MLflow Tracking (Part 2)$(NC)"
	@make train-mlflow
	@echo ""
	@echo "$(YELLOW)Step 3/5: Training PySpark Pipeline (Part 3)$(NC)"
	@make train-pyspark
	@echo ""
	@echo "$(YELLOW)Step 4/5: Testing Inference Pipeline$(NC)"
	@make test-inference
	@echo ""
	@echo "$(YELLOW)Step 5/5: Comparing Frameworks$(NC)"
	@make compare-frameworks
	@echo ""
	@echo "$(GREEN)========================================"
	@echo "✓ Complete pipeline finished!"
	@echo "========================================$(NC)"
	@echo ""
	@echo "Next steps:"
	@echo "  - View MLflow experiments: make mlflow-ui"
	@echo "  - Start Airflow orchestration: make airflow-up"
	@echo "  - Generate reports: make generate-report"

run-sklearn-only: ## 🎯 Run only Scikit-Learn components (Part 1 & 2)
	@echo "$(YELLOW)Running Scikit-Learn components...$(NC)"
	@make train-sklearn
	@make train-mlflow
	@make test-inference
	@echo "$(GREEN)✓ Scikit-Learn pipeline complete$(NC)"

run-pyspark-only: check-java ## 🎯 Run only PySpark components (Part 3)
	@echo "$(YELLOW)Running PySpark components...$(NC)"
	@make train-pyspark
	@make compare-frameworks
	@echo "$(GREEN)✓ PySpark pipeline complete$(NC)"

# ============================================================================
# TESTING & VALIDATION
# ============================================================================

test: ## 🧪 Run all tests
	@echo "$(YELLOW)Running tests...$(NC)"
	@source $(VENV) && pytest tests/ -v
	@echo "$(GREEN)✓ Tests complete$(NC)"

test-coverage: ## 📊 Run tests with coverage report
	@echo "$(YELLOW)Running tests with coverage...$(NC)"
	@source $(VENV) && pytest tests/ --cov=src --cov-report=html --cov-report=term
	@echo "$(GREEN)✓ Coverage report generated in htmlcov/$(NC)"

lint: ## 🔍 Run code linting
	@echo "$(YELLOW)Running linters...$(NC)"
	@source $(VENV) && flake8 src/ --max-line-length=100 || true
	@source $(VENV) && pylint src/ --max-line-length=100 || true
	@echo "$(GREEN)✓ Linting complete$(NC)"

# ============================================================================
# JUPYTER NOTEBOOKS
# ============================================================================

notebook: ## 📓 Start Jupyter notebook server
	@echo "$(YELLOW)Starting Jupyter notebook server...$(NC)"
	@source $(VENV) && jupyter notebook --notebook-dir=notebooks
	@echo "$(GREEN)✓ Jupyter stopped$(NC)"

notebook-bg: ## 📓 Start Jupyter in background
	@echo "$(YELLOW)Starting Jupyter in background...$(NC)"
	@source $(VENV) && jupyter notebook --notebook-dir=notebooks --no-browser > /dev/null 2>&1 &
	@sleep 3
	@echo "$(GREEN)✓ Jupyter running. Check terminal for access URL$(NC)"

# ============================================================================
# REPORTING & DOCUMENTATION
# ============================================================================

generate-report: ## 📄 Generate executive summary report
	@echo "$(YELLOW)Generating executive summary...$(NC)"
	@echo "Report generated in reports/EXECUTIVE_SUMMARY.md"
	@echo "$(GREEN)✓ Report complete$(NC)"

screenshot-checklist: ## 📸 Print screenshot checklist
	@echo "$(YELLOW)========================================"
	@echo "Screenshot Checklist for Submission"
	@echo "========================================$(NC)"
	@echo ""
	@echo "MLflow UI (Part 2):"
	@echo "  ☐ Experiments list page"
	@echo "  ☐ Run comparison view"
	@echo "  ☐ Metrics comparison chart"
	@echo "  ☐ Model artifacts page"
	@echo "  ☐ Confusion matrix plots"
	@echo ""
	@echo "Airflow UI (Part 4):"
	@echo "  ☐ DAG graph view"
	@echo "  ☐ DAG tree view"
	@echo "  ☐ Successful run status"
	@echo "  ☐ Task logs"
	@echo "  ☐ Gantt chart view"
	@echo ""
	@echo "Save screenshots to: reports/"

submission-check: ## ✅ Verify submission completeness
	@echo "$(YELLOW)========================================"
	@echo "Submission Checklist"
	@echo "========================================$(NC)"
	@echo ""
	@echo "Part 1: Scikit-Learn Pipelines (25 marks)"
	@test -f src/pipelines/sklearn_pipeline.py && echo "  $(GREEN)✓$(NC) sklearn_pipeline.py" || echo "  $(RED)✗$(NC) sklearn_pipeline.py"
	@test -f src/pipelines/inference_pipeline.py && echo "  $(GREEN)✓$(NC) inference_pipeline.py" || echo "  $(RED)✗$(NC) inference_pipeline.py"
	@test -f src/models/sklearn_pipeline.pkl && echo "  $(GREEN)✓$(NC) Model saved" || echo "  $(RED)✗$(NC) Model not saved"
	@echo ""
	@echo "Part 2: MLflow Tracking (25 marks)"
	@test -f src/pipelines/sklearn_mlflow_pipeline.py && echo "  $(GREEN)✓$(NC) sklearn_mlflow_pipeline.py" || echo "  $(RED)✗$(NC) sklearn_mlflow_pipeline.py"
	@test -d mlruns && echo "  $(GREEN)✓$(NC) MLflow tracking data" || echo "  $(RED)✗$(NC) MLflow tracking missing"
	@test -d reports/figures && echo "  $(GREEN)✓$(NC) Visualization plots" || echo "  $(RED)✗$(NC) Plots missing"
	@echo ""
	@echo "Part 3: PySpark MLlib (30 marks)"
	@test -f src/pipelines/pyspark_pipeline.py && echo "  $(GREEN)✓$(NC) pyspark_pipeline.py" || echo "  $(RED)✗$(NC) pyspark_pipeline.py"
	@test -d src/models/pyspark_model && echo "  $(GREEN)✓$(NC) PySpark model saved" || echo "  $(RED)✗$(NC) PySpark model missing"
	@echo ""
	@echo "Part 4: Airflow Orchestration (20 marks)"
	@test -f airflow-docker/docker-compose.yml && echo "  $(GREEN)✓$(NC) docker-compose.yml" || echo "  $(RED)✗$(NC) docker-compose.yml"
	@test -f airflow-docker/dags/telco_churn_pipeline_dag.py && echo "  $(GREEN)✓$(NC) DAG file" || echo "  $(RED)✗$(NC) DAG file missing"
	@echo ""
	@echo "Documentation (Bonus 5 marks)"
	@test -f README.md && echo "  $(GREEN)✓$(NC) README.md" || echo "  $(RED)✗$(NC) README.md"
	@test -f reports/EXECUTIVE_SUMMARY.md && echo "  $(GREEN)✓$(NC) Executive Summary" || echo "  $(RED)✗$(NC) Executive Summary"
	@test -f config/config.yaml && echo "  $(GREEN)✓$(NC) Configuration file" || echo "  $(RED)✗$(NC) Config missing"
	@test -f requirements.txt && echo "  $(GREEN)✓$(NC) requirements.txt" || echo "  $(RED)✗$(NC) requirements.txt"

# ============================================================================
# CLEANUP & MAINTENANCE
# ============================================================================

clean: ## 🧹 Clean generated files and artifacts
	@echo "$(YELLOW)Cleaning up...$(NC)"
	@rm -rf src/models/*.pkl
	@rm -rf src/models/pyspark_model
	@rm -rf mlruns
	@rm -rf reports/figures/*.png
	@rm -rf data/processed/*
	@rm -rf data/predictions/*
	@rm -rf .pytest_cache
	@rm -rf htmlcov
	@rm -rf .coverage
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "$(GREEN)✓ Cleanup complete$(NC)"

clean-all: clean airflow-down ## 🧹 Deep clean (including Docker)
	@echo "$(YELLOW)Performing deep clean...$(NC)"
	@cd airflow-docker && docker-compose down -v
	@rm -rf .venv
	@echo "$(GREEN)✓ Deep clean complete$(NC)"
	@echo "Run 'make install' to set up again"

reset: clean ## 🔄 Reset project (keep venv and data)
	@echo "$(YELLOW)Resetting project...$(NC)"
	@make clean
	@echo "$(GREEN)✓ Project reset complete$(NC)"
	@echo "Run 'make run-all' to start fresh"

# ============================================================================
# UTILITY TARGETS
# ============================================================================

disk-usage: ## 💾 Show disk usage of project components
	@echo "$(YELLOW)Disk usage by component:$(NC)"
	@du -sh data/ 2>/dev/null || echo "data/: N/A"
	@du -sh mlruns/ 2>/dev/null || echo "mlruns/: N/A"
	@du -sh src/models/ 2>/dev/null || echo "src/models/: N/A"
	@du -sh .venv/ 2>/dev/null || echo ".venv/: N/A"
	@du -sh airflow-docker/ 2>/dev/null || echo "airflow-docker/: N/A"

ports-check: ## 🔌 Check if required ports are available
	@echo "$(YELLOW)Checking port availability...$(NC)"
	@lsof -i:$(MLFLOW_PORT) > /dev/null 2>&1 && \
		echo "$(RED)✗ Port $(MLFLOW_PORT) (MLflow) is in use$(NC)" || \
		echo "$(GREEN)✓ Port $(MLFLOW_PORT) (MLflow) is available$(NC)"
	@lsof -i:$(AIRFLOW_PORT) > /dev/null 2>&1 && \
		echo "$(RED)✗ Port $(AIRFLOW_PORT) (Airflow) is in use$(NC)" || \
		echo "$(GREEN)✓ Port $(AIRFLOW_PORT) (Airflow) is available$(NC)"
	@lsof -i:8888 > /dev/null 2>&1 && \
		echo "$(RED)✗ Port 8888 (Jupyter) is in use$(NC)" || \
		echo "$(GREEN)✓ Port 8888 (Jupyter) is available$(NC)"

project-info: ## ℹ️  Show project information
	@echo "$(YELLOW)========================================"
	@echo "Project Information"
	@echo "========================================$(NC)"
	@echo "Project: Telco Customer Churn Prediction"
	@echo "Type: Production ML System"
	@echo "Components: Scikit-Learn, PySpark, MLflow, Airflow"
	@echo ""
	@echo "Marks Breakdown:"
	@echo "  Part 1 (Scikit-Learn): 25 marks"
	@echo "  Part 2 (MLflow): 25 marks"
	@echo "  Part 3 (PySpark): 30 marks"
	@echo "  Part 4 (Airflow): 20 marks"
	@echo "  Bonus (Documentation): 5 marks"
	@echo "  Total: 105 marks"

quick-demo: ## 🎬 Quick demo run (fast version for testing)
	@echo "$(YELLOW)Running quick demo...$(NC)"
	@make train-sklearn
	@make mlflow-ui-bg
	@sleep 3
	@echo ""
	@echo "$(GREEN)✓ Demo complete!$(NC)"
	@echo "MLflow UI: http://localhost:$(MLFLOW_PORT)"
	@echo "Stop MLflow: make mlflow-stop"

# ============================================================================
# ALIASES & SHORTCUTS
# ============================================================================

init: install setup-dirs download-data ## 🚀 Complete initial setup
	@echo "$(GREEN)========================================"
	@echo "✓ Initial setup complete!"
	@echo "========================================$(NC)"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Activate environment: source .venv/bin/activate"
	@echo "  2. Run pipeline: make run-all"
	@echo "  3. View experiments: make mlflow-ui"
	@echo "  4. Start orchestration: make airflow-up"

dev: install setup-dirs ## 🛠️  Development setup
	@make verify-env
	@make ports-check
	@echo "$(GREEN)✓ Development environment ready$(NC)"

prod: run-all ## 🏭 Production run (all components)
	@echo "$(GREEN)✓ Production pipeline complete$(NC)"
	@make submission-check