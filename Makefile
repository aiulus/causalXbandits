.PHONY: help
help:
	@echo ""
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@echo "  init            Initialize submodules"
	@echo "  update          Pull latest changes for all submodules"
	@echo "  status          Show current submodule status"
	@echo "  install         Install submodules as editable packages"
	@echo "  clean           Remove build artifacts (optional)"
	@echo "  venv            Create and activate virtual environment (optional)"
	@echo ""

# Initialize and update submodules
.PHONY: init
init:
	git submodule update --init --recursive

# Pull latest from each submodule
.PHONY: update
update:
	@git submodule foreach 'echo Updating $$name... && git pull origin main'

# Show status of submodules
.PHONY: status
status:
	@git submodule status

# Install submodules as editable packages (e.g., for Python)
.PHONY: install
install:
	@echo "Installing causal_models..."
	pip install -e causal_models
	@echo "Installing mab..."
	pip install -e mab

# Optional: Set up virtual environment
.PHONY: venv
venv:
	@echo "Creating venv..."
	python -m venv .venv
	@echo "To activate: source .venv/bin/activate (Linux/Mac) or .venv\\Scripts\\activate (Windows)"

# Optional: Clean build caches
.PHONY: clean
clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
