.PHONY: help dev-up dev-down prod-up prod-down test lint format clean

help:
	@echo "Available commands:"
	@echo "  make prod-up    - Start full production stack via docker compose"
	@echo "  make prod-down  - Stop production stack"
	@echo "  make dev-up     - Start development infrastructure stack"
	@echo "  make dev-down   - Stop development infrastructure stack"
	@echo "  make test       - Run Python unit & algorithm tests"
	@echo "  make lint       - Run ruff and clang-format checks"
	@echo "  make format     - Automatically format code"
	@echo "  make clean      - Clean cache and temporary files"

prod-up:
	docker compose -f deployments/docker/docker-compose.prod.yml up -d

prod-down:
	docker compose -f deployments/docker/docker-compose.prod.yml down

dev-up:
	docker compose -f deployments/docker/docker-compose.dev.yml up -d

dev-down:
	docker compose -f deployments/docker/docker-compose.dev.yml down

test:
	pytest services/analytics_api/tests/ -v

lint:
	ruff check services/analytics_api/ tools/

format:
	ruff format services/analytics_api/ tools/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
