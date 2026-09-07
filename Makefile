.PHONY: install lint format test run docker-build

install:
	python -m pip install -e ".[dev]"
	pre-commit install

lint:
	ruff check .
	ruff format --check .

format:
	ruff check --fix .
	ruff format .

test:
	pytest

run:
	uvicorn bikeflow.api.main:app --reload --host 0.0.0.0 --port 8000

docker-build:
	docker build --tag bikeflow:local .
