PYTHON ?= python3.12
VENV ?= .venv
BIN := $(VENV)/bin

.PHONY: install test eval lint

install:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/python -m pip install -e ".[dev]"

test:
	$(BIN)/pytest -W error --ignore=tests/eval

eval:
	$(BIN)/pytest -W error tests/eval

lint:
	$(BIN)/ruff check .
