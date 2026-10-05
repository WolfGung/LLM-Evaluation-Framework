PYTHON ?= python3.12
VENV ?= .venv
BIN := $(VENV)/bin

.PHONY: install test eval estimate status record prune lint

install:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/python -m pip install -e ".[dev]"

# Unit, app and repository tests. No network, no key.
test:
	$(BIN)/pytest -W error --ignore=tests/eval

# Replay the recorded run: write results/, then compare every case with the
# baseline. Without cassettes/manifest.json both say "pending first recorded run".
eval:
	$(BIN)/llmeval eval
	$(BIN)/pytest -W error tests/eval

# The call plan, the free-quota days and the estimated cost. No key needed.
estimate:
	$(BIN)/llmeval estimate

# Calls planned and recorded, and the free requests left today when
# OPENROUTER_API_KEY is set.
status:
	$(BIN)/llmeval status

# Record every planned call with the real API. Reads OPENROUTER_API_KEY from
# the environment (for example: set -a; . ./.env; set +a). Exit code 75 means
# the free quota or a rate limit stopped it: rerun later to continue. Ctrl-C
# stops it cleanly (exit code 130); every recorded call is kept.
record:
	$(BIN)/llmeval record

# Remove the recorded entries the current plan no longer has, for example the
# judge calls of an older judge config: runs llmeval prune --yes. To see the
# list first and change nothing, run .venv/bin/llmeval prune. The cassettes are
# in git, so git can bring removed entries back. Prune never edits the
# manifest; make record rewrites it once the current plan is fully recorded.
prune:
	$(BIN)/llmeval prune --yes

lint:
	$(BIN)/ruff check .
