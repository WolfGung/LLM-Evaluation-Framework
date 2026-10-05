PYTHON ?= python3.12
VENV ?= .venv
BIN := $(VENV)/bin

.PHONY: install test eval baseline gate readme docs estimate status record prune label lint

install:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/python -m pip install -e ".[dev]"

# Unit, app and repository tests. No network, no key.
test:
	$(BIN)/pytest -W error --ignore=tests/eval

# Replay the recorded run: write results/, then the judge's agreement with the
# owner's labels (results/judge-agreement.json; "pending human labels" until
# labels/human.jsonl has labels), then compare every case with the baseline.
# Without cassettes/manifest.json all three say "pending first recorded run".
eval:
	$(BIN)/llmeval eval
	$(BIN)/llmeval agreement
	$(BIN)/pytest -W error tests/eval

# Write results/baseline.json from the replay results in results/ and the
# manifest: each case's known failures and the key metrics. Refuses missing or
# stale results (run make eval first). Update it deliberately: the eval tests
# and the gate compare with it.
baseline:
	$(BIN)/llmeval baseline

# Compare the key rates of results/ with results/baseline.json, within the
# tolerances of config/gate.yaml; exit 1 on a regression. A replay always
# equals the baseline; the tolerances matter for live (drift) results.
gate:
	$(BIN)/llmeval gate

# Write every generated block of README.md and docs/*.md (between
# <!-- NAME:start --> and <!-- NAME:end -->) from results/, the manifest, the
# gate tolerances and the datasets. A repository test fails when a file
# differs; python -m tools.render --check says the same. make docs is the same.
readme:
	$(BIN)/python -m tools.render --write

docs:
	$(BIN)/python -m tools.render --write

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

# The owner's labelling tool: shows each answer of labels/sample.json (the
# question, the documents the assistant was given, the answer; never the
# judge's verdict) and appends the owner's pass or fail, with a comment, to
# labels/human.jsonl. Only the owner runs it. Ctrl-C keeps every saved label;
# run it again to go on. Ctrl-C makes llmeval exit with 130 after its own
# message; make takes that as a clean stop. Any other error still fails.
# Ctrl-C reaches the recipe's shell too: bash waits for llmeval and runs the
# || part, where dash would die first and make would print "Interrupt".
label: SHELL := /bin/bash
label:
	@$(BIN)/llmeval label || [ $$? -eq 130 ]

lint:
	$(BIN)/ruff check .
