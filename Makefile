# Developer workflow shortcuts. These are conveniences, not required — every command
# below is a plain invocation you can also run by hand (see the README).
#
# MLFLOW_TRACKING_URI is pinned to the same SQLite store the app defaults to
# (agent/observability.py::_DEFAULT_TRACKING_URI) so `make screen` and `make mlflow-ui`
# always agree on where traces live.

MLFLOW_TRACKING_URI ?= sqlite:///mlflow.db
MLFLOW_SERVER_URL   ?= http://127.0.0.1:5000
ARGS                ?=

.PHONY: screen screen-voice screen-es screen-server mlflow-ui mlflow-server test eval

# --- run the agent ---------------------------------------------------------------

screen:                ## Traced screening; forward cli.py flags via ARGS, e.g. ARGS="--voice"
	MLFLOW_TRACING=1 python cli.py $(ARGS)

screen-voice:          ## Shortcut for `make screen ARGS="--voice"`
	$(MAKE) screen ARGS="--voice"

screen-es:             ## Shortcut for `make screen ARGS="--lang es --voice"` (ES voice)
	$(MAKE) screen ARGS="--lang es --voice"

screen-server:         ## Traced run that POSTs to `make mlflow-server` (start that first)
	MLFLOW_TRACKING_URI=$(MLFLOW_SERVER_URL) python cli.py $(ARGS)

# --- observability ---------------------------------------------------------------

mlflow-ui:             ## Browse traces at http://127.0.0.1:5000 (reads the SQLite DB directly)
	MLFLOW_TRACKING_URI=$(MLFLOW_TRACKING_URI) mlflow ui

mlflow-server:         ## Server-first model: start this, THEN `make screen-server` (separate shell)
	mlflow server --backend-store-uri $(MLFLOW_TRACKING_URI) --port 5000

# --- tests / eval ----------------------------------------------------------------

test:                  ## Unit suite (no network/credentials required)
	pytest

eval:                  ## Extraction precision + false-positive report
	python -m eval.harness
