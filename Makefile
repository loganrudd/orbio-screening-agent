# Developer workflow shortcuts. These are conveniences, not required — every command
# below is a plain invocation you can also run by hand (see the README).
#
# MLFLOW_TRACKING_URI is pinned to the same SQLite store the app defaults to
# (agent/observability.py::_DEFAULT_TRACKING_URI) so `make screen` and `make mlflow-ui`
# always agree on where traces live.

MLFLOW_TRACKING_URI ?= sqlite:///mlflow.db
MLFLOW_SERVER_URL   ?= http://127.0.0.1:5000

.PHONY: screen screen-voice screen-es screen-server mlflow-ui mlflow-server test eval

# --- run the agent ---------------------------------------------------------------

screen:                ## Text screening with MLflow tracing on
	MLFLOW_TRACING=1 python cli.py

screen-voice:          ## Voice screening (Deepgram STT/TTS) with tracing on
	MLFLOW_TRACING=1 python cli.py --voice

screen-es:             ## Spanish voice screening (greeting starts in ES) with tracing on
	MLFLOW_TRACING=1 python cli.py --lang es --voice

screen-server:         ## Traced run that POSTs to `make mlflow-server` (start that first)
	MLFLOW_TRACKING_URI=$(MLFLOW_SERVER_URL) python cli.py

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
