# Developer workflow shortcuts. These are conveniences, not required — every command
# below is a plain invocation you can also run by hand (see the README).
#
# MLFLOW_TRACKING_URI is pinned to the same SQLite store the app defaults to
# (agent/observability.py::_DEFAULT_TRACKING_URI) so `make screen` and `make mlflow-ui`
# always agree on where traces live.

MLFLOW_TRACKING_URI ?= sqlite:///mlflow.db

.PHONY: screen screen-voice screen-es mlflow-ui test eval

# --- run the agent ---------------------------------------------------------------

screen:                ## Text screening with MLflow tracing on
	MLFLOW_TRACING=1 python cli.py

screen-voice:          ## Voice screening (Deepgram STT/TTS) with tracing on
	MLFLOW_TRACING=1 python cli.py --voice

screen-es:             ## Spanish voice screening (greeting starts in ES) with tracing on
	MLFLOW_TRACING=1 python cli.py --lang es --voice

# --- observability ---------------------------------------------------------------

mlflow-ui:             ## Browse traces at http://127.0.0.1:5000 (same DB as `make screen`)
	MLFLOW_TRACKING_URI=$(MLFLOW_TRACKING_URI) mlflow ui

# --- tests / eval ----------------------------------------------------------------

test:                  ## Unit suite (no network/credentials required)
	pytest

eval:                  ## Extraction precision + false-positive report
	python -m eval.harness
