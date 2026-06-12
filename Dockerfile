# Proportionate, deployable artifact. The service is stateless (state lives behind
# ConversationStore), so this image scales horizontally behind a load balancer.
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY agent/ ./agent/
COPY eval/ ./eval/
COPY cli.py .

# Default to the CLI; in production this would launch the FastAPI app (stretch).
ENTRYPOINT ["python", "cli.py"]
