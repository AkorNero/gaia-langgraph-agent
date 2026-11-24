# Image for the run_python sandbox. Build once:
#   docker build -t agent-sandbox -f sandbox.Dockerfile .
FROM python:3.12-slim
RUN pip install --no-cache-dir numpy pandas openpyxl tabulate
USER 65534:65534
WORKDIR /tmp
