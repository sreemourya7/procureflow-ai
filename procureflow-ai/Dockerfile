FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY procureflow ./procureflow
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && pip install --no-cache-dir .
COPY data ./data
COPY artifacts ./artifacts
COPY results ./results
RUN useradd --create-home reviewer && mkdir -p /app/runtime && chown -R reviewer:reviewer /app
USER reviewer
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "procureflow.server:app", "--host", "0.0.0.0", "--port", "8000"]
