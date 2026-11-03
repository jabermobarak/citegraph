FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && pip install --no-cache-dir .
COPY artifacts ./artifacts
RUN python -m citegraph.cli prepare && chmod -R a+rX /app/artifacts /app/data && useradd --create-home researcher
USER researcher
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"
CMD ["python", "-m", "uvicorn", "citegraph.api:app", "--host", "0.0.0.0", "--port", "8000"]
