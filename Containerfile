# pqnld sidecar — a thin HTTP layer over an OpenAI-compatible vLLM endpoint.
FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --no-compile .

EXPOSE 11560
ENTRYPOINT ["pqnld"]
CMD ["--vllm-url", "http://127.0.0.1:11542", "--host", "0.0.0.0", "--port", "11560"]
