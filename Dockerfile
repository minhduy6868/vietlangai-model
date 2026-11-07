FROM python:3.11-slim
ARG EXTRAS=""
WORKDIR /app
COPY pyproject.toml ./
COPY app ./app
COPY data ./data
RUN if [ -n "$EXTRAS" ]; then pip install --no-cache-dir ".[$EXTRAS]"; else pip install --no-cache-dir .; fi
EXPOSE 8100
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8100"]
