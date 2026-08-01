FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# CBC (the MILP solver) ships inside the pulp wheel — no system solver needed.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e ".[api]"

EXPOSE 8000
CMD ["uvicorn", "zephyrus.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
