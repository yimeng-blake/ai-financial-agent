# ============================================================================
# AI Financial Agent — Production Docker Image
# ============================================================================
FROM python:3.11-slim AS base

# Prevent Python from buffering stdout/stderr (important for Docker logs)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# ----------------------------------------------------------------------------
# Install system dependencies
# pymupdf ships pre-built wheels so no build tools needed
# ----------------------------------------------------------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ----------------------------------------------------------------------------
# Install Python dependencies (layer cached — changes only when pyproject changes)
# ----------------------------------------------------------------------------
COPY pyproject.toml ./

# Install pip and poetry, then export requirements and pip install
# (avoids needing poetry in the final image)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi --no-root && \
    pip uninstall -y poetry && \
    rm -rf /root/.cache

# ----------------------------------------------------------------------------
# Copy application code
# ----------------------------------------------------------------------------
COPY app.py main.py ./
COPY src/ ./src/
COPY static/ ./static/
COPY templates/ ./templates/

# ----------------------------------------------------------------------------
# Runtime configuration
# ----------------------------------------------------------------------------
EXPOSE 8000

# Run with uvicorn — no --reload in production
# Workers: 1 (in-memory session store for PDF uploads is per-process)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
