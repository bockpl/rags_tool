FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System dependencies for scientific stack
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install python dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    qdrant-client \
    openai \
    pydantic \
    pydantic-settings \
    tiktoken \
    scikit-learn \
    numpy \
    markdown2 \
    beautifulsoup4 \
    html2text \
    PyPDF2

COPY app.py settings.py ./

EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
