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
    PyPDF2 \
    spacy

RUN python -m spacy download pl_core_news_sm

COPY app ./app
COPY main.py ./
COPY templates ./templates
COPY tools ./tools

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
