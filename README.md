# SummRAG (0.2.0)

Dwustopniowy serwis RAG zbudowany na FastAPI. System wspiera streszczanie dokumentów, indeksowanie w Qdrant oraz wyszukiwanie hybrydowe (dense + TF-IDF).

## Nowości w 0.2.0

- Hybrydowy etap recall na streszczeniach — wyniki dense są łączone z dopasowaniem TF-IDF po streszczeniach i sygnaturach dokumentów.
- Podczas ingestu zapisywane są sparse wektory streszczeń i treści, co stabilizuje ranking hybrydowy w obu etapach wyszukiwania.

## Wymagania

- Python 3.11+
- Zewnętrzna instancja Qdrant (SaaS lub self-hosted)
- Endpoint zgodny z protokołem OpenAI zapewniający embeddingi i model konwersacyjny (np. vLLM, Ollama, OpenAI)

## Konfiguracja środowiska

Ustaw wymagane zmienne środowiskowe przed uruchomieniem aplikacji:

```bash
export QDRANT_URL="http://127.0.0.1:6333"
export QDRANT_API_KEY=""
export EMBEDDING_API_URL="http://127.0.0.1:8000/v1"
export EMBEDDING_API_KEY="sk-embed-xxx"
export EMBEDDING_MODEL="BAAI/bge-m3"
export SUMMARY_API_URL="http://127.0.0.1:8001/v1"
export SUMMARY_API_KEY="sk-summary-xxx"
export SUMMARY_MODEL="gpt-4o-mini"
export COLLECTION_NAME="rags_tool"
export VECTOR_STORE_DIR=".rags_tool_store"
export DEBUG="false"  # ustaw na "true", aby włączyć logi debugujące
```

Możesz także umieścić te wartości w pliku `.env`; aplikacja wczyta je automatycznie dzięki `pydantic-settings`. W repo znajdziesz przykładowy plik `.env`, który możesz skopiować i dostosować. Flaga `DEBUG=true` włącza szczegółowe logi z przebiegu ingestu (po jednym wpisie na dokument).

## Uruchomienie lokalne

1. Utwórz i aktywuj wirtualne środowisko.
2. Zainstaluj zależności:
   ```bash
   pip install fastapi uvicorn qdrant-client openai pydantic pydantic-settings tiktoken scikit-learn markdown2 beautifulsoup4 html2text PyPDF2
   ```
3. Start serwera:
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8080
   ```
4. Dokumentacja API jest dostępna pod `/docs` (Swagger UI) oraz `/openapi.json`.

## Uruchomienie w Dockerze

1. Zbuduj obraz:
   ```bash
   docker build -t rags_tool:latest .
   ```
2. Uruchom kontener, przekazując wymagane zmienne środowiskowe:
   ```bash
   docker run --rm -p 8080:8080 \
     --env-file .env \
    -v $(pwd)/.rags_tool_store:/app/.rags_tool_store \
    rags_tool:latest
   ```

> **Uwaga**: Qdrant i modele LLM muszą działać poza kontenerem i być dostępne pod adresami przekazanymi w zmiennych środowiskowych.

## Najważniejsze endpointy

- `GET /about` – metadane usługi
- `GET /health` – sprawdzenie dostępności Qdrant
- `POST /collections/init` – utworzenie/aktualizacja kolekcji w Qdrant
- `POST /ingest/scan` – skanowanie katalogu z dokumentami
- `POST /ingest/build` – pełny pipeline ingestu (streszczenia + embedding + indeks)
- `POST /summaries/generate` – generowanie streszczeń pojedynczych plików
- `POST /documents/upsert` – wstawianie dokumentu wraz z gotowymi danymi
- `POST /search/query` – zapytania dwustopniowe z hybrydowym rankingiem

## Licencja

MIT
