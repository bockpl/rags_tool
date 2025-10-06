"""OpenAI-compatible reranker client (HTTP), minimal dependency version.

Użycie:
    client = OpenAIReranker(base_url, api_key, model)
    results = client.rerank(query="zapytanie", documents=["passage1", "passage2"], top_n=50)

Zwraca listę słowników: {"index": int, "relevance_score": float} zgodnie z konwencją OpenAI‑/Cohere‑like `/v1/rerank`.

Założenia:
- Endpoint: POST {base_url}/v1/rerank
- Nagłówki: Authorization: Bearer <API_KEY>, Content-Type: application/json
- Body: {"model": str, "query": str, "documents": List[str], "top_n": int, "return_documents": false}
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from urllib import request as _rq


class OpenAIReranker:
    """Prosty klient HTTP dla endpointu rerank (OpenAI‑compatible).

    Celowo oparty o 'urllib' aby uniknąć dodatkowych zależności.
    """

    # Stałe ścieżki i timeout (ms)
    # Domyślna ścieżka endpointu; jeśli base_url kończy się na '/v1',
    # dołączamy tylko '/rerank' (unikanie podwójnego '/v1/v1').
    REQUEST_PATH = "/v1/rerank"
    TIMEOUT_SECONDS = 8.0  # 8000 ms

    def __init__(self, base_url: str, api_key: Optional[str], model: str):
        # Bazowy URL bez końcowego '/'
        self.base_url = (base_url or "").rstrip("/")
        self.api_key = api_key or ""
        self.model = model

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _build_url(self) -> str:
        base = self.base_url.rstrip("/")
        if base.endswith("/v1"):
            return f"{base}/rerank"
        return f"{base}{self.REQUEST_PATH}"

    def rerank(self, *, query: str, documents: List[str], top_n: int) -> List[Dict[str, Any]]:
        """Wywołaj endpoint rerank i zwróć listę {index, relevance_score}.

        Uwagi implementacyjne (opis kroków):
        1) Budujemy JSON: model + query + documents + top_n; nie prosimy o return_documents.
        2) Wysyłamy POST pod {base_url}/v1/rerank z Bearer (jeśli dostępny).
        3) Odczytujemy JSON, walidujemy minimalne pola.
        4) Zwracamy listę elementów z zachowaniem 'index' wejściowych dokumentów.
        """
        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": int(max(1, top_n)),
            "return_documents": False,
        }
        data = json.dumps(payload).encode("utf-8")
        url = self._build_url()
        req = _rq.Request(url, data=data, method="POST", headers=self._headers())
        with _rq.urlopen(req, timeout=self.TIMEOUT_SECONDS) as resp:
            raw = resp.read().decode("utf-8")
        obj = json.loads(raw)
        # Akceptujemy zarówno 'data' (niektóre serwery) jak i 'results' (np. vLLM/Cohere‑like)
        data_list = obj.get("data") or obj.get("results") or []
        results: List[Dict[str, Any]] = []
        for item in data_list:
            try:
                idx = int(item.get("index"))
                score = float(item.get("relevance_score"))
            except Exception:
                # Pomijamy niepoprawny wpis — zachowawczo
                continue
            results.append({"index": idx, "relevance_score": score})
        return results
