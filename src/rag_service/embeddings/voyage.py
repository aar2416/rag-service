_KNOWN_DIMS = {
    "voyage-3.5": 1024,
    "voyage-3.5-lite": 1024,
    "voyage-3-large": 1024,
    "voyage-code-3": 1024,
}


class VoyageEmbedder:
    """Voyage AI hosted embeddings (pip install voyageai)."""

    def __init__(self, model: str, api_key: str | None):
        import voyageai

        self.client = voyageai.Client(api_key=api_key)  # None -> VOYAGE_API_KEY env var
        self.model = model
        self.dim = _KNOWN_DIMS.get(model) or len(self.embed_query("dimension probe"))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.client.embed(texts, model=self.model, input_type="document").embeddings

    def embed_query(self, text: str) -> list[float]:
        return self.client.embed([text], model=self.model, input_type="query").embeddings[0]
