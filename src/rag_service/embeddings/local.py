from ..config import get_settings


class LocalEmbedder:
    """sentence-transformers model, loaded from the HF hub or a local folder (e.g. ./model/...)."""

    def __init__(self, model_name_or_path: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name_or_path)
        self.dim = self.model.get_sentence_embedding_dimension()
        # BGE models expect an instruction prefix on queries only
        self.query_prefix = (
            "Represent this sentence for searching relevant passages: "
            if "bge" in model_name_or_path.lower()
            else ""
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vecs = self.model.encode(
            texts,
            batch_size=get_settings().embedding_batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vecs.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(self.query_prefix + text, normalize_embeddings=True).tolist()
