import logging

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


class EmbeddingGenerator:
    def __init__(self, model_name: str = MODEL_NAME):
        logger.info(f"Loading embedding model: {model_name}")
        try:
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            raise RuntimeError(f"Failed to load embedding model '{model_name}': {e}") from e
        logger.info(f"Embedding model loaded (dim={EMBEDDING_DIM})")

    def encode(self, text: str) -> list[float]:
        return self.model.encode(text, show_progress_bar=False).tolist()

    def encode_batch(
        self,
        texts: list[str],
        batch_size: int = 64,
        show_progress_bar: bool = True,
        device: str | None = None,
    ) -> list[list[float]]:
        kwargs = {
            "batch_size": batch_size,
            "show_progress_bar": show_progress_bar,
        }
        if device is not None:
            kwargs["device"] = device
        embeddings = self.model.encode(texts, **kwargs)
        return embeddings.tolist()