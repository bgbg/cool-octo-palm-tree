#!/usr/bin/env python
from typing import List, Optional
import logging
import os
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from cachier import cachier
from datetime import timedelta

# Configure logging
logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating embeddings from text using a sentence transformer model."""

    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize the embedding service with a specific model.

        Args:
            model_name: Name of the model to use (from HuggingFace)
            device: Device to use for inference ("cpu", "cuda", or "auto")
        """
        # Get configuration from environment variables with defaults
        recommended_models = [
            # General-purpose Hebrew SBERT model (best for semantic search)
            "MPA/sambert",
            # General-purpose, distilled from LaBSE, optimized for Hebrew
            "imvladikon/sentence-transformers-alephbert",
            # Default multilingual model that works with Hebrew
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        ]

        default_model = recommended_models[0]
        self._model_name = model_name or os.getenv(
            "EMBEDDING_MODEL_NAME", default_model
        )
        self._device = device or os.getenv("EMBEDDING_DEVICE", "auto")

        # Auto-detect device if set to "auto"
        if self._device == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load the model
        self.model = SentenceTransformer(self._model_name, device=self._device)
        self.model_version = f"{self.model.__class__.__name__}_{self.model.get_sentence_embedding_dimension()}"
        logger.info(
            f"Loaded embedding model: {self._model_name} (version: {self.model_version}) on {self._device}"
        )

    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate an embedding for a single text string.

        Args:
            text: The text to embed

        Returns:
            Numpy array containing the embedding vector
        """
        return self.model.encode(text)

    def generate_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings for multiple text strings.

        Args:
            texts: List of texts to embed

        Returns:
            List of numpy arrays containing the embedding vectors
        """
        return self.model.encode(texts)


class OpenAIEmbeddingService:
    """Service for generating embeddings from text using the OpenAI embedding API (text-embedding-3-large by default)."""

    OPENAI_EMBEDDING_PRICING = {
        # model: price per 1K tokens (USD)
        "text-embedding-3-large": 0.00013,
        # Add more models if needed
    }

    def __init__(self, model_name: Optional[str] = None, api_key: Optional[str] = None):
        """
        Initialize the OpenAI embedding service.

        Args:
            model_name: Name of the OpenAI embedding model to use (default: text-embedding-3-large)
            api_key: OpenAI API key (default: from environment variable OPENAI_API_KEY)
        """
        import openai

        self.openai = openai
        self._model_name = model_name or os.getenv(
            "OPENAI_EMBEDDING_MODEL_NAME", "text-embedding-3-large"
        )
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY must be set in the environment or passed explicitly."
            )
        self.openai.api_key = self.api_key

    @cachier(stale_after=timedelta(days=1))
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate an embedding for a single text string using OpenAI API.

        Args:
            text: The text to embed

        Returns:
            Numpy array containing the embedding vector
        """
        response = self.openai.embeddings.create(input=text, model=self._model_name)
        # Log token usage and cost
        usage = getattr(response, "usage", None)
        if usage:

            def extract(key, default=0):
                try:
                    if isinstance(usage, dict):
                        return usage.get(key, default)
                    return getattr(usage, key, default)
                except Exception:
                    try:
                        return usage[key]
                    except Exception:
                        return default

            prompt_tokens = extract("prompt_tokens", 0)
            total_tokens = extract("total_tokens", 0)
            price_per_1k = self.OPENAI_EMBEDDING_PRICING.get(self._model_name, 0)
            cost = (total_tokens / 1000) * price_per_1k
            logger.info(
                f"OpenAI embedding | model={self._model_name} | prompt_tokens={prompt_tokens} | total_tokens={total_tokens} | cost_usd={cost:.8f}"
            )
        return np.array(response.data[0].embedding, dtype=np.float32)

    def generate_embeddings(
        self, texts: List[str], batch_size: int = 100
    ) -> List[np.ndarray]:
        """
        Generate embeddings for multiple text strings using OpenAI API, parallelized in batches.

        Args:
            texts: List of texts to embed
            batch_size: Number of texts per API call (default: 100, max: 2048)

        Returns:
            List of numpy arrays containing the embedding vectors
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def batch_embed(batch):
            response = self.openai.embeddings.create(
                input=batch, model=self._model_name
            )
            # Log token usage and cost
            usage = getattr(response, "usage", None)
            if usage:

                def extract(key, default=0):
                    try:
                        if isinstance(usage, dict):
                            return usage.get(key, default)
                        return getattr(usage, key, default)
                    except Exception:
                        try:
                            return usage[key]
                        except Exception:
                            return default

                prompt_tokens = extract("prompt_tokens", 0)
                total_tokens = extract("total_tokens", 0)
                price_per_1k = self.OPENAI_EMBEDDING_PRICING.get(self._model_name, 0)
                cost = (total_tokens / 1000) * price_per_1k
                logger.info(
                    f"OpenAI embedding | model={self._model_name} | prompt_tokens={prompt_tokens} | total_tokens={total_tokens} | cost_usd={cost:.8f}"
                )
            return [np.array(d.embedding, dtype=np.float32) for d in response.data]

        if len(texts) <= batch_size:
            return batch_embed(texts)

        batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
        results = [None] * len(batches)
        with ThreadPoolExecutor() as executor:
            future_to_idx = {
                executor.submit(batch_embed, batch): idx
                for idx, batch in enumerate(batches)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()
        # Flatten results
        return [vec for batch in results for vec in batch]


if __name__ == "__main__":
    # embedding_service = OpenAIEmbeddingService()
    # embedding_service.generate_embedding("Hello, world!")
    # print(embedding_service.generate_embedding("Hello, world!"))

    import numpy as np

    model_names = [
        "dicta-il/dictabert-large-char",
        "intfloat/e5-large-v2",
        "haguy77/sdictabert-heq",
        "yam-peleg/Hebrew-Mistral-7B",
    ]

    for model_name in model_names:
        service = EmbeddingService(model_name=model_name)
        embedding = service.generate_embedding("שלום")
        print(embedding.shape)
        print(np.linalg.norm(embedding))
