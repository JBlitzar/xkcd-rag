import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
from typing import Dict, List, Optional
import os
import requests
from pathlib import Path


class _QuantizedEmbedder:
    def __init__(
        self,
        model_name: str = "nomic-ai/nomic-embed-text-v1.5",
        cache_dir: str = "onnx_cache",
        num_threads: int = 4,  # NEW: configurable thread count
    ):
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.num_threads = num_threads

        # Download and cache the quantized ONNX model
        self.model_path = self._download_quantized_model()

        # Load tokenizer
        self.tokenizer = Tokenizer.from_pretrained(model_name)

        # CRITICAL: Disable tokenizer parallelism to avoid thread explosion
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # Create ONNX Runtime session with optimizations
        providers = ["CPUExecutionProvider"]
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        # FIXED: Use controlled thread count instead of all CPUs
        sess_options.intra_op_num_threads = self.num_threads
        sess_options.inter_op_num_threads = 1  # NEW: Disable inter-op parallelism
        sess_options.execution_mode = (
            ort.ExecutionMode.ORT_SEQUENTIAL
        )  # NEW: Sequential execution

        self.session = ort.InferenceSession(
            str(self.model_path), sess_options=sess_options, providers=providers
        )

        # Get input/output names
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]

    def _download_quantized_model(self) -> Path:
        """Download the quantized ONNX model if not cached."""
        model_filename = "model_quantized.onnx"
        local_path = self.cache_dir / model_filename

        if local_path.exists():
            # Removed print for production use
            return local_path

        # Download from HuggingFace
        url = f"https://huggingface.co/{self.model_name}/resolve/main/onnx/{model_filename}"
        print(f"Downloading quantized model from {url}...")

        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(local_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"Downloaded quantized model to {local_path}")
        return local_path

    def _tokenize(
        self, texts: List[str], max_length: int = 8192
    ) -> Dict[str, np.ndarray]:
        """Tokenize texts for ONNX model input."""

        encoded = self.tokenizer.encode_batch(texts)

        # Convert to the expected input format for ONNX
        inputs = {}

        # Extract input_ids and attention_mask from the encoded batch
        input_ids = np.array([enc.ids for enc in encoded])
        attention_mask = np.array([enc.attention_mask for enc in encoded])

        # Add to inputs dict based on model requirements
        if "input_ids" in self.input_names:
            inputs["input_ids"] = input_ids.astype(np.int64)
        if "attention_mask" in self.input_names:
            inputs["attention_mask"] = attention_mask.astype(np.int64)
        if "token_type_ids" in self.input_names:
            # Create token_type_ids (all zeros for single sequence)
            token_type_ids = np.zeros_like(input_ids)
            inputs["token_type_ids"] = token_type_ids.astype(np.int64)

        return inputs

    def _mean_pooling(
        self, token_embeddings: np.ndarray, attention_mask: np.ndarray
    ) -> np.ndarray:
        """Apply mean pooling to get sentence embeddings."""
        # Expand attention mask to match token embeddings dimensions
        input_mask_expanded = np.expand_dims(attention_mask, axis=-1)
        input_mask_expanded = np.broadcast_to(
            input_mask_expanded, token_embeddings.shape
        )

        # Apply mask and compute mean
        masked_embeddings = token_embeddings * input_mask_expanded
        sum_embeddings = np.sum(masked_embeddings, axis=1)
        sum_mask = np.sum(input_mask_expanded, axis=1)

        # Avoid division by zero
        sum_mask = np.maximum(sum_mask, 1e-9)
        mean_embeddings = sum_embeddings / sum_mask

        return mean_embeddings

    def encode(self, texts: List[str] | str, normalize: bool = True) -> np.ndarray:
        """Encode texts to embeddings using quantized ONNX model."""
        if isinstance(texts, str):
            texts = [texts]

        # Tokenize
        inputs = self._tokenize(texts)

        # Run inference
        outputs = self.session.run(self.output_names, inputs)

        # Extract embeddings (usually the first output)
        token_embeddings = outputs[0]

        # Apply mean pooling if attention mask is available
        if "attention_mask" in inputs:
            embeddings = self._mean_pooling(token_embeddings, inputs["attention_mask"])
        else:
            # Fallback to simple mean along sequence dimension
            embeddings = np.mean(token_embeddings, axis=1)

        # Normalize embeddings if requested
        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)  # Avoid division by zero
            embeddings = embeddings / norms

        return embeddings.astype(np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        """Encode a search query with the appropriate prefix."""
        prefixed_text = f"search_query: {text}"
        return self.encode([prefixed_text])[0]

    def encode_document(self, text: str) -> np.ndarray:
        """Encode a document with the appropriate prefix."""
        prefixed_text = f"search_document: {text}"
        return self.encode([prefixed_text])[0]


embedder = None


def getQuantizedEmbedder() -> _QuantizedEmbedder:
    # singleton pattern
    global embedder
    if embedder is None:
        embedder = _QuantizedEmbedder(num_threads=4)  # TUNABLE: Start with 4
    return embedder


if __name__ == "__main__":
    embedder = getQuantizedEmbedder()
    texts = ["This is a test sentence.", "Another sentence for embedding."]
    embeddings = embedder.encode(texts)
    print("Embeddings shape:", embeddings.shape)
    print(embeddings)
