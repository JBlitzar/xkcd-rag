import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from typing import Dict, List, Optional
import os
import requests
from pathlib import Path

class QuantizedEmbedder:
    def __init__(
        self, 
        model_name: str = "nomic-ai/nomic-embed-text-v1.5",
        cache_dir: str = "onnx_cache"
    ):
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Download and cache the quantized ONNX model
        self.model_path = self._download_quantized_model()
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Create ONNX Runtime session with optimizations
        providers = ['CPUExecutionProvider']
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = os.cpu_count()
        
        self.session = ort.InferenceSession(
            str(self.model_path), 
            sess_options=sess_options, 
            providers=providers
        )
        
        # Get input/output names
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
    
    def _download_quantized_model(self) -> Path:
        """Download the quantized ONNX model if not cached."""
        model_filename = "model_quantized.onnx"
        local_path = self.cache_dir / model_filename
        
        if local_path.exists():
            print(f"Using cached quantized model: {local_path}")
            return local_path
        
        # Download from HuggingFace
        url = f"https://huggingface.co/{self.model_name}/resolve/main/onnx/{model_filename}"
        print(f"Downloading quantized model from {url}...")
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Downloaded quantized model to {local_path}")
        return local_path
    
    def _tokenize(self, texts: List[str], max_length: int = 8192) -> Dict[str, np.ndarray]:
        """Tokenize texts for ONNX model input."""
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="np"
        )
        
        # Convert to the expected input format for ONNX
        inputs = {}
        for key, value in encoded.items():
            if key in self.input_names:
                inputs[key] = value.astype(np.int64)
        
        return inputs
    
    def _mean_pooling(self, token_embeddings: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        """Apply mean pooling to get sentence embeddings."""
        # Expand attention mask to match token embeddings dimensions
        input_mask_expanded = np.expand_dims(attention_mask, axis=-1)
        input_mask_expanded = np.broadcast_to(input_mask_expanded, token_embeddings.shape)
        
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
        if 'attention_mask' in inputs:
            embeddings = self._mean_pooling(token_embeddings, inputs['attention_mask'])
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


if __name__ == "__main__":
    embedder = QuantizedEmbedder()
    texts = [
        "This is a test sentence.",
        "Another sentence for embedding."
    ]
    embeddings = embedder.encode(texts)
    print("Embeddings shape:", embeddings.shape)
    print(embeddings)