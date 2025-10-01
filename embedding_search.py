import numpy as np
import os
import json
import shutil
from explainxkcd import ExplainXKCDScraper
from tqdm import tqdm
from typing import Dict, Tuple, List, Optional
from sentence_transformers import SentenceTransformer


class EmbeddingCache:
    """Stores all embeddings in a single .npy plus an index.json mapping.

    - Combined array saved to <cache_dir>/embeddings.npy (shape: N x vector_size, dtype=float32)
    - Index mapping saved to <cache_dir>/index.json: {"<comic_number>": row_index}
    - Caches arrays and index in memory to avoid reloading each query.
    """

    def __init__(
        self,
        cache_dir: str = "embeddings_cache",
        model: str = "nomic-ai/nomic-embed-text-v1.5",
        vector_size: int = 768,
    ):
        self.cache_dir = cache_dir
        self.model = SentenceTransformer(model, trust_remote_code=True)
        self.vector_size = vector_size
        os.makedirs(self.cache_dir, exist_ok=True)

        self.combined_filename = "embeddings.npy"
        self.index_filename = "index.json"

        # in-memory caches
        self._combined_cache: Optional[np.ndarray] = None
        self._index_cache: Optional[Dict[str, int]] = None
        # cached mapping of comic_number -> embedding (loaded from combined array)
        self._doc_embeddings: Optional[Dict[int, np.ndarray]] = None

        # If there are many per-comic .npy files and no combined file, migrate them.
        self._maybe_migrate_individual_files()

    def _combined_path(self) -> str:
        return os.path.join(self.cache_dir, self.combined_filename)

    def _index_path(self) -> str:
        return os.path.join(self.cache_dir, self.index_filename)

    def _load_index(self) -> Dict[str, int]:
        if self._index_cache is not None:
            return self._index_cache
        if os.path.exists(self._index_path()):
            with open(self._index_path(), "r", encoding="utf-8") as f:
                self._index_cache = json.load(f)
                return self._index_cache
        self._index_cache = {}
        return self._index_cache

    def _save_index(self, index: Dict[str, int]) -> None:
        self._index_cache = index
        with open(self._index_path(), "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=2)

    def _load_combined(self) -> Optional[np.ndarray]:
        if self._combined_cache is not None:
            return self._combined_cache
        if os.path.exists(self._combined_path()):
            # Security: disable pickle on load
            arr = np.load(self._combined_path(), allow_pickle=False)
            self._combined_cache = arr
            return arr
        return None

    def has(self, key: str) -> bool:
        index = self._load_index()
        return key in index

    def load(self, key: str) -> np.ndarray:
        index = self._load_index()
        if key not in index:
            raise KeyError(f"Key {key} not found in embedding index")
        arr = self._load_combined()
        if arr is None:
            raise RuntimeError("Combined embeddings file not found")
        row = index[key]
        return arr[row].astype(np.float32)

    def save(self, key: str, embedding: np.ndarray) -> None:
        """Add or update a single embedding in the combined store."""
        embedding = self._ensure_vector_size(np.asarray(embedding, dtype=np.float32))
        arr = self._load_combined()
        index = self._load_index()

        if key in index:
            # update existing row
            idx = index[key]
            if arr is None:
                raise RuntimeError("Index exists but combined array missing")
            arr[idx] = embedding
        else:
            # append new row
            if arr is None:
                arr = np.expand_dims(embedding, axis=0)
            else:
                arr = np.vstack([arr, np.expand_dims(embedding, axis=0)])
            index[key] = int(arr.shape[0] - 1)

        # update in-memory caches
        self._combined_cache = arr
        self._index_cache = index
        # invalidate per-doc cache
        self._doc_embeddings = None

        # Save atomically: write to temp files then move
        tmp_arr = self._combined_path() + ".tmp"
        tmp_idx = self._index_path() + ".tmp"
        np.save(tmp_arr, arr)
        with open(tmp_idx, "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=2)
        os.replace(tmp_arr + ".npy", self._combined_path())
        os.replace(tmp_idx, self._index_path())

    def compute_embedding(self, text: str) -> np.ndarray:
        query_embedding = self.model.encode("search_query: " + text)
        arr = np.array(query_embedding, dtype=np.float32)
        return self._ensure_vector_size(arr)

    def compute_embedding_doc(self, text: str) -> np.ndarray:
        query_embedding = self.model.encode("search_document: " + text)
        arr = np.array(query_embedding, dtype=np.float32)
        return self._ensure_vector_size(arr)

    def _ensure_vector_size(self, emb: np.ndarray) -> np.ndarray:
        """Pad with zeros or truncate to self.vector_size and return float32 1-D array."""
        emb = np.asarray(emb).astype(np.float32).ravel()
        if emb.size == self.vector_size:
            return emb
        if emb.size > self.vector_size:
            return emb[: self.vector_size]
        # pad
        out = np.zeros(self.vector_size, dtype=np.float32)
        out[: emb.size] = emb
        return out

    def get_or_compute(self, key: str, text: str, force: bool = False) -> np.ndarray:
        if self.has(key) and not force:
            return self.load(key)
        emb = self.compute_embedding_doc(text)
        self.save(key, emb)
        return emb

    def build_for_documents(self, docs: Dict[int, str]) -> Dict[int, np.ndarray]:
        embeddings: Dict[int, np.ndarray] = {}
        for comic_number, text in tqdm(
            docs.items(), desc="Building/loading embeddings"
        ):
            key = str(comic_number)
            embeddings[comic_number] = self.get_or_compute(key, text)
        return embeddings

    def get_doc_embeddings(self, docs: Dict[int, str]) -> Dict[int, np.ndarray]:
        """Return a mapping comic_number -> embedding (cached in memory).

        This avoids rebuilding or reloading embeddings repeatedly.
        """
        if self._doc_embeddings is not None:
            return self._doc_embeddings

        # Try to load combined array and index and map rows to comic numbers
        arr = self._load_combined()
        index = self._load_index()
        if arr is None or not index:
            # fallback to building per-document
            self._doc_embeddings = self.build_for_documents(docs)
            return self._doc_embeddings

        mapping: Dict[int, np.ndarray] = {}
        for comic_str, row in index.items():
            try:
                comic_number = int(comic_str)
            except ValueError:
                continue
            if row < 0 or row >= arr.shape[0]:
                continue
            mapping[comic_number] = arr[row].astype(np.float32)

        # For any docs missing from index, compute and append
        missing: List[int] = [c for c in docs.keys() if c not in mapping]
        if missing:
            # compute missing embeddings and save them
            for comic_number in missing:
                text = docs[comic_number]
                emb = self.compute_embedding_doc(text)
                self.save(str(comic_number), emb)
                mapping[comic_number] = emb

        self._doc_embeddings = mapping
        return mapping

    def _maybe_migrate_individual_files(self) -> None:
        """If individual <n>.npy files exist and no combined file exists, migrate them.

        After migration, move original files into a backup folder to avoid re-migration.
        """
        combined_exists = os.path.exists(self._combined_path())
        # find numeric .npy files (e.g. 1.npy, 42.npy)
        all_files = [f for f in os.listdir(self.cache_dir) if f.endswith(".npy")]
        individual_files = [f for f in all_files if f != self.combined_filename]
        if combined_exists or not individual_files:
            return

        index: Dict[str, int] = {}
        entries: List[tuple] = []
        for fname in individual_files:
            name = fname[:-4]
            if not name.isdigit():
                continue
            try:
                comic_number = int(name)
                path = os.path.join(self.cache_dir, fname)
                # security: disallow pickle
                emb = np.load(path, allow_pickle=False)
                emb = self._ensure_vector_size(emb)
                entries.append((comic_number, emb))
            except Exception as e:
                print(f"Skipping {fname} during migration: {e}")

        if not entries:
            return

        # sort by comic number for deterministic ordering
        entries.sort(key=lambda x: x[0])
        arr = np.vstack([e[1] for e in entries]).astype(np.float32)
        for i, (comic_number, _) in enumerate(entries):
            index[str(comic_number)] = i

        # write combined and index and populate caches
        np.save(self._combined_path(), arr)
        self._save_index(index)
        self._combined_cache = arr
        self._index_cache = index

        # move originals to backup
        backup_dir = os.path.join(self.cache_dir, "backup_individual_npys")
        os.makedirs(backup_dir, exist_ok=True)
        for fname in individual_files:
            src = os.path.join(self.cache_dir, fname)
            dst = os.path.join(backup_dir, fname)
            try:
                shutil.move(src, dst)
            except Exception as e:
                print(f"Failed to move {src} to backup: {e}")


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# Module-level singleton for the embedding cache
_embedding_cache_singleton: Optional[EmbeddingCache] = None


def get_embedding_cache() -> EmbeddingCache:
    global _embedding_cache_singleton
    if _embedding_cache_singleton is None:
        _embedding_cache_singleton = EmbeddingCache()
    return _embedding_cache_singleton


# Module-level cached explanations
_explanations_cache: Optional[Dict[int, str]] = None


def get_explanations(refresh: bool = False) -> Dict[int, str]:
    global _explanations_cache
    if _explanations_cache is not None and not refresh:
        return _explanations_cache
    scraper = ExplainXKCDScraper()
    # hydrate cache if needed (will only download what's missing)
    scraper.hydrate_and_refresh_cache()
    explanations = scraper.load_cache()
    _explanations_cache = explanations or {}
    return _explanations_cache


def query_xkcd(text: str, top_k: int = 3) -> List[Tuple[int, str, float]]:
    print("Querying xkcd explanations...", text)
    explanations = get_explanations()
    if not explanations:
        return []

    emb_cache = get_embedding_cache()
    doc_embeddings = emb_cache.get_doc_embeddings(explanations)

    query_emb = emb_cache.compute_embedding(text)

    sims: List[Tuple[int, str, float]] = []
    for comic_number, doc_emb in doc_embeddings.items():
        score = cosine_similarity(query_emb, doc_emb)
        sims.append((comic_number, explanations[comic_number], score))

    sims.sort(key=lambda x: x[2], reverse=True)
    return sims[:top_k]


if __name__ == "__main__":
    query = """man pages are so confusing"""
    results = query_xkcd(query)
    for comic_number, explanation, score in results:
        print(
            f"Comic: {comic_number} Score: {score:.4f}. Explanation: {explanation[:100]}\n"
        )
