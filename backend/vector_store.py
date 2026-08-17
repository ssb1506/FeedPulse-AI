"""
Vector Store — FAISS-based in-memory vector store for semantic search over posts.
Used by the AI agent to find relevant posts when answering questions.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import threading


class VectorStore:
    """In-memory FAISS vector store for social media posts."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print("[VectorStore] Loading embedding model...")
        self.model = SentenceTransformer(model_name)
        self.dimension = 384  # all-MiniLM-L6-v2 output dimension

        # FAISS index
        self.index = faiss.IndexFlatL2(self.dimension)

        # Parallel list to store post metadata (FAISS only stores vectors)
        self.posts: list[dict] = []
        self._lock = threading.Lock()

        print("[VectorStore] Ready")

    def add_post(self, post: dict):
        """Embed a post and add to the index."""
        text = f"{post['username']}: {post['text']}"
        embedding = self.model.encode([text])[0].astype(np.float32)

        with self._lock:
            self.index.add(np.array([embedding]))
            self.posts.append(post)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Semantic search — find posts most similar to the query.
        Skips posts with text identical to one already in the results,
        so repeated template-generated posts don't crowd out variety."""
        if self.index.ntotal == 0:
            return []

        query_embedding = self.model.encode([query])[0].astype(np.float32)

        with self._lock:
            # Search more candidates than we need, since some will be
            # deduped away — cap at ntotal so we never ask FAISS for more
            # than exists.
            k = min(top_k * 4, self.index.ntotal)
            distances, indices = self.index.search(np.array([query_embedding]), k)

            results = []
            seen_text = set()
            for i, idx in enumerate(indices[0]):
                if idx >= len(self.posts):
                    continue
                post = self.posts[idx]
                if post["text"] in seen_text:
                    continue
                seen_text.add(post["text"])

                post = post.copy()
                post["relevance_score"] = round(1.0 / (1.0 + float(distances[0][i])), 3)
                results.append(post)

                if len(results) >= top_k:
                    break

        return results

    def get_count(self) -> int:
        """Number of posts in the vector store."""
        return self.index.ntotal
