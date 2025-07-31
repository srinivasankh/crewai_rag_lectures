import os
import faiss
from langchain.embeddings.openai import OpenAIEmbeddings
import numpy as np
import pickle

class EmbedStore:
    def __init__(self, persist_path="vector.index", meta_path="vector_meta.pkl"):
        self.persist_path = persist_path
        self.meta_path = meta_path
        self.embeddings = OpenAIEmbeddings()
        self.index = None
        self.metadata = []

    def build_index(self, texts, metadatas):
        embs = self.embeddings.embed_documents(texts)
        vecs = np.array(embs).astype("float32")
        dim = vecs.shape[1]   # Make sure to get the dimension
        self.index = faiss.IndexFlatL2(dim)   # FIXED: pass dim as argument!
        self.index.add(vecs)
        self.metadata = metadatas
        faiss.write_index(self.index, self.persist_path)
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.metadata, f)

    def load_index(self):
        if os.path.exists(self.persist_path):
            self.index = faiss.read_index(self.persist_path)
        else:
            self.index = None
        if os.path.exists(self.meta_path):
            with open(self.meta_path, "rb") as f:
                self.metadata = pickle.load(f)
        else:
            self.metadata = []

    def query(self, query: str, top_k=3):
        if self.index is None:
            raise ValueError("Index not initialized. Build or load the index before querying.")
        q_emb = np.array(self.embeddings.embed_query(query)).astype("float32").reshape(1, -1)
        D, I = self.index.search(q_emb, top_k)  # FIXED: only q_emb and top_k
        results = []
        for idx in I[0]:
            results.append(self.metadata[idx])
        return results
