from lib.semantic_search import *
import numpy as np, json, os

class ChunkedSemanticSearch(SemanticSearch):
    def __init(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        super().__init__(self.model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        document_list = []
        chunk_list: list[str] = []
        chunk_metadata_list: list[dict] = []
        for document in documents:
                    self.document_map[document["id"]] = document
                    document_list.append(f"{document['title']}: {document['description']}")
                    if document["text"] is None:
                        continue
                    else:
                        chunks = semantic_chunk(document["description"], 4, 1)
                        chunk_counter = 0
                        for chunk in chunks:
                            chunk_list.append(chunk)
                            chunk_data = {
                                    "movie_idx": document["id"],
                                    "chunk_idx": chunk_counter,
                                    "total_chunks": len(chunks)
                            }
                            chunk_metadata_list.append(chunk_data)
                            chunk_counter += 1
        self.chunk_embeddings = self.model.encode(chunk_list)
        self.chunk_metadata = chunk_metadata_list
        with open("cache/chunk_embeddings.npy", "wb") as f:
                np.save(f, self.chunk_embeddings)
        with open("cache/chunk_metadata.json", "w") as f:
                json.dump({"chunks": self.chunk_metadata, "total_chunks": len(chunk_list)}, f, indent=2)
        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        for document in documents:
            self.document_map[document["id"]] = document
        if os.path.exists("cache/chunk_embeddings.npy") and os.path.exists("cache/chunk_metadata.json"):
            with open("cache/chunk_embeddings.npy", "rb") as f:
                self.chunk_embeddings = np.load(f)
            with open("cache/chunk_metadata.json", "r") as f:
                self.chunk_metadata = json.load(f)
            return self.chunk_embeddings
        else:
             return self.build_chunk_embeddings(documents)
    


                    