from sentence_transformers import SentenceTransformer
import numpy as np, os, json

def verify_model():
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print(f"Model loaded: {model}")
    print(f"Max sequence length: {model.max_seq_length}")


def embed_text(text):
    search = SemanticSearch()
    embedding = search.generate_embedding(text)

    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings():
    search = SemanticSearch()
    with open("data/movies.json") as f:
        movies_dict = json.load(f)
        documents = movies_dict["movies"]
    embeddings = search.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )

def embed_query_text(query):
    search = SemanticSearch()
    embedding = search.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")

class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def generate_embedding(self, text):
        if text is None or text.strip() == "":
            raise ValueError("Empty/whitespace string provided!")

        embedding = self.model.encode([text])[0]
        

        return embedding

    def build_embeddings(self, documents):
        self.documents = documents
        document_list = []
        for document in documents:
            self.document_map[document["id"]] = document
            document_list.append(f"{document['title']}: {document['description']}")
        self.embeddings = self.model.encode(document_list, show_progress_bar=True)

        with open("cache/movie_embeddings.npy", "wb") as embed_file:
            np.save(embed_file, self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents):
        self.documents = documents
        for document in documents:
            self.document_map[document["id"]] = document

        if os.path.exists("cache/movie_embeddings.npy"):
            with open("cache/movie_embeddings.npy", "rb") as embed_file:
                self.embeddings = np.load(embed_file)
            if len(self.embeddings) == len(documents):
                return self.embeddings
        return self.build_embeddings(documents)