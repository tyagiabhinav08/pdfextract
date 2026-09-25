
import chromadb
from sentence_transformers import SentenceTransformer


# 1. Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# 2. Connect to our existing ChromaDB
client = chromadb.PersistentClient(
    path="chroma_db"
)


# 3. Get our collection
collection = client.get_collection(
    name="pdf_documents"
)

question=input("Ask a question: ")
question_embedding=model.encode(question)
results = collection.query(
    query_embeddings=[question_embedding.tolist()],
    n_results=1
)
print("\nRelevant chunks:\n")
for i, document in enumerate(results["documents"][0]):
    print(f"--- Result {i + 1} ---")
    print(document)
    print()