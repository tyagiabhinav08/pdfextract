import pymupdf
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


# 1. Open the PDF
doc = pymudpdf.open("data/res.pdf")


# 2. Extract text
text = ""

for page in doc:
    text += page.get_text()


# 3. Split text into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50
)

chunks = splitter.split_text(text)


# 4. Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# 5. Create embeddings
embeddings = model.encode(chunks)


# 6. Create ChromaDB client
client = chromadb.PersistentClient(
    path="chroma_db"
)


# 7. Create a collection
collection = client.get_or_create_collection(
    name="pdf_documents"
)


# 8. Store chunks and embeddings
collection.add(
    ids=[str(i) for i in range(len(chunks))],
    documents=chunks,
    embeddings=embeddings.tolist()
)


# 9. Check how many documents are stored
print("Documents stored:", collection.count())