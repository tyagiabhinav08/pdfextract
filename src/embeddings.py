import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


# 1. Open the PDF
doc = pymupdf.open("data/sample.pdf")


# 2. Extract the text
text = ""

for page in doc:
    text += page.get_text()


# 3. Split the text into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_text(text)


# 4. Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# 5. Convert chunks into embeddings
embeddings = model.encode(chunks)


# 6. Display the results
print("Number of chunks:", len(chunks))
print("Number of embeddings:", len(embeddings))
print("Embedding size:", len(embeddings[0]))

print("\nFirst chunk:")
print(chunks[0])

print("\nFirst embedding:")
print(embeddings[0])