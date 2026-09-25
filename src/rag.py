import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv
import os


# 1. Load environment variables
load_dotenv()


# 2. Create the Groq client
client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


# 3. Load the embedding model
embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# 4. Connect to ChromaDB
chroma_client = chromadb.PersistentClient(
    path="chroma_db"
)


# 5. Get our collection
collection = chroma_client.get_collection(
    name="pdf_documents"
)


# 6. Ask the user a question
question = input("Ask a question: ")


# 7. Convert the question into an embedding
question_embedding = embedding_model.encode(question)


# 8. Retrieve the most relevant chunk
results = collection.query(
    query_embeddings=[question_embedding.tolist()],
    n_results=1
)


# 9. Get the retrieved text
context = results["documents"][0][0]


# 10. Create the prompt
prompt = f"""
Answer the user's question using ONLY the context provided below.

If the answer is not present in the context, say:
"I don't know based on the provided document."

Keep the answer concise.

Context:
{context}

Question:
{question}

Answer:
"""


# 11. Send the prompt to the LLM
response = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ],
    temperature=0
)


# 12. Print the answer
answer = response.choices[0].message.content

print("\nAnswer:")
print(answer)