import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter


# 1. Open the PDF
doc = pymupdf.open("data/res.pdf")


# 2. Extract all text
text = ""

for page in doc:
    text += page.get_text()


# 3. Create the text splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50
)


# 4. Split the text
chunks = splitter.split_text(text)


# 5. Display the chunks
print("Number of chunks:", len(chunks))

for i, chunk in enumerate(chunks):
    print(f"\n--- Chunk {i + 1} ---")
    print(chunk)