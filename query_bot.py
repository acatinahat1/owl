import chromadb
from chromadb.utils import embedding_functions

# 1. Load the exact same multilingual embedding model
multilingual_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# 2. Connect to your existing local ChromaDB database
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection(
    name="bangla_subject_books",
    embedding_function=multilingual_ef
)

# 3. Type your question here (Bangla or English!)
user_question = "তোমার বইগুলোতে কি কি বিষয় আলোচনা করা হয়েছে?"

print(f"\nSearching database for: '{user_question}'...\n")

# 4. Search ChromaDB for the top 3 most relevant book chunks
results = collection.query(
    query_texts=[user_question],
    n_results=3
)

# 5. Print the retrieved book snippets
print("--- RETRIEVED SNIPPETS FROM YOUR BOOKS ---")
for index, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
    print(f"\nResult {index + 1}, Page: {meta['page']}:")
    print(f"{doc[:300]}...")  # Prints the first 300 characters of the page
print("\n-------------------------------------------")