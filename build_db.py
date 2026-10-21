import json
import os
import chromadb
from chromadb.utils import embedding_functions

# 1. Initialize local ChromaDB client
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# 2. Set up your multilingual embedding model
multilingual_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# 3. Create a fresh collection for all your school books
collection_name = "bangla_subject_books"
collection = chroma_client.get_or_create_collection(
    name=collection_name,
    embedding_function=multilingual_ef
)

# 4. List all your pre-processed subject files here
# (Make sure these .jsonl or .txt files are in your RAG folder)
subject_files = [
    {"file": "bangla.jsonl", "subject": "Bangla"},
    {"file": "english.jsonl", "subject": "English"},
    {"file": "math.jsonl", "subject": "Math"}
]

total_added = 0

for item in subject_files:
    file_path = item["file"]
    subject_name = item["subject"]
    
    if os.path.exists(file_path):
        print(f"Reading chunks for {subject_name} from {file_path}...")
        with open(file_path, "r", encoding="utf-8") as f:
            documents = []
            metadatas = []
            ids = []
            
            for index, line in enumerate(f):
                if line.strip():
                    record = json.loads(line)
                    
                    # Extract fields from ChatGPT's JSONL format
                    text_content = record.get("text") or record.get("content") or str(record)
                    page_num = record.get("page") or record.get("page_number", index + 1)
                    chapter = record.get("chapter", "Unknown")
                    
                    doc_id = f"{subject_name.lower()}_page_{page_num}_idx_{index}"
                    
                    documents.append(text_content)
                    metadatas.append({
                        "subject": subject_name,
                        "page": str(page_num),
                        "chapter": str(chapter)
                    })
                    ids.append(doc_id)
            
            # Add this subject's chunks to ChromaDB
            if documents:
                collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                total_added += len(documents)
                print(f"✅ Added {len(documents)} chunks for {subject_name}.")
    else:
        print(f"⚠️ Warning: File '{file_path}' not found. Skipping {subject_name}.")

print(f"\n🎉 Done! Total chunks across all subjects loaded into ChromaDB: {total_added}")