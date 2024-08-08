import chromadb
from chromadb.config import Settings

# Set the Chroma DB implementation to DuckDB+Parquet and specify a custom persist directory
settings = Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="db"
)

# Create the Chroma client
client = chromadb.Client(settings=settings)

# Create a collection
collection = client.create_collection("all-my-documents")

# Add documents to the collection
collection.add(
    documents=["This is document1", "This is document2"],
    metadatas=[{"source": "notion"}, {"source": "google-docs"}],
    ids=["doc1", "doc2"],
)

# Query the collection
results = collection.query(
    query_texts=["This is a query document"],
    n_results=2,
)

# Persist the client data
client.persist()

print(results)
