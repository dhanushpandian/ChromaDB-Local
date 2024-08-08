import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

# Load the Chroma client and collection
settings = Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="db"
)
client = chromadb.Client(settings=settings)
collection = client.get_collection("all-my-documents")

# Retrieve the documents and metadata
documents = collection.get("documents")
metadatas = collection.get("metadatas")

# Get the embeddings for the documents
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction()
embeddings = embedding_function.embed(documents)

# Use the embeddings for your specific use case
# For example, you can use the embeddings for semantic search
# Use the first document's embedding as the query
new_query_embedding = embeddings[0]
results = collection.query(
    query_embeddings=[new_query_embedding],
    n_results=3
)

# Print the results
for doc, meta, dist in zip(results.documents, results.metadatas, results.distances):
    print(f"Document: {doc}, Metadata: {meta}, Distance: {dist}")
