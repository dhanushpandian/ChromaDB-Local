import chromadb
import numpy as np
from transformers import BertModel, BertTokenizer
import torch

# Initialize ChromaDB client
client = chromadb.Client()

# Define collection
collection_name = 'user_vectors'
if collection_name not in client.list_collections():
    collection = client.create_collection(collection_name)
else:
    collection = client.get_collection(collection_name)

# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)


def embed_text(text):
    inputs = tokenizer(text, return_tensors='pt',
                       truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()


# Insert vectors into the collection
documents = ["Document 1", "Document 2", "Document 3"]
for doc_id, text in enumerate(documents, start=1):
    vector = embed_text(text).tolist()
    collection.insert({
        'id': f'doc_{doc_id}',
        'vector': vector,
        'metadata': {'text': text}
    })

# Query the collection
query_text = "Query text"
query_vector = embed_text(query_text).tolist()
results = collection.query(query_vector, top_k=5)

# Print results
for result in results:
    print(
        f"ID: {result['id']}, Distance: {result['distance']}, Text: {result['metadata']['text']}")
