import json
import pinecone
from data_processor import ProductDataProcessor
from config import EMBEDDING_MODEL, SENTIMENT_MODEL, PINECONE_API_KEY, PINECONE_INDEX_NAME


# Helper to clean metadata for Pinecone
def clean_metadata(md):
    cleaned = {}
    for k, v in md.items():
        if isinstance(v, (str, int, float, bool)):
            cleaned[k] = v
        elif isinstance(v, list) and all(isinstance(i, str) for i in v):
            cleaned[k] = v
        else:
            cleaned[k] = json.dumps(v, ensure_ascii=False)
    return cleaned


# Load products
with open("products.json", "r", encoding="utf-8") as f:
    products = json.load(f)

# Process products into chunks
processor = ProductDataProcessor(EMBEDDING_MODEL, sentiment_model_name=SENTIMENT_MODEL)
chunks = processor.process_products(products)

# Init Pinecone
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# Fetch existing IDs from Pinecone
print("Fetching existing IDs from Pinecone...")
existing_ids = set()

# The list() method returns a generator that yields lists of IDs
for ids_batch in index.list(namespace=""):
    existing_ids.update(ids_batch)  # Add all IDs from the batch to the set

print(f"Found {len(existing_ids)} existing vectors. Skipping duplicates.")


print(f"Found {len(existing_ids)} existing vectors. Skipping duplicates.")

# Prepare vectors for upload
vectors = []
for i, chunk in enumerate(chunks):
    vec_id = f"chunk_{i}"
    if vec_id in existing_ids:
        continue
    md = chunk["metadata"].copy()
    md["type"] = chunk.get("type")
    md["product_id"] = chunk.get("product_id")
    md = clean_metadata(md)
    vectors.append({
        "id": vec_id,
        "values": chunk["embedding"],
        "metadata": md
    })

# Upload in batches
batch_size = 100
total_uploaded = 0
for i in range(0, len(vectors), batch_size):
    batch = vectors[i:i + batch_size]
    index.upsert(batch)
    total_uploaded += len(batch)
    print(f"Uploaded {total_uploaded}/{len(vectors)} new vectors")

print(f"Done. Products: {len(products)}  Chunks processed: {len(chunks)}  New uploaded: {total_uploaded}")
