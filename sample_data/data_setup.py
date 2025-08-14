# data_setup.py - Run once to populate Pinecone
import json
from data_processor import ProductDataProcessor
import pinecone

with open("products.json", "r", encoding="utf-8") as f:
    products = json.load(f)

processor = ProductDataProcessor('all-MiniLM-L6-v2')
chunks = processor.process_products(products)

pc = pinecone.Pinecone(api_key="your_key")
index = pc.Index("ecommerce-products")

vectors = []
for i, chunk in enumerate(chunks):
    vectors.append({
        "id": f"chunk_{i}",
        "values": chunk["embedding"],
        "metadata": {
            "text": chunk["text"],
            "type": chunk["type"],
            "product_id": chunk["product_id"],
            **chunk["metadata"]
        }
    })

batch_size = 100
for i in range(0, len(vectors), batch_size):
    batch = vectors[i:i + batch_size]
    index.upsert(batch)
    print(f"Uploaded batch {i//batch_size + 1}/{(len(vectors) + batch_size - 1)//batch_size}")

print(f"Successfully uploaded {len(products)} products with {len(vectors)} vector chunks to Pinecone!")
