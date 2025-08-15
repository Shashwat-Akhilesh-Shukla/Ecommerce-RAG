import json
import sys
import pinecone
from data_processor import ProductDataProcessor
from config import (
    EMBEDDING_MODEL, SENTIMENT_MODEL,
    PINECONE_API_KEY, PINECONE_INDEX_NAME
)

def upsert_products(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        products = json.load(f)
    proc = ProductDataProcessor(EMBEDDING_MODEL, sentiment_model_name=SENTIMENT_MODEL)
    chunks = proc.process_products(products)

    pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)

    vectors = []
    for i, c in enumerate(chunks):
        md = c["metadata"].copy()
        md["type"] = c.get("type")
        md["product_id"] = c.get("product_id")
        vectors.append({"id": f"update_{i}", "values": c["embedding"], "metadata": md})

    for i in range(0, len(vectors), 100):
        index.upsert(vectors[i:i+100])
        print(f"Upserted {min(i+100, len(vectors))}/{len(vectors)}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python update_index.py path_to_products.json")
        sys.exit(1)
    upsert_products(sys.argv[1])
