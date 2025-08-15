import os

from dotenv import load_dotenv
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
PINECONE_INDEX_NAME = "ecommerce-products"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 200
CHUNK_OVERLAP = 50


SENTIMENT_MODEL = os.getenv("SENTIMENT_MODEL", "distilbert-base-uncased-finetuned-sst-2-english")

PREFERENCES_FILE = "user_profiles.json"
METRICS_FILE = "metrics.json"

RELATED_CATEGORIES = {
    "Smartphones": ["Smartwatches", "Headphones"],
    "Laptops": ["Monitors", "Headphones"],
    "Headphones": ["Smartphones", "Laptops"],
    "Smartwatches": ["Smartphones"],
    "Gaming": ["Headphones", "Monitors"],
    "Tablets": ["Keyboards", "Headphones"],
    "Smart Home": ["Smartphones"],
    "Cameras": ["Headphones", "Storage"]
}

WEIGHT_PINECONE = 1.0      
WEIGHT_SENTIMENT = 0.15
WEIGHT_PREF_CATEGORY = 0.2
WEIGHT_PREF_BRAND = 0.2
WEIGHT_PREF_PRICE = 0.15
