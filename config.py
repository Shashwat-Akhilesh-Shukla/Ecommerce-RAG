import os
from dotenv import load_dotenv

load_dotenv()


PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "ecommerce-products")


EMBEDDING_MODEL = "all-MiniLM-L6-v2"  
SENTIMENT_MODEL = os.getenv("SENTIMENT_MODEL", "distilbert-base-uncased-finetuned-sst-2-english")


CHUNK_SIZE = 200  
CHUNK_OVERLAP = 50  


PREFERENCES_FILE = "user_profiles.json"
METRICS_FILE = "metrics.json"
PRODUCTS_FILE = "products.json"


RELATED_CATEGORIES = {
    "Smartphones": ["Smartwatches", "Headphones", "Tablets", "Smart Home"],
    "Laptops": ["Monitors", "Headphones", "Keyboards", "Gaming"],
    "Headphones": ["Smartphones", "Laptops", "Gaming", "Tablets"],
    "Smartwatches": ["Smartphones", "Headphones"],
    "Gaming": ["Headphones", "Monitors", "Laptops"],
    "Tablets": ["Keyboards", "Headphones", "Smart Home"],
    "Smart Home": ["Smartphones", "Tablets"],
    "Cameras": ["Headphones", "Storage", "Laptops"]
}


WEIGHT_PINECONE = 1.0           
WEIGHT_SENTIMENT = 0.15         
WEIGHT_PREF_CATEGORY = 0.25     
WEIGHT_PREF_BRAND = 0.20        
WEIGHT_PREF_PRICE = 0.15        


WEIGHT_RATING_THRESHOLD = 0.10  
WEIGHT_INTERACTION_HISTORY = 0.15  
WEIGHT_INTENT_ALIGNMENT = 0.10  


MAX_CHUNKS_PER_PRODUCT = 5      
MIN_PRODUCTS_RETRIEVED = 3      
MAX_PRODUCTS_RETRIEVED = 15     
DEFAULT_TOP_K = 12              


MAX_COMPARISON_PRODUCTS = 6     
MAX_DISPLAY_PRODUCTS = 8        
DEFAULT_RESPONSE_TIMEOUT = 30   


MAX_INTERACTION_HISTORY = 50    
MAX_PREFERRED_CATEGORIES = 5    
MAX_PREFERRED_BRANDS = 8        


SLOW_QUERY_THRESHOLD_MS = 3000  
MAX_METRICS_ENTRIES = 100       


COMPARISON_KEYWORDS = ["compare", "vs", "versus", "difference", "better", "best", "which"]
BUDGET_KEYWORDS = ["cheap", "budget", "affordable", "under", "below", "less than"]
PREMIUM_KEYWORDS = ["premium", "high-end", "luxury", "professional", "best quality"]
GAMING_KEYWORDS = ["gaming", "gamer", "game", "fps", "performance"]
PROFESSIONAL_KEYWORDS = ["professional", "work", "business", "office", "productivity"]


FEATURE_PATTERNS = {
    "battery_life": ["battery", "battery life", "long lasting", "all day"],
    "camera_quality": ["camera", "photography", "photo", "video", "megapixel"],
    "performance": ["performance", "speed", "fast", "powerful", "processor"],
    "display": ["display", "screen", "resolution", "4k", "hd", "oled"],
    "audio": ["audio", "sound", "speaker", "music", "bass"],
    "connectivity": ["wifi", "bluetooth", "wireless", "5g", "connection"],
    "durability": ["durable", "waterproof", "rugged", "military grade"],
    "portability": ["portable", "lightweight", "compact", "travel"]
}


PRICE_RANGES = {
    "budget": (0, 300),
    "mid_range": (300, 800),
    "premium": (800, 1500),
    "luxury": (1500, float('inf'))
}


RATING_THRESHOLDS = {
    "excellent": 4.5,
    "good": 4.0,
    "average": 3.5,
    "below_average": 3.0
}


UI_CONFIG = {
    "example_queries": [
        ("📱 Budget Smartphones", "Best smartphone under $500 with good camera"),
        ("💻 Gaming Laptops", "Compare gaming laptops under $2000 with RTX graphics"),
        ("🎧 Noise Cancelling", "Wireless headphones with active noise cancellation"),
        ("⌚ Fitness Watches", "Smartwatch for fitness tracking and Android compatibility"),
        ("📷 Professional Camera", "Best mirrorless camera for wedding photography"),
        ("🎮 Console Gaming", "PlayStation 5 vs Xbox Series X for exclusive games")
    ],
    "metrics_to_display": ["Total Queries", "Avg Response Time", "Popular Categories"],
    "max_chat_history": 20,  
    "auto_scroll_chat": True,
    "show_debug_info": False,  
}


ERROR_MESSAGES = {
    "no_api_keys": "⚠️ Please configure PINECONE_API_KEY and PERPLEXITY_API_KEY in your environment",
    "system_init_failed": "❌ Failed to initialize RAG system. Check your API keys and connection.",
    "no_results": "🔍 No relevant products found. Try rephrasing your query with different keywords.",
    "api_timeout": "⏰ Request timed out. Please try again.",
    "general_error": "❌ An error occurred. Please try again or contact support."
}


SUCCESS_MESSAGES = {
    "system_initialized": "✅ RAG System initialized successfully!",
    "preference_updated": "✅ Your preferences have been updated!",
    "query_processed": "🎯 Found relevant products for you!"
}


DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
ENABLE_CACHING = os.getenv("ENABLE_CACHING", "true").lower() == "true"


def validate_config():
    """Validate critical configuration parameters"""
    errors = []
    
    if not PINECONE_API_KEY:
        errors.append("PINECONE_API_KEY is not set")
    
    if not PERPLEXITY_API_KEY:
        errors.append("PERPLEXITY_API_KEY is not set")
    
    if CHUNK_SIZE < 50 or CHUNK_SIZE > 500:
        errors.append("CHUNK_SIZE should be between 50 and 500")
    
    if CHUNK_OVERLAP >= CHUNK_SIZE:
        errors.append("CHUNK_OVERLAP should be less than CHUNK_SIZE")
    
    
    total_weight = (WEIGHT_SENTIMENT + WEIGHT_PREF_CATEGORY + 
                   WEIGHT_PREF_BRAND + WEIGHT_PREF_PRICE)
    if total_weight > 1.0:
        errors.append(f"Total preference weights ({total_weight}) should not exceed 1.0")
    
    return errors

if __name__ == "__main__":
    
    validation_errors = validate_config()
    if validation_errors:
        print("Configuration Errors:")
        for error in validation_errors:
            print(f"  - {error}")
    else:
        print("✅ Configuration validated successfully!")