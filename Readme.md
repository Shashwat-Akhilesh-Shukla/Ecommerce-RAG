# 🛒 E-commerce Product Recommendation RAG System

A sophisticated **Retrieval-Augmented Generation (RAG)** system designed specifically for e-commerce product recommendations. This system combines product descriptions, user reviews, specifications, and sentiment analysis to provide personalized, intelligent product recommendations with detailed comparisons.

<img width="1916" height="964" alt="Screenshot 2025-08-15 221105" src="https://github.com/user-attachments/assets/e06112bf-283c-4157-8931-8f6712e51d72" />
<img width="1919" height="965" alt="Screenshot 2025-08-15 221049" src="https://github.com/user-attachments/assets/02982b60-0b27-478d-9973-cc6772b6944d" />
<img width="1916" height="979" alt="Screenshot 2025-08-15 221128" src="https://github.com/user-attachments/assets/a0d7d026-6c90-4838-8a08-fffec1de55e0" />




## 🎯 Key Features

### ✨ Core Capabilities
- **Multi-source Data Integration**: Combines product descriptions, reviews, specifications, and ratings
- **Personalized Recommendations**: Learns from user interactions and builds preference profiles
- **Intelligent Comparison**: Provides detailed side-by-side product comparisons
- **Sentiment-Aware**: Integrates review sentiment analysis for better recommendations
- **Real-time Adaptation**: Updates recommendations based on user feedback

### 🧠 Advanced AI Features
- **Intent Detection**: Understands query context (comparison, budget, specific features)
- **Cross-category Recommendations**: Suggests related products across categories
- **Diversity Optimization**: Ensures variety in brand and category representation
- **Context-Aware Generation**: Provides tailored responses based on user needs

### 🎨 User Experience
- **Interactive Web Interface**: Clean, modern Streamlit-based UI
- **One-click Example Queries**: Predefined queries for quick exploration
- **Visual Analytics**: Charts and metrics for price and rating comparisons
- **Preference Learning**: Like/dislike system builds user profiles

## 🏗️ System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit UI  │    │  RAG System     │    │  Vector Store   │
│                 │────│                 │────│   (Pinecone)    │
│ • Chat Interface│    │ • Query Processing    │ • Product Chunks │
│ • Visualizations│    │ • Retrieval     │    │ • Embeddings    │
│ • User Profiles │    │ • Reranking     │    │ • Metadata      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ User Preferences│    │  LLM Response   │    │ Product Database│
│                 │    │   (Perplexity)  │    │                 │
│ • Categories    │    │                 │    │ • Descriptions  │
│ • Brands        │    │ • Structured    │    │ • Specifications│
│ • Price Range   │    │   Comparisons   │    │ • Reviews       │
│ • History       │    │ • Recommendations    │ • Ratings       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🔧 Technical Implementation Overview

This section explains the key technical choices made to implement the features, tailored for rookie AI engineers:

### **Data Processing & Chunking Strategy**

**Why Smart Chunking?** Product data is heterogeneous - descriptions, specs, and reviews have different structures. We implemented smart chunking in `data_processor.py` that creates separate chunks for:
- Core product info (name, price, rating, brand)
- Description text (split by sentences, max 200 chars)
- Specifications (structured key-value pairs)
- Customer reviews (with sentiment analysis)

**Technical Choice:** This approach ensures each chunk type can be retrieved independently, providing targeted context to the LLM.

### **Embedding & Vector Store**

**Why Sentence Transformers?** We use `all-MiniLM-L6-v2` because it's:
- Lightweight (22MB model)
- Fast inference
- Good semantic understanding for product descriptions
- Pre-trained on diverse text types

**Why Pinecone?** Managed vector database that handles:
- Automatic scaling
- Fast similarity search (sub-100ms)
- Metadata filtering
- No infrastructure management needed

### **Intent Detection & Query Enhancement**

**Simple but Effective Approach:** We use regex patterns and keyword matching to detect:
```python
# Price intent: "under $500", "budget of $1000"
price_patterns = [r'under \$?(\d+)', r'budget of \$?(\d+)']

# Comparison intent: "compare", "vs", "better"
comparison_words = ["compare", "vs", "versus", "difference"]
```

**Why Not ML-based Intent?** For an MVP, rule-based intent detection is:
- Faster to implement
- More interpretable
- Easier to debug
- Sufficient for common e-commerce queries

### **Related Categories Mapping**

**Strategic Design Choice:** Hardcoded mappings like:
```python
RELATED_CATEGORIES = {
    "Smartphones": ["Smartwatches", "Headphones"],
    "Laptops": ["Gaming", "Monitors", "Headphones"]
}
```

**Why This Works:** 
- Captures domain expertise about product ecosystems
- Expands retrieval when primary category has few results
- Mimics real shopping behavior (phone → accessories)

### **Sentiment-Enhanced Ranking**

**Multi-factor Scoring:** We combine multiple signals:
```python
final_score = (
    base_similarity_score * 1.0 +           # Vector similarity
    sentiment_score * 0.15 +                # Review sentiment
    preference_score * 0.25 +               # User preferences
    intent_bonus * 0.1                      # Query intent match
)
```

**Why Weighted Scoring?** Different signals have different reliability - vector similarity is most trustworthy, so it gets the highest weight.

### **Prompt Engineering for Structured Output**

**Critical Design:** We enforce JSON schema in prompts:
```python
prompt = f"""Return ONLY valid JSON with this EXACT format:
{{
  "summary": "Brief overview",
  "comparisons": [
    {{
      "name": "Product Name",
      "price": numerical_value,
      "rating": numerical_value,
      "key_features": ["Feature 1", "Feature 2"]
    }}
  ]
}}"""
```

**Why This Matters:** Structured output enables rich UI rendering - tables, charts, and interactive elements.

### **User Personalization Architecture**

**Profile-based Learning:** We track user interactions:
```python
profile = {
    "preferred_categories": ["Smartphones", "Laptops"],
    "preferred_brands": ["Apple", "Google"],
    "max_price": 1500,
    "interaction_history": [...]
}
```

**Why JSON Files?** For an MVP, file-based storage is:
- Simple to implement
- Version controllable
- No database setup required
- Sufficient for hundreds of users

### **UI/UX Technical Choices**

**Why Streamlit?** 
- Rapid prototyping (UI in hours, not days)
- Built-in components (chat, charts, buttons)
- Auto-refresh on code changes
- Perfect for ML demos and MVPs

**Interactive Visualizations:** We use Plotly for:
- Price comparison bar charts
- Rating scatter plots
- Responsive, interactive charts

## 🚀 Quick Start

### Prerequisites
- Python 3.12 recommended
- Pinecone API key
- Perplexity AI API key

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Shashwat-Akhilesh-Shukla/Ecommerce-RAG.git
cd ecommerce-rag
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
Create a `.env` file in the root directory:
```env
PINECONE_API_KEY=your_pinecone_api_key_here
PERPLEXITY_API_KEY=your_perplexity_api_key_here
PINECONE_INDEX_NAME=ecommerce-products
DEBUG_MODE=false
```

4. **Initialize the vector database**
```bash
python data_setup.py
```

5. **Run the application**
```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`
