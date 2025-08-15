# 🛒 E-commerce Product Recommendation RAG System

A sophisticated **Retrieval-Augmented Generation (RAG)** system designed specifically for e-commerce product recommendations. This system combines product descriptions, user reviews, specifications, and sentiment analysis to provide personalized, intelligent product recommendations with detailed comparisons.

![System Architecture](https://img.shields.io/badge/Architecture-RAG%20System-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-green)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

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

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Pinecone API key
- Perplexity AI API key

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/ecommerce-rag-system.git
cd ecommerce-rag-system
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