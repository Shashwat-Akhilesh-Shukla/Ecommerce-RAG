#streamlit interface for the app

import streamlit as st
import json
import os
from rag_system import ECommerceRAG
from data_processor import ProductDataProcessor
import pinecone

# Page config
st.set_page_config(
    page_title="E-commerce Product Recommendation RAG",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'messages' not in st.session_state:
    st.session_state.messages = []

def initialize_system():
    """Initialize the RAG system"""
    pinecone_key = st.secrets.get("PINECONE_API_KEY") or os.getenv("PINECONE_API_KEY")
    perplexity_key = st.secrets.get("PERPLEXITY_API_KEY") or os.getenv("PERPLEXITY_API_KEY")
    
    if not pinecone_key or not perplexity_key:
        st.error("Please set up your API keys in Streamlit secrets!")
        return None
    
    try:
        rag = ECommerceRAG(
            pinecone_key=pinecone_key,
            perplexity_key=perplexity_key,
            index_name="ecommerce-products"
        )
        return rag
    except Exception as e:
        st.error(f"Error initializing system: {str(e)}")
        return None

def main():
    st.title("🛒 E-commerce Product Recommendation RAG")
    st.markdown("*Powered by Perplexity AI, Pinecone, and HuggingFace Transformers*")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ System Info")
        st.write("- **LLM**: Perplexity AI")
        st.write("- **Vector DB**: Pinecone")
        st.write("- **Embeddings**: all-MiniLM-L6-v2")
        st.write("- **Cost**: $0 (Free tiers)")
        
        if st.button("🔄 Initialize System"):
            with st.spinner("Initializing RAG system..."):
                st.session_state.rag_system = initialize_system()
                if st.session_state.rag_system:
                    st.success("System initialized successfully!")
    
    # Main interface
    if st.session_state.rag_system is None:
        st.warning("Please initialize the system using the sidebar.")
        return
    
    # Chat interface
    st.header("💬 Ask for Product Recommendations")
    
    # Example queries
    st.subheader("💡 Example Queries")
    example_queries = [
        "I need a good smartphone under $500 with excellent camera",
        "Compare the best laptops for gaming",
        "Show me wireless headphones with noise cancellation",
        "I want eco-friendly home appliances",
        "Best running shoes for marathon training"
    ]
    
    cols = st.columns(3)
    for i, query in enumerate(example_queries):
        if cols[i % 3].button(f"📱 {query[:25]}...", key=f"example_{i}"):
            st.session_state.messages.append({"role": "user", "content": query})
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about products..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Finding best products for you..."):
                try:
                    result = st.session_state.rag_system.get_recommendations(prompt)
                    
                    # Display response
                    st.write(result['response'])
                    
                    # Display recommended products
                    if result['products']:
                        st.subheader("🎯 Top Recommendations")
                        
                        for i, product in enumerate(result['products'], 1):
                            with st.expander(f"{i}. {product['name']} - ${product['price']} ⭐{product['rating']}"):
                                col1, col2, col3 = st.columns(3)
                                col1.metric("Price", f"${product['price']}")
                                col2.metric("Rating", f"{product['rating']}/5")
                                col3.metric("Category", product['category'])
                    
                    st.session_state.messages.append({"role": "assistant", "content": result['response']})
                    
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

if __name__ == "__main__":
    main()
# 