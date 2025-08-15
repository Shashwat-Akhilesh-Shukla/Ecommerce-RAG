import streamlit as st
import os
import json
from rag_system import ECommerceRAG
from config import PREFERENCES_FILE, METRICS_FILE
from dotenv import load_dotenv
import time
import plotly.express as px
import pandas as pd

load_dotenv()

st.set_page_config(
    page_title="🛒 E-commerce Product Recommendation RAG",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
    .stButton button {
        width: 100%;
        height: 3rem;
        border-radius: 10px;
        border: 1px solid 
        background: linear-gradient(90deg, 
        color: white;
        font-weight: 500;
        margin: 5px 0;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    .comparison-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 10px 0;
        border-left: 4px solid 
    }
    .metric-card {
        background: linear-gradient(135deg, 
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        margin: 5px;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'user_id' not in st.session_state:
        st.session_state.user_id = "demo_user"
    if 'system_initialized' not in st.session_state:
        st.session_state.system_initialized = False
    if 'current_query' not in st.session_state:
        st.session_state.current_query = ""

def auto_initialize_system():
    """Auto-initialize the system on first load"""
    if not st.session_state.system_initialized:
        
        pinecone_key = (
            st.secrets.get("PINECONE_API_KEY") if "PINECONE_API_KEY" in st.secrets 
            else os.getenv("PINECONE_API_KEY")
        )
        perplexity_key = (
            st.secrets.get("PERPLEXITY_API_KEY") if "PERPLEXITY_API_KEY" in st.secrets 
            else os.getenv("PERPLEXITY_API_KEY")
        )

        if pinecone_key and perplexity_key:
            try:
                with st.spinner("🚀 Auto-initializing RAG system..."):
                    st.session_state.rag_system = ECommerceRAG(
                        pinecone_key=pinecone_key,
                        perplexity_key=perplexity_key,
                        index_name=os.getenv("PINECONE_INDEX_NAME", "ecommerce-products")
                    )
                st.session_state.system_initialized = True
                st.success("✅ RAG System initialized successfully!")
                return True
            except Exception as e:
                st.error(f"❌ Error initializing system: {e}")
                return False
        else:
            st.error("⚠️ Please set PINECONE_API_KEY and PERPLEXITY_API_KEY in .env or .streamlit/secrets.toml")
            return False
    return True

def render_enhanced_comparison(structured):
    """Enhanced comparison rendering with better UI"""
    comparisons = structured.get("comparisons", [])
    
    if not comparisons:
        st.warning("⚠️ No structured comparisons available")
        return
    
    st.subheader("📊 Product Comparison")
    
    
    tab1, tab2, tab3 = st.tabs(["📋 Table View", "📈 Price Analysis", "⭐ Rating Analysis"])
    
    with tab1:
        
        df_data = []
        for comp in comparisons:
            features_str = " • ".join(comp.get("key_features", [])[:3])
            df_data.append({
                "🏷️ Name": comp.get("name", ""),
                "🏢 Brand": comp.get("brand", ""),
                "📁 Category": comp.get("category", ""),
                "💰 Price": f"${comp.get('price', 'N/A')}",
                "⭐ Rating": f"{comp.get('rating', 'N/A')}/5",
                "🔥 Key Features": features_str
            })
        
        df = pd.DataFrame(df_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
    
    with tab2:
        
        if len(comparisons) > 1:
            price_data = [
                {"Product": comp.get("name", "")[:20], "Price": comp.get("price", 0)}
                for comp in comparisons if isinstance(comp.get("price"), (int, float))
            ]
            if price_data:
                fig = px.bar(price_data, x="Product", y="Price", 
                           title="💰 Price Comparison",
                           color="Price", color_continuous_scale="viridis")
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        
        if len(comparisons) > 1:
            rating_data = [
                {"Product": comp.get("name", "")[:20], "Rating": comp.get("rating", 0)}
                for comp in comparisons if isinstance(comp.get("rating"), (int, float))
            ]
            if rating_data:
                fig = px.scatter(rating_data, x="Product", y="Rating",
                               title="⭐ Rating Comparison", size="Rating",
                               color="Rating", color_continuous_scale="RdYlGn")
                fig.update_layout(xaxis_tickangle=-45, yaxis=dict(range=[0, 5]))
                st.plotly_chart(fig, use_container_width=True)

def handle_example_query(query):
    """Handle example query button clicks"""
    st.session_state.current_query = query
    
    process_user_query(query)

def process_user_query(query):
    """Process user query and display results"""
    st.session_state.messages.append({"role": "user", "content": query})
    
    with st.chat_message("user"):
        st.write(query)
    
    with st.chat_message("assistant"):
        start_time = time.time()
        
        with st.spinner("🔍 Retrieving and analyzing products..."):
            result = st.session_state.rag_system.get_recommendations(
                query, user_id=st.session_state.user_id
            )
            
        processing_time = time.time() - start_time
        
        
        st.write("**💬 AI Response:**")
        st.write(result['response'])
        
        
        structured = result.get("structured", {})
        if structured and structured.get("comparisons"):
            render_enhanced_comparison(structured)
        
        
        products = result.get('products', [])
        if products:
            st.subheader("🎯 Top Recommendations")
            
            cols = st.columns(min(len(products), 3))
            for i, product in enumerate(products[:6]):  
                col_idx = i % 3
                with cols[col_idx]:
                    with st.container():
                        st.markdown(f"""
                        <div class="comparison-card">
                            <h4>🏷️ {product['name']}</h4>
                            <p><strong>💰 Price:</strong> ${product['price']}</p>
                            <p><strong>⭐ Rating:</strong> {product['rating']}/5</p>
                            <p><strong>🏢 Brand:</strong> {product['brand']}</p>
                            <p><strong>📁 Category:</strong> {product['category']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("👍 Like", key=f"like_{product['product_id']}_{i}", 
                                       help="Add to preferences"):
                                st.session_state.rag_system.update_preferences(
                                    st.session_state.user_id, product, like=True
                                )
                                st.success("✅ Added to preferences!")
                        with col2:
                            if st.button("👎 Dislike", key=f"dislike_{product['product_id']}_{i}",
                                       help="Remove from preferences"):
                                st.session_state.rag_system.update_preferences(
                                    st.session_state.user_id, product, like=False
                                )
                                st.info("ℹ️ Preference updated!")
        
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>⚡ Response Time</h3>
                <h2>{processing_time:.2f}s</h2>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>🎯 Products Found</h3>
                <h2>{len(products)}</h2>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            comparisons_count = len(structured.get("comparisons", []))
            st.markdown(f"""
            <div class="metric-card">
                <h3>📊 Comparisons</h3>
                <h2>{comparisons_count}</h2>
            </div>
            """, unsafe_allow_html=True)
    
    st.session_state.messages.append({"role": "assistant", "content": result['response']})

def render_sidebar():
    """Enhanced sidebar with better organization"""
    with st.sidebar:
        st.header("🔧 System Status")
        
        if st.session_state.system_initialized:
            st.success("✅ RAG System Active")
            st.info("🔗 Vector DB: Pinecone")
            st.info("🤖 LLM: Perplexity Sonar")
            st.info("🧠 Embeddings: MiniLM-L6-v2")
        else:
            st.error("❌ System Not Initialized")
            if st.button("🔄 Retry Initialization"):
                auto_initialize_system()
        
        st.divider()
        
        
        st.header("👤 User Profile")
        user_id = st.text_input("User ID", value=st.session_state.user_id, 
                               help="Change to switch user profiles")
        if user_id != st.session_state.user_id:
            st.session_state.user_id = user_id
            st.rerun()
        
        
        if os.path.exists(PREFERENCES_FILE):
            with open(PREFERENCES_FILE, "r", encoding="utf-8") as f:
                prefs = json.load(f).get(st.session_state.user_id, {})
            
            if prefs:
                st.write("**🎯 Your Preferences:**")
                if prefs.get("preferred_categories"):
                    st.write("📁 Categories:", ", ".join(prefs["preferred_categories"]))
                if prefs.get("preferred_brands"):
                    st.write("🏢 Brands:", ", ".join(prefs["preferred_brands"]))
                if prefs.get("max_price"):
                    st.write(f"💰 Max Price: ${prefs['max_price']}")
            else:
                st.info("💡 Like products to build your preference profile!")
        
        st.divider()
        
        
        st.header("📈 Analytics")
        if os.path.exists(METRICS_FILE):
            try:
                with open(METRICS_FILE, "r", encoding="utf-8") as f:
                    metrics = json.load(f)
                
                if metrics:
                    st.metric("📊 Total Queries", len(metrics))
                    
                    
                    recent_queries = [m.get("query", "")[:30] + "..." 
                                    for m in metrics[-5:]]
                    if recent_queries:
                        st.write("**🔍 Recent Queries:**")
                        for query in reversed(recent_queries):
                            st.text(f"• {query}")
                else:
                    st.info("No analytics data yet")
            except Exception:
                st.error("Error loading analytics")
        
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

def main():
    init_session_state()
    
    
    st.title("🛒 E-commerce Product Recommendation RAG")
    st.caption("🚀 Powered by Perplexity AI, Pinecone, and HuggingFace • Personalized product discovery made simple")
    
    
    if not auto_initialize_system():
        st.stop()
    
    
    render_sidebar()
    
    
    st.header("🔍 Find Your Perfect Product")
    
    
    st.subheader("💡 Try These Popular Queries")
    
    example_queries = [
        ("📱 Smartphones", "I need a good smartphone under $800 with excellent camera"),
        ("💻 Gaming Laptops", "Compare the best laptops for gaming under $2000"),
        ("🎧 Audio", "Show me wireless headphones with noise cancellation"),
        ("⌚ Wearables", "I want a premium smartwatch for Android users"),
        ("📷 Photography", "Best cameras for professional wedding photography"),
        ("🎮 Gaming", "Compare PlayStation 5 vs Xbox Series X for gaming")
    ]
    
    cols = st.columns(3)
    for i, (category, query) in enumerate(example_queries):
        with cols[i % 3]:
            if st.button(f"{category}", key=f"example_{i}", help=query):
                handle_example_query(query)
    
    st.divider()
    
    
    st.subheader("💬 Chat Interface")
    
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    
    if prompt := st.chat_input("Ask about products... (e.g., 'Best budget laptops for students')"):
        process_user_query(prompt)

if __name__ == "__main__":
    main()