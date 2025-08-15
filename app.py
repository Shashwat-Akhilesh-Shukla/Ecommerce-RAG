import streamlit as st
import os
import json
from rag_system import ECommerceRAG
from config import PREFERENCES_FILE
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

st.set_page_config(
    page_title="E-commerce Product Recommendation RAG",
    page_icon=":shopping_trolley:",
    layout="wide",
    initial_sidebar_state="expanded"
)

if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'user_id' not in st.session_state:
    st.session_state.user_id = "demo_user"

def initialize_system():
    # Priority: secrets.toml → .env
    pinecone_key = (
        st.secrets.get("PINECONE_API_KEY") if "PINECONE_API_KEY" in st.secrets else os.getenv("PINECONE_API_KEY")
    )
    perplexity_key = (
        st.secrets.get("PERPLEXITY_API_KEY") if "PERPLEXITY_API_KEY" in st.secrets else os.getenv("PERPLEXITY_API_KEY")
    )

    if not pinecone_key or not perplexity_key:
        st.error("Please set PINECONE_API_KEY and PERPLEXITY_API_KEY in .env or .streamlit/secrets.toml")
        return None

    try:
        return ECommerceRAG(
            pinecone_key=pinecone_key,
            perplexity_key=perplexity_key,
            index_name=os.getenv("PINECONE_INDEX_NAME", "ecommerce-products")
        )
    except Exception as e:
        st.error(f"Error initializing system: {e}")
        return None

def render_comparison(structured):
    print(f"🔍 DEBUG render_comparison called with: {type(structured)}")
    print(f"🔍 DEBUG structured keys: {list(structured.keys()) if structured else 'None'}")
    
    comps = structured.get("comparisons", [])
    print(f"🔍 DEBUG comparisons: {len(comps)} items")
    
    if not comps:
        st.warning("⚠️ No comparisons found in structured data")
        st.write("**Available structured data:**")
        st.json(structured)
        return
    
    st.subheader("📊 Structured Comparison")
    st.success(f"✅ Found {len(comps)} products to compare")
    
    # Build a compact table
    headers = ["Name", "Brand", "Category", "Price", "Rating", "Key Features"]
    rows = []
    
    for i, c in enumerate(comps[:5]):
        print(f"🔍 DEBUG processing product {i+1}: {c.get('name', 'Unknown')}")
        
        # Handle key features properly
        features = c.get("key_features", [])
        features_str = ""
        if isinstance(features, list):
            features_str = " • ".join(features[:3])
        else:
            features_str = str(features)
        
        rows.append([
            c.get("name", ""),
            c.get("brand", ""),
            c.get("category", ""),
            c.get("price", ""),
            c.get("rating", ""),
            features_str
        ])
    
    # Create the table
    import pandas as pd
    df = pd.DataFrame(rows, columns=headers)
    st.dataframe(df, use_container_width=True)
    
    # Also show as regular table for comparison
    st.write("**Alternative table view:**")
    st.table([headers] + rows)

def main():
    st.title("E-commerce Product Recommendation RAG")
    st.caption("Powered by Perplexity AI, Pinecone, and HuggingFace.")

    with st.sidebar:
        st.header("System")
        st.write("LLM: Perplexity Sonar small")
        st.write("Vector DB: Pinecone")
        st.write("Embeddings: all-MiniLM-L6-v2")
        if st.button("Initialize System"):
            with st.spinner("Initializing..."):
                st.session_state.rag_system = initialize_system()
                if st.session_state.rag_system:
                    st.success("Initialized")

        st.divider()
        st.header("User")
        user_id = st.text_input("User ID", value=st.session_state.user_id)
        if user_id:
            st.session_state.user_id = user_id
        if os.path.exists(PREFERENCES_FILE):
            with open(PREFERENCES_FILE, "r", encoding="utf-8") as f:
                prefs = json.load(f).get(st.session_state.user_id, {})
            st.write("Preferences")
            st.json(prefs or {"message": "No preferences yet"})

    if st.session_state.rag_system is None:
        st.warning("Initialize the system from the sidebar.")
        return

    st.header("Ask for Product Recommendations")
    examples = [
        "I need a good smartphone under 800 with excellent camera",
        "Compare the best laptops for gaming",
        "Show me wireless headphones with noise cancellation",
        "I want a premium smartwatch for Android",
        "Best cameras for weddings"
    ]
    cols = st.columns(3)
    for i, q in enumerate(examples):
        if cols[i % 3].button(q):
            st.session_state.messages.append({"role": "user", "content": q})

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.write(m["content"])

    if prompt := st.chat_input("Ask about products..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.write(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Retrieving and ranking..."):
                result = st.session_state.rag_system.get_recommendations(
                    prompt, user_id=st.session_state.user_id
                )
                
                # 🔍 DEBUG: Show what we got back
                st.write("🔍 **DEBUG INFO:**")
                st.write(f"- Response keys: {list(result.keys())}")
                st.write(f"- Response type: {type(result.get('response'))}")
                st.write(f"- Structured keys: {list(result.get('structured', {}).keys())}")
                st.write(f"- Products count: {len(result.get('products', []))}")
                
                structured = result.get("structured", {})
                if structured:
                    st.write(f"- Structured type: {type(structured)}")
                    st.write(f"- Comparisons: {len(structured.get('comparisons', []))}")
                    
                    # Show raw structured data
                    st.write("**Raw Structured Data:**")
                    st.json(structured)
                
                st.divider()
                
                # Display the actual response
                st.write("**LLM Response:**")
                st.write(result['response'])
                
                # Try to render comparison
                st.write("**Attempting to render comparison...**")
                render_comparison(result.get("structured", {}))
                
                # Show products
                products = result.get('products', [])
                if products:
                    st.subheader("Top Recommendations")
                    for p in products:
                        pid = p.get("product_id")
                        with st.expander(f"{p['name']} - Price: {p['price']} - Rating: {p['rating']}"):
                            c1, c2, c3, c4 = st.columns(4)
                            c1.metric("Relevance", f"{p['relevance']:.3f}")
                            c2.metric("Avg Sentiment", f"{p['avg_sentiment']:.3f}")
                            c3.metric("Category", p.get("category",""))
                            c4.metric("Brand", p.get("brand",""))
                            
                            lc, dc = st.columns(2)
                            if lc.button("Like", key=f"like_{pid}"):
                                st.session_state.rag_system.update_preferences(
                                    st.session_state.user_id, p, like=True
                                )
                                st.success("Preference updated")
                            
                            if dc.button("Dislike", key=f"dislike_{pid}"):
                                st.session_state.rag_system.update_preferences(
                                    st.session_state.user_id, p, like=False
                                )
                                st.info("Preference updated")
        
        st.session_state.messages.append({"role": "assistant", "content": result['response']})


if __name__ == "__main__":
    main()
