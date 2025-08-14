import pinecone
import requests
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import json

class ECommerceRAG:
    def __init__(self, pinecone_key: str, perplexity_key: str, index_name: str):
        # Initialize Pinecone
        pc = pinecone.Pinecone(api_key=pinecone_key)
        self.index = pc.Index(index_name)
        
        # Initialize models
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.perplexity_key = perplexity_key
        
    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for user query"""
        return self.embedding_model.encode([query])[0].tolist()
    
    def retrieve_relevant_products(self, query: str, top_k: int = 10) -> List[Dict]:
        """Retrieve relevant product chunks from Pinecone"""
        query_embedding = self.embed_query(query)
        
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            include_values=False
        )
        
        return results['matches']
    
    def generate_response(self, query: str, context_chunks: List[Dict]) -> str:
        """Generate response using Perplexity API"""
        
        # Prepare context
        context = "\n\n".join([
            f"Product Info: {match['metadata'].get('text', '')}"
            for match in context_chunks[:5]  # Limit context for API efficiency
        ])
        
        prompt = f"""You are an expert e-commerce product recommendation assistant. Based on the following product information, provide helpful recommendations and detailed comparisons.

Product Context:
{context}

User Query: {query}

Please provide:
1. Direct answer to the user's question
2. Specific product recommendations with reasons
3. Key features comparison if multiple products
4. Price and rating information when available

Keep your response helpful, concise, and focused on the user's needs."""

        try:
            response = requests.post(
                'https://api.perplexity.ai/chat/completions',
                headers={
                    'Authorization': f'Bearer {self.perplexity_key}',
                    'Content-Type': 'application/json'
                },
                json={
                    'model': 'llama-3.1-sonar-small-128k-online',
                    'messages': [{'role': 'user', 'content': prompt}],
                    'max_tokens': 1000,
                    'temperature': 0.1
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content']
            else:
                return f"Sorry, I couldn't generate a response. Please try again."
                
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def get_recommendations(self, query: str) -> Dict:
        """Main method to get recommendations"""
        
        # Retrieve relevant chunks
        relevant_chunks = self.retrieve_relevant_products(query)
        
        if not relevant_chunks:
            return {
                'response': "Sorry, I couldn't find any relevant products for your query.",
                'products': []
            }
        
        # Generate response
        response = self.generate_response(query, relevant_chunks)
        
        # Extract unique products from chunks
        unique_products = {}
        for chunk in relevant_chunks[:10]:
            metadata = chunk.get('metadata', {})
            product_id = metadata.get('product_id')
            if product_id and product_id not in unique_products:
                unique_products[product_id] = {
                    'name': metadata.get('name', 'Unknown'),
                    'category': metadata.get('category', 'Unknown'),
                    'price': metadata.get('price', 'N/A'),
                    'rating': metadata.get('rating', 'N/A'),
                    'score': chunk.get('score', 0)
                }
        
        return {
            'response': response,
            'products': list(unique_products.values())[:5]
        }
