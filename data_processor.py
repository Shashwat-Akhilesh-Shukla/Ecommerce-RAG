import pandas as pd
import re
from typing import List, Dict
from sentence_transformers import SentenceTransformer

class ProductDataProcessor:
    def __init__(self, model_name: str):
        self.embedding_model = SentenceTransformer(model_name)
        
    def smart_chunk_product(self, product: Dict) -> List[Dict]:
        """Optimized chunking strategy for e-commerce products"""
        chunks = []
        
        # Core product info chunk
        core_info = f"""
        Product: {product.get('name', '')}
        Category: {product.get('category', '')}
        Price: ${product.get('price', 'N/A')}
        Brand: {product.get('brand', '')}
        Rating: {product.get('rating', 'N/A')}/5
        """.strip()
        
        chunks.append({
            'text': core_info,
            'type': 'core_info',
            'product_id': product.get('id'),
            'metadata': {
                'name': product.get('name'),
                'category': product.get('category'),
                'price': product.get('price'),
                'rating': product.get('rating')
            }
        })
        
        # Description chunks (split by sentences)
        description = product.get('description', '')
        if description:
            desc_sentences = re.split(r'[.!?]+', description)
            desc_chunks = []
            current_chunk = ""
            
            for sentence in desc_sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                    
                if len(current_chunk + sentence) < 200:
                    current_chunk += sentence + ". "
                else:
                    if current_chunk:
                        desc_chunks.append(current_chunk.strip())
                    current_chunk = sentence + ". "
            
            if current_chunk:
                desc_chunks.append(current_chunk.strip())
            
            for i, chunk in enumerate(desc_chunks):
                chunks.append({
                    'text': f"Product: {product.get('name', '')} - {chunk}",
                    'type': 'description',
                    'product_id': product.get('id'),
                    'chunk_index': i,
                    'metadata': product
                })
        
        # Specifications chunk
        specs = product.get('specifications', {})
        if specs:
            spec_text = f"Product: {product.get('name', '')} Specifications:\n"
            spec_text += "\n".join([f"- {k}: {v}" for k, v in specs.items()])
            
            chunks.append({
                'text': spec_text,
                'type': 'specifications',
                'product_id': product.get('id'),
                'metadata': product
            })
        
        # Reviews summary chunk (aggregate top reviews)
        reviews = product.get('reviews', [])
        if reviews:
            # Take top 3 most helpful reviews
            top_reviews = sorted(reviews, key=lambda x: x.get('helpful_count', 0), reverse=True)[:3]
            review_text = f"Product: {product.get('name', '')} Customer Reviews:\n"
            
            for review in top_reviews:
                sentiment = "Positive" if review.get('rating', 3) >= 4 else "Negative" if review.get('rating', 3) <= 2 else "Neutral"
                review_text += f"- {sentiment} Review ({review.get('rating', 'N/A')}/5): {review.get('text', '')[:100]}...\n"
            
            chunks.append({
                'text': review_text,
                'type': 'reviews',
                'product_id': product.get('id'),
                'metadata': product
            })
        
        return chunks
    
    def process_products(self, products: List[Dict]) -> List[Dict]:
        """Process all products and create embeddings"""
        all_chunks = []
        
        for product in products:
            chunks = self.smart_chunk_product(product)
            all_chunks.extend(chunks)
        
        # Generate embeddings in batches
        texts = [chunk['text'] for chunk in all_chunks]
        embeddings = self.embedding_model.encode(texts, batch_size=32, show_progress_bar=True)
        
        for i, chunk in enumerate(all_chunks):
            chunk['embedding'] = embeddings[i].tolist()
        
        return all_chunks
