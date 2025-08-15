import re
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from transformers import pipeline

class ProductDataProcessor:
    def __init__(self, model_name: str, sentiment_model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
        self.embedding_model = SentenceTransformer(model_name)
        
        self.sentiment = pipeline("sentiment-analysis", model=sentiment_model_name, truncation=True)

    @staticmethod
    def _sentiment_to_score(label: str, score: float) -> float:
        
        if label.upper().startswith("POS"):
            return float(score)
        if label.upper().startswith("NEG"):
            return float(-score)
        return 0.0

    def smart_chunk_product(self, product: Dict) -> List[Dict]:
        chunks = []

        core_info = f"""Product: {product.get('name', '')}
            Category: {product.get('category', '')}
            Price: ${product.get('price', 'N/A')}
            Brand: {product.get('brand', '')}
            Rating: {product.get('rating', 'N/A')}/5""".strip()

        base_meta = {
            'name': product.get('name'),
            'category': product.get('category'),
            'price': product.get('price'),
            'rating': product.get('rating'),
            'brand': product.get('brand'),
            'product_id': product.get('id')
        }

        chunks.append({
            'text': core_info,
            'type': 'core_info',
            'product_id': product.get('id'),
            'metadata': {**base_meta}
        })

        description = product.get('description', '')
        if description:
            desc_sentences = re.split(r'[.!?]+', description)
            desc_chunks, current = [], ""
            for s in desc_sentences:
                s = s.strip()
                if not s:
                    continue
                if len(current) + len(s) + 2 < 200:
                    current += s + ". "
                else:
                    if current:
                        desc_chunks.append(current.strip())
                    current = s + ". "
            if current:
                desc_chunks.append(current.strip())

            for i, ch in enumerate(desc_chunks):
                chunks.append({
                    'text': f"Product: {product.get('name', '')} - {ch}",
                    'type': 'description',
                    'product_id': product.get('id'),
                    'chunk_index': i,
                    'metadata': {**base_meta}
                })

        specs = product.get('specifications', {})
        if specs:
            spec_text = f"Product: {product.get('name', '')} Specifications:\n" + \
                        "\n".join([f"- {k}: {v}" for k, v in specs.items()])
            chunks.append({
                'text': spec_text,
                'type': 'specifications',
                'product_id': product.get('id'),
                'metadata': {**base_meta, 'specifications': specs}
            })

        
        reviews = product.get('reviews', [])
        avg_sent = 0.0
        if reviews:
            top = sorted(reviews, key=lambda x: x.get('helpful_count', 0), reverse=True)[:5]
            sentiments = self.sentiment([r.get('text', '')[:300] for r in top])
            scores = [self._sentiment_to_score(s['label'], float(s['score'])) for s in sentiments]
            avg_sent = sum(scores) / max(1, len(scores))

            txt = f"Product: {product.get('name', '')} Customer Reviews (avg_sentiment={avg_sent:.3f}):\n"
            for r, s in zip(top, scores):
                sign = "Positive" if s > 0.2 else "Negative" if s < -0.2 else "Neutral"
                txt += f"- {sign} ({r.get('rating','N/A')}/5): {r.get('text','')[:120]}...\n"

            chunks.append({
                'text': txt,
                'type': 'reviews',
                'product_id': product.get('id'),
                'metadata': {**base_meta, 'avg_sentiment': avg_sent}
            })

        
        for c in chunks:
            c['metadata'].setdefault('avg_sentiment', avg_sent)

        return chunks

    def process_products(self, products: List[Dict]) -> List[Dict]:
        all_chunks = []
        for p in products:
            all_chunks.extend(self.smart_chunk_product(p))

        texts = [c['text'] for c in all_chunks]
        embeddings = self.embedding_model.encode(texts, batch_size=32, show_progress_bar=True)
        for i, c in enumerate(all_chunks):
            c['embedding'] = embeddings[i].tolist()
        return all_chunks
