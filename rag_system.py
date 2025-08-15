import json
import os
import time
from typing import List, Dict, Any
import pinecone
import requests
from sentence_transformers import SentenceTransformer

from config import (
    PREFERENCES_FILE, METRICS_FILE, RELATED_CATEGORIES,
    WEIGHT_PINECONE, WEIGHT_SENTIMENT, WEIGHT_PREF_CATEGORY,
    WEIGHT_PREF_BRAND, WEIGHT_PREF_PRICE
)

class ECommerceRAG:
    def __init__(self, pinecone_key: str, perplexity_key: str, index_name: str):
        pc = pinecone.Pinecone(api_key=pinecone_key)
        self.index = pc.Index(index_name)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.perplexity_key = perplexity_key

        
        if not os.path.exists(PREFERENCES_FILE):
            with open(PREFERENCES_FILE, "w", encoding="utf-8") as f:
                json.dump({}, f)
        if not os.path.exists(METRICS_FILE):
            with open(METRICS_FILE, "w", encoding="utf-8") as f:
                json.dump([], f)

    
    def _load_prefs(self) -> Dict[str, Any]:
        with open(PREFERENCES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_prefs(self, data: Dict[str, Any]) -> None:
        with open(PREFERENCES_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        data = self._load_prefs()
        return data.get(user_id, {
            "preferred_categories": [],
            "preferred_brands": [],
            "max_price": None
        })

    def update_preferences(self, user_id: str, product: Dict[str, Any], like: bool) -> None:
        data = self._load_prefs()
        prof = data.get(user_id, {"preferred_categories": [], "preferred_brands": [], "max_price": None})
        cat = product.get("category")
        brand = product.get("brand")
        price = product.get("price")

        if like:
            if cat and cat not in prof["preferred_categories"]:
                prof["preferred_categories"].append(cat)
            if brand and brand not in prof["preferred_brands"]:
                prof["preferred_brands"].append(brand)
            if price:
                if prof["max_price"] is None:
                    prof["max_price"] = price
                else:
                    prof["max_price"] = max(prof["max_price"], price)
        else:
            
            if brand in prof["preferred_brands"]:
                prof["preferred_brands"].remove(brand)

        data[user_id] = prof
        self._save_prefs(data)

    
    def embed_query(self, query: str) -> List[float]:
        return self.embedding_model.encode([query])[0].tolist()

    def _query_index(self, vector: List[float], top_k: int) -> List[Dict]:
        res = self.index.query(
            vector=vector,
            top_k=top_k,
            include_metadata=True,
            include_values=False
        )
        return res.get("matches", [])

    def retrieve_with_related(self, query: str, user_profile: Dict[str, Any], top_k: int = 12) -> List[Dict]:
        qvec = self.embed_query(query)
        primary = self._query_index(qvec, top_k)

        
        expanded = []
        cats = list({m["metadata"].get("category") for m in primary if m.get("metadata")})
        for c in cats:
            for rc in RELATED_CATEGORIES.get(c, []):
                expanded.append(rc)
        expanded = list(set(expanded))

        
        results = primary[:]
        if expanded:
            
            for rc in expanded[:3]:
                biased_vec = self.embed_query(f"{query}. Category: {rc}")
                results.extend(self._query_index(biased_vec, max(3, top_k // 4)))

        
        seen = set()
        uniq = []
        for m in results:
            mid = m.get("id") or f"{m['metadata'].get('product_id')}::{m['metadata'].get('text','')[:24]}"
            if mid in seen:
                continue
            seen.add(mid)
            uniq.append(m)
        return uniq

    
    def _pref_score(self, meta: Dict[str, Any], prof: Dict[str, Any]) -> float:
        score = 0.0
        if meta.get("category") in set(prof.get("preferred_categories", [])):
            score += WEIGHT_PREF_CATEGORY
        if meta.get("brand") in set(prof.get("preferred_brands", [])):
            score += WEIGHT_PREF_BRAND
        
        mp = prof.get("max_price")
        price = meta.get("price")
        if mp and price:
            
            if price <= mp:
                score += WEIGHT_PREF_PRICE
            else:
                over = max(0.0, price - mp)
                
                score -= min(WEIGHT_PREF_PRICE, over / max(1.0, mp + 1.0) * WEIGHT_PREF_PRICE)
        return score

    def rerank(self, matches: List[Dict], prof: Dict[str, Any]) -> List[Dict]:
        ranked = []
        for m in matches:
            base = float(m.get("score", 0.0)) * WEIGHT_PINECONE
            meta = m.get("metadata", {}) or {}
            sent = float(meta.get("avg_sentiment", 0.0)) * WEIGHT_SENTIMENT
            pref = self._pref_score(meta, prof)
            final = base + sent + pref
            ranked.append((final, m))
        ranked.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in ranked]

    
    def _build_structured_prompt(self, query: str, chunks: List[Dict]) -> str:
        context = "\n\n".join([
            f"{i+1}. {m.get('metadata', {}).get('name', 'Unknown Product')}: {m.get('metadata', {}).get('text', '')[:300]}"
            for i, m in enumerate(chunks[:6])
        ])
        
        prompt = f"""You are an expert e-commerce assistant. Use ONLY the provided context to answer.

    CONTEXT:
    {context}

    USER QUERY: {query}

    INSTRUCTIONS:
    1. Analyze the products in the context
    2. Provide a helpful summary
    3. Create a structured comparison of the most relevant products
    4. Return ONLY valid JSON with this exact structure:

    {{
    "summary": "Brief helpful summary of recommendations",
    "comparisons": [
        {{
        "name": "Product Name",
        "price": 1299,
        "rating": 4.5,
        "category": "Product Category",
        "brand": "Brand Name",
        "key_features": ["Feature 1", "Feature 2", "Feature 3"]
        }}
    ]
    }}

    IMPORTANT: Return ONLY the JSON object, no additional text before or after."""
        
        print(f"🔍 DEBUG: Final prompt:")
        print(prompt)
        return prompt


    def generate_response(self, query: str, context_chunks: List[Dict]) -> Dict[str, Any]:
        prompt = self._build_structured_prompt(query, context_chunks)
        
        print(f"🔍 DEBUG: Sending prompt to LLM...")
        print(f"📝 Prompt length: {len(prompt)} characters")
        
        try:
            r = requests.post(
                "https://api.perplexity.ai/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.perplexity_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "sonar",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 900,
                    "temperature": 0.1
                },
                timeout=30
            )
            
            print(f"🔍 DEBUG: LLM Response Status: {r.status_code}")
            
            if r.status_code != 200:
                print(f"❌ ERROR: LLM API failed with status {r.status_code}")
                print(f"Response: {r.text}")
                return {"summary": "Unable to generate structured response.", "comparisons": []}
            
            content = r.json()["choices"][0]["message"]["content"]
            content = content.strip()
            
            print(f"🔍 DEBUG: Raw LLM Response:")
            print(f"📄 Content length: {len(content)} characters")
            print(f"📄 First 500 chars: {content[:500]}...")
            print(f"📄 Last 200 chars: ...{content[-200:]}")
            
            # Try to find JSON in the response
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                print("❌ ERROR: No JSON found in response")
                return {"summary": content, "comparisons": []}
            
            json_content = content[json_start:json_end]
            print(f"🔍 DEBUG: Extracted JSON content:")
            print(f"📄 JSON: {json_content}")
            
            try:
                parsed_json = json.loads(json_content)
                print(f"✅ SUCCESS: JSON parsed successfully")
                print(f"📊 Parsed structure: {list(parsed_json.keys())}")
                
                if 'comparisons' in parsed_json:
                    print(f"📋 Comparisons count: {len(parsed_json['comparisons'])}")
                    for i, comp in enumerate(parsed_json['comparisons'][:3]):
                        print(f"   Product {i+1}: {comp.get('name', 'Unknown')}")
                
                return parsed_json
                
            except json.JSONDecodeError as e:
                print(f"❌ JSON DECODE ERROR: {e}")
                print(f"📄 Problematic JSON: {json_content}")
                return {"summary": content, "comparisons": []}
                
        except Exception as e:
            print(f"❌ GENERAL ERROR in generate_response: {e}")
            return {"summary": f"Error generating response: {e}", "comparisons": []}

    
    def log_metrics(self, query: str, retrieved: List[Dict], response_ms: float) -> None:
        try:
            with open(METRICS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = []
        log = {
            "query": query,
            "top_product_ids": [m.get("metadata", {}).get("product_id") for m in retrieved[:10]],
            "latency_ms": int(response_ms),
            "timestamp": int(time.time())
        }
        data.append(log)
        with open(METRICS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    
    def get_recommendations(self, query: str, user_id: str = "demo_user") -> Dict[str, Any]:
        t0 = time.time()
        profile = self.get_user_profile(user_id)
        matches = self.retrieve_with_related(query, profile, top_k=12)
        if not matches:
            self.log_metrics(query, [], (time.time() - t0) * 1000)
            return {'response': "No relevant products found.", 'products': [], 'structured': {}}

        reranked = self.rerank(matches, profile)
        gen = self.generate_response(query, reranked)

        
        unique = {}
        for m in reranked:
            meta = m.get("metadata", {}) or {}
            pid = meta.get("product_id")
            if not pid or pid in unique:
                continue
            unique[pid] = {
                "product_id": pid,
                "name": meta.get("name", "Unknown"),
                "category": meta.get("category", "Unknown"),
                "brand": meta.get("brand", "Unknown"),
                "price": meta.get("price", "N/A"),
                "rating": meta.get("rating", "N/A"),
                "relevance": float(m.get("score", 0.0)),
                "avg_sentiment": float(meta.get("avg_sentiment", 0.0))
            }
            if len(unique) >= 5:
                break

        self.log_metrics(query, reranked, (time.time() - t0) * 1000)
        
        response_text = gen.get("summary", "Generated.")
        return {
            'response': response_text,
            'products': list(unique.values()),
            'structured': gen
        }
