import json
import os
import time
import re
from typing import List, Dict, Any, Optional
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
        self.pc = pinecone.Pinecone(api_key=pinecone_key)
        self.index = self.pc.Index(index_name)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.perplexity_key = perplexity_key
        
        
        self._ensure_files_exist()
        
        
        self.category_keywords = {
            "smartphones": ["phone", "mobile", "android", "iphone", "cellular"],
            "laptops": ["laptop", "notebook", "computer", "gaming pc", "workstation"],
            "headphones": ["headphones", "earbuds", "audio", "wireless", "noise cancel"],
            "smartwatches": ["watch", "wearable", "fitness tracker"],
            "gaming": ["gaming", "console", "playstation", "xbox", "nintendo"],
            "cameras": ["camera", "photography", "dslr", "mirrorless", "lens"]
        }

    def _ensure_files_exist(self) -> None:
        """Ensure preference and metrics files exist"""
        if not os.path.exists(PREFERENCES_FILE):
            with open(PREFERENCES_FILE, "w", encoding="utf-8") as f:
                json.dump({}, f)
        if not os.path.exists(METRICS_FILE):
            with open(METRICS_FILE, "w", encoding="utf-8") as f:
                json.dump([], f)

    def _load_prefs(self) -> Dict[str, Any]:
        """Load user preferences from file"""
        try:
            with open(PREFERENCES_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_prefs(self, data: Dict[str, Any]) -> None:
        """Save user preferences to file"""
        try:
            with open(PREFERENCES_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving preferences: {e}")

    def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """Get user profile with preferences"""
        data = self._load_prefs()
        return data.get(user_id, {
            "preferred_categories": [],
            "preferred_brands": [],
            "max_price": None,
            "min_rating": None,
            "interaction_history": []
        })

    def update_preferences(self, user_id: str, product: Dict[str, Any], like: bool) -> None:
        """Update user preferences based on interactions"""
        data = self._load_prefs()
        profile = self.get_user_profile(user_id)
        
        category = product.get("category")
        brand = product.get("brand")
        price = product.get("price")
        rating = product.get("rating")
        
        
        interaction = {
            "product_id": product.get("product_id"),
            "action": "like" if like else "dislike",
            "timestamp": int(time.time()),
            "category": category,
            "brand": brand,
            "price": price
        }
        profile.setdefault("interaction_history", []).append(interaction)
        
        
        profile["interaction_history"] = profile["interaction_history"][-50:]
        
        if like:
            
            if category and category not in profile["preferred_categories"]:
                profile["preferred_categories"].append(category)
            if brand and brand not in profile["preferred_brands"]:
                profile["preferred_brands"].append(brand)
            
            
            if isinstance(price, (int, float)):
                if profile["max_price"] is None:
                    profile["max_price"] = price
                else:
                    profile["max_price"] = max(profile["max_price"], price)
            
            if isinstance(rating, (int, float)):
                if profile["min_rating"] is None:
                    profile["min_rating"] = rating
                else:
                    profile["min_rating"] = min(profile["min_rating"], rating)
        else:
            
            if brand in profile["preferred_brands"]:
                profile["preferred_brands"].remove(brand)
        
        data[user_id] = profile
        self._save_prefs(data)

    def _detect_query_intent(self, query: str) -> Dict[str, Any]:
        """Detect user intent from query"""
        query_lower = query.lower()
        
        intent = {
            "type": "general",
            "categories": [],
            "comparison_requested": False,
            "budget_mentioned": False,
            "price_range": None,
            "specific_features": []
        }
        
        
        comparison_words = ["compare", "vs", "versus", "difference", "better", "best"]
        if any(word in query_lower for word in comparison_words):
            intent["comparison_requested"] = True
            intent["type"] = "comparison"
        
        
        for category, keywords in self.category_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                intent["categories"].append(category)
        
        
        price_patterns = [
            r'under \$?(\d+)',
            r'below \$?(\d+)', 
            r'less than \$?(\d+)',
            r'budget of \$?(\d+)',
            r'\$(\d+) budget'
        ]
        
        for pattern in price_patterns:
            match = re.search(pattern, query_lower)
            if match:
                intent["budget_mentioned"] = True
                intent["price_range"] = int(match.group(1))
                break
        
        
        feature_keywords = [
            "noise cancel", "wireless", "gaming", "professional", "camera", 
            "battery", "fast charging", "waterproof", "lightweight"
        ]
        
        for feature in feature_keywords:
            if feature in query_lower:
                intent["specific_features"].append(feature)
        
        return intent

    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for query"""
        return self.embedding_model.encode([query])[0].tolist()

    def _query_index(self, vector: List[float], top_k: int, filter_dict: Optional[Dict] = None) -> List[Dict]:
        """Query Pinecone index with optional filtering"""
        try:
            res = self.index.query(
                vector=vector,
                top_k=top_k,
                include_metadata=True,
                include_values=False,
                filter=filter_dict
            )
            return res.get("matches", [])
        except Exception as e:
            print(f"Error querying index: {e}")
            return []

    def retrieve_with_intent(self, query: str, user_profile: Dict[str, Any], intent: Dict[str, Any], top_k: int = 15) -> List[Dict]:
        """Enhanced retrieval with intent awareness"""
        query_vector = self.embed_query(query)
        
        
        primary_results = self._query_index(query_vector, top_k)
        
        
        if intent["categories"]:
            
            for category in intent["categories"]:
                enhanced_query = f"{query}. Category: {category}"
                cat_vector = self.embed_query(enhanced_query)
                cat_results = self._query_index(cat_vector, max(5, top_k // 3))
                primary_results.extend(cat_results)
        
        
        if intent["budget_mentioned"] and intent["price_range"]:
            price_filtered = []
            for result in primary_results:
                metadata = result.get("metadata", {})
                price = metadata.get("price")
                if isinstance(price, (int, float)) and price <= intent["price_range"]:
                    price_filtered.append(result)
            if price_filtered:
                primary_results = price_filtered
        
        
        expanded_results = primary_results[:]
        categories_found = set()
        for result in primary_results:
            cat = result.get("metadata", {}).get("category")
            if cat:
                categories_found.add(cat)
        
        for category in categories_found:
            related_cats = RELATED_CATEGORIES.get(category, [])
            for related_cat in related_cats[:2]:  
                related_query = f"{query}. Related category: {related_cat}"
                related_vector = self.embed_query(related_query)
                related_results = self._query_index(related_vector, max(3, top_k // 5))
                expanded_results.extend(related_results)
        
        
        seen_ids = set()
        unique_results = []
        for result in expanded_results:
            result_id = result.get("id", "")
            if result_id not in seen_ids:
                seen_ids.add(result_id)
                unique_results.append(result)
                if len(unique_results) >= top_k * 2:  
                    break
        
        return unique_results

    def _enhanced_preference_score(self, metadata: Dict[str, Any], profile: Dict[str, Any]) -> float:
        """Enhanced preference scoring with interaction history"""
        score = 0.0
        
        
        if metadata.get("category") in profile.get("preferred_categories", []):
            score += WEIGHT_PREF_CATEGORY
        
        
        if metadata.get("brand") in profile.get("preferred_brands", []):
            score += WEIGHT_PREF_BRAND
        
        
        max_price = profile.get("max_price")
        price = metadata.get("price")
        if max_price and isinstance(price, (int, float)):
            if price <= max_price:
                score += WEIGHT_PREF_PRICE
            else:
                
                overage = (price - max_price) / max_price
                score -= min(WEIGHT_PREF_PRICE, overage * WEIGHT_PREF_PRICE)
        
        
        min_rating = profile.get("min_rating")
        rating = metadata.get("rating")
        if min_rating and isinstance(rating, (int, float)):
            if rating >= min_rating:
                score += 0.1  
        
        
        interaction_history = profile.get("interaction_history", [])
        product_category = metadata.get("category")
        product_brand = metadata.get("brand")
        
        
        recent_interactions = interaction_history[-20:]  
        positive_similar = sum(1 for interaction in recent_interactions 
                             if (interaction.get("action") == "like" and 
                                 (interaction.get("category") == product_category or 
                                  interaction.get("brand") == product_brand)))
        
        if positive_similar > 0:
            score += min(0.15, positive_similar * 0.05)  
        
        return score

    def rerank_with_diversity(self, matches: List[Dict], profile: Dict[str, Any], intent: Dict[str, Any]) -> List[Dict]:
        """Enhanced reranking with diversity consideration"""
        scored_results = []
        
        for match in matches:
            base_score = float(match.get("score", 0.0)) * WEIGHT_PINECONE
            metadata = match.get("metadata", {}) or {}
            
            
            sentiment_score = float(metadata.get("avg_sentiment", 0.0)) * WEIGHT_SENTIMENT
            
            
            preference_score = self._enhanced_preference_score(metadata, profile)
            
            
            intent_bonus = 0.0
            if intent["comparison_requested"]:
                
                product_category = metadata.get("category", "").lower()
                if any(cat in product_category for cat in intent.get("categories", [])):
                    intent_bonus += 0.1
            
            
            if intent["specific_features"]:
                product_text = metadata.get("text", "").lower()
                feature_matches = sum(1 for feature in intent["specific_features"] 
                                    if feature in product_text)
                intent_bonus += feature_matches * 0.05
            
            final_score = base_score + sentiment_score + preference_score + intent_bonus
            scored_results.append((final_score, match))
        
        
        scored_results.sort(key=lambda x: x[0], reverse=True)
        
        
        final_results = []
        seen_brands = set()
        seen_categories = set()
        
        for score, result in scored_results:
            metadata = result.get("metadata", {})
            brand = metadata.get("brand", "")
            category = metadata.get("category", "")
            
            
            brand_count = sum(1 for r in final_results 
                            if r.get("metadata", {}).get("brand") == brand)
            category_count = sum(1 for r in final_results 
                               if r.get("metadata", {}).get("category") == category)
            
            
            max_same_category = 4 if category.lower() in intent.get("categories", []) else 2
            
            if brand_count < 2 and category_count < max_same_category:
                final_results.append(result)
                seen_brands.add(brand)
                seen_categories.add(category)
                
                if len(final_results) >= 15:  
                    break
        
        return final_results

    def _build_enhanced_prompt(self, query: str, chunks: List[Dict], intent: Dict[str, Any]) -> str:
        """Build enhanced prompt with intent awareness"""
        
        
        products_info = {}
        for chunk in chunks[:10]:  
            metadata = chunk.get("metadata", {})
            product_id = metadata.get("product_id")
            if not product_id:
                continue
                
            if product_id not in products_info:
                products_info[product_id] = {
                    "name": metadata.get("name", "Unknown"),
                    "category": metadata.get("category", "Unknown"),
                    "brand": metadata.get("brand", "Unknown"),
                    "price": metadata.get("price", "N/A"),
                    "rating": metadata.get("rating", "N/A"),
                    "features": [],
                    "reviews_summary": "",
                    "specifications": {}
                }
            
            
            chunk_text = metadata.get("text", "")
            if "specifications" in chunk.get("type", "").lower():
                
                specs = metadata.get("specifications", {})
                if specs:
                    products_info[product_id]["specifications"].update(specs)
            elif "reviews" in chunk.get("type", "").lower():
                products_info[product_id]["reviews_summary"] = chunk_text[:200]
        
        
        context_parts = []
        for i, (product_id, info) in enumerate(products_info.items(), 1):
            context_part = f"{i}. **{info['name']}** ({info['brand']})\n"
            context_part += f"   - Category: {info['category']}\n"
            context_part += f"   - Price: ${info['price']}\n"
            context_part += f"   - Rating: {info['rating']}/5\n"
            
            if info['specifications']:
                context_part += f"   - Key Specs: {', '.join([f'{k}: {v}' for k, v in list(info['specifications'].items())[:3]])}\n"
            
            if info['reviews_summary']:
                context_part += f"   - Customer Feedback: {info['reviews_summary'][:150]}...\n"
            
            context_parts.append(context_part)
        
        context = "\n".join(context_parts)
        
        
        if intent["comparison_requested"]:
            instruction_type = "comparison"
        elif len(products_info) > 3:
            instruction_type = "ranking"
        else:
            instruction_type = "recommendation"
        
        
        if instruction_type == "comparison":
            specific_instruction = """
    4. Create a detailed comparison focusing on:
       - Key differences in features and specifications
       - Price-to-value analysis
       - Use cases for each product
       - Clear winner recommendations for different user needs
    """
        elif instruction_type == "ranking":
            specific_instruction = """
    4. Create a ranking-style response that:
       - Ranks products by overall value and relevance
       - Explains why each product earned its position
       - Highlights the best use case for each product
       - Provides clear "best for" categories
    """
        else:
            specific_instruction = """
    4. Create personalized recommendations that:
       - Match the user's specific requirements
       - Explain why each product is recommended
       - Highlight key benefits and potential drawbacks
       - Suggest the best overall choice
    """
        
        prompt = f"""You are an expert e-commerce product advisor. Analyze the products and provide helpful recommendations based on the user's query.

CONTEXT (Available Products):
{context}

USER QUERY: {query}

INSTRUCTIONS:
1. Analyze the user's needs from their query
2. Evaluate each product's relevance to those needs
3. Consider price, features, ratings, and user feedback
{specific_instruction}
5. Provide a structured JSON response with this EXACT format:

{{
    "summary": "Brief 2-3 sentence overview of your recommendations addressing the user's specific needs",
    "comparisons": [
        {{
            "name": "Exact Product Name",
            "price": numerical_price_only,
            "rating": numerical_rating_only,
            "category": "Product Category",
            "brand": "Brand Name",
            "key_features": ["Feature 1", "Feature 2", "Feature 3"],
            "recommended_for": "Brief description of ideal user/use case",
            "pros": ["Pro 1", "Pro 2"],
            "cons": ["Con 1", "Con 2"]
        }}
    ],
    "top_pick": "Name of the best overall product for this user",
    "budget_pick": "Name of the best budget option (if applicable)"
}}

CRITICAL REQUIREMENTS:
- Use ONLY information from the provided context
- Include 3-6 products in comparisons array
- Ensure all price and rating values are numbers (not strings)
- Be specific and actionable in recommendations
- Focus on the user's actual needs, not generic features

Return ONLY the JSON object, no additional text."""
        
        return prompt

    def generate_enhanced_response(self, query: str, context_chunks: List[Dict], intent: Dict[str, Any]) -> Dict[str, Any]:
        """Generate enhanced response with better error handling"""
        prompt = self._build_enhanced_prompt(query, context_chunks, intent)
        
        try:
            response = requests.post(
                "https://api.perplexity.ai/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.perplexity_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "sonar",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 1200,
                    "temperature": 0.1
                },
                timeout=30
            )
            
            if response.status_code != 200:
                print(f"API Error: {response.status_code} - {response.text}")
                return self._generate_fallback_response(context_chunks)
            
            content = response.json()["choices"][0]["message"]["content"].strip()
            
            
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                return self._generate_fallback_response(context_chunks)
            
            json_content = content[json_start:json_end]
            
            try:
                parsed_response = json.loads(json_content)
                
                
                if not isinstance(parsed_response.get("comparisons"), list):
                    parsed_response["comparisons"] = []
                
                if not parsed_response.get("summary"):
                    parsed_response["summary"] = "Here are the product recommendations based on your query."
                
                return parsed_response
                
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {e}")
                return self._generate_fallback_response(context_chunks)
                
        except Exception as e:
            print(f"Error generating response: {e}")
            return self._generate_fallback_response(context_chunks)

    def _generate_fallback_response(self, context_chunks: List[Dict]) -> Dict[str, Any]:
        """Generate fallback response when LLM fails"""
        products = {}
        
        for chunk in context_chunks[:6]:
            metadata = chunk.get("metadata", {})
            product_id = metadata.get("product_id")
            
            if product_id and product_id not in products:
                products[product_id] = {
                    "name": metadata.get("name", "Unknown Product"),
                    "price": metadata.get("price", 0),
                    "rating": metadata.get("rating", 0),
                    "category": metadata.get("category", "Unknown"),
                    "brand": metadata.get("brand", "Unknown"),
                    "key_features": ["High quality", "Popular choice", "Good reviews"],
                    "recommended_for": "General use",
                    "pros": ["Well-rated", "Reliable brand"],
                    "cons": ["Limited information available"]
                }
        
        return {
            "summary": "Here are some relevant product recommendations based on your search.",
            "comparisons": list(products.values()),
            "top_pick": list(products.values())[0]["name"] if products else "No products found",
            "budget_pick": min(products.values(), key=lambda x: x.get("price", float('inf')))["name"] if products else "No budget option found"
        }

    def log_enhanced_metrics(self, query: str, intent: Dict[str, Any], retrieved: List[Dict], response_time: float) -> None:
        """Enhanced metrics logging"""
        try:
            with open(METRICS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = []
        
        log_entry = {
            "query": query,
            "intent": intent,
            "products_retrieved": len(retrieved),
            "top_product_ids": [r.get("metadata", {}).get("product_id") for r in retrieved[:5]],
            "categories_found": list(set(r.get("metadata", {}).get("category") for r in retrieved[:10] if r.get("metadata", {}).get("category"))),
            "latency_ms": int(response_time * 1000),
            "timestamp": int(time.time())
        }
        
        data.append(log_entry)
        
        
        if len(data) > 100:
            data = data[-100:]
        
        try:
            with open(METRICS_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving metrics: {e}")

    def get_recommendations(self, query: str, user_id: str = "demo_user") -> Dict[str, Any]:
        """Main recommendation method with enhanced features"""
        start_time = time.time()
        
        
        profile = self.get_user_profile(user_id)
        intent = self._detect_query_intent(query)
        
        
        matches = self.retrieve_with_intent(query, profile, intent, top_k=15)
        
        if not matches:
            self.log_enhanced_metrics(query, intent, [], time.time() - start_time)
            return {
                'response': "I couldn't find any relevant products for your query. Please try rephrasing or using different keywords.",
                'products': [],
                'structured': {"summary": "No products found", "comparisons": []}
            }
        
        
        reranked = self.rerank_with_diversity(matches, profile, intent)
        
        
        structured_response = self.generate_enhanced_response(query, reranked, intent)
        
        
        unique_products = {}
        for match in reranked:
            metadata = match.get("metadata", {}) or {}
            product_id = metadata.get("product_id")
            
            if product_id and product_id not in unique_products:
                unique_products[product_id] = {
                    "product_id": product_id,
                    "name": metadata.get("name", "Unknown"),
                    "category": metadata.get("category", "Unknown"),
                    "brand": metadata.get("brand", "Unknown"),
                    "price": metadata.get("price", "N/A"),
                    "rating": metadata.get("rating", "N/A"),
                    "relevance": float(match.get("score", 0.0)),
                    "avg_sentiment": float(metadata.get("avg_sentiment", 0.0))
                }
                
                if len(unique_products) >= 8:  
                    break
        
        
        self.log_enhanced_metrics(query, intent, reranked, time.time() - start_time)
        
        return {
            'response': structured_response.get("summary", "Here are my recommendations."),
            'products': list(unique_products.values()),
            'structured': structured_response
        }