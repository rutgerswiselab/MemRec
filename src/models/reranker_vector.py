"""
Vector-based Reranker for Stage-ReRank
Uses Sentence Transformer for semantic vector scoring (no LLM calls)
"""
import numpy as np
from typing import Dict, List
from collections import Counter


class VectorReranker:
    """Use Sentence Transformer vector similarity to rerank candidate items"""
    
    def __init__(self):
        """Initialize Vector Reranker with Sentence Transformer"""
        try:
            from sentence_transformers import SentenceTransformer
            # Use lightweight but effective model
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.use_transformer = True
            print("  ✓ Sentence Transformer loaded: all-MiniLM-L6-v2")
        except ImportError:
            print("  ⚠ sentence-transformers not found, falling back to naive method")
            self.use_transformer = False
        except Exception as e:
            print(f"  ⚠ Failed to load Sentence Transformer: {e}, using naive method")
            self.use_transformer = False
    
    def extract_item_vector(
        self,
        item: Dict,
        item_mem = None  # Can be Dict or str
    ) -> Dict[str, float]:
        """
        Extract vector representation from item metadata and ItemMem
        
        Args:
            item: Item information {'id', 'title', 'tags', ...}
            item_mem: Item memory (optional, can be Dict or str)
            
        Returns:
            {key: value, ...} Vector representation
        """
        vector = {}
        
        # 1. Extract from tags
        if 'tags' in item and item['tags']:
            for tag in item['tags']:
                vector[tag.lower()] = 1.0
        
        # 2. Extract keywords from title (simple tokenization)
        if 'title' in item:
            words = item['title'].lower().split()
            for word in words:
                if len(word) > 3:  # Filter short words
                    vector[word] = vector.get(word, 0) + 0.5
        
        # 3. Extract from ItemMem
        if item_mem:
            # Handle string format item_mem (returned from storage.get_item_memory)
            if isinstance(item_mem, str):
                # Extract keywords from text
                words = item_mem.lower().split()
                for word in words:
                    if len(word) > 3:  # Filter short words
                        vector[word] = vector.get(word, 0) + 0.5
            # Handle dict format item_mem (old format, backward compatible)
            elif isinstance(item_mem, dict):
                for key, entry in item_mem.items():
                    # Use key and value
                    vector[key.lower()] = entry.get('confidence', 0.5) if isinstance(entry, dict) else 0.5
                    if isinstance(entry, dict) and 'value' in entry:
                        val_str = str(entry['value']).lower()
                        vector[val_str] = entry.get('confidence', 0.5)
        
        return vector
    
    def cosine_similarity(
        self,
        vec1: Dict[str, float],
        vec2: Dict[str, float]
    ) -> float:
        """
        Calculate cosine similarity between two sparse vectors
        
        Args:
            vec1, vec2: {key: value, ...}
            
        Returns:
            Similarity [0..1]
        """
        # Calculate dot product
        dot_product = 0.0
        for key in vec1:
            if key in vec2:
                dot_product += vec1[key] * vec2[key]
        
        # Calculate norms
        norm1 = np.sqrt(sum(v**2 for v in vec1.values()))
        norm2 = np.sqrt(sum(v**2 for v in vec2.values()))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def rerank(
        self,
        user_id: int,
        retrieval_bundle: Dict,  # Stage-R output
        candidates: List[Dict],  # [{'id', 'title', 'tags', ...}]
        item_mems: Dict[int, str] = None,  # item_id -> memory string (from storage)
        debug_logger = None
    ) -> List[Dict]:
        """
        Rerank candidate items using vector similarity
        
        Args:
            user_id: User ID
            retrieval_bundle: Stage-R output {'facets', 'vector_profile', 'support_edges'}
            candidates: List of candidate items
            item_mems: Item memories (optional)
            debug_logger: Debug logger
            
        Returns:
            [{'item_id': int, 'score': float, 'rationale': str}, ...]
        """
        if self.use_transformer:
            return self._rerank_with_transformer(user_id, retrieval_bundle, candidates, item_mems, debug_logger)
        else:
            return self._rerank_naive(user_id, retrieval_bundle, candidates, item_mems, debug_logger)
    
    def _rerank_with_transformer(
        self,
        user_id: int,
        retrieval_bundle: Dict,
        candidates: List[Dict],
        item_mems: Dict[int, Dict] = None,
        debug_logger = None
    ) -> List[Dict]:
        """Use Sentence Transformer for semantic matching"""
        # Build user preference text (from facets)
        facets = retrieval_bundle.get('facets', [])
        user_texts = []
        for facet in facets:
            text = facet.get('facet', facet.get('text', ''))
            if text:
                user_texts.append(text)
        
        # If no facets, use default
        if not user_texts:
            user_profile_text = "General recommendations"
        else:
            user_profile_text = ". ".join(user_texts)
        
        # Build candidate item texts
        candidate_texts = []
        for candidate in candidates:
            # Combine title, tags, item_mem
            parts = []
            if 'title' in candidate and candidate['title']:
                parts.append(candidate['title'])
            if 'tags' in candidate and candidate['tags']:
                parts.append(", ".join(candidate['tags'][:5]))  # First 5 tags
            
            # Add item memory
            if item_mems:
                item_mem = item_mems.get(candidate['id'], None)
                if item_mem:
                    # Handle string format item_mem
                    if isinstance(item_mem, str):
                        # Use string directly (may be long, truncate)
                        mem_text = item_mem[:150] if len(item_mem) > 150 else item_mem
                        parts.append(mem_text)
                    # Handle dict format item_mem (backward compatible)
                    elif isinstance(item_mem, dict):
                        mem_parts = [f"{k}: {v.get('value', '')}" for k, v in list(item_mem.items())[:3]]
                        if mem_parts:
                            parts.append(". ".join(mem_parts))
            
            candidate_text = ". ".join(parts) if parts else "Unknown item"
            candidate_texts.append(candidate_text)
        
        # Encode
        user_embedding = self.model.encode([user_profile_text], convert_to_numpy=True)[0]
        item_embeddings = self.model.encode(candidate_texts, convert_to_numpy=True)
        
        # Calculate cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity([user_embedding], item_embeddings)[0]
        
        # Build results
        scores = []
        for i, candidate in enumerate(candidates):
            score = float(similarities[i])
            scores.append({
                'item_id': candidate['id'],
                'score': score,
                'rationale': f"Semantic similarity: {score:.3f}"
            })
        
        # Sort
        scores.sort(key=lambda x: x['score'], reverse=True)
        
        if debug_logger:
            debug_logger(f"SentenceTransformer: scored {len(scores)} items, top score={scores[0]['score']:.3f}", level="INFO")
        
        return scores
    
    def _rerank_naive(
        self,
        user_id: int,
        retrieval_bundle: Dict,
        candidates: List[Dict],
        item_mems: Dict[int, Dict] = None,
        debug_logger = None
    ) -> List[Dict]:
        """Naive bag-of-words method (fallback)"""
        # Build user_vector from facets
        facets = retrieval_bundle.get('facets', [])
        user_vector = {}
        for facet in facets:
            text = facet.get('facet', facet.get('text', ''))
            conf = facet.get('confidence', 0.5)
            # Simple tokenization
            words = text.lower().split()
            for word in words:
                if len(word) > 3:
                    user_vector[word] = max(user_vector.get(word, 0), conf)
        
        # Score each candidate
        scores = []
        for candidate in candidates:
            item_id = candidate['id']
            item_mem = item_mems.get(item_id, None) if item_mems else None
            
            # Extract item vector
            item_vector = self.extract_item_vector(candidate, item_mem)
            
            # Calculate similarity
            score = self.cosine_similarity(user_vector, item_vector)
            
            # Generate simple rationale
            common_keys = set(user_vector.keys()) & set(item_vector.keys())
            if common_keys:
                top_common = sorted(common_keys, key=lambda k: user_vector[k] * item_vector[k], reverse=True)[:3]
                rationale = f"Match on: {', '.join(top_common)}"
            else:
                rationale = "No strong match"
            
            scores.append({
                'item_id': item_id,
                'score': float(score),
                'rationale': rationale
            })
        
        # Sort
        scores.sort(key=lambda x: x['score'], reverse=True)
        
        if debug_logger:
            debug_logger(f"VectorReranker (naive): scored {len(scores)} items, top score={scores[0]['score']:.3f}", level="INFO")
        
        return scores

