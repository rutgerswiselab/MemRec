"""
Books Domain Pruning Rules
Content-driven with stable long-term preferences
"""
import numpy as np
from typing import Dict


class BooksRules:
    """
    InstructRec-Books pruning rules
    
    Key characteristics:
    - Content-driven (genre/author similarity is critical)
    - Stable preferences (long-term reader tastes)
    - Sparse interactions (limited user-user overlap)
    """
    
    @staticmethod
    def apply_rules(features: Dict[str, float]) -> float:
        """
        Apply Books-specific pruning rules
        
        Args:
            features: Dict with keys:
                - edge_weight: Base CF score
                - metadata_overlap_score: Content similarity (0-1)
                - co_interaction_count: Number of co-rated books
                - memory_similarity_score: Memory similarity (0-1)
                - recency_days: Days since last interaction
                - neighbor_type: 'user' or 'item'
        
        Returns:
            final_score: Weighted score for neighbor selection
        """
        edge_weight = features.get('edge_weight', 0.0)
        metadata_overlap = features.get('metadata_overlap_score', 0.0)
        co_count = features.get('co_interaction_count', 0)
        memory_sim = features.get('memory_similarity_score', 0.0)
        recency_days = features.get('recency_days', 0)
        neighbor_type = features.get('neighbor_type', 'item')
        
        # Rule 1: Content Similarity Boost
        content_boost = 1.0
        if metadata_overlap > 0.6:
            content_boost = 2.5
        
        # Rule 2: Collaborative Filtering with Threshold
        cf_boost = 1.0
        if co_count > 3:
            cf_boost = 1.8
            if memory_sim > 0.5:
                cf_boost *= 1.5
        
        # Rule 3: Mild Recency Decay
        recency_decay = 1.0
        if recency_days > 180:
            recency_decay = np.exp(-0.004 * recency_days)
        
        # Rule 4: Memory-Enhanced Ranking
        if neighbor_type == 'item':
            memory_boost = 1.0 + 1.2 * memory_sim
        else:  # user
            memory_boost = 1.0 + 0.8 * memory_sim
        
        # Rule 5: Combined Score
        final_score = edge_weight * content_boost * cf_boost * recency_decay * memory_boost
        
        return final_score
    
    @staticmethod
    def get_description() -> str:
        return "Books: Content-driven (2.5×), mild recency decay (e^-0.004), CF threshold > 3"

