"""
GoodReads Domain Pruning Rules
Social book community with dense graph and series-awareness
"""
import numpy as np
from typing import Dict


class GoodReadsRules:
    """
    InstructRec-GoodReads pruning rules
    
    Key characteristics:
    - Very dense graph (52.7 books/user, 10.8 users/item)
    - Series-aware reading patterns
    - Strong community effects (book clubs, genre specialization)
    """
    
    @staticmethod
    def apply_rules(features: Dict[str, float]) -> float:
        """
        Apply GoodReads-specific pruning rules
        
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
        
        # Rule 1: High Co-interaction Boost
        social_boost = 1.0
        memory_boost = 1.0 + memory_sim
        if co_count > 10:
            social_boost = 2.0
            memory_boost = 1.0 + 1.5 * memory_sim
        
        # Rule 2: Series Detection
        series_boost = 1.0
        if metadata_overlap > 0.8:  # Likely series match
            series_boost = 3.0
        
        # Rule 3: Social Signal Priority
        if co_count > 15:
            # Favor explicit social overlap over pure CF
            edge_weight = edge_weight * 0.7
            social_boost *= 1.5
        
        # Rule 4: Minimal Recency Decay
        recency_decay = 1.0
        if recency_days > 365:
            recency_decay = np.exp(-0.002 * recency_days)
        
        # Rule 5: Combined Score
        final_score = edge_weight * social_boost * series_boost * memory_boost * recency_decay
        
        return final_score
    
    @staticmethod
    def get_description() -> str:
        return "GoodReads: Social (2.0×) + Series (3.0×), minimal decay (e^-0.002), CF threshold > 10"

