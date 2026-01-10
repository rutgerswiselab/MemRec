"""
Yelp Domain Pruning Rules
Categorical dominance with very strong recency decay
"""
import numpy as np
from typing import Dict


class YelpRules:
    """
    InstructRec-Yelp pruning rules
    
    Key characteristics:
    - Categorical constraints (cuisine, price, location)
    - Recency is critical (restaurants close, quality changes)
    - Context-rich metadata (ambience, features)
    """
    
    @staticmethod
    def apply_rules(features: Dict[str, float]) -> float:
        """
        Apply Yelp-specific pruning rules
        
        Args:
            features: Dict with keys:
                - edge_weight: Base CF score
                - metadata_overlap_score: Multi-faceted similarity (0-1)
                      (cuisine + price + attributes)
                - co_interaction_count: Number of co-visited restaurants
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
        
        # Rule 1: Categorical Dominance
        categorical_boost = 1.0
        if metadata_overlap > 0.7:  # Same cuisine + price
            categorical_boost = 3.5
        if metadata_overlap > 0.85:  # + attribute match
            categorical_boost = 4.5
        
        # Rule 2: Very Strong Recency Decay
        if recency_days <= 90:
            recency_decay = 1.0
        elif recency_days <= 180:
            recency_decay = np.exp(-0.028 * recency_days)
        else:
            recency_decay = np.exp(-0.028 * recency_days) * 0.5  # Additional penalty
        
        # Rule 3: Attribute-Aware Memory
        memory_boost = 1.0 + memory_sim
        if metadata_overlap > 0.7:  # Attribute match
            memory_boost = 1.0 + 2.2 * memory_sim
        
        # Rule 4: Sparse CF Handling
        cf_boost = 0.5  # Default: downweight CF
        if co_count >= 2:  # Rare but meaningful
            cf_boost = 2.0
        
        # Rule 5: Category Filter (strong penalty for mismatch)
        category_filter = 1.0
        if metadata_overlap < 0.4:  # Different cuisine/price
            category_filter = 0.2  # Strong penalty
        
        # Rule 6: Combined Score
        final_score = (edge_weight * cf_boost * categorical_boost * 
                      memory_boost * recency_decay * category_filter)
        
        return final_score
    
    @staticmethod
    def get_description() -> str:
        return "Yelp: Categorical dominant (3.5×), very strong decay (e^-0.028), CF threshold >= 2"

