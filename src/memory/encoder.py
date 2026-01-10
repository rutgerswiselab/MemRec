"""
Facet Encoder
Convert text facets to vector representations for enhancing LightGCN/SASRec
"""
import numpy as np
from typing import List, Dict
import torch


class FacetEncoder:
    """Simple encoder that encodes facets into vectors"""
    
    def __init__(self, embedding_dim: int = 64):
        """
        Initialize encoder
        
        Args:
            embedding_dim: Output vector dimension (should match embedding dimension of LightGCN/SASRec)
        """
        self.embedding_dim = embedding_dim
        self.encoder_type = 'simple'  # v0: simple weighted average; v1: can use sentence transformer
    
    def encode_facets(
        self, 
        facets: List[Dict],
        method: str = 'weighted_mean'
    ) -> np.ndarray:
        """
        Encode facets into vectors
        
        Args:
            facets: List of facets, each facet has 'facet', 'confidence', 'supporting_neighbors'
            method: Encoding method
                - 'weighted_mean': Weighted average based on confidence (v0 simple implementation)
                - 'transformer': Use sentence transformer (v1 optional)
        
        Returns:
            Vector representation (embedding_dim,)
        """
        if not facets:
            # Return zero vector
            return np.zeros(self.embedding_dim)
        
        if method == 'weighted_mean':
            return self._encode_weighted_mean(facets)
        elif method == 'transformer':
            # TODO: v1 implementation
            raise NotImplementedError("Transformer encoding not implemented in v0")
        else:
            raise ValueError(f"Unknown encoding method: {method}")
    
    def _encode_weighted_mean(self, facets: List[Dict]) -> np.ndarray:
        """
        Simple weighted average encoding (v0)
        
        Use simple hashing trick to map text to vector space
        """
        # Initialize vector
        vector = np.zeros(self.embedding_dim)
        total_weight = 0.0
        
        for facet_dict in facets:
            facet_text = facet_dict.get('facet', '')
            confidence = facet_dict.get('confidence', 1.0)
            
            # Simple hashing: map each word of text to a dimension
            words = facet_text.lower().split()
            facet_vector = np.zeros(self.embedding_dim)
            
            for word in words:
                # Use simple string hashing
                hash_val = hash(word) % self.embedding_dim
                facet_vector[hash_val] += 1.0
            
            # Normalize
            if np.linalg.norm(facet_vector) > 0:
                facet_vector = facet_vector / np.linalg.norm(facet_vector)
            
            # Weighted accumulation
            vector += confidence * facet_vector
            total_weight += confidence
        
        # Normalize
        if total_weight > 0:
            vector = vector / total_weight
        
        if np.linalg.norm(vector) > 0:
            vector = vector / np.linalg.norm(vector)
        
        return vector
    
    def encode_facets_batch(
        self, 
        facets_list: List[List[Dict]],
        method: str = 'weighted_mean'
    ) -> np.ndarray:
        """
        Batch encode facets
        
        Args:
            facets_list: Facets for multiple users
            method: Encoding method
            
        Returns:
            Vector matrix (batch_size, embedding_dim)
        """
        vectors = []
        for facets in facets_list:
            vector = self.encode_facets(facets, method)
            vectors.append(vector)
        
        return np.array(vectors)
    
    def to_torch(self, vector: np.ndarray, device: str = 'cpu') -> torch.Tensor:
        """Convert numpy vector to torch tensor"""
        return torch.from_numpy(vector).float().to(device)


class TextBundleBuilder:
    """Build text bundles (for iAgent/i2Agent rerank/explain)"""
    
    @staticmethod
    def build_collaborative_summary(
        facets: List[Dict],
        neighbor_context: str = "",
        max_length: int = 500
    ) -> str:
        """
        Build collaborative summary from facets and neighbor context
        
        Args:
            facets: List of facets
            neighbor_context: Neighbor context (optional)
            max_length: Maximum length
            
        Returns:
            Text summary
        """
        if not facets:
            return ""
        
        # Build summary
        summary_parts = ["**Collaborative Preferences:**"]
        
        for i, facet_dict in enumerate(facets[:5], 1):  # Limit to 5
            facet_text = facet_dict.get('facet', '')
            confidence = facet_dict.get('confidence', 0.0)
            summary_parts.append(f"{i}. {facet_text} (confidence: {confidence:.2f})")
        
        summary = "\n".join(summary_parts)
        
        # Truncate
        if len(summary) > max_length:
            summary = summary[:max_length] + "..."
        
        return summary
    
    @staticmethod
    def build_candidate_explanation(
        candidate_id: int,
        candidate_notes: List[Dict],
        facets: List[Dict]
    ) -> str:
        """
        Build explanation for candidate item
        
        Args:
            candidate_id: Candidate item ID
            candidate_notes: Notes for all candidates
            facets: List of facets
            
        Returns:
            Explanation text
        """
        # Find the note for this candidate
        note = None
        for n in candidate_notes:
            if n.get('candidate_id') == candidate_id:
                note = n
                break
        
        if not note:
            return f"Item {candidate_id} is recommended based on collaborative signals."
        
        fit = note.get('fit', 0.0)
        rationale = note.get('rationale', '')
        
        explanation = f"Item {candidate_id} (fit: {fit:.2f}): {rationale}"
        
        return explanation

