"""
Neighbor scoring and Top-k selector
v2: Supports both parameterized MLP and hybrid rule modes
"""
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from pathlib import Path


class PrunerMLP(nn.Module):
    """2-layer MLP for neighbor scoring"""
    def __init__(self, input_dim: int = 10, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x).squeeze(-1)


class NeighborPruner:
    """Parameterized graph pruner (v2)
    
    Supports two modes:
    - hybrid_rule: Rule-based scoring (default)
    - learned_mlp: MLP-based learned scoring (requires checkpoint)
    """
    
    def __init__(
        self,
        k: int = 16,
        item_weight: float = 1.0,
        user_weight: float = 0.5,
        recency_weight: float = 0.5,
        degree_weight: float = 0.3,
        overlap_weight: float = 0.7,
        mix_min_users: int = 4,
        mix_min_items: int = 6,
        mode: str = "hybrid_rule",
        checkpoint: Optional[str] = None
    ):
        """
        Initialize pruner
        
        Args:
            k: Total number of neighbors to select (items + users)
            item_weight: Weight for item neighbors
            user_weight: Weight for user neighbors
            recency_weight: Weight for temporal decay
            degree_weight: Weight for node degree (higher degree is more important)
            overlap_weight: Weight for overlap (more common items is more important)
            mix_min_users: Minimum number of user neighbors (mixing constraint)
            mix_min_items: Minimum number of item neighbors (mixing constraint)
            mode: Scoring mode ("hybrid_rule" or "learned_mlp")
            checkpoint: Checkpoint path for MLP model (required when mode="learned_mlp")
        """
        self.k = k
        self.item_weight = item_weight
        self.user_weight = user_weight
        self.recency_weight = recency_weight
        self.degree_weight = degree_weight
        self.overlap_weight = overlap_weight
        self.mix_min_users = mix_min_users
        self.mix_min_items = mix_min_items
        self.mode = mode
        self.checkpoint = checkpoint
        
        # Initialize MLP (if using learned mode)
        self.mlp = None
        if mode == "learned_mlp":
            self.mlp = PrunerMLP(input_dim=10, hidden_dim=32)
            if checkpoint and Path(checkpoint).exists():
                print(f"  Loading pruner MLP from {checkpoint}")
                self.mlp.load_state_dict(torch.load(checkpoint, map_location='cpu'))
                self.mlp.eval()
            else:
                print(f"  WARNING: MLP mode requested but no valid checkpoint. Falling back to hybrid_rule.")
                self.mode = "hybrid_rule"
    
    def extract_features(
        self,
        neighbor_type: str,
        neighbor_id: int,
        user_id: int,
        graph,
        recency: float = None,
        overlap_count: int = None
    ) -> np.ndarray:
        """Extract feature vector for neighbor (for MLP)
        
        Features:
        - [0] type (0=item, 1=user)
        - [1] degree (normalized)
        - [2] recency (for items)
        - [3] overlap (for users)
        - [4-9] reserved for future features
        """
        features = np.zeros(10, dtype=np.float32)
        
        if neighbor_type == 'item':
            features[0] = 0.0
            item_degree = graph.get_item_degree(neighbor_id)
            max_item_degree = max(graph.item_degrees.values()) if graph.item_degrees else 1
            features[1] = item_degree / max_item_degree
            features[2] = recency if recency is not None else 0.0
        else:  # user
            features[0] = 1.0
            user_degree = graph.get_user_degree(neighbor_id)
            max_user_degree = max(graph.user_degrees.values()) if graph.user_degrees else 1
            features[1] = user_degree / max_user_degree
            if overlap_count is not None:
                user_item_count = graph.get_user_degree(user_id)
                features[3] = overlap_count / user_item_count if user_item_count > 0 else 0.0
        
        return features
    
    def score_neighbors_mlp(
        self,
        user_id: int,
        graph,
        candidates: List[int] = None
    ) -> Dict:
        """Score neighbors using MLP"""
        neighbors = graph.get_user_neighbors(user_id, max_items=100, max_users=50)
        
        item_scores = []
        for item_id, recency in neighbors['item_neighbors']:
            features = self.extract_features('item', item_id, user_id, graph, recency=recency)
            with torch.no_grad():
                score = self.mlp(torch.from_numpy(features).unsqueeze(0)).item()
            item_scores.append((item_id, score))
        
        user_scores = []
        for neighbor_user_id, overlap_count in neighbors['user_neighbors']:
            features = self.extract_features('user', neighbor_user_id, user_id, graph, overlap_count=overlap_count)
            with torch.no_grad():
                score = self.mlp(torch.from_numpy(features).unsqueeze(0)).item()
            user_scores.append((neighbor_user_id, score))
        
        item_scores.sort(key=lambda x: x[1], reverse=True)
        user_scores.sort(key=lambda x: x[1], reverse=True)
        
        return {
            'item_scores': item_scores,
            'user_scores': user_scores
        }
    
    def score_neighbors(
        self,
        user_id: int,
        graph,
        candidates: List[int] = None
    ) -> Dict:
        """
        Score all neighbors of a user (choose method based on mode)
        
        Args:
            user_id: Target user
            graph: UserItemGraph instance
            candidates: List of candidate items (optional, for candidate-aware scoring)
            
        Returns:
            {
                'item_scores': [(item_id, score), ...],
                'user_scores': [(user_id, score), ...]
            }
        """
        # Select scoring method
        if self.mode == "learned_mlp" and self.mlp is not None:
            return self.score_neighbors_mlp(user_id, graph, candidates)
        else:
            return self.score_neighbors_hybrid(user_id, graph, candidates)
    
    def score_neighbors_hybrid(
        self,
        user_id: int,
        graph,
        candidates: List[int] = None
    ) -> Dict:
        """
        Score neighbors using hybrid rule
        
        Args:
            user_id: Target user
            graph: UserItemGraph instance
            candidates: List of candidate items (optional)
            
        Returns:
            {
                'item_scores': [(item_id, score), ...],
                'user_scores': [(user_id, score), ...]
            }
        """
        # Get neighbors
        neighbors = graph.get_user_neighbors(
            user_id, 
            max_items=100,  # Get more neighbors first, then sort
            max_users=50
        )
        
        # 1. Score item neighbors
        item_scores = []
        for item_id, recency in neighbors['item_neighbors']:
            # Feature 1: Temporal decay (already between 0-1)
            recency_score = recency
            
            # Feature 2: Item degree normalized (more popular items may be more representative)
            item_degree = graph.get_item_degree(item_id)
            max_item_degree = max(graph.item_degrees.values()) if graph.item_degrees else 1
            degree_score = item_degree / max_item_degree
            
            # Feature 3: Candidate-aware (if candidates provided)
            candidate_score = 0.0
            if candidates:
                # Simple proxy: item-candidate co-occurring user overlap
                item_users = set(graph.get_item_users(item_id))
                max_overlap = 0
                for cand_id in candidates:
                    cand_users = set(graph.get_item_users(cand_id))
                    overlap = len(item_users & cand_users)
                    max_overlap = max(max_overlap, overlap)
                candidate_score = max_overlap / len(item_users) if item_users else 0.0
            
            # Combined score
            score = (
                self.recency_weight * recency_score +
                self.degree_weight * degree_score +
                (0.5 * candidate_score if candidates else 0.0)
            )
            item_scores.append((item_id, score))
        
        # 2. Score user neighbors
        user_scores = []
        for neighbor_user_id, overlap_count in neighbors['user_neighbors']:
            # Feature 1: Normalized overlap
            user_item_count = graph.get_user_degree(user_id)
            overlap_score = overlap_count / user_item_count if user_item_count > 0 else 0.0
            
            # Feature 2: Neighbor user's degree
            neighbor_degree = graph.get_user_degree(neighbor_user_id)
            max_user_degree = max(graph.user_degrees.values()) if graph.user_degrees else 1
            degree_score = neighbor_degree / max_user_degree
            
            # Combined score
            score = (
                self.overlap_weight * overlap_score +
                self.degree_weight * degree_score
            )
            user_scores.append((neighbor_user_id, score))
        
        # Sort
        item_scores.sort(key=lambda x: x[1], reverse=True)
        user_scores.sort(key=lambda x: x[1], reverse=True)
        
        return {
            'item_scores': item_scores,
            'user_scores': user_scores
        }
    
    def prune(
        self,
        user_id: int,
        graph,
        candidates: List[int] = None
    ) -> Dict:
        """
        Select top-k neighbors
        
        Args:
            user_id: Target user
            graph: UserItemGraph instance
            candidates: List of candidate items (optional)
            
        Returns:
            {
                'user_id': int,
                'neighbors': [
                    {'type': 'item' or 'user', 'id': int, 'score': float, 'features': dict},
                    ...
                ],
                'n_items': int,
                'n_users': int
            }
        """
        # Score
        scored = self.score_neighbors(user_id, graph, candidates)
        
        # Merge and apply type weights
        all_neighbors = []
        
        for item_id, score in scored['item_scores']:
            weighted_score = score * self.item_weight
            all_neighbors.append({
                'type': 'item',
                'id': item_id,
                'score': weighted_score,
                'features': {
                    'recency': graph.get_item_recency(user_id, item_id),
                    'degree': graph.get_item_degree(item_id)
                }
            })
        
        for neighbor_user_id, score in scored['user_scores']:
            weighted_score = score * self.user_weight
            all_neighbors.append({
                'type': 'user',
                'id': neighbor_user_id,
                'score': weighted_score,
                'features': {
                    'degree': graph.get_user_degree(neighbor_user_id)
                }
            })
        
        # Apply mixing constraint: ensure at least mix_min_users and mix_min_items
        item_neighbors = [n for n in all_neighbors if n['type'] == 'item']
        user_neighbors = [n for n in all_neighbors if n['type'] == 'user']
        
        # Sort by score
        item_neighbors.sort(key=lambda x: x['score'], reverse=True)
        user_neighbors.sort(key=lambda x: x['score'], reverse=True)
        
        # First ensure minimum quantities
        selected = []
        selected.extend(item_neighbors[:self.mix_min_items])
        selected.extend(user_neighbors[:self.mix_min_users])
        
        # Remaining quota: select highest-scoring from both
        remaining_items = item_neighbors[self.mix_min_items:]
        remaining_users = user_neighbors[self.mix_min_users:]
        remaining_pool = remaining_items + remaining_users
        remaining_pool.sort(key=lambda x: x['score'], reverse=True)
        
        remaining_quota = self.k - len(selected)
        if remaining_quota > 0:
            selected.extend(remaining_pool[:remaining_quota])
        
        # Re-sort final results by score
        selected.sort(key=lambda x: x['score'], reverse=True)
        
        # Statistics
        n_items = sum(1 for n in selected if n['type'] == 'item')
        n_users = sum(1 for n in selected if n['type'] == 'user')
        
        return {
            'user_id': user_id,
            'neighbors': selected,
            'n_items': n_items,
            'n_users': n_users,
            'meta': {
                'k': self.k,
                'mode': self.mode,
                'mix_min_users': self.mix_min_users,
                'mix_min_items': self.mix_min_items,
                'checkpoint': self.checkpoint if self.mode == "learned_mlp" else None
            }
        }
