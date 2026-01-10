"""
User-Item Graph Builder
Build lightweight collaborative graph from training set, supporting neighbor queries and temporal decay
"""
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import numpy as np


class UserItemGraph:
    """User-item bipartite graph"""
    
    def __init__(self, dataset):
        """
        Build graph from dataset
        
        Args:
            dataset: RecDataset instance containing train_data
        """
        self.dataset = dataset
        
        # Build adjacency lists
        self.items_by_user = defaultdict(list)  # user -> [items]
        self.users_by_item = defaultdict(list)  # item -> [users]
        self.recency = defaultdict(dict)  # user -> {item: recency_score}
        
        # Build graph from training data
        self._build_from_train()
        
        # Compute statistics
        self.user_degrees = {u: len(items) for u, items in self.items_by_user.items()}
        self.item_degrees = {i: len(users) for i, users in self.users_by_item.items()}
        
    def _build_from_train(self):
        """Build graph from training data"""
        for user_id, item_list in self.dataset.train_data.items():
            # Add temporal decay score for each user's items
            n_items = len(item_list)
            for pos, item_id in enumerate(item_list):
                # Record user-item edge
                self.items_by_user[user_id].append(item_id)
                self.users_by_item[item_id].append(user_id)
                
                # Compute recency: newer items have higher scores (0 to 1)
                # pos=0 is earliest, pos=n_items-1 is most recent
                recency_score = (pos + 1) / n_items if n_items > 0 else 1.0
                self.recency[user_id][item_id] = recency_score
    
    def get_user_items(self, user_id: int) -> List[int]:
        """Get items the user has interacted with"""
        return self.items_by_user.get(user_id, [])
    
    def get_item_users(self, item_id: int) -> List[int]:
        """Get users who have interacted with the item"""
        return self.users_by_item.get(item_id, [])
    
    def get_item_recency(self, user_id: int, item_id: int) -> float:
        """Get temporal decay score for user-item interaction"""
        return self.recency.get(user_id, {}).get(item_id, 0.0)
    
    def get_user_degree(self, user_id: int) -> int:
        """Get user degree (number of interacted items)"""
        return self.user_degrees.get(user_id, 0)
    
    def get_item_degree(self, item_id: int) -> int:
        """Get item degree (number of interacting users)"""
        return self.item_degrees.get(item_id, 0)
    
    def get_user_neighbors(
        self, 
        user_id: int, 
        max_items: int = 20, 
        max_users: int = 10
    ) -> Dict:
        """
        Get neighborhood subgraph for user
        
        Args:
            user_id: Target user
            max_items: Maximum number of item neighbors to return (sorted by temporal decay)
            max_users: Maximum number of user neighbors to return (discovered through common items)
            
        Returns:
            {
                'item_neighbors': [(item_id, recency_score), ...],
                'user_neighbors': [(neighbor_user_id, overlap_count), ...]
            }
        """
        # 1. Item neighbors: items from user's interaction history, sorted by recency
        user_items = self.get_user_items(user_id)
        item_neighbors = []
        for item_id in user_items:
            recency = self.get_item_recency(user_id, item_id)
            item_neighbors.append((item_id, recency))
        
        # Sort by recency in descending order, take top max_items
        item_neighbors.sort(key=lambda x: x[1], reverse=True)
        item_neighbors = item_neighbors[:max_items]
        
        # 2. User neighbors: users discovered through common items, sorted by overlap
        user_item_set = set(user_items)
        neighbor_overlap = defaultdict(int)
        
        for item_id in user_items:
            # Find other users who also interacted with this item
            co_users = self.get_item_users(item_id)
            for co_user in co_users:
                if co_user != user_id:
                    neighbor_overlap[co_user] += 1
        
        # Sort by overlap, take top max_users
        user_neighbors = sorted(
            neighbor_overlap.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:max_users]
        
        return {
            'item_neighbors': item_neighbors,
            'user_neighbors': user_neighbors
        }
    
    def get_stats(self) -> Dict:
        """Get graph statistics"""
        n_users = len(self.items_by_user)
        n_items = len(self.users_by_item)
        n_edges = sum(len(items) for items in self.items_by_user.values())
        
        avg_user_degree = n_edges / n_users if n_users > 0 else 0
        avg_item_degree = n_edges / n_items if n_items > 0 else 0
        
        return {
            'n_users': n_users,
            'n_items': n_items,
            'n_edges': n_edges,
            'avg_user_degree': avg_user_degree,
            'avg_item_degree': avg_item_degree
        }
    
    def __repr__(self):
        stats = self.get_stats()
        return (f"UserItemGraph(users={stats['n_users']}, "
                f"items={stats['n_items']}, "
                f"edges={stats['n_edges']}, "
                f"avg_user_deg={stats['avg_user_degree']:.2f})")
