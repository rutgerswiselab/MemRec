"""
Evaluation metrics for recommendation systems
"""
import torch
import numpy as np
from typing import List, Dict


def hit_at_k(predictions: torch.Tensor, targets: torch.Tensor, k: int) -> float:
    """
    Hit@K: whether the target item is in top-K predictions
    
    Args:
        predictions: (batch_size, n_items) - predicted scores
        targets: (batch_size,) - target item indices
        k: Top-K
    
    Returns:
        hit_rate: Hit rate in [0, 1]
    """
    _, topk_indices = torch.topk(predictions, k, dim=1)
    hits = (topk_indices == targets.unsqueeze(1)).any(dim=1).float()
    return hits.mean().item()


def recall_at_k(predictions: torch.Tensor, targets: torch.Tensor, k: int) -> float:
    """
    Recall@K: same as Hit@K for single target item
    
    Args:
        predictions: (batch_size, n_items) - predicted scores
        targets: (batch_size,) - target item indices
        k: Top-K
    
    Returns:
        recall: Recall in [0, 1]
    """
    return hit_at_k(predictions, targets, k)


def ndcg_at_k(predictions: torch.Tensor, targets: torch.Tensor, k: int) -> float:
    """
    NDCG@K: Normalized Discounted Cumulative Gain
    
    Args:
        predictions: (batch_size, n_items) - predicted scores
        targets: (batch_size,) - target item indices
        k: Top-K
    
    Returns:
        ndcg: NDCG in [0, 1]
    """
    _, topk_indices = torch.topk(predictions, k, dim=1)
    
    # Find position of target in top-K (0-indexed)
    positions = (topk_indices == targets.unsqueeze(1)).nonzero(as_tuple=True)
    
    # Compute DCG for hits
    ndcg_scores = torch.zeros(predictions.size(0), device=predictions.device)
    if len(positions[0]) > 0:
        # positions[0] = batch indices, positions[1] = positions in top-K
        batch_indices = positions[0]
        rank_positions = positions[1]
        # DCG = 1 / log2(position + 2)
        ndcg_scores[batch_indices] = 1.0 / torch.log2(rank_positions.float() + 2)
    
    return ndcg_scores.mean().item()


def mrr_at_k(predictions: torch.Tensor, targets: torch.Tensor, k: int) -> float:
    """
    MRR@K: Mean Reciprocal Rank
    
    Args:
        predictions: (batch_size, n_items) - predicted scores
        targets: (batch_size,) - target item indices
        k: Top-K
    
    Returns:
        mrr: MRR in [0, 1]
    """
    _, topk_indices = torch.topk(predictions, k, dim=1)
    
    # Find position of target in top-K (0-indexed)
    positions = (topk_indices == targets.unsqueeze(1)).nonzero(as_tuple=True)
    
    # Compute reciprocal rank for hits
    mrr_scores = torch.zeros(predictions.size(0), device=predictions.device)
    if len(positions[0]) > 0:
        batch_indices = positions[0]
        rank_positions = positions[1]
        # RR = 1 / (position + 1)
        mrr_scores[batch_indices] = 1.0 / (rank_positions.float() + 1)
    
    return mrr_scores.mean().item()


def evaluate_ranking(predictions: torch.Tensor, targets: torch.Tensor, 
                     ks: List[int] = [1, 5, 10],
                     metric_names: List[str] = ['Hit', 'NDCG']) -> Dict[str, float]:
    """
    Compute ranking metrics
    
    Args:
        predictions: (batch_size, n_items) - predicted scores
        targets: (batch_size,) - target item indices
        ks: List of K values
        metric_names: List of metric names to compute (default: ['Hit', 'NDCG'])
    
    Returns:
        metrics: Dictionary of metric values
    """
    metrics = {}
    
    for k in ks:
        if 'Hit' in metric_names:
            metrics[f'Hit@{k}'] = hit_at_k(predictions, targets, k)
        if 'Recall' in metric_names:
            metrics[f'Recall@{k}'] = recall_at_k(predictions, targets, k)
        if 'NDCG' in metric_names:
            metrics[f'NDCG@{k}'] = ndcg_at_k(predictions, targets, k)
        if 'MRR' in metric_names:
            metrics[f'MRR@{k}'] = mrr_at_k(predictions, targets, k)
    
    return metrics


def format_metrics(metrics: Dict[str, float]) -> str:
    """Format metrics for printing"""
    items = [f"{k}: {v:.4f}" for k, v in metrics.items()]
    return ", ".join(items)


