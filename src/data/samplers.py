"""
Samplers for training LightGCN and SASRec
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple


class BPRSampler(Dataset):
    """BPR sampler for LightGCN (pairwise ranking)"""
    
    def __init__(self, rec_dataset, n_negatives: int = 1):
        """
        Args:
            rec_dataset: RecDataset instance
            n_negatives: Number of negative samples per positive
        """
        self.dataset = rec_dataset
        self.n_negatives = n_negatives
        
        # Get all training interactions
        self.interactions = rec_dataset.get_train_interactions()
        
        # OPTIMIZED: Pre-sample by user (MUCH faster than per-interaction)
        print("Pre-sampling negative items (optimized batch mode)...")
        
        # Step 1: Group interactions by user
        from collections import defaultdict
        user_interaction_indices = defaultdict(list)
        for idx, (user_id, _) in enumerate(self.interactions):
            user_interaction_indices[user_id].append(idx)
        
        # Step 2: Sample negatives per user (batch operation)
        self.presampled_negatives = {}
        n_users = len(user_interaction_indices)
        processed_interactions = 0
        
        for i, (user_id, indices) in enumerate(user_interaction_indices.items()):
            # Sample enough negatives for ALL interactions of this user at once
            n_interactions = len(indices)
            n_total = n_interactions * n_negatives * 10  # 10x buffer per interaction
            negatives = rec_dataset.sample_negative_items(user_id, n_total)
            
            # Distribute negatives to each interaction (each gets 10x buffer)
            chunk_size = n_negatives * 10
            for j, idx in enumerate(indices):
                start = j * chunk_size
                end = start + chunk_size
                self.presampled_negatives[idx] = negatives[start:end] if end <= len(negatives) else negatives[start:]
            
            processed_interactions += n_interactions
            
            if (i + 1) % 500 == 0:
                print(f"  Processed {i + 1}/{n_users} users ({processed_interactions}/{len(self.interactions)} interactions)...")
        
        print(f"✓ Pre-sampling completed: {len(self.interactions)} interactions from {n_users} users")
        
    def __len__(self):
        return len(self.interactions)
    
    def __getitem__(self, idx):
        user_id, pos_item = self.interactions[idx]
        
        # Use pre-sampled negatives (MUCH faster!)
        presampled = self.presampled_negatives[idx]
        # Randomly select one negative from pre-sampled pool
        import random
        neg_item = random.choice(presampled) if len(presampled) > 0 else 0
        
        return {
            'user': user_id,
            'pos_item': pos_item,
            'neg_items': neg_item  # Single integer, not list
        }


class SequenceSampler(Dataset):
    """Sequence sampler for SASRec"""
    
    def __init__(self, rec_dataset, max_seq_len: int = 50, n_negatives: int = 1):
        """
        Args:
            rec_dataset: RecDataset instance
            max_seq_len: Maximum sequence length
            n_negatives: Number of negative samples per position
        """
        self.dataset = rec_dataset
        self.max_seq_len = max_seq_len
        self.n_negatives = n_negatives
        
        # Build sequences from training data
        self.sequences = []
        for user_id, items in rec_dataset.train_data.items():
            if len(items) >= 2:  # Need at least 2 items for training
                self.sequences.append((user_id, items))
        
        # OPTIMIZED: Pre-sample by user (MUCH faster)
        print("Pre-sampling negative items (optimized batch mode)...")
        
        # Step 1: Group sequences by user
        from collections import defaultdict
        user_sequence_indices = defaultdict(list)
        user_total_negatives = {}
        
        for idx, (user_id, items) in enumerate(self.sequences):
            # Calculate needed negatives for this sequence
            if len(items) > max_seq_len:
                items = items[-max_seq_len:]
            seq_len = len(items) - 1
            n_needed = seq_len * n_negatives * 3  # 3x buffer
            
            user_sequence_indices[user_id].append((idx, n_needed))
            user_total_negatives[user_id] = user_total_negatives.get(user_id, 0) + n_needed
        
        # Step 2: Sample negatives per user (batch operation)
        self.presampled_negatives = {}
        n_users = len(user_sequence_indices)
        processed_sequences = 0
        
        for i, (user_id, seq_list) in enumerate(user_sequence_indices.items()):
            # Sample ALL negatives for this user at once
            n_total = user_total_negatives[user_id]
            negatives = rec_dataset.sample_negative_items(user_id, n_total)
            
            # Distribute negatives to each sequence
            offset = 0
            for idx, n_needed in seq_list:
                self.presampled_negatives[idx] = negatives[offset:offset + n_needed]
                offset += n_needed
            
            processed_sequences += len(seq_list)
            
            if (i + 1) % 500 == 0:
                print(f"  Processed {i + 1}/{n_users} users ({processed_sequences}/{len(self.sequences)} sequences)...")
        
        print(f"✓ Pre-sampling completed: {len(self.sequences)} sequences from {n_users} users")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        user_id, items = self.sequences[idx]
        
        # Truncate if too long
        if len(items) > self.max_seq_len:
            items = items[-self.max_seq_len:]
        
        # Input sequence: all but last
        # Target: next item for each position
        seq_len = len(items)
        input_seq = items[:-1]
        target_seq = items[1:]
        
        # Use pre-sampled negatives (MUCH faster!)
        presampled = self.presampled_negatives[idx]
        neg_items = []
        for i in range(len(target_seq)):
            # Get negatives from pre-sampled pool
            start_idx = i * self.n_negatives
            end_idx = start_idx + self.n_negatives
            if end_idx <= len(presampled):
                negs = presampled[start_idx:end_idx]
            else:
                # Wrap around if needed
                negs = [presampled[j % len(presampled)] for j in range(start_idx, end_idx)]
            neg_items.append(negs)
        
        return {
            'user': user_id,
            'input_seq': input_seq,
            'target_seq': target_seq,
            'neg_items': neg_items,
            'seq_len': len(input_seq)
        }


def bpr_collate_fn(batch):
    """Collate function for BPR sampler"""
    users = torch.LongTensor([item['user'] for item in batch])
    pos_items = torch.LongTensor([item['pos_item'] for item in batch])
    neg_items = torch.LongTensor([item['neg_items'] for item in batch])  # neg_items is now a single int
    
    return {
        'users': users,
        'pos_items': pos_items,
        'neg_items': neg_items
    }


def sequence_collate_fn(batch):
    """Collate function for sequence sampler with padding"""
    max_len = max(item['seq_len'] for item in batch)
    batch_size = len(batch)
    
    users = torch.LongTensor([item['user'] for item in batch])
    
    # Pad sequences
    input_seqs = torch.zeros(batch_size, max_len, dtype=torch.long)
    target_seqs = torch.zeros(batch_size, max_len, dtype=torch.long)
    seq_lens = torch.LongTensor([item['seq_len'] for item in batch])
    
    # Pad negative items
    n_negatives = len(batch[0]['neg_items'][0]) if batch[0]['neg_items'] else 1
    neg_items = torch.zeros(batch_size, max_len, n_negatives, dtype=torch.long)
    
    for i, item in enumerate(batch):
        seq_len = item['seq_len']
        input_seqs[i, :seq_len] = torch.LongTensor(item['input_seq'])
        target_seqs[i, :seq_len] = torch.LongTensor(item['target_seq'])
        
        for j in range(seq_len):
            neg_items[i, j] = torch.LongTensor(item['neg_items'][j])
    
    return {
        'users': users,
        'input_seqs': input_seqs,
        'target_seqs': target_seqs,
        'neg_items': neg_items,
        'seq_lens': seq_lens
    }


def get_bpr_dataloader(rec_dataset, batch_size: int, shuffle: bool = True, 
                       num_workers: int = 4) -> DataLoader:
    """
    Create DataLoader for BPR training
    
    Args:
        rec_dataset: RecDataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of workers for data loading (default: 4)
    """
    sampler = BPRSampler(rec_dataset, n_negatives=1)
    return DataLoader(
        sampler,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=bpr_collate_fn,
        pin_memory=True,  # Speed up CPU->GPU transfer
        persistent_workers=True if num_workers > 0 else False
    )


def get_sequence_dataloader(rec_dataset, max_seq_len: int, batch_size: int,
                            shuffle: bool = True, num_workers: int = 4) -> DataLoader:
    """
    Create DataLoader for sequence training
    
    Args:
        rec_dataset: RecDataset instance
        max_seq_len: Maximum sequence length
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of workers for data loading (default: 4)
    """
    sampler = SequenceSampler(rec_dataset, max_seq_len=max_seq_len, n_negatives=1)
    return DataLoader(
        sampler,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=sequence_collate_fn,
        pin_memory=True,  # Speed up CPU->GPU transfer
        persistent_workers=True if num_workers > 0 else False
    )



