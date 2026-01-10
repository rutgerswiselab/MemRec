#!/usr/bin/env python3
"""
Generate a fixed sampled user list for ablation experiments.
This ensures all ablation studies use the same 1000 users for fair comparison.
"""
import json
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.dataset_base import RecDataset


def generate_user_sample(dataset_name: str, n_samples: int = 1000, seed: int = 42):
    """
    Generate a fixed sampled user list from a dataset's test set.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'instructrec-books')
        n_samples: Number of users to sample (default: 1000)
        seed: Random seed for reproducibility (default: 42)
    """
    # Paths
    project_root = Path(__file__).parent.parent
    data_path = project_root / f"data/processed/{dataset_name}/{dataset_name}.inter"
    output_path = project_root / f"eval_user_sample_{n_samples//1000}k_{dataset_name}.json"
    
    print(f"Loading dataset: {dataset_name}")
    print(f"Data path: {data_path}")
    
    # Load dataset
    dataset = RecDataset(str(data_path), seed=seed)
    
    # Get all test users
    test_users = list(dataset.test_data.keys())
    n_test_users = len(test_users)
    
    print(f"\nDataset statistics:")
    print(f"  Total test users: {n_test_users}")
    print(f"  Sampling: {n_samples} users")
    
    # Sample users
    if n_samples >= n_test_users:
        print(f"  Warning: Requested {n_samples} users, but only {n_test_users} available.")
        print(f"  Using all {n_test_users} test users.")
        sampled_users = test_users
    else:
        rng = np.random.RandomState(seed)
        sampled_users = sorted(rng.choice(test_users, size=n_samples, replace=False).tolist())
    
    # Save to JSON
    output_data = {
        "dataset": dataset_name,
        "n_samples": len(sampled_users),
        "n_total_test_users": n_test_users,
        "seed": seed,
        "user_ids": sampled_users
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✓ Saved sampled user list to: {output_path}")
    print(f"  Sampled {len(sampled_users)} users from {n_test_users} test users")
    print(f"  User ID range: [{min(sampled_users)}, {max(sampled_users)}]")
    
    return output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate a fixed sampled user list for experiments")
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Dataset name (e.g., instructrec-books, instructrec-yelp)'
    )
    parser.add_argument(
        '--n',
        type=int,
        default=1000,
        help='Number of users to sample (default: 1000)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Generate user sample
    generate_user_sample(args.dataset, n_samples=args.n, seed=args.seed)
    
    print("\n" + "="*60)
    print("✓ User sample generation complete!")
    print("="*60)

