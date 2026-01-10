#!/usr/bin/env python3
"""
Main training/evaluation script for MemRec Agent
"""
import argparse
import sys
import os
import json
import csv
from pathlib import Path
from datetime import datetime

import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import set_seed, load_config, get_device
from src.data import RecDataset
from src.train import MemRecTrainer


def save_results(config, valid_metrics, test_metrics, save_dir):
    """Save training results"""
    # Create results directory
    results_dir = Path(save_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save per-run JSON
    dataset_name = config['dataset']
    model_name = config['model']
    seed = config['seed']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    run_file = results_dir / f"{dataset_name}_{model_name}_seed{seed}_{timestamp}.json"
    
    results = {
        'config': config,
        'valid_metrics': valid_metrics,
        'test_metrics': test_metrics,
        'timestamp': timestamp
    }
    
    with open(run_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {run_file}")
    
    # Append to summary CSV
    summary_file = results_dir / "summary.csv"
    
    file_exists = summary_file.exists()
    
    # Define metric order based on config
    topk = config.get('topk', [1, 5, 10])
    metrics = config.get('metrics', ['Hit', 'NDCG'])
    metric_cols = [f'{metric}@{k}' for k in topk for metric in metrics]
    
    # Add MRR if it exists in test_metrics
    if 'MRR' in test_metrics:
        metric_cols.append('MRR')
    
    with open(summary_file, 'a', newline='') as f:
        # Add evaluation parameters and MemRec-specific fields
        fieldnames = ['timestamp', 'dataset', 'model', 'seed', 'n_eval_users', 'n_eval_candidates'] + metric_cols
        
        # Add MemRec-specific fields
        memrec_fields = ['pruner_type', 'k', 'tau', 'llm_model', 'n_facets', 'reranker_mode']
        fieldnames.extend(memrec_fields)
        
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        
        if not file_exists:
            writer.writeheader()
        
        # Determine pruner_type
        pruner_mode = config.get('memrec', {}).get('pruner', {}).get('mode', 'hybrid_rule')
        if pruner_mode == 'llm_rules':
            pruner_type = 'llm_generated_rules'
        elif pruner_mode == 'hybrid_rule':
            pruner_type = 'predefined_rules'
        elif pruner_mode == 'learned_mlp':
            pruner_type = 'learned_mlp'
        else:
            pruner_type = 'unknown'
        
        memrec_config = config.get('memrec', {})
        row = {
            'timestamp': timestamp,
            'dataset': dataset_name,
            'model': model_name,
            'seed': seed,
            'n_eval_users': config.get('n_eval_users', 'all'),
            'n_eval_candidates': config.get('n_eval_candidates', 100),
            **test_metrics,
            'pruner_type': pruner_type,
            'k': memrec_config.get('k', 16),
            'tau': memrec_config.get('tau_tokens', 1800),
            'llm_model': config.get('llm_model', 'gpt-4o-mini'),
            'n_facets': memrec_config.get('n_facets', 7),
            'reranker_mode': memrec_config.get('reranker_mode', 'llm')
        }
        
        writer.writerow(row)
    
    print(f"Summary appended to: {summary_file}")


def main():
    parser = argparse.ArgumentParser(description="Train/Evaluate MemRec Agent")
    
    parser.add_argument(
        '--model',
        type=str,
        default='memrec_agent',
        choices=['memrec_agent'],
        help='Model to run (currently only memrec_agent)'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Dataset name (e.g., instructrec-books, instructrec-yelp)'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default=None,
        help='Data directory (default: data/processed/<dataset>)'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to config YAML file'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='Device (cuda:0, cpu, etc.)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed (overrides config)'
    )
    parser.add_argument(
        '--n_eval_candidates',
        type=int,
        default=None,
        help='Number of candidates during evaluation (overrides config, default: 10)'
    )
    parser.add_argument(
        '--n_eval_users',
        type=int,
        default=None,
        help='Number of users to evaluate on (default: None = all users in eval set)'
    )
    parser.add_argument(
        '--max_workers',
        type=int,
        default=None,
        help='Number of parallel workers for LLM API calls (default: 8, recommended: 8-32)'
    )
    parser.add_argument(
        '--save_llm_conversations',
        action='store_true',
        help='Save all LLM conversations to JSONL file for debugging and analysis'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory for saving results (memory.jsonl, llm_conversations.jsonl, etc.)'
    )
    parser.add_argument(
        '--use_pregenerated_candidates',
        action='store_true',
        help='Use pre-generated ranked_lists from data (base recommender output)'
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        default=None,
        help='Directory to save models (default: results/runs/<dataset>_<model>)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode with detailed logging'
    )
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Enable parallel evaluation (faster, no debug output)'
    )
    parser.add_argument(
        '--parallel_workers',
        type=int,
        default=8,
        help='Number of parallel workers for evaluation (default: 8)'
    )
    parser.add_argument(
        '--debug_log_file',
        type=str,
        default=None,
        help='Path to debug log file (default: results/runs/<dataset>_<model>/debug.txt)'
    )
    # Hyperparameter overrides for MemRec
    parser.add_argument(
        '--neighbor_k',
        type=int,
        default=None,
        help='Override MemRec neighbor count k (config.memrec.k)'
    )
    parser.add_argument(
        '--n_facets',
        type=int,
        default=None,
        help='Override MemRec synthesis facet count N_f (config.memrec.n_facets)'
    )
    parser.add_argument(
        '--tau_tokens',
        type=int,
        default=None,
        help='Override MemRec token budget (config.memrec.tau_tokens)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    print("=" * 80)
    print("MemRec Agent Evaluation")
    print("=" * 80)
    print(f"\nLoading config from: {args.config}")
    config = load_config(args.config)
    
    # Ensure seed exists for downstream logging
    if 'seed' not in config:
        config['seed'] = args.seed if args.seed is not None else 42
    
    # Override config with command line arguments
    if args.seed is not None:
        config['seed'] = args.seed
    
    if args.n_eval_candidates is not None:
        config['n_eval_candidates'] = args.n_eval_candidates
    
    if args.n_eval_users is not None:
        config['n_eval_users'] = args.n_eval_users
    
    if args.max_workers is not None:
        config['max_workers'] = args.max_workers
    
    # Handle output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        config['output_dir'] = str(output_dir)
        
        # Set conversation file path if saving conversations
        if args.save_llm_conversations or config.get('save_llm_conversations', False):
            config['save_llm_conversations'] = True
            config['conversation_file'] = str(output_dir / 'llm_conversations.jsonl')
    else:
        # Use default run directory
        if args.save_llm_conversations:
            config['save_llm_conversations'] = True
            run_dir = f"results/runs/{args.dataset}_memrec_agent"
            os.makedirs(run_dir, exist_ok=True)
            config['conversation_file'] = f"{run_dir}/llm_conversations.jsonl"
    
    # Override use_pregenerated_candidates if flag is set
    if args.use_pregenerated_candidates:
        config['use_pregenerated_candidates'] = True
    
    # Override debug settings if flag is set
    if args.debug:
        config['debug'] = True
        if args.debug_log_file:
            config['debug_log_file'] = args.debug_log_file
    
    config['model'] = 'memrec_agent'
    config['dataset'] = args.dataset
    
    # Override MemRec hyperparameters if provided
    memrec_cfg = config.get('memrec', {})
    if args.neighbor_k is not None:
        memrec_cfg['k'] = args.neighbor_k
    if args.n_facets is not None:
        memrec_cfg['n_facets'] = args.n_facets
    if args.tau_tokens is not None:
        memrec_cfg['tau_tokens'] = args.tau_tokens
    config['memrec'] = memrec_cfg
    
    # Set random seed
    seed = config.get('seed', 42)
    set_seed(seed)
    print(f"Random seed set to: {seed}")
    
    # Get device
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    # Load dataset
    if args.data_dir is None:
        data_dir = PROJECT_ROOT / "data" / "processed" / args.dataset
    else:
        data_dir = Path(args.data_dir)
    
    data_file = data_dir / f"{args.dataset}.inter"
    
    if not data_file.exists():
        print(f"\nError: Data file not found: {data_file}")
        print("Please prepare the data first or download from Google Drive.")
        print("See README.md for instructions.")
        sys.exit(1)
    
    print(f"\nLoading dataset from: {data_file}")
    dataset = RecDataset(str(data_file), seed=seed)
    print(dataset)
    
    stats = dataset.get_stats()
    print(f"\nDataset statistics:")
    print(f"  Users: {stats['n_users']}")
    print(f"  Items: {stats['n_items']}")
    print(f"  Interactions: {stats['n_interactions']}")
    print(f"  Train users: {stats['n_train_users']}")
    print(f"  Train interactions: {stats['n_train_interactions']}")
    print(f"  Density: {stats['density']:.6f}")
    
    # Create save directory
    if args.output_dir:
        save_dir = Path(args.output_dir)
    elif args.save_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_dir = PROJECT_ROOT / "results" / f"{args.dataset}_memrec_agent_seed{seed}_{timestamp}"
    else:
        save_dir = Path(args.save_dir)
    
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nResults will be saved to: {save_dir}")
    
    # Create MemRec trainer
    print(f"\nInitializing MemRec Agent...")
    trainer = MemRecTrainer(None, dataset, config, device)
    
    # Train (warmup + evaluation)
    best_state = trainer.train(save_dir=str(save_dir))
    
    # Test
    test_metrics = trainer.test(parallel=args.parallel, n_workers=args.parallel_workers)
    
    # Save results
    valid_metrics = best_state.get('metrics', {})
    save_results(config, valid_metrics, test_metrics, PROJECT_ROOT / "results")
    
    print("\n" + "=" * 80)
    print("Evaluation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
