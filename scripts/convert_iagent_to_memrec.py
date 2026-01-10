#!/usr/bin/env python3
"""
Convert iAgent (InstructRec) dataset to MemRec format

Usage:
    python scripts/convert_iagent_to_memrec.py \
        --input /path/to/booksAll_recagent.pkl \
        --mapping /path/to/combined_books_asin_mapping.csv \
        --output data/processed/instructrec-books \
        --domain books \
        --mode full
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from rich.console import Console
from rich.table import Table

console = Console()


def load_iagent_data(pkl_path, mapping_path):
    """Load iAgent data"""
    console.print(f"[cyan]Loading iAgent data from:[/cyan] {pkl_path}")
    
    # Load main data
    df = pd.read_pickle(pkl_path)
    
    # Load mapping
    console.print(f"[cyan]Loading mapping from:[/cyan] {mapping_path}")
    mapping = pd.read_csv(mapping_path)
    
    console.print(f"[green]✓[/green] Loaded successfully!")
    console.print(f"  Samples: {len(df)}")
    console.print(f"  Products in mapping: {len(mapping)}")
    
    return df, mapping


def convert_to_basic_format(df, output_dir, dataset_name):
    """
    Convert to basic format (ratings only)
    
    Output: user_id, item_id, rating, timestamp
    """
    console.print("\n[bold cyan]Converting to basic format (ratings only)...[/bold cyan]")
    
    interactions = []
    user_id_counter = 0
    
    for idx, row in df.iterrows():
        asin_list = row['asin']  # Product ID list
        
        # Each product as one interaction
        for position, item_id in enumerate(asin_list):
            interactions.append({
                'user_id': user_id_counter,
                'item_id': item_id,
                'rating': 5.0,  # Implicit positive feedback
                'timestamp': position + 1  # Sequence position as time
            })
        
        user_id_counter += 1
        
        if (idx + 1) % 1000 == 0:
            console.print(f"  Processed {idx + 1}/{len(df)} users...")
    
    # Save - use dataset_name as filename
    inter_df = pd.DataFrame(interactions)
    output_path = output_dir / f'{dataset_name}.inter'
    inter_df.to_csv(output_path, sep='\t', index=False)
    
    console.print(f"\n[green]✓[/green] Saved to: {output_path}")
    console.print(f"  Users: {inter_df['user_id'].nunique():,}")
    console.print(f"  Items: {inter_df['item_id'].nunique():,}")
    console.print(f"  Interactions: {len(inter_df):,}")
    
    return inter_df


def convert_to_full_format(df, mapping, output_dir, dataset_name, source_pkl_path=None):
    """
    Convert to full format (with text)
    
    Output:
        - .inter: basic interaction data
        - .text: review text
        - .meta: product metadata
        - .instruction: user instructions
        - {domain}All_recagent.pkl: original pkl file (for ranked_lists)
    """
    console.print("\n[bold cyan]Converting to full format (with text)...[/bold cyan]")
    
    # 0. Copy original pkl file (keep ranked_lists etc. original data)
    if source_pkl_path:
        import shutil
        console.print(f"\n[cyan]Copying original pkl file...[/cyan]")
        pkl_filename = Path(source_pkl_path).name
        dest_pkl_path = output_dir / pkl_filename
        shutil.copy2(source_pkl_path, dest_pkl_path)
        console.print(f"[green]✓[/green] Copied to: {dest_pkl_path}")
    
    # 1. Basic interaction data
    inter_df = convert_to_basic_format(df, output_dir, dataset_name)
    
    # 2. Review text
    console.print("\n[cyan]Extracting review texts...[/cyan]")
    reviews = []
    user_id_counter = 0
    
    for idx, row in df.iterrows():
        asin_list = row['asin']
        review_list = row.get('reviewText', [None] * len(asin_list))
        
        for item_id, review_text in zip(asin_list, review_list):
            if review_text is not None:
                reviews.append({
                    'user_id': user_id_counter,
                    'item_id': item_id,
                    'review_text': str(review_text)
                })
        
        user_id_counter += 1
        
        if (idx + 1) % 1000 == 0:
            console.print(f"  Processed {idx + 1}/{len(df)} users...")
    
    review_df = pd.DataFrame(reviews)
    review_path = output_dir / f'{dataset_name}.text'
    review_df.to_csv(review_path, sep='\t', index=False)
    console.print(f"[green]✓[/green] Saved reviews to: {review_path}")
    console.print(f"  Total reviews: {len(review_df):,}")
    
    # 3. Product metadata
    console.print("\n[cyan]Extracting item metadata...[/cyan]")
    
    # Get from mapping file
    meta_df = mapping.copy()
    
    # Unify column names
    if 'index' in meta_df.columns:
        meta_df = meta_df.rename(columns={'index': 'item_id'})
    
    # Ensure required columns
    required_cols = ['item_id', 'asin', 'title']
    if not all(col in meta_df.columns for col in required_cols):
        console.print("[yellow]Warning:[/yellow] Missing required columns in mapping file")
    
    # Add description (if not exists, use empty string)
    if 'description' not in meta_df.columns:
        meta_df['description'] = ""
    
    meta_path = output_dir / f'{dataset_name}.meta'
    meta_df.to_csv(meta_path, sep='\t', index=False)
    console.print(f"[green]✓[/green] Saved metadata to: {meta_path}")
    console.print(f"  Total items: {len(meta_df):,}")
    
    # 4. User instructions
    console.print("\n[cyan]Extracting user instructions...[/cyan]")
    
    instructions = []
    user_id_counter = 0  # Use counter to ensure consistency with .inter file
    for idx, row in df.iterrows():
        instructions.append({
            'user_id': user_id_counter,  # FIX: Use counter instead of idx
            'instruction': str(row.get('instruction', '')),
            'persona': str(row.get('persona', ''))
        })
        
        user_id_counter += 1
        
        if (user_id_counter) % 1000 == 0:
            console.print(f"  Processed {user_id_counter}/{len(df)} users...")
    
    inst_df = pd.DataFrame(instructions)
    inst_path = output_dir / f'{dataset_name}.instruction'
    inst_df.to_csv(inst_path, sep='\t', index=False)
    console.print(f"[green]✓[/green] Saved instructions to: {inst_path}")
    console.print(f"  Total instructions: {len(inst_df):,}")
    
    return inter_df, review_df, meta_df, inst_df


def generate_statistics(df, output_dir):
    """Generate data statistics"""
    console.print("\n[bold cyan]═══════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]Dataset Statistics[/bold cyan]")
    console.print("[bold cyan]═══════════════════════════════════════[/bold cyan]")
    
    n_users = df['user_id'].nunique()
    n_items = df['item_id'].nunique()
    n_interactions = len(df)
    density = n_interactions / (n_users * n_items)
    avg_items_per_user = n_interactions / n_users
    avg_users_per_item = n_interactions / n_items
    
    # Create statistics table
    table = Table(title="")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="green", justify="right")
    
    table.add_row("Users", f"{n_users:,}")
    table.add_row("Items", f"{n_items:,}")
    table.add_row("Interactions", f"{n_interactions:,}")
    table.add_row("Density", f"{density:.6f}")
    table.add_row("Avg items/user", f"{avg_items_per_user:.2f}")
    table.add_row("Avg users/item", f"{avg_users_per_item:.2f}")
    
    console.print(table)
    
    # Save statistics
    stats_path = output_dir / 'statistics.txt'
    with open(stats_path, 'w') as f:
        f.write("Dataset Statistics\n")
        f.write("=" * 50 + "\n")
        f.write(f"Users: {n_users:,}\n")
        f.write(f"Items: {n_items:,}\n")
        f.write(f"Interactions: {n_interactions:,}\n")
        f.write(f"Density: {density:.6f}\n")
        f.write(f"Avg items/user: {avg_items_per_user:.2f}\n")
        f.write(f"Avg users/item: {avg_users_per_item:.2f}\n")
    
    console.print(f"\n[green]✓[/green] Statistics saved to: {stats_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert iAgent dataset to MemRec format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic conversion (ratings only)
  python scripts/convert_iagent_to_memrec.py \\
      --input /path/to/booksAll_recagent.pkl \\
      --mapping /path/to/combined_books_asin_mapping.csv \\
      --output data/processed/iagent-books \\
      --mode basic
  
  # Full conversion (with text)
  python scripts/convert_iagent_to_memrec.py \\
      --input /path/to/booksAll_recagent.pkl \\
      --mapping /path/to/combined_books_asin_mapping.csv \\
      --output data/processed/iagent-books \\
      --mode full

Output files:
  - iagent.inter: Basic interaction data (user_id, item_id, rating, timestamp)
  - iagent.text: Review texts (only in full mode)
  - iagent.meta: Item metadata (only in full mode)
  - iagent.instruction: User instructions (only in full mode)
  - statistics.txt: Dataset statistics
        """
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to iAgent .pkl file (e.g., booksAll_recagent.pkl)'
    )
    parser.add_argument(
        '--mapping',
        type=str,
        required=True,
        help='Path to mapping CSV file (e.g., combined_books_asin_mapping.csv)'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory for converted data'
    )
    parser.add_argument(
        '--domain',
        type=str,
        required=True,
        help='Domain name (books, movietv, goodreads, yelp)'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['basic', 'full'],
        default='full',
        help='Conversion mode: basic (ratings only) or full (with text)'
    )
    
    args = parser.parse_args()
    
    # Show configuration
    console.print("\n[bold cyan]════════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]iAgent to MemRec Converter[/bold cyan]")
    console.print("[bold cyan]════════════════════════════════════════[/bold cyan]")
    console.print(f"[cyan]Input file:[/cyan] {args.input}")
    console.print(f"[cyan]Mapping file:[/cyan] {args.mapping}")
    console.print(f"[cyan]Output directory:[/cyan] {args.output}")
    console.print(f"[cyan]Domain:[/cyan] {args.domain}")
    console.print(f"[cyan]Mode:[/cyan] {args.mode}")
    console.print()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate dataset name: instructrec-{domain}
    dataset_name = f"instructrec-{args.domain}"
    
    try:
        # Load data
        df, mapping = load_iagent_data(args.input, args.mapping)
        
        # Convert
        if args.mode == 'basic':
            inter_df = convert_to_basic_format(df, output_dir, dataset_name)
            generate_statistics(inter_df, output_dir)
        else:
            inter_df, review_df, meta_df, inst_df = convert_to_full_format(
                df, mapping, output_dir, dataset_name, 
                source_pkl_path=args.input  # Pass original pkl path
            )
            generate_statistics(inter_df, output_dir)
        
        console.print("\n[bold green]═══════════════════════════════════════[/bold green]")
        console.print("[bold green]✓ Conversion complete![/bold green]")
        console.print("[bold green]═══════════════════════════════════════[/bold green]")
        console.print(f"\n[cyan]Next steps:[/cyan]")
        console.print("  1. Check the generated files in the output directory")
        console.print("  2. Run MemRec training:")
        console.print(f"     python scripts/run_train.py \\")
        console.print(f"         --model lightgcn \\")
        console.print(f"         --dataset {Path(args.output).name} \\")
        console.print(f"         --config configs/lightgcn_iagent.yaml \\")
        console.print(f"         --device cuda:0")
        console.print()
        
    except Exception as e:
        console.print(f"\n[red]Error:[/red] {e}")
        raise


if __name__ == "__main__":
    main()

