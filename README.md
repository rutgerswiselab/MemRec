# MemRec: Collaborative Memory-Augmented Agentic Recommender System

MemRec is a memory-augmented intelligent recommender system that achieves efficient personalized recommendations through collaborative memory mechanisms and large language models.

## 📁 Project Structure

```
memrec/
├── configs/             # Experiment configurations
├── scripts/             # Run scripts (train, eval, data processing)
└── src/
    ├── memory/          # Memory mechanisms (Storage, Pruner, Graph)
    ├── models/          # MemRec Agent & LLM Clients
    ├── train/           # Trainer & Metrics
    └── data/            # Dataset loaders & Samplers
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create conda environment
conda create -n memrec python=3.10
conda activate memrec

# Install dependencies
pip install -r requirements.txt
```

**Requirements:**
- Python 3.10+
- PyTorch 2.9.0+
- CUDA 12.1+ (recommended for accelerating candidate retrieval models)
- LLM API support (Azure OpenAI, local vLLM, etc.)

### 2. Configure API Keys

MemRec requires LLM API access. Set environment variables:

```bash
# Azure OpenAI (recommended)
export AZURE_OPENAI_ENDPOINT="https://your-endpoint.openai.azure.com/"
export AZURE_OPENAI_API_KEY="your-api-key"
```

### 3. Download Datasets

Download the InstructRec datasets published by [iAgent](https://github.com/agiresearch/iAgent):

**📦 Google Drive Link:** [InstructRec Datasets](https://drive.google.com/drive/folders/1-3kHU9D4IH210kSYL-m2cCgWbcY5ilBI?usp=sharing)

After downloading, extract the datasets to the `data/iagent/` directory:

```bash
# Create iagent directory if it doesn't exist
mkdir -p data/iagent

# Extract datasets to data/iagent/ directory
# Place all downloaded files (*.pkl and *.csv) into data/iagent/

# Convert all InstructRec datasets from iAgent format to MemRec format
bash scripts/convert_all_instructrec.sh

# Verify processed datasets
ls data/processed/
# Should see: instructrec-books, instructrec-goodreads, instructrec-movietv, instructrec-yelp
```

**Supported Datasets:**
- **instructrec-books**: Book recommendations
- **instructrec-goodreads**: Goodreads books
- **instructrec-movietv**: Movie and TV recommendations
- **instructrec-yelp**: Yelp business recommendations

### 4. Run MemRec

#### Basic Usage

```bash
python scripts/run_train.py \
  --model memrec_agent \
  --dataset instructrec-books \
  --config configs/memrec_instructrec-books.yaml \
  --device cuda:0
```

#### Custom Configuration

```bash
python scripts/run_train.py \
  --model memrec_agent \
  --dataset instructrec-books \
  --config configs/memrec_instructrec-books.yaml \
  --device cuda:0 \
  --n_eval_users 100 \        # Number of evaluation users
  --n_eval_candidates 10 \    # Number of candidate items
  --parallel \                # Enable parallel evaluation
  --parallel_workers 8        # Number of parallel workers
```
### 5. View Results

```bash
# View detailed results
cat results/runs/instructrec-books_memrec_agent_seed42_*.json | python -m json.tool

# View LLM conversation logs (if --save_llm_conversations was enabled)
ls results/runs/*/llm_conversations/
```

## LLM Configuration

```yaml
provider:
  name: azure_openai         # azure_openai, qwen, llama, etc.
  model: gpt-4o-mini
  endpoint: ${ENV:AZURE_OPENAI_ENDPOINT}
  api_key: ${ENV:AZURE_OPENAI_API_KEY}
```

## 📊 Evaluation Metrics

MemRec uses the following metrics for evaluation (default K ∈ {1, 3, 5, 10}):

- **Hit@K**: Whether the target item is in the Top-K
- **NDCG@K**: Normalized Discounted Cumulative Gain, considering ranking positions

## 🧠 Core Modules

### 1. Memory Manager

Manages user and item memories:
- Dynamic memory content updates
- Cross-user knowledge sharing support
- Automatic pruning of expired or low-quality memories

```python
from src.memory.manager import MemoryManager

memory_manager = MemoryManager(config)
memory_manager.warmup(train_data)  # Warm-up phase
recommendations = memory_manager.recommend(user_id, candidates)
```

### 2. Memory Pruner

Selects the most relevant memories for context construction:
- **llm_rules**: Uses LLM-generated domain rules
- **hybrid_rule**: Feature-weighted hybrid rules

```python
from src.memory.pruner import MemoryPruner

pruner = MemoryPruner(mode='llm_rules')
selected_memories = pruner.prune(candidate_memories, target_user, budget)
```

### 3. LLM Client

Unified LLM interface supporting multiple providers:

```python
from src.models.llm_client import LLMClient

llm_client = LLMClient(provider='azure_openai', model='gpt-4o-mini')
response = llm_client.generate(prompt, max_tokens=4000)
```

### 4. Reranker Module

Performs precise ranking of candidate items:
- **LLM Reranker**: Uses LLM to understand reasons and rank
- **Vector Reranker**: Fast ranking based on vector similarity

```python
from src.models.reranker_llm import LLMReranker

reranker = LLMReranker(llm_client)
ranked_items = reranker.rerank(user_profile, candidates, reasons)
```

## 🔬 Advanced Usage

### Custom Domain Rules

Add new domain rule files in `src/memory/domain_rules/`:

```python
# src/memory/domain_rules/custom_rules.py
def get_custom_rules():
    return {
        'user_preference': 'weight=0.8',
        'item_quality': 'weight=0.7',
        'recency': 'weight=0.6',
        # Add more rules...
    }
```

### Parallel Evaluation Optimization

Increase parallel workers to accelerate evaluation:

```bash
python scripts/run_train.py \
  --model memrec_agent \
  --dataset instructrec-books \
  --config configs/memrec_instructrec-books.yaml \
  --parallel \
  --parallel_workers 32  # Adjust based on CPU cores
```

## ❓ FAQ

### Q1: LLM API Call Failure

**Solution:**
- Check if environment variables are correctly set
- Verify API key validity
- Check network connection and API quota

```bash
# Verify environment variables
echo $AZURE_OPENAI_ENDPOINT
echo $AZURE_OPENAI_API_KEY
```

### Q2: Slow Evaluation Speed

**Solution:**
- Use vector reranker: set `reranker_mode: vector` in config
- Reduce evaluation users: `--n_eval_users 100`
- Increase parallel threads: `--parallel --parallel_workers 16`

### Q3: How to Reproduce Paper Results

Ensure using the same configuration:
```bash
# For full evaluation (all test users)
python scripts/run_train.py \
  --model memrec_agent \
  --dataset instructrec-books \
  --config configs/memrec_instructrec-books.yaml \
  --seed 42

# For 1k sampled users evaluation (for reproducibility)
# Note: The config file already specifies eval_user_list: eval_user_sample_1k_instructrec-books.json
# Make sure the JSON file is in the project root directory
python scripts/run_train.py \
  --model memrec_agent \
  --dataset instructrec-books \
  --config configs/memrec_instructrec-books_1k.yaml \
  --seed 42
```

**Note for 1k evaluation:** The `memrec_instructrec-books_1k.yaml` config file already includes `eval_user_list: eval_user_sample_1k_instructrec-books.json`. Ensure this JSON file is placed in the project root directory (same level as `scripts/` and `configs/`).

### Q4: Custom Dataset

1. Prepare data in unified format (user_id, item_id, rating, timestamp)
2. Map IDs to 0-based integers
3. Save as `.inter` file to `data/processed/your-dataset/`
4. Copy and modify configuration file
5. Run training script
