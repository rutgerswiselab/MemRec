"""
Budgeted snippet packer
Greedily select most valuable neighbor snippets under token budget constraint
"""
from typing import Dict, List, Tuple
import json


class SnippetPacker:
    """Greedy packer based on token budget"""
    
    def __init__(self, tau: int = 1800):
        """
        Initialize packer
        
        Args:
            tau: Token budget (default 1800, leaving enough space for facets + candidate_notes)
        """
        self.tau = tau
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text
        Simple heuristic: 1 token ≈ 4 characters (accurate for English)
        
        Args:
            text: Input text
            
        Returns:
            Estimated token count
        """
        return len(text) // 4
    
    def build_neighbor_snippet(
        self, 
        neighbor: Dict,
        dataset,
        max_len: int = 200
    ) -> str:
        """
        Build text snippet for a neighbor
        
        Args:
            neighbor: Neighbor dict {'type': 'item'/'user', 'id': int, 'score': float, ...}
            dataset: RecDataset instance (for retrieving metadata)
            max_len: Maximum snippet length (characters)
            
        Returns:
            Formatted snippet string
        """
        neighbor_type = neighbor['type']
        neighbor_id = neighbor['id']
        score = neighbor['score']
        
        if neighbor_type == 'item':
            # Item snippet: title + description (truncated)
            if dataset.item_metadata and neighbor_id in dataset.item_metadata:
                meta = dataset.item_metadata[neighbor_id]
                title = meta.get('title', f'Item-{neighbor_id}')
                desc = meta.get('description', '')
                
                # Truncate
                title = title[:80] if len(title) > 80 else title
                desc = desc[:120] if len(desc) > 120 else desc
                
                snippet = f"[Item-{neighbor_id}] {title}"
                if desc:
                    snippet += f" | {desc}"
                snippet += f" (score={score:.3f})"
            else:
                snippet = f"[Item-{neighbor_id}] (score={score:.3f})"
        
        else:  # user
            # User snippet: simple description + their popular items
            snippet = f"[User-{neighbor_id}] (overlap_score={score:.3f})"
            
            # Optional: list this user's most recent 2-3 items
            if hasattr(dataset, 'train_data') and neighbor_id in dataset.train_data:
                user_items = dataset.train_data[neighbor_id]
                recent_items = user_items[-3:] if len(user_items) >= 3 else user_items
                if recent_items and dataset.item_metadata:
                    item_titles = []
                    for iid in recent_items:
                        if iid in dataset.item_metadata:
                            title = dataset.item_metadata[iid].get('title', f'Item-{iid}')
                            item_titles.append(title[:40])
                    if item_titles:
                        snippet += f" - Recent: {', '.join(item_titles)}"
        
        return snippet[:max_len]
    
    def pack(
        self,
        pruned_subgraph: Dict,
        dataset,
        candidates: List[int] = None,
        user_memory_summary: str = ""
    ) -> Dict:
        """
        Pack neighbor snippets within token budget
        
        Args:
            pruned_subgraph: Pruned subgraph (from pruner)
            dataset: RecDataset instance
            candidates: List of candidate items
            user_memory_summary: User memory summary (from storage)
            
        Returns:
            {
                'context_text': str,  # Complete context string
                'neighbors_text': str,  # Neighbor table/list
                'candidates_text': str,  # Candidate list
                'memory_text': str,  # Memory summary
                'n_neighbors': int,
                'estimated_tokens': int
            }
        """
        neighbors = pruned_subgraph['neighbors']
        user_id = pruned_subgraph['user_id']
        
        # 1. Build neighbor snippets
        neighbor_snippets = []
        for neighbor in neighbors:
            snippet = self.build_neighbor_snippet(neighbor, dataset)
            estimated = self.estimate_tokens(snippet)
            neighbor_snippets.append({
                'text': snippet,
                'tokens': estimated,
                'score': neighbor['score']
            })
        
        # 2. Greedy selection: sort by score, add incrementally until budget exceeded
        neighbor_snippets.sort(key=lambda x: x['score'], reverse=True)
        
        selected_snippets = []
        current_tokens = 0
        
        # Reserve space for other parts
        reserve_for_candidates = 300 if candidates else 0
        reserve_for_memory = 200 if user_memory_summary else 0
        reserve_for_output = 600  # Reserve for facets + candidate_notes output
        available_budget = self.tau - reserve_for_candidates - reserve_for_memory - reserve_for_output
        
        for snippet_info in neighbor_snippets:
            if current_tokens + snippet_info['tokens'] <= available_budget:
                selected_snippets.append(snippet_info['text'])
                current_tokens += snippet_info['tokens']
            else:
                break
        
        # 3. Build neighbors text (table format)
        if selected_snippets:
            neighbors_text = "**Collaborative Neighbors:**\n" + "\n".join(
                f"{i+1}. {s}" for i, s in enumerate(selected_snippets)
            )
        else:
            neighbors_text = "**Collaborative Neighbors:** (none available)"
        
        # 4. Build candidates text
        candidates_text = ""
        if candidates:
            cand_items = []
            for cand_id in candidates[:10]:  # Limit to 10
                if dataset.item_metadata and cand_id in dataset.item_metadata:
                    meta = dataset.item_metadata[cand_id]
                    title = meta.get('title', f'Item-{cand_id}')
                    cand_items.append(f"[{cand_id}] {title[:60]}")
                else:
                    cand_items.append(f"[{cand_id}]")
            candidates_text = "**Candidates to Rank:**\n" + "\n".join(
                f"{i+1}. {c}" for i, c in enumerate(cand_items)
            )
        
        # 5. Build memory text
        memory_text = ""
        if user_memory_summary:
            memory_text = f"**User Memory Summary:**\n{user_memory_summary[:200]}"
        
        # 6. Assemble complete context
        context_parts = []
        if memory_text:
            context_parts.append(memory_text)
        context_parts.append(neighbors_text)
        if candidates_text:
            context_parts.append(candidates_text)
        
        context_text = "\n\n".join(context_parts)
        
        # 7. Estimate total tokens
        estimated_tokens = self.estimate_tokens(context_text)
        
        return {
            'context_text': context_text,
            'neighbors_text': neighbors_text,
            'candidates_text': candidates_text,
            'memory_text': memory_text,
            'n_neighbors': len(selected_snippets),
            'estimated_tokens': estimated_tokens
        }
