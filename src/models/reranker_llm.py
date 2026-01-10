"""
LLM-based Reranker for Stage-ReRank
Scores candidate items based on facets/vector_profile output from Stage-R
"""
import json
from typing import Dict, List


class LLMReranker:
    """Use LLM to rerank candidate items"""
    
    def __init__(self, llm_client):
        """
        Initialize LLM Reranker
        
        Args:
            llm_client: LLMClient instance
        """
        self.llm = llm_client
    
    def build_rerank_prompt(
        self,
        user_id: int,
        facets: List[Dict],
        candidates: List[Dict],  # [{'id': int, 'title': str, 'tags': [...]}]
        item_mems: Dict[int, Dict] = None,  # {item_id: ItemMem}
        instruction: str = None,  # User instruction (like iAgent)
        vanilla_mode: bool = False  # Vanilla mode: no memory, only item descriptions
    ) -> List[Dict[str, str]]:
        """
        Build reranking prompt
        
        Args:
            user_id: User ID
            facets: Facets from Stage-R output (empty in vanilla mode)
            candidates: List of candidate items (with metadata)
            item_mems: Item memories (optional, empty in vanilla mode)
            instruction: User instruction/intent (optional, from dataset)
            vanilla_mode: Whether vanilla mode (no memory)
            
        Returns:
            List of messages
        """
        # Build paper-friendly single prompt
        if vanilla_mode:
            # Vanilla mode: no memory, only user instruction and item descriptions
            prompt_parts = [
                f"You are an intelligent recommendation scoring system. Your task is to evaluate how well each candidate item matches the target user's preferences.",
                f"\n**Target User:** User {user_id}"
            ]
            
            # Add user instruction/persona if available
            if instruction:
                prompt_parts.append(f"\n**User Profile:**\n{instruction}")
            else:
                prompt_parts.append(f"\n**User Profile:**\nNo specific user profile provided.")
            
            # Format candidate items with descriptions
            prompt_parts.append("\n**Candidate Items:**")
            for c in candidates:
                cid = c['id']
                title = c.get('title', f'Item {cid}')
                description = c.get('description', '')
                tags = c.get('tags', [])
                if description:
                    # Truncate long descriptions
                    if len(description) > 200:
                        description = description[:200] + "..."
                    prompt_parts.append(f"  • Item {cid}: {title}. {description}")
                elif tags:
                    tags_str = ", ".join(tags[:5])  # Limit to 5 tags
                    prompt_parts.append(f"  • Item {cid}: {title} (Tags: {tags_str})")
                else:
                    prompt_parts.append(f"  • Item {cid}: {title}")
            
            # Task description for vanilla mode
            prompt_parts.append("""
**Your Task:**
For each of the candidate items listed above, provide a relevance score between 0 and 1 that indicates how well the item matches the user's profile:
  • 1.0 = Excellent match, highly aligned with user's preferences
  • 0.5 = Moderate match, partially relevant
  • 0.0 = Poor match, not aligned with user's interests

For each item, provide a brief rationale explaining your scoring decision based on the user's profile and item characteristics.

**Expected Output Format:**
Your response should be a JSON object with a single field:
- "scores": An array of scoring objects, each containing:
  * "item_id": The item's ID (integer)
  * "score": Your relevance score between 0 and 1 (number)
  * "rationale": A brief explanation of your scoring (string)
""")
        else:
            # MemRec mode: with memory and facets
            prompt_parts = [
                f"You are an intelligent recommendation scoring system. Your task is to evaluate how well each candidate item matches the target user's preferences based on their personal memory and collaborative signals.",
                f"\n**Target User:** User {user_id}"
            ]
            
            # Add user instruction if available
            if instruction:
                prompt_parts.append(f"\n**User's Current Request:**\n{instruction}")
            
            # Format preference patterns (extracted from collaborative memories)
            prompt_parts.append("\n**User Preferences (Extracted from Collaborative Memories):**")
            prompt_parts.append("Based on collaborative signals from neighboring users and items, we have identified the following preference patterns:")
            if facets:
                for i, f in enumerate(facets[:10], 1):
                    facet_text = f.get('facet', f.get('text', 'N/A'))
                    conf = f.get('confidence', 0)
                    prompt_parts.append(f"  {i}. {facet_text} (confidence: {conf:.2f})")
            else:
                prompt_parts.append("  (No facets extracted)")
            
            # Format Item memories
            prompt_parts.append("\n**Candidate Item Memories:**")
            for c in candidates:
                cid = c['id']
                title = c.get('title', f'Item {cid}')
                memory = ""
                if item_mems and cid in item_mems:
                    memory = item_mems[cid]
                    if len(memory) > 150:
                        memory = memory[:150] + "..."
                else:
                    memory = "(No memory recorded)"
                prompt_parts.append(f"  • Item {cid} ({title}): {memory}")
            
            # Task description for MemRec mode
            prompt_parts.append("""
**Your Task:**
For each of the candidate items listed above, provide a relevance score between 0 and 1 that indicates how well the item matches the user's preferences:
  • 1.0 = Excellent match, highly aligned with user's facets and memory
  • 0.5 = Moderate match, partially relevant
  • 0.0 = Poor match, not aligned with user's interests

For each item, provide a brief rationale explaining your scoring decision based on the user's preference facets and personal memory.

**Expected Output Format:**
Your response should be a JSON object with a single field:
- "scores": An array of scoring objects, each containing:
  * "item_id": The item's ID (integer)
  * "score": Your relevance score between 0 and 1 (number)
  * "rationale": A brief explanation of your scoring (string)
""")
        
        prompt = "".join(prompt_parts)
        
        # Single message format
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        return messages
    
    def get_rerank_schema(self) -> Dict:
        """Get reranking JSON schema"""
        return {
            "scores": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "item_id": {"type": "integer"},
                        "score": {"type": "number"},
                        "rationale": {"type": "string"}
                    },
                    "required": ["item_id", "score", "rationale"],
                    "additionalProperties": False
                }
            }
        }
    
    def rerank(
        self,
        user_id: int,
        retrieval_bundle: Dict,  # Stage-R output (empty in vanilla mode)
        candidates: List[Dict],  # [{'id', 'title', 'tags', ...}]
        item_mems: Dict[int, Dict] = None,
        instruction: str = None,  # User instruction (like iAgent)
        temperature: float = 0.0,
        max_tokens: int = 4000,
        debug_logger = None,
        vanilla_mode: bool = False  # Vanilla mode: no memory
    ) -> List[Dict]:
        """
        Rerank candidate items
        
        Args:
            user_id: User ID
            retrieval_bundle: Stage-R output {'facets', 'vector_profile', 'support_edges'} (empty in vanilla mode)
            candidates: List of candidate items
            item_mems: Item memories (optional, empty in vanilla mode)
            instruction: User instruction/intent (optional)
            temperature: LLM temperature
            max_tokens: Maximum tokens
            debug_logger: Debug logger
            vanilla_mode: Whether vanilla mode (no memory)
            
        Returns:
            [{'item_id': int, 'score': float, 'rationale': str}, ...]
        """
        # Build prompt
        messages = self.build_rerank_prompt(
            user_id=user_id,
            facets=retrieval_bundle.get('facets', []) if not vanilla_mode else [],
            candidates=candidates,
            item_mems=item_mems if not vanilla_mode else {},
            instruction=instruction,
            vanilla_mode=vanilla_mode
        )
        
        # Get schema
        properties = self.get_rerank_schema()
        
        # Call LLM (use stageRR cache)
        try:
            response = self.llm.generate_json(
                messages=messages,
                properties=properties,
                temperature=temperature,
                max_tokens=max_tokens,
                debug_logger=debug_logger
            )
            
            scores = response.get('scores', [])
            
            # Debug log: record scoring results
            if debug_logger and scores:
                debug_logger(f"\n📊 Rerank Scores (Stage-ReRank):")
                for i, score_entry in enumerate(scores[:10], 1):
                    item_id = score_entry.get('item_id')
                    score = score_entry.get('score', 0)
                    rationale = score_entry.get('rationale', 'N/A')[:80]
                    debug_logger(f"  [{i}] Item {item_id}: {score:.3f} - {rationale}")
                debug_logger("")  # Empty line separator
            
            return scores
        except Exception as e:
            print(f"Error in LLM Reranker for user {user_id}: {e}")
            # Return default scores
            return [{'item_id': c['id'], 'score': 0.5, 'rationale': 'Error'} for c in candidates]

