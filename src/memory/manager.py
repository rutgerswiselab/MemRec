"""
MemRec Manager: Two-stage LM calls (Stage-R and Stage-W)
Responsible for building prompts, defining JSON schemas, calling LLM, and validating responses
"""
import json
from typing import Dict, List, Optional


class MemRecManager:
    """MemRec two-stage manager"""
    
    def __init__(self, llm_client):
        """
        Initialize manager
        
        Args:
            llm_client: LLMClient instance
        """
        self.llm = llm_client
    
    # ========================================================================
    # Stage-R: Retrieval (generate facets, evidence, vector_profile, candidate_notes)
    # ========================================================================
    
    def build_stage_r_prompt(
        self,
        user_id: int,
        user_mem_bullets: str,
        neighbor_table_json: str,
        candidates_json: str = "",
        n_facets: int = 7
    ) -> List[Dict[str, str]]:
        """
        Build Stage-R prompt (paper-friendly version)
        
        Args:
            user_id: User ID
            user_mem_bullets: User memory summary (bullet points)
            neighbor_table_json: Neighbor table (JSON format)
            candidates_json: Candidate items (optional, JSON format)
            n_facets: Number of facets requested
            
        Returns:
            List of messages
        """
        # Merge system and user message into one natural language prompt
        prompt = f"""You are an intelligent memory retrieval system for personalized recommendation. Your task is to analyze the user's personal memory and collaborative memories from their neighbors to extract preference facets.

**Target User:** User {user_id}

**User's Personal Memory:**
{user_mem_bullets if user_mem_bullets else "(No personal memory recorded yet for this user)"}

**Collaborative Neighbor Memories:**
The following neighboring users and items provide collaborative signals for understanding this user's preferences:

{neighbor_table_json}
"""
        
        if candidates_json:
            prompt += f"""
**Context (Candidate Items):**
{candidates_json}
(Note: These candidates are for context only, do not score them)
"""
        
        prompt += f"""

**Your Task:**
Analyze the user's personal memory and the collaborative memories from neighboring users and items to identify {n_facets} distinct preference facets that characterize this user's interests and tastes.

For each preference facet, provide:
1. A concise natural language description of the preference (e.g., "interest in mystery novels with strong female protagonists")
2. A confidence score between 0 and 1 indicating how strongly this facet is supported by the evidence
3. A list of supporting neighbors (user IDs or item IDs) that provide evidence for this facet

Additionally, identify the collaborative edges between neighboring users/items and the target user, with edge weights (0-1) indicating the strength of collaborative signal.

**Expected Output Format:**
Your response should be a JSON object with two fields:
- "facets": An array of facet objects, each containing:
  * "facet": A string describing the preference
  * "confidence": A number between 0 and 1
  * "supporting_neighbors": An array of neighbor IDs (e.g., ["User-123", "Item-456"])
  
- "support_edges": An array of edge objects, each containing:
  * "from": The source neighbor ID (string)
  * "to": The target user ID (string)
  * "w": The edge weight between 0 and 1 (number)
"""
        
        # Single message format (more natural)
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        return messages
    
    def get_stage_r_schema(self) -> Dict:
        """Get Stage-R JSON schema (v2: no candidate scoring)"""
        return {
            "facets": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "facet": {"type": "string"},
                        "confidence": {"type": "number"},
                        "supporting_neighbors": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["facet", "confidence", "supporting_neighbors"],
                    "additionalProperties": False
                }
            },
            "support_edges": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "from": {"type": "string"},
                        "to": {"type": "string"},
                        "w": {"type": "number"}
                    },
                    "required": ["from", "to", "w"],
                    "additionalProperties": False
                }
            }
        }
    
    def run_stage_r(
        self,
        user_id: int,
        packed_context: Dict,
        n_facets: int = 7,
        temperature: float = 0.0,
        max_tokens: int = 4000,
        debug_logger = None
    ) -> Dict:
        """
        Run Stage-R (Retrieval)
        
        Args:
            user_id: User ID
            packed_context: Packed context (from packer)
            n_facets: Number of facets
            temperature: LLM temperature
            max_tokens: Maximum number of tokens
            debug_logger: Optional debug logger
            
        Returns:
            {
                'facets': [...],
                'support_edges': [...]
            }
        """
        # Build prompt
        messages = self.build_stage_r_prompt(
            user_id=user_id,
            user_mem_bullets=packed_context.get('memory_text', ''),
            neighbor_table_json=packed_context.get('neighbors_text', ''),
            candidates_json=packed_context.get('candidates_text', ''),
            n_facets=n_facets
        )
        
        # Get schema
        properties = self.get_stage_r_schema()
        
        # Call LLM (use stageR cache)
        try:
            response = self.llm.generate_json(
                messages=messages,
                properties=properties,
                temperature=temperature,
                max_tokens=max_tokens,
                debug_logger=debug_logger
            )
            return response
        except Exception as e:
            print(f"Error in Stage-R for user {user_id}: {e}")
            # Return empty response
            return {
                "facets": [],
                "support_edges": []
            }
    
    # ========================================================================
    # Stage-W: Write (generate user_patches, neighbor_patches, propagation_policy)
    # ========================================================================
    
    def build_stage_w_prompt(
        self,
        user_id: int,
        feedback: Dict,
        recent_facets: List[Dict],
        user_mem_keys: List[str],
        item_mem_keys: Dict[int, List[str]],
        neighbor_ids: List[str],
        neighbor_details: List[Dict] = None,  # New: neighbor detailed information
        fanout_cap: int = 6,
        clicked_item_info: str = None
    ) -> List[Dict[str, str]]:
        """
        Build Stage-W prompt
        
        Args:
            user_id: User ID
            feedback: Feedback information {'action': 'CLICK'/'SKIP'/'RATING', 'item_id': int, 'value': ...}
            recent_facets: Recent facets (from Stage-R)
            user_mem_keys: User's current memory keys
            item_mem_keys: Item's current memory keys {item_id: [keys]}
            neighbor_ids: Neighbor IDs (for propagation, bounded)
            fanout_cap: Propagation fanout cap
            clicked_item_info: Clicked item metadata (title)
            
        Returns:
            List of messages
        """
        # Get current user profile (if available)
        current_user_memory = user_mem_keys[0] if user_mem_keys else None
        
        # Format clicked item's current description (if available)
        current_item_memory = None
        if item_mem_keys:
            for item_id, mem_list in item_mem_keys.items():
                if mem_list:
                    current_item_memory = mem_list[0]
                    break
        
        # Build paper-friendly single prompt
        prompt_parts = [
            f"You are an intelligent memory management system for collaborative recommendation. Your task is to update the personal memories of the user, the clicked item, and relevant collaborative neighbors based on this new interaction.",
            f"\n**Interaction Context:**",
            f"User {user_id} has just interacted with (clicked) Item {feedback.get('item_id', 'unknown')} ({clicked_item_info or 'Unknown Item'})."
        ]
        
        # Add extracted preference patterns (extracted from collaborative memories)
        prompt_parts.append(f"\n**User Preferences (Extracted from Collaborative Memories):**")
        prompt_parts.append("The following preference patterns were identified for this user:")
        if recent_facets:
            for i, f in enumerate(recent_facets[:5], 1):
                facet_text = f.get('facet', f.get('text', 'N/A'))
                conf = f.get('confidence', 1.0)
                prompt_parts.append(f"  {i}. {facet_text} (confidence: {conf:.2f})")
        else:
            prompt_parts.append("  (No preference patterns extracted for this interaction)")
        
        # Add current memories
        prompt_parts.append(f"\n**Current Personal Memory of User {user_id}:**")
        if current_user_memory and current_user_memory.strip():
            prompt_parts.append(current_user_memory)
        else:
            prompt_parts.append("(No memory recorded yet for this user)")
        
        clicked_item_id = feedback.get('item_id', 'unknown')
        prompt_parts.append(f"\n**Current Memory of Item {clicked_item_id} ({clicked_item_info or 'Unknown'}):**")
        if current_item_memory and current_item_memory.strip():
            prompt_parts.append(current_item_memory)
        else:
            prompt_parts.append("(No memory recorded yet for this item)")
        
        # Add collaborative neighbor information
        prompt_parts.append(f"\n**Collaborative Neighbors Available for Memory Propagation:**")
        if neighbor_ids and len(neighbor_ids) > 0:
            n_neighbors = min(len(neighbor_ids), fanout_cap)
            prompt_parts.append(f"The following {n_neighbors} collaborative neighbors are available for potential memory updates:")
            
            for i, nb_id in enumerate(neighbor_ids[:fanout_cap], 1):
                # Find neighbor detailed information
                nb_detail = None
                if neighbor_details:
                    nb_detail = next((d for d in neighbor_details if d.get('id_str') == nb_id), None)
                
                if nb_detail:
                    score = nb_detail.get('score', 0.0)
                    memory_preview = nb_detail.get('memory', 'N/A')
                    if memory_preview and len(memory_preview) > 150:
                        memory_preview = memory_preview[:150] + "..."
                    prompt_parts.append(f"  {i}. {nb_id} (collaborative strength: {score:.3f})")
                    prompt_parts.append(f"     Current memory: {memory_preview if memory_preview and memory_preview != 'N/A' else '(empty)'}")
                else:
                    prompt_parts.append(f"  {i}. {nb_id}")
                    prompt_parts.append(f"     Current memory: (unknown)")
        else:
            prompt_parts.append("  (No collaborative neighbors available for this interaction)")
        
        # Task description
        prompt_parts.append("""

**Your Task:**
Generate UPDATED memories for:
1. **The current user** (synthesize current memory + facets + clicked item)
2. **The clicked item** (describe what it is and who might enjoy it)
3. **Selected neighbors** (IMPORTANT: collaborative propagation is key!)
   * Analyze the available neighbors and their current memories
   * Select neighbors that are RELEVANT to this interaction (e.g., similar themes, related topics)
   * Update their memories to reflect new insights from this interaction
   * This helps the system learn collaboratively!

**Output Requirements:**

- "user_memory": Concise natural language description of user's interests and preferences
  * Synthesize themes (e.g., "holistic health", "children's education")
  * Be specific (e.g., "interested in Reiki and aromatherapy")
  * DON'T just list item titles
  * Keep it focused (typically a few sentences)
  
- "item_memory": Concise description of the clicked item
  * What it's about and who might enjoy it
  * Keep it brief but informative
  
- "neighbor_updates": Array of neighbor memory updates (OPTIONAL but recommended)
  * Select neighbors that are MOST relevant to this interaction
  * Choose as many as needed (typically a few, but flexible)
  * For each neighbor, provide updated memory content (NOT just appending text)
  * Rationale explains why this neighbor is relevant
  
**Example (GOOD):**
{{
  "user_memory": "A book enthusiast interested in holistic health and wellness, particularly Reiki, chakra work, and aromatherapy. Shows preference for instructional materials that promote personal development. Recently explored children's books about organization.",
  "item_memory": "A children's book teaching organizing skills through engaging storytelling. Helps parents guide children to keep rooms tidy.",
  "neighbor_updates": [
    {{
      "neighbor_id": "User-XXXX",  ← IMPORTANT: Replace with ACTUAL neighbor ID from the list above!
      "memory_update": "Interested in holistic wellness and personal development. Similar tastes to User-{user_id} in self-improvement content.",
      "rationale": "Similar user with overlapping interests in wellness"
    }},
    {{
      "neighbor_id": "Item-YYYY",  ← IMPORTANT: Replace with ACTUAL neighbor ID from the list above!
      "memory_update": "Office organizing guide. Related to tidiness and organization themes, complementary to children's organizing books.",
      "rationale": "Related item about organization"
    }}
  ]
}}

**CRITICAL**: You MUST select neighbor IDs from the "Available Neighbors for Propagation" list shown above. Do NOT use IDs from this example directly!

**Example (BAD - Missing propagation!):**
{{
  "user_memory": "Clicked item 29300. Likes books.",  ← Too vague!
  "item_memory": "Benji's Messy Room",  ← Just the title!
  "neighbor_updates": []  ← ❌ NO propagation (this is bad!)
}}

**Note**: Always try to propagate to at least 1-3 relevant neighbors when possible. This helps build a collaborative knowledge graph!

**Expected Output Format:**
Your response should be a JSON object with three fields:
- "user_memory": The updated personal memory for the user (string)
- "item_memory": The updated memory for the clicked item (string)  
- "neighbor_updates": An array of neighbor update objects (may be empty)
""")
        
        prompt = "".join(prompt_parts)
        
        # Single message format
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        return messages
    
    def get_stage_w_schema(self) -> Dict:
        """Get Stage-W JSON schema (supports neighbor propagation)"""
        return {
            "user_memory": {
                "type": "string",
                "description": "Concise natural language description of user's interests and preferences"
            },
            "item_memory": {
                "type": "string",
                "description": "Concise description of the clicked item's content and appeal"
            },
            "neighbor_updates": {
                "type": "array",
                "description": "Updates to propagate to collaborative neighbors (select relevant neighbors)",
                "items": {
                    "type": "object",
                    "properties": {
                        "neighbor_id": {
                            "type": "string",
                            "description": "Neighbor ID (e.g., 'User-123' or 'Item-456')"
                        },
                        "memory_update": {
                            "type": "string",
                            "description": "New memory content for this neighbor (concise description)"
                        },
                        "rationale": {
                            "type": "string",
                            "description": "Why this neighbor should be updated"
                        }
                    },
                    "required": ["neighbor_id", "memory_update", "rationale"],
                    "additionalProperties": False  # Azure OpenAI requirement
                }
            }
        }
    
    def run_stage_w(
        self,
        user_id: int,
        feedback: Dict,
        recent_facets: List[Dict],
        user_mem_keys: List[str],
        item_mem_keys: Dict[int, List[str]],
        neighbor_ids: List[str],
        neighbor_details: List[Dict] = None,  # New: neighbor detailed information
        fanout_cap: int = 6,
        clicked_item_info: str = None,
        temperature: float = 0.0,
        max_tokens: int = 4000,
        debug_logger = None
    ) -> Dict:
        """
        Run Stage-W (Write)
        
        Args:
            user_id: User ID
            feedback: Feedback information
            recent_facets: Recent facets
            user_mem_keys: User memory keys
            item_mem_keys: Item memory keys
            neighbor_ids: Neighbor IDs
            fanout_cap: Propagation fanout cap
            clicked_item_info: Clicked item metadata (title)
            temperature: LLM temperature
            max_tokens: Maximum number of tokens
            debug_logger: Optional debug logger
            
        Returns:
            {
                'user_patches': [...],
                'neighbor_patches': [...],
                'propagation_policy': {...}
            }
        """
        # Build prompt
        messages = self.build_stage_w_prompt(
            user_id=user_id,
            feedback=feedback,
            recent_facets=recent_facets,
            user_mem_keys=user_mem_keys,
            item_mem_keys=item_mem_keys,
            neighbor_ids=neighbor_ids,
            neighbor_details=neighbor_details,  # New
            fanout_cap=fanout_cap,
            clicked_item_info=clicked_item_info
        )
        
        # Get schema
        properties = self.get_stage_w_schema()
        
        # Call LLM (use stageW cache)
        try:
            response = self.llm.generate_json(
                messages=messages,
                properties=properties,
                temperature=temperature,
                max_tokens=max_tokens,
                debug_logger=debug_logger
            )
            return response
        except Exception as e:
            error_msg = f"❌ Error in Stage-W for user {user_id}: {e}"
            print(error_msg)
            if debug_logger:
                debug_logger(error_msg)
                import traceback
                debug_logger(f"Traceback:\n{traceback.format_exc()}")
            # Return empty response
            return {
                "user_memory": "",
                "item_memory": "",
                "neighbor_updates": []
            }
