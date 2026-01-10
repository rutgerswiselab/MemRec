"""
Simplified memory storage - single text mode
User Memory: single profile field (natural language description)
Item Memory: single description field (natural language description)
"""
import json
from typing import Dict, Optional
from pathlib import Path


class MemoryStorage:
    """Simplified memory storage: each entity stores only one piece of text"""
    
    def __init__(self, **kwargs):
        """
        Initialize storage
        user_profiles: {user_id: str}  # User profiles (natural language)
        item_descriptions: {item_id: str}  # Item descriptions (natural language)
        """
        self.user_profiles = {}  # {user_id: "A book enthusiast interested in..."}
        self.item_descriptions = {}  # {item_id: "A children's book about..."}
        self.n_updates = 0  # Count of updates
    
    def get_user_memory(self, user_id: int) -> Optional[str]:
        """Get user memory (natural language profile)"""
        return self.user_profiles.get(user_id, None)
    
    def get_item_memory(self, item_id: int) -> Optional[str]:
        """Get item memory (natural language description)"""
        return self.item_descriptions.get(item_id, None)
    
    def update_user_memory(self, user_id: int, new_profile: str):
        """Update user memory (complete overwrite)"""
        if new_profile and new_profile.strip():
            self.user_profiles[user_id] = new_profile.strip()
            self.n_updates += 1
    
    def update_item_memory(self, item_id: int, new_description: str):
        """Update item memory (complete overwrite)"""
        if new_description and new_description.strip():
            self.item_descriptions[item_id] = new_description.strip()
            self.n_updates += 1
    
    def initialize_item_descriptions(self, item_metadata: Dict[int, Dict]):
        """
        Initialize item descriptions from metadata
        
        Args:
            item_metadata: {item_id: {'title': ..., 'description': ...}}
        """
        print("Initializing item descriptions from metadata...")
        for item_id, meta in item_metadata.items():
            if item_id in self.item_descriptions:
                continue  # Already has description, skip
            
            title = meta.get('title', f'Item-{item_id}')
            desc = meta.get('description', '')
            
            # Construct initial description (handle list or string format)
            if desc:
                # If description is list (reviews), extract first one and clean
                if isinstance(desc, list):
                    # Filter empty strings, take first non-empty review
                    non_empty = [d.strip() for d in desc if d and d.strip() and d.strip().lower() not in ['nan', 'n/a']]
                    if non_empty:
                        desc_text = non_empty[0][:300]  # Take first, limit length
                    else:
                        desc_text = ""
                else:
                    desc_text = str(desc)[:300]
                
                if desc_text:
                    description = f"{title}. {desc_text}"
                else:
                    description = title
            else:
                description = title
            
            self.item_descriptions[item_id] = description
        
        print(f"  Initialized descriptions for {len(self.item_descriptions)} items")
    
    def render_user_summary(self, user_id: int) -> str:
        """
        Render user memory summary (for Stage-R prompt)
        
        Returns:
            User profile text, or default prompt if none exists
        """
        profile = self.get_user_memory(user_id)
        if profile:
            return profile
        else:
            return "No specific memory recorded for this user yet."
    
    def get_stats(self) -> Dict:
        """Get statistics"""
        return {
            'n_users': len(self.user_profiles),
            'n_items': len(self.item_descriptions),
            'n_updates': self.n_updates,
            'storage': {
                'n_users': len(self.user_profiles),
                'n_items': len(self.item_descriptions),
                'n_updates': self.n_updates
            }
        }
    
    def save(self, save_path: str):
        """Save to file"""
        data = {
            'user_profiles': {str(k): v for k, v in self.user_profiles.items()},
            'item_descriptions': {str(k): v for k, v in self.item_descriptions.items()},
            'n_updates': self.n_updates
        }
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Memory saved to {save_path}")
    
    def load(self, load_path: str):
        """Load from file"""
        with open(load_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.user_profiles = {int(k): v for k, v in data.get('user_profiles', {}).items()}
        self.item_descriptions = {int(k): v for k, v in data.get('item_descriptions', {}).items()}
        self.n_updates = data.get('n_updates', 0)
        
        print(f"Loaded {len(self.user_profiles)} users, {len(self.item_descriptions)} items")
    
    def save_to_jsonl(self, save_path: str, dataset=None):
        """
        Save in JSONL format (save all updated memories)
        
        Args:
            save_path: Save path
            dataset: Optional dataset object, used to check if item was updated
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        n_saved = 0
        n_items_saved = 0
        
        with open(save_path, 'w', encoding='utf-8') as f:
            # Save user profiles
            for user_id, profile in self.user_profiles.items():
                entry = {
                    'type': 'user',
                    'id': user_id,
                    'memory': profile
                }
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                n_saved += 1
            
            # Save updated item descriptions
            # If dataset provided, only save updated items
            # Otherwise save all items (not recommended when too many)
            if dataset and hasattr(dataset, 'item_metadata'):
                for item_id, desc in self.item_descriptions.items():
                    # Check if updated (different from metadata)
                    if item_id in dataset.item_metadata:
                        original = dataset.item_metadata[item_id].get('title', '')
                        # If description differs from original title, it was updated
                        if desc != original and desc not in [str([original]), str(['', original])]:
                            entry = {
                                'type': 'item',
                                'id': item_id,
                                'memory': desc
                            }
                            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                            n_saved += 1
                            n_items_saved += 1
            else:
                # No dataset, only save user profiles
                # (item_descriptions too many, don't save)
                pass
        
        print(f"Memory saved to {save_path} ({n_saved} entries: {len(self.user_profiles)} users, {n_items_saved} items)")
