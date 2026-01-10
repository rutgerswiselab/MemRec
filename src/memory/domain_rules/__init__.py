"""
Domain-specific pruning rules for different datasets
"""
from .books_rules import BooksRules
from .goodreads_rules import GoodReadsRules
from .movietv_rules import MovieTVRules
from .yelp_rules import YelpRules

# Registry mapping dataset names to rule classes
DOMAIN_RULES = {
    'instructrec-books': BooksRules,
    'instructrec-goodreads': GoodReadsRules,
    'instructrec-movietv': MovieTVRules,
    'instructrec-yelp': YelpRules,
}

def get_domain_rules(dataset_name: str):
    """Get the appropriate rules class for a dataset"""
    # Extract base dataset name
    base_name = dataset_name.lower().strip()
    
    if base_name in DOMAIN_RULES:
        return DOMAIN_RULES[base_name]
    
    # Try to match partial names
    for key in DOMAIN_RULES:
        if key in base_name or base_name in key:
            return DOMAIN_RULES[key]
    
    # Default to Books rules if no match
    print(f"Warning: No rules found for dataset '{dataset_name}', using Books rules as default")
    return BooksRules

__all__ = [
    'BooksRules',
    'GoodReadsRules',
    'MovieTVRules',
    'YelpRules',
    'DOMAIN_RULES',
    'get_domain_rules',
]

