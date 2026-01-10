#!/usr/bin/env python3
"""
Utility script: Update config file by adding eval_user_list or n_eval_users
"""
import sys
import json
from pathlib import Path

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    # If yaml is not available, use simple string processing

def update_config_with_user_list(config_path, user_list_path, output_path, method_name, parallel_workers=16):
    """
    Update config file by adding user list settings
    
    Args:
        config_path: Original config file path
        user_list_path: User list JSON file path
        output_path: Output config file path
        method_name: Method name (used to determine whether to use eval_user_list or n_eval_users)
        parallel_workers: Number of parallel workers (default: 16)
    """
    # Read original config
    with open(config_path, 'r', encoding='utf-8') as f:
        config_lines = f.readlines()
    
    # Read user list file and get number of users
    user_list_file = Path(user_list_path)
    if not user_list_file.is_absolute():
        # Relative path, relative to project root
        project_root = Path(__file__).parent.parent
        user_list_file = project_root / user_list_path
    
    n_users = None
    if user_list_file.exists():
        with open(user_list_file, 'r', encoding='utf-8') as f:
            user_data = json.load(f)
            if 'user_ids' in user_data:
                n_users = len(user_data['user_ids'])
            elif 'n_samples' in user_data:
                n_users = user_data['n_samples']
    
    # Determine if method supports eval_user_list
    # MemRec supports eval_user_list
    methods_with_user_list = ['memrec_agent']
    
    # If PyYAML is not available, use simple string processing
    if not HAS_YAML:
        # Check if eval_user_list or n_eval_users already exists
        has_user_setting = any('eval_user_list:' in line or 'n_eval_users:' in line for line in config_lines)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            if has_user_setting:
                # Replace existing settings
                for line in config_lines:
                    if 'eval_user_list:' in line:
                        if method_name.lower() in methods_with_user_list:
                            f.write(f"eval_user_list: '{user_list_path}'\n")
                        else:
                            f.write(line)  # Keep original line, don't modify
                    elif 'n_eval_users:' in line:
                        if method_name.lower() not in methods_with_user_list and n_users:
                            f.write(f"n_eval_users: {n_users}\n")
                        else:
                            f.write(line)  # Keep original line
                    else:
                        f.write(line)
            else:
                # Append new settings
                f.writelines(config_lines)
                f.write("\n")
                if method_name.lower() in methods_with_user_list:
                    f.write(f"eval_user_list: '{user_list_path}'\n")
                elif n_users:
                    f.write(f"n_eval_users: {n_users}\n")
        return
    
    # Use PyYAML for parsing and modification (more reliable)
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}
    except Exception as e:
        print(f"Warning: Failed to parse YAML, using simple method: {e}", file=sys.stderr)
        # If parsing fails, copy directly and append
        with open(output_path, 'w', encoding='utf-8') as f:
            f.writelines(config_lines)
            if method_name.lower() in methods_with_user_list:
                f.write(f"\neval_user_list: '{user_list_path}'\n")
            elif n_users:
                f.write(f"\nn_eval_users: {n_users}\n")
        return
    
    # Set user list
    if method_name.lower() in methods_with_user_list:
        config['eval_user_list'] = user_list_path
    elif n_users:
        config['n_eval_users'] = n_users
    
    # For MemRec, update parallel workers
    if method_name.lower() == 'memrec_agent':
        config['parallel_workers'] = parallel_workers
    
    # Save config (try to maintain original format)
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    except Exception as e:
        print(f"Warning: YAML save failed, using simple append: {e}", file=sys.stderr)
        # Fallback: append directly to original file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.writelines(config_lines)
            if method_name.lower() in methods_with_user_list:
                f.write(f"\neval_user_list: '{user_list_path}'\n")
            elif n_users:
                f.write(f"\nn_eval_users: {n_users}\n")

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print(f"Usage: {sys.argv[0]} <config_path> <user_list_path> <output_path> <method_name> [parallel_workers]")
        sys.exit(1)
    
    config_path = sys.argv[1]
    user_list_path = sys.argv[2]
    output_path = sys.argv[3]
    method_name = sys.argv[4]
    parallel_workers = int(sys.argv[5]) if len(sys.argv) > 5 else 16
    
    update_config_with_user_list(config_path, user_list_path, output_path, method_name, parallel_workers)

