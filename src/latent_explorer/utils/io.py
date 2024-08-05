
from os import path, makedirs
from json import dump
from typing import List, Dict

def save_results(results: List[Dict[str, str]], folder_path:str, file_name:str = 'results.json'):
    
    # Create the folder if it does not exist
    if not path.exists(folder_path):
        makedirs(folder_path)
    
    file_path = path.join(folder_path, file_name)
    with open(file_path, mode = 'w', encoding='utf-8') as f:
        dump(results, f, indent = 4, ensure_ascii = False)
    
    print(f"\nThe results are saved in '{file_path}'!\n")