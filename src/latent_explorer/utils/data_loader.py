from json import load
from litgpt.config import configs
from pkg_resources import resource_filename

EXAMPLE_FILEPATH = resource_filename('latent_explorer', resource_name = 'resources/in_context_examples.json')

def load_examples(file_path: str = EXAMPLE_FILEPATH) -> list[dict]:
    with open(file_path, mode = 'r', encoding="utf-8") as f:
        examples = load(f)
    return examples

def all_supported_models(verbose = True) -> list[str]:
    supported_models = [f"{config['hf_config']['org']}/{config['hf_config']['name']}" for config in configs]
    supported_models = list(sorted(supported_models))
    
    if verbose:
        print("\n".join(f'[{i}] {model}'for i, model in enumerate(supported_models)))
        
    return supported_models