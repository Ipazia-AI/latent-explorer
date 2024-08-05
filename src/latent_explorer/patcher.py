import torch
from typing import Dict

# Local imports
from .utils.litgpt_minimal.model import GPT
from .utils.litgpt_minimal.tokenizer import Tokenizer
from .utils.utils import clean_token
from .llm import LLM

class Patcher(LLM):
    def __init__(self, model:GPT, tokenizer:Tokenizer, prompt: str, placeholder_token:str):
        self.model = model
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.placeholder_token = placeholder_token

        # Initialize the device
        self._init_device()
        
        # Set the default generated parameters
        self.generated_params = self._generated_params(top_k = 50, temperature = 0.0, max_returned_tokens = 100)
    
    def merged_hs(self, hidden_states: torch.Tensor, weights: torch.Tensor, verbose = True) -> torch.Tensor:

        # Merge the hidden states of each hidden layer
        merged_hs = list()
        for i in range(hidden_states.shape[0]):
            
            # Merge the hidden states using a weighted sum
            ws = torch.sum(hidden_states[i] * weights, dim = 0)
            merged_hs.append(ws)
            
            if verbose:
                print(f'L{i}:', hidden_states[i].shape, '-->', ws.shape)
        merged_hs = torch.stack(merged_hs)

        return merged_hs
    
    def _single_patching(self, hidden_states: torch.Tensor):

        # Tokenize the target prompt
        input_ids, input_tokens = self._tokenize(self.prompt, return_tokens = True)
        
        # Retrive the position of the placeholder token
        patching_token_pos = [idk for idk, token in enumerate(input_tokens) if clean_token(token) == self.placeholder_token]

        if len(patching_token_pos) != 1:
            raise ValueError(f"The placeholder token '{self.placeholder_token}' should appear once in the prompt! TOKENS:{input_tokens}")
        patching_token_pos = patching_token_pos[0]

        assert clean_token(input_tokens[patching_token_pos]) == self.placeholder_token, f"The placeholder token '{self.placeholder_token}' should appear at the position {patching_token_pos} in the prompt! FOUND INSTEAD: {clean_token(input_tokens[patching_token_pos])}"
        
        # Inject the hidden states into the model
        self.model.load_source_hidden_states(hidden_states, patching_token_pos = patching_token_pos)
        
        # Perform inference patched with the hidden states
        generated_text, _ = self._generate(input_ids, parse_output=True)

        return generated_text

    def probing(self, hidden_states: torch.Tensor) -> Dict[str, str]:
        
        generated_texts = dict()
        
        # Apply patching for each hidden layer
        num_layers = hidden_states.shape[0]
        
        # TODO CAN BE PARALLELIZED 
        for i in range(num_layers):
            generated_texts[f'L{i}'] = self._single_patching(hidden_states[i])

        return generated_texts