from typing import List, Dict
import torch
from collections import defaultdict
from tqdm import tqdm 
import spacy
from json import dump 
from os import path, makedirs

# LOCAL IMPORTS
from .llm import LLM
from .patcher import Patcher
from .utils.utils import clean_token

class LatentExplorer(LLM):
    def __init__(self, model_name:str, inputs: list[str], hf_access_token:str = None):
        super().__init__(model_name, inputs, hf_access_token)
        
        # Perform part-of-speech tagging
        self._pos_tagging()
        
        # Weigh the input tokens based on their part-of-speech tags
        self.tag_weights = {'NOUN': 1, 'PROPN': 1, 'VERB': 1}
        
    def _pos_tagging(self):
    
        # Check if the spacy model is installed
        if not spacy.util.is_package("en_core_web_trf"):
            spacy.cli.download("en_core_web_trf")
        
        # Load the spacy model
        spacy.prefer_gpu()
        nlp = spacy.load("en_core_web_trf", exclude=["lemmatizer", "ner"])
        
        # Perform part-of-speech tagging
        self.pos_tags = list()
        for item in nlp.pipe(self.inputs):
            
            # Extract the part-of-speech tags for each input
            item_pos_tags = defaultdict(list)
            for token in item:
                item_pos_tags[token.pos_].append(token.text)
            self.pos_tags.append(dict(item_pos_tags))
    
    def _weigh_input_tokens(self, hidden_states:torch.Tensor, verbose:bool = False) -> List[torch.Tensor]:
               
        item_weights = []
        item_considered_tokens = []
        for i in range(len(self.inputs)):
            
            # Get the input text
            input_text = self.inputs[i]
            
            # Get the part-of-speech tags for the input tokens
            pos_tags = self.pos_tags[i]
            
            # Filter the eligible words
            word_weights = {word: self.tag_weights[pos_tag] 
                            for pos_tag, words in pos_tags.items() 
                            for word in words 
                            if pos_tag in self.tag_weights.keys()}
            
            # EXPERIMENT exp1: CONSIDER ALL THE WORDS
            #word_weights = {word: 1 for words in pos_tags.values() for word in words}
            
            # Initialize the weights to zeros
            num_tokens = hidden_states[i].shape[1]
            weights = torch.zeros(num_tokens)
            
            # Get the input tokens (pos: token)
            input_tokens = self.input_tokens[i]
            
            # Compute the weights
            for pos, token in input_tokens.items():
                
                # Clean the token by removing the special characters
                token = clean_token(token)
                
                # Skip the token if it only contains one character
                if len(token) < 2: # exp1
                    continue
                
                # Check if the token includes the the end of the considered words
                for word, word_weight in word_weights.items():
                    if word.endswith(token):
                    #if token in word: # exp1 & exp2
                        weights[pos] = word_weight

            # The weights before the first considered token should be zeros
            assert  torch.argwhere(weights > 0).flatten()[0] >= list(input_tokens.keys())[0], f"The token weights before the input text ({torch.argwhere(weights > 0).flatten()[0]} < {list(input_tokens.keys())[0]}) should be zeros!"

            # Save the considered tokens
            considered_tokens = {token: weights[pos].item() for pos, token in input_tokens.items()}

            # Save the considered tokens and their weights 
            item_considered_tokens.append(considered_tokens)
            item_weights.append(weights)
            
            if verbose:
                print("\nINPUT:", input_text)
                print("POS TAGS:", pos_tags)
                print("WORD WEIGHTS:", word_weights)
                print("CONSIDERED TOKENS:", considered_tokens, "\n")
                #print("WEIGHTS:", weights, "\n")

        return item_weights, item_considered_tokens
        
    
    def probe_hidden_states(self, verbose:bool = False) -> List[Dict[str, str]]:
        
        # Extract the hidden states
        hidden_states = [item['hs'] for item in self.outputs]
        
        # Weigh the input tokens
        token_weights, considered_tokens = self._weigh_input_tokens(hidden_states, verbose = verbose)
        
        # Initialize the patcher class
        patcher = Patcher(model = self.model, tokenizer = self.tokenizer, prompt = self.target_prompt, placeholder_token = self.placeholder_token)

        # For each input text
        outputs = list()
        
        # TODO CAN BE PARALLELIZED
        for i in tqdm(range(len(self.inputs))):
  
            # Merge the latent representation of the tokens in each layer: [num_layers, num_tokens, hidden_dim]) --> [num_layers, hidden_dim]
            merged_hs = patcher.merged_hs(hidden_states[i], token_weights[i].reshape(-1, 1), verbose = verbose)
            
            # Perform the patching operation for each hidden layer
            generated_texts = patcher.probing(merged_hs)
            
            # Save the output by appending the generated text from the inference and the original input text
            outputs.append({
                'STATS': {'INPUT': self.inputs[i], 'CONSIDERED_TOKENS': considered_tokens[i]},
                '_INFERENCE': self.outputs[i]['text'],
                'HS': generated_texts,
            })
            
        return outputs
    
    def save_stats(self, folder_path:str):
        
        stats = {
            "num_inputs": len(self.inputs),
            "model": self.model_name,
            "num_layers": self.model.config.n_layer,
            "device": self.fabric.device.type.upper() + f"(x{self.num_devices})",
            "prompts": {'source': self.prompts[0].replace(self.inputs[0], "$INPUT"), 'target': self.target_prompt},
            #"model_params": self.model.__dict__
        }
        
        # Create the folder if it does not exist
        if not path.exists(folder_path):
            makedirs(folder_path)
        
        with open(folder_path + "/stats.json", mode = "w", encoding='utf-8') as f:
            dump(stats, f, indent = 4, ensure_ascii=False)