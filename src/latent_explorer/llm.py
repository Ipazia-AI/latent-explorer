from logging import warn
from multiprocessing.pool import ThreadPool 
from pathlib import Path
from typing import Union, Tuple, List, Dict
from lightning import Fabric
import torch
import sys

# LITGPT IMPORTS
from litgpt import Config
from litgpt.scripts.download import download_from_hub
from litgpt.utils import check_valid_checkpoint_dir, get_default_supported_precision
from litgpt.generate.sequentially import sequential

# LOCAL IMPORTS
from .utils.litgpt_minimal.model import GPT
from .utils.litgpt_minimal.base import generate
from .utils.litgpt_minimal.tokenizer import Tokenizer
from .utils.utils import print_title, text2json, generate_prompt_template, search_subvector, clean_token
from .utils.data_loader import load_examples

class LLM:
    
    DEFAULT_SOURCE_SYS_INST = "You are a journalist with expertise in fact-checking. Your role is to evaluate the truthfulness of factual claims. To uphold journalistic integrity, you must produce a report containing a binary assessment and all the factual information that supports your evaluation."
    
    DEFAULT_TARGET_SYS_INST = "You are an assistant with expertise in fact-checking. Your role is to assess claims."
    
    def __init__(self, model_name:str, inputs: list[str], hf_access_token:str = None):
        self.model_name = model_name
        self.inputs = inputs
        
        # Initialize the device
        self._init_device()

        # Access token for downloading the model
        self.hf_access_token = hf_access_token
        
        # Set the placeholder token
        self.placeholder_token = "x"
        
        # Visualize the model information
        print_title(self.__str__())
        
        # Load the model
        self._download_model()
        self._load_model()
        
    def __str__(self) -> str:
        return f'LatentExplorer: MODEL={self.model_name}, DATA={len(self.inputs)}, DEVICE={self.fabric.device.type.upper()}(x{self.num_devices})'
    
    def get_model_name(self) -> str:
        return list(reversed(self.model.config.hf_config.values()))

    def _init_device(self):
        precision = get_default_supported_precision(training=False)
        
        self.fabric = Fabric(devices = 1, precision=precision)
        self.num_devices = self.fabric.accelerator.auto_device_count()
        
    def _download_model(self):
       
        # Create the model folder if it does not exist  
        model_folder = Path('.models')
        model_folder.mkdir(parents=True, exist_ok=True)
        
        # Download the model if not exists
        self.model_path = model_folder.joinpath(self.model_name)
        if not self.model_path.exists():
            print(f"\n[INFO] Downloading the model from Hugging Face: {self.model_name}\n")
            
            try:
                download_from_hub(repo_id = self.model_name, access_token = self.hf_access_token, checkpoint_dir = model_folder) 
            except ValueError as e:
                if self.hf_access_token is None:
                    print(f'[ERROR] {self.model_name} requires authentication, please set the `hf_access_token` variable in the LatentExplorer class. You can find your token by visiting https://huggingface.co/settings/tokens')
                else:
                    print(f'[ERROR] {e}')
                sys.exit(1)
            print(f'\n[INFO] Model downloaded: {self.model_name}')
    
    def _load_model(self):
        
        # Load the metadata
        check_valid_checkpoint_dir(self.model_path)
        config = Config.from_file(self.model_path / "model_config.yaml")
        checkpoint_path = self.model_path / "lit_model.pth"

        # Load the tokenizer
        self.tokenizer = Tokenizer(self.model_path)

        # Initalize the model
        with self.fabric.init_tensor(), torch.device("meta"):
            model = GPT(config)
            model.set_kv_cache(batch_size = 1)

        # Load the model checkpoints
        state_dict = torch.load(str(checkpoint_path), mmap=True, map_location="cpu", weights_only=False)
        model.load_state_dict(state_dict, assign=True)

        # Setup the model
        model = self.fabric.setup_module(model, move_to_device=False)

        # TURNAROUND for NotImplementedError: Only balanced partitioning is implemented
        while model.config.n_layer % self.num_devices != 0:
            self.num_devices-= 1
            
        # Partition the transformer blocks across all your devices and running them sequentially (https://github.com/Lightning-AI/litgpt/blob/main/tutorials/inference.md)
        print(f'Loading the model "{self.model_name}" on {self.num_devices} {self.fabric.device.type.upper()} devices:')
        self.model = sequential(model, root = self.fabric.device, max_seq_length = config.block_size, devices = self.num_devices)
        
        if self.fabric.device.type == "cuda":
            self.fabric.print(f"\nCUDA memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")
            
        # Set the default generated parameters
        self.generated_params = self._generated_params(top_k = 50, top_p = 0.9, temperature = 0.0, max_returned_tokens = 512)
    
    def _generated_params(self, **kwargs):
        params = dict(eos_id = self.tokenizer.eos_id)
        if len(kwargs.keys()) > 0:
            params.update(kwargs)

        return params
    
    def _tokenize(self, prompt:str, return_tokens:bool) -> torch.Tensor:
        
        # Input encoding
        input_ids = self.tokenizer.encode(prompt, device = self.fabric.device, add_special_tokens = False)
        
        if return_tokens:
            tokens = [self.tokenizer.id_to_token(token.item()) for token in input_ids]
            return input_ids, tokens
        
        return input_ids
    
    def _generate(self, input_ids:torch.Tensor, parse_output:bool) -> Tuple[str, torch.Tensor]:
        
        # Set the generated parameters
        params = self.generated_params.copy()
        params['max_returned_tokens'] += input_ids.size(0)
        
        if params['max_returned_tokens'] > self.model.config.block_size:
            warn(f" Setting 'max_returned_tokens' to {self.model.config.block_size} since it is greater than the block size: {params['max_returned_tokens']} > {self.model.config.block_size}\n")
            params['max_returned_tokens'] = self.model.config.block_size
        
        # Generate the output
        output_ids, hidden_states = generate(model = self.model, prompt = input_ids, output_hidden_states = True, **params)
        
        # Decode the output
        output_ids = output_ids[input_ids.size(0):]
        generated_text = self.tokenizer.decode(output_ids).strip()
        
        # Parse the output text to JSON
        if parse_output and len(generated_text) > 0:
            generated_text = text2json(generated_text, max_attempts = 3)
        
        return generated_text, hidden_states
    
    def _inference(self, input_ids:torch.Tensor, parse_output:bool = True, output_hidden_states:bool = True) -> Union[str, Tuple[str, torch.Tensor]]:
        
        # Generate the output
        generated_text, hidden_states = self._generate(input_ids, parse_output = parse_output)
        
        if output_hidden_states:
            return generated_text, hidden_states
        else:
            return generated_text
    
    # --------------------------------------------
    # ---------- PUBLIC METHODS ------------------ 
    # --------------------------------------------
    def generate_prompts(self, source_sys_inst:str = None, target_sys_inst:str = None, in_context_examples:list[dict] = None, verbose = False):
        
        # Set the system instructions
        source_sys_inst = source_sys_inst if source_sys_inst is not None else self.DEFAULT_SOURCE_SYS_INST
        target_sys_inst = target_sys_inst if target_sys_inst is not None else self.DEFAULT_TARGET_SYS_INST
        models_w_sys_roles = ['llama']
        
        # Load the default in-context examples
        if not isinstance(in_context_examples, list) or len(in_context_examples) < 1:
            in_context_examples = load_examples()
    
        # Split the examples into source and target examples (1st example for the source, the others for the target)
        if in_context_examples == 1:
            source_examples = target_examples = in_context_examples
        else:
            source_examples = [in_context_examples[0]]
            target_examples = in_context_examples[1:]
        
        # Generate the source prompts for the inputs using different threads
        generate_prompt = lambda input: self.tokenizer.apply_chat_template(
            conversation = generate_prompt_template(
                sys_prompt = source_sys_inst, 
                examples = source_examples, 
                input_text = input, 
                role_sys_available = any([m in self.model_name.lower() for m in models_w_sys_roles])), 
            tokenize = False)
        
        with ThreadPool(processes = len(self.inputs)) as pool:
            result = pool.map_async(func = generate_prompt, iterable = self.inputs)
            self.prompts = [value for value in result.get()]
            
        # Generate the target prompts for the inputs using different threads
        self.target_prompt = self.tokenizer.apply_chat_template(
            conversation = generate_prompt_template(
                sys_prompt = target_sys_inst, 
                examples = target_examples, 
                input_text = self.placeholder_token, 
                role_sys_available = any([m in self.model_name.lower() for m in models_w_sys_roles])), 
            tokenize = False)
        
        # Save the tokenized input texts (for part-of-speech tagging)
        self.input_tokens = [self._tokenize(input, return_tokens=True)[1] for input in self.inputs] 
        
        if verbose:
            print_title('SOURCE PROMPT TEMPLATE')
            print(self.prompts[0].replace(self.inputs[0], "$INPUT"), end = '\n' + '-' * 100 + '\n\n')
            print_title('TARGET PROMPT')
            print(self.target_prompt, end = '\n' + '-' * 100 + '\n\n')
        
    def inference(self, parse_output:bool = True, output_hidden_states:bool = True, verbose:bool = False)-> List[Dict[str, Union[str, torch.Tensor]]]:

        # Perform inference for each prompt
        self.outputs = list()
        for idk, prompt in enumerate(self.prompts):
            
            # Tokenize the input text
            input_ids, input_tokens = self._tokenize(prompt, return_tokens=True)
            
            # Find the input tokens within the prompt
            pos = search_subvector(vector = input_tokens, sub = self.input_tokens[idk])
            self.input_tokens[idk] = dict(zip(pos, self.input_tokens[idk]))        
            
            # Perform inference
            generated_text, hidden_states = self._inference(
                input_ids = input_ids, 
                parse_output = parse_output, 
                output_hidden_states = output_hidden_states
                )
            
            # Save the output as a dictionary
            self.outputs.append({'text': generated_text, 'hs': hidden_states})
            
            if verbose:
                print_title('PROMPT')
                print(prompt)
                print_title(f'OUTPUT [hs: {tuple(hidden_states.size())}]')
                print(generated_text, end = '\n\n')

        return self.outputs