# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

from typing import Optional
from torch import Tensor

from litgpt.config import Config
from litgpt import GPT as litGPT

class GPT(litGPT):
    def __init__(self, config: Config) -> None:
        super().__init__(config)
        
        self.init_hidden_states()
        
    # ------------------ NEW CODE [PATCHING] ------------------
    def load_source_hidden_states(self, hidden_states:Tensor = None, patching_token_pos:int = None):
        self.source_hidden_states = hidden_states
        self.patching_token_pos = patching_token_pos
    
    def init_hidden_states(self):
        self.load_source_hidden_states(hidden_states = None, patching_token_pos = None)
        
    def _inject_source_hidden_states(self, hidden_states: Tensor) -> Tensor:
        hidden_states[0, self.patching_token_pos, :] = self.source_hidden_states
        return hidden_states
    # ------------------ NEW CODE [PATCHING] ------------------

    def forward(self, idx: Tensor, input_pos: Optional[Tensor] = None, output_hidden_states:bool = False) -> Tensor:
        T = idx.size(1)

        # ------------------ NEW CODE ------------------
        all_hidden_states = () if output_hidden_states else None
        # ------------------ NEW CODE ------------------
        
        if self.max_seq_length < T:
            raise ValueError(f"Cannot forward sequence of length {T}, max seq length is only {self.max_seq_length}.")

        if input_pos is not None:  # use the kv cache
            cos = self.cos.index_select(0, input_pos)
            sin = self.sin.index_select(0, input_pos)
            if self.mask_cache is None:
                raise TypeError("You need to call `gpt.set_kv_cache()`")
            mask = self.mask_cache.index_select(2, input_pos)
        else:
            cos = self.cos[:T]
            sin = self.sin[:T]
            mask = None

        x = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        if self.config.scale_embeddings:
            x = x * (self.config.n_embd**0.5)
            
        # ------------------ NEW CODE [PATCHING] ------------------
        first_pass = True if len(input_pos) > 1 else False 
        if first_pass and self.source_hidden_states is not None:
            x = self._inject_source_hidden_states(x)
        # ------------------ NEW CODE [PATCHING] ------------------

        for block in self.transformer.h:
            
            # ------------------ NEW CODE ------------------
            if output_hidden_states:
                all_hidden_states += (x,)
            # ------------------ NEW CODE ------------------
                
            x = block(x, cos, sin, mask, input_pos)
            
        # ------------------ NEW CODE ------------------
        if output_hidden_states:
            all_hidden_states += (x,)
        # ------------------ NEW CODE ------------------
            
        x = self.transformer.ln_f(x)

        if output_hidden_states:
            return self.lm_head(x), all_hidden_states
        else:
            return self.lm_head(x)  # (b, t, vocab_size)