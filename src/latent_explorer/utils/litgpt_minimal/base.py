# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
# SOURCE: litgpt/generate/base.py || VERSION: 0.4.8 || DATA: 2024-08-07

from typing import Any, Optional

import torch
import torch._dynamo.config
import torch._inductor.config

from litgpt import GPT
from litgpt.generate.base import sample

@torch.inference_mode()
def generate(
    model: GPT,
    prompt: torch.Tensor,
    max_returned_tokens: int,
    output_hidden_states: bool = False,
    *,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: float = 1.0,
    eos_id: Optional[int] = None,
    include_prompt: bool = True,
    **kwargs: Any
) -> torch.Tensor:
    """
    Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.
    The implementation of this function is modified from A. Karpathy's nanoGPT.

    Args:
        model: The model to use.
        prompt: Tensor of shape (T) with indices of the prompt sequence.
        max_returned_tokens: The maximum number of tokens to return (given plus generated).
        temperature: Scales the predicted logits by 1 / temperature.
        top_k: If specified, only sample among the tokens with the k highest probabilities.
        top_p: If specified, it represents the cumulative probability threshold to consider in the sampling process.
            In top-p sampling, the next token is sampled from the highest probability tokens
            whose cumulative probability exceeds the threshold `top_p`. When specified,
            it must be `0 <= top_p <= 1`. Here, `top_p=0` is equivalent
            to sampling the most probable token, while `top_p=1` samples from the whole distribution.
            It can be used in conjunction with `top_k` and `temperature` with the following order
            of application:

            1. `top_k` sampling
            2. `temperature` scaling
            3. `top_p` sampling

            For more details, see https://arxiv.org/abs/1904.09751
            or https://huyenchip.com/2024/01/16/sampling.html#top_p
        eos_id: If specified, stop generating any more token once the <eos> token is triggered.
        include_prompt: If true (default) prepends the prompt (after applying the prompt style) to the output.
    """
    T = prompt.size(0)
    assert max_returned_tokens > T
    if model.max_seq_length < max_returned_tokens - 1:
        # rolling the kv cache based on the `input_pos` value would be necessary. However, doing so would introduce a
        # data dependency on the `input_pos` tensor and impact model compilation. Since this setting is uncommon, we do
        # not support it to avoid negatively impacting the overall speed
        raise NotImplementedError(f"max_seq_length {model.max_seq_length} needs to be >= {max_returned_tokens - 1}")

    device = prompt.device
    tokens = [prompt] if include_prompt else []
    input_pos = torch.tensor([T], device=device)
    
    # ------------------ MODIFIED CODE ---------------------
    # Generate the first token
    token, hidden_states = next_token(
        model = model, 
        input_pos = torch.arange(0, T, device=device), 
        x = prompt.view(1, -1), 
        output_hidden_states = True,
        temperature=temperature, 
        top_k=top_k, 
        # **kwargs     # TODO wait support for other parameters
    )
    token = token.clone()
    tokens.append(token)
    # ------------------ MODIFIED CODE ---------------------
    
    # Generate the rest of the tokens
    for _ in range(2, max_returned_tokens - T + 1):
        token = next_token(model, input_pos, x = token.view(1, -1), temperature=temperature, top_k=top_k).clone() # **kwargs
        tokens.append(token)
        if token == eos_id:
            break
        input_pos = input_pos.add_(1)
    
    # ------------------ MODIFIED CODE --------------------
    if output_hidden_states:
        return torch.cat(tokens), hidden_states
    else:
        return torch.cat(tokens)
    # ------------------ MODIFIED CODE --------------------


def next_token(model: GPT, input_pos: torch.Tensor, x: torch.Tensor, output_hidden_states:bool = False, **kwargs: Any) -> torch.Tensor:
    
    # ------------------ MODIFIED CODE --------------------
    if output_hidden_states:
        logits, hidden_states = model(x, input_pos, output_hidden_states = True)
        hidden_states = [hs.cpu() for hs in hidden_states]
        hidden_states = torch.cat(hidden_states, dim = 0)
    else:
        logits = model(x, input_pos)
    # ------------------ MODIFIED CODE --------------------
    

    next = sample(logits, **kwargs).to(dtype=x.dtype)
    
    # ------------------ MODIFIED CODE --------------------
    if output_hidden_states:
        return next, hidden_states
    else:
        return next
    # ------------------ MODIFIED CODE --------------------
