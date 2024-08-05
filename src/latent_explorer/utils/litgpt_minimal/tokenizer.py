# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
import json
from pathlib import Path
from typing import Optional, Union, List, Dict, Any
import torch

from functools import lru_cache
from litgpt import Tokenizer as litTokenizer

class Tokenizer(litTokenizer):
    
    def __init__(self, checkpoint_dir: Union[Path, str]) -> None:
        checkpoint_dir = Path(checkpoint_dir)
        
        if not checkpoint_dir.exists():
            raise NotADirectoryError(f"The checkpoint directory does not exist: {str(checkpoint_dir)}")

        # Initialize the tokenizer
        self.use_bos = self.check_if_bos_token_used(checkpoint_dir)
        self.bos_id = None
        self.eos_id = None
        
        # Special token path
        if (special_tokens_path := checkpoint_dir / "tokenizer_config.json").is_file():
            with open(special_tokens_path) as fp:
                token_config = json.load(fp)
        
            # Load the special tokens
            self.bos_token = token_config.get("bos_token")
            self.eos_token = token_config.get("eos_token")
            self.additional_special_tokens = token_config.get("additional_special_tokens")
            self.tokenizer_class = token_config.get("tokenizer_class")
            self.chat_template = token_config.get("chat_template")

        # Load the vocabylary and the bos and eos tokens (some checkpoints have both files, `.model` takes precedence)
        
        # TODO ISSUE: The SentencePiece tokenizer does not recognize the special tokens in the prompt chat template!
        # if (vocabulary_path := checkpoint_dir / "tokenizer.model").is_file():
        #    from sentencepiece import SentencePieceProcessor

        #    self.processor = SentencePieceProcessor(model_file=str(vocabulary_path)) #  add_bos=False, add_eos=False
        #    self.backend = "sentencepiece"
            
        #    self.bos_id = self.processor.bos_id()
        #    self.eos_id = self.processor.eos_id()

        if (vocabulary_path := checkpoint_dir / "tokenizer.json").is_file():
            from tokenizers import Tokenizer as HFTokenizer

            self.processor = HFTokenizer.from_file(str(vocabulary_path))
            self.backend = "huggingface"

            if (special_tokens_path := checkpoint_dir / "tokenizer_config.json").is_file():
                
                # Load the special tokens ids (beginning and end of sentence)
                if self.bos_token is not None:
                    if isinstance(self.bos_token, dict):
                        self.bos_id = self.token_to_id(self.bos_token['content'])
                    elif isinstance(self.bos_token, str):
                        self.bos_id = self.token_to_id(self.bos_token)
                        
                if self.eos_token is not None:
                    if isinstance(self.eos_token, dict):
                        self.eos_id = self.token_to_id(self.eos_token['content'])
                    elif isinstance(self.eos_token, str):
                        self.eos_id = self.token_to_id(self.eos_token)
           
        
            #if (special_tokens_path := checkpoint_dir / "special_tokens_map.json").is_file():
            #    with open(special_tokens_path) as fp:
            #        self.special_tokens_map = json.load(fp)
            #else:
            #    self.special_tokens_map = dict()

            if (special_tokens_path := checkpoint_dir / "generation_config.json").is_file():
                with open(special_tokens_path) as fp:
                    gen_config = json.load(fp)
                    
                if self.bos_id is None:
                    self.bos_id = gen_config.get("bos_token_id")
                if self.eos_id is None:
                    self.eos_id = gen_config.get("eos_token_id")
        else:
            raise NotImplementedError('The tokenizer file ("tokenizer_config.json") is not found in the checkpoint directory.')
    
    def encode(
        self,
        string: str,
        device: Optional[torch.device] = None,
        add_special_tokens: bool = True,
    ) -> torch.Tensor:
        if self.backend == "huggingface":
            tokens = self.processor.encode(string, add_special_tokens = add_special_tokens).ids
        elif self.backend == "sentencepiece":
            tokens = self.processor.encode(string)
        else:
            raise RuntimeError

        return torch.tensor(tokens, dtype=torch.int, device=device)
    
    def id_to_token(self, id: int) -> str:
        if self.backend == "huggingface":
            token = self.processor.id_to_token(id)
        elif self.backend == "sentencepiece":
            token = self.processor.IdToPiece(id)
        else:
            raise RuntimeError
        if id is None:
            raise ValueError(f"id {id!r} not found in the collection.")
        return token
    
    # -----------------------------------------------------------------------------
    # The following methods is copied from the `transformers` library, for applying chat templates
    # -----------------------------------------------------------------------------
    def apply_chat_template(
        self,
        conversation: Union[List[Dict[str, str]], "Conversation"],
        chat_template: Optional[str] = None,
        add_generation_prompt: bool = False,
        tokenize: bool = True,
        padding: bool = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
        return_tensors = None,
        return_dict: bool = False,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Union[str, List[int]]:
        """
        Converts a Conversation object or a list of dictionaries with `"role"` and `"content"` keys to a list of token
        ids. This method is intended for use with chat models, and will read the tokenizer's chat_template attribute to
        determine the format and control tokens to use when converting. When chat_template is None, it will fall back
        to the default_chat_template specified at the class level.

        Args:
            conversation (Union[List[Dict[str, str]], "Conversation"]): A Conversation object or list of dicts
                with "role" and "content" keys, representing the chat history so far.
            chat_template (str, *optional*): A Jinja template to use for this conversion. If
                this is not passed, the model's default chat template will be used instead.
            add_generation_prompt (bool, *optional*): Whether to end the prompt with the token(s) that indicate
                the start of an assistant message. This is useful when you want to generate a response from the model.
                Note that this argument will be passed to the chat template, and so it must be supported in the
                template for this argument to have any effect.
            tokenize (`bool`, defaults to `True`):
                Whether to tokenize the output. If `False`, the output will be a string.
            padding (`bool`, defaults to `False`):
                Whether to pad sequences to the maximum length. Has no effect if tokenize is `False`.
            truncation (`bool`, defaults to `False`):
                Whether to truncate sequences at the maximum length. Has no effect if tokenize is `False`.
            max_length (`int`, *optional*):
                Maximum length (in tokens) to use for padding or truncation. Has no effect if tokenize is `False`. If
                not specified, the tokenizer's `max_length` attribute will be used as a default.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Has no effect if tokenize is `False`. Acceptable
                values are:
                - `'tf'`: Return TensorFlow `tf.Tensor` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.
            return_dict (`bool`, *optional*, defaults to `False`):
                Whether to return a dictionary with named outputs. Has no effect if tokenize is `False`.
            tokenizer_kwargs (`Dict[str: Any]`, *optional*): Additional kwargs to pass to the tokenizer.
            **kwargs: Additional kwargs to pass to the template renderer. Will be accessible by the chat template.

        Returns:
            `List[int]`: A list of token ids representing the tokenized chat so far, including control tokens. This
            output is ready to pass to the model, either directly or via methods like `generate()`.
        """

        if hasattr(conversation, "messages"):
            # Indicates it's a Conversation object
            conversation = conversation.messages

        if tokenizer_kwargs is None:
            tokenizer_kwargs = {}

        # First, handle the cases when the model has a dict of multiple templates
        if isinstance(self.chat_template, dict) or (
            self.chat_template is None and isinstance(self.default_chat_template, dict)
        ):
            template_dict = self.chat_template or self.default_chat_template
            if chat_template is not None and chat_template in template_dict:
                # The user can pass the name of a template to the chat template argument instead of an entire template
                chat_template = template_dict[chat_template]
            elif chat_template is None and "default" in template_dict:
                chat_template = template_dict["default"]
            elif chat_template is None:
                raise ValueError(
                    "This model has multiple chat templates with no default specified! Please either pass a chat "
                    "template or the name of the template you wish to use to the `chat_template` argument. Available "
                    f"template names are {sorted(template_dict.keys())}."
                )
        elif chat_template is None:
            # These are the cases when the model has a single template
            # priority: `chat_template` argument > `tokenizer.chat_template` > `tokenizer.default_chat_template
            if self.chat_template is not None:
                chat_template = self.chat_template
            else:
                chat_template = self.default_chat_template

        # Compilation function uses a cache to avoid recompiling the same template
        compiled_template = self._compile_jinja_template(chat_template)

        template_kwargs = {**self.special_tokens_map, **kwargs}  # kwargs overwrite special tokens if both are present
        rendered = compiled_template.render(
            messages=conversation, add_generation_prompt=add_generation_prompt, **template_kwargs
        )

        if padding is True:
            padding = "max_length"  # There's only one sequence here, so "longest" makes no sense
        if tokenize:
            if return_dict:
                return self(
                    rendered,
                    padding=padding,
                    truncation=truncation,
                    max_length=max_length,
                    add_special_tokens=False,
                    return_tensors=return_tensors,
                    **tokenizer_kwargs,
                )
            else:
                return self.encode(
                    rendered,
                    #padding=padding,
                    #truncation=truncation,
                    #max_length=max_length,
                    bos = False,
                    eos = False,
                    #add_special_tokens=False,
                    #return_tensors=return_tensors,
                    #**tokenizer_kwargs,
                )
        else:
            return rendered
        
    @lru_cache
    def _compile_jinja_template(self, chat_template):
        try:
            import jinja2
            from jinja2.exceptions import TemplateError
            from jinja2.sandbox import ImmutableSandboxedEnvironment
        except ImportError:
            raise ImportError("apply_chat_template requires jinja2 to be installed.")

        #if version.parse(jinja2.__version__) < version.parse("3.0.0"):
            #raise ImportError("apply_chat_template requires jinja2>=3.0.0 to be installed. Your version is " f"{jinja2.__version__}.")

        def raise_exception(message):
            raise TemplateError(message)

        jinja_env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        jinja_env.globals["raise_exception"] = raise_exception
        return jinja_env.from_string(chat_template)
    
    @property
    def default_chat_template(self):
        """
        This template formats inputs in the standard ChatML format. See
        https://github.com/openai/openai-python/blob/main/chatml.md
        """

        if self.tokenizer_class in ['GPTNeoXTokenizer']:
            
            # Copied from transformers.models.gpt2.tokenization_gpt2.GPT2Tokenizer.default_chat_template
            return "{% for message in messages %}" "{{ message.content }}{{ eos_token }}" "{% endfor %}"
        else:
            return (
                "{% for message in messages %}"
                "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
                "{% endfor %}"
                "{% if add_generation_prompt %}"
                "{{ '<|im_start|>assistant\n' }}"
                "{% endif %}"
            )

    @property
    def special_tokens_map(self) -> Dict[str, Union[str, List[str]]]:
        """
        `Dict[str, Union[str, List[str]]]`: A dictionary mapping special token class attributes (`cls_token`,
        `unk_token`, etc.) to their values (`'<unk>'`, `'<cls>'`, etc.).

        Convert potential tokens of `tokenizers.AddedToken` type to string.
        """
        
        set_attr = {}
        for attr in ["bos_token", "eos_token", "unk_token", "sep_token", "pad_token", "cls_token", "mask_token", "additional_special_tokens"]:
            if attr in self.__dir__():
                attr_value = getattr(self, attr)

                if attr_value:
                    if isinstance(attr_value, dict) and "content" in attr_value.keys():
                        set_attr[attr] = attr_value['content']
                    elif attr_value is not None:
                        set_attr[attr] = attr_value
        return set_attr