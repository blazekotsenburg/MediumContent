from dataclasses import dataclass
from dataclasses import dataclass, field

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

@dataclass
class Orchestrator:
    model_id: str                       # required when you instantiate
    device:   str = "cpu"               # "cuda" or "cpu"

    # Internal objects created later; exclude from __init__ & repr
    tokenizer: AutoTokenizer         = field(init=False, repr=False)
    llm:        AutoModelForCausalLM = field(init=False, repr=False)

    def __post_init__(self):
        # -- Tokenizer ----------------------------------------------------
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        # some Llama/Mistral checkpoints miss a pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # -- Model --------------------------------------------------------
        self.llm = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
            device_map={"": 0} if "cuda" in self.device else None
        ).to(self.device)

        # keep model & tokenizer in sync for padding
        self.llm.config.pad_token_id = self.tokenizer.pad_token_id

    def invoke(self,
               messages: list[dict[str, str]],
               device: str,
               max_new_tokens: int = 128) -> str:
        """
        Parameters
        ----------
        messages : [{'role': ..., 'content': ...}, ...]
        device   : "cuda" or "cpu"  (kept for backward-compat)
        Returns
        -------
        str : the model's completion (plain text, JSON-safe)
        """
        device = device or self.device   # stay on the default if None

        # -- Build prompt as *string* (good for logging / DALL·E)
        prompt_str: str = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,               # keep TEXT
            add_generation_prompt=True    # works for chat checkpoints
        )

        # -- Tokenise for generation (tensor IDs)
        enc = self.tokenizer(
            prompt_str,
            return_tensors="pt",
            padding=True
        ).to(device)

        # -- Generate (same call pattern you had)
        out_ids = self.llm.generate(
            enc["input_ids"],
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id
        )

        # -- Decode ONLY the new tokens (skip the prompt part)
        resp_text = self.tokenizer.decode(
            out_ids[0][enc["input_ids"].shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        ).strip()

        return resp_text