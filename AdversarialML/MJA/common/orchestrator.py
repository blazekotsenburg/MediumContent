from dataclasses import dataclass

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

@dataclass
class Orchestrator:

    _llm: transformers.Pipeline

    def __init__(self, model_id):
        # self._llm = transformers.pipeline(
        #                 "text-generation",
        #                 model=model_id,
        #                 # model_kwargs={"torch_dtype": torch.bfloat16},
        #                 device_map="auto",
        #             )
        
        # self._terminators = [
        #                         self._llm.tokenizer.eos_token_id,
        #                         self._llm.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        #                     

        self._llm = AutoModelForCausalLM.from_pretrained(model_id)
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        
    def invoke(self, messages: list[dict]) -> str:
        # prompt = self._tokenizer.apply_chat_template(messages, tokenize=False)
        inputs = self._tokenizer.apply_chat_template(messages, return_tensors="pt").to(self._llm.device)
        outputs = self._llm.generate(inputs, max_length=1000)

        print(outputs)
        return outputs
        # self._llm(
        #     messages,
        #     max_new_tokens=512,
        #     eos_token_id=self._terminators,
        #     do_sample=True,
        #     temperature=0.7,
        #     # top_p=0.9,
        # )
