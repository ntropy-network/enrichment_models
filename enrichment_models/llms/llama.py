from pathlib import Path

import torch
from peft import PeftModel  # type: ignore
from tqdm import tqdm  # type: ignore
from transformers import GenerationConfig  # type: ignore
from transformers import LlamaForCausalLM, LlamaTokenizer

from enrichment_models.llms.llm import LLM
from enrichment_models.llms.prompter import LLMPrompter
from enrichment_models.llms.utils import (
    DEFAULT_GENERATION_CONFIG,
    fix_special_tokens,
    validate_llm_inputs,
)


class LLAMA(LLM):
    finetuning_config_filename: str = "finetuning_config.json"
    tokenizer_folder: str = "tokenizer"

    def __init__(
        self,
        base_model: str = "yahma/llama-7b-hf",
        lora_weights: str = "LouisML/llama_categorization7B",
        generation_config: GenerationConfig = DEFAULT_GENERATION_CONFIG,
        load_in_8bit: bool = True,
        batch_size: int = 1,
        prompt_template: str = "alpaca_short",
        cache_dir: str | None = None,
    ) -> None:
        if torch.cuda.is_available():
            device: str = "cuda"
        else:
            device = "cpu"

        try:
            if torch.backends.mps.is_available():
                device = "mps"
        except:  # noqa: E722
            pass
        self.device = device
        self.batch_size = batch_size
        self.generation_config = generation_config
        self.prompter = LLMPrompter(prompt_template)

        if device == "cuda":
            model = LlamaForCausalLM.from_pretrained(
                base_model,
                load_in_8bit=load_in_8bit,
                torch_dtype=torch.float16,
                device_map="auto",
                cache_dir=cache_dir,
            )
            model = PeftModel.from_pretrained(
                model, lora_weights, torch_dtype=torch.float16, cache_dir=cache_dir
            )
        elif device == "mps":
            model = LlamaForCausalLM.from_pretrained(
                base_model,
                device_map={"": device},
                torch_dtype=torch.float16,
                cache_dir=cache_dir,
            )
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                device_map={"": device},
                torch_dtype=torch.float16,
                cache_dir=cache_dir,
            )
        else:
            model = LlamaForCausalLM.from_pretrained(
                base_model,
                device_map={"": device},
                low_cpu_mem_usage=True,
                cache_dir=cache_dir,
            )
            model = PeftModel.from_pretrained(
                model, lora_weights, device_map={"": device}, cache_dir=cache_dir
            )
        # If adapter got a tokenizer, use it, otherwise use the base one
        if (Path(lora_weights) / self.tokenizer_folder).exists():
            tokenizer: LlamaTokenizer = LlamaTokenizer.from_pretrained(
                Path(lora_weights) / self.tokenizer_folder
            )
        elif (Path(base_model) / self.tokenizer_folder).exists():
            tokenizer = LlamaTokenizer.from_pretrained(
                Path(base_model) / self.tokenizer_folder
            )
        else:
            tokenizer = LlamaTokenizer.from_pretrained(base_model, padding_side="left")

        self.model, self.tokenizer = fix_special_tokens(
            model=model, tokenizer=tokenizer, base_model=base_model
        )

    def _predict(self, instructions: list[str], inputs: list[str]) -> list[str]:
        prompts = [
            self.prompter.generate_prompt(instruction, input)
            for instruction, input in zip(instructions, inputs)
        ]
        encodings = self.tokenizer(prompts, return_tensors="pt", padding=True).to(
            self.device
        )
        with torch.no_grad():
            generation_output = self.model.generate(
                **encodings,
                generation_config=self.generation_config,
                return_dict_in_generate=True,
                output_scores=False,
            )
        outputs = [
            self.tokenizer.decode(s, skip_special_tokens=True)
            for s in generation_output.sequences
        ]
        return [self.prompter.get_response(output) for output in outputs]

    def predict(
        self, instructions: str | list[str], inputs: str | list[str]
    ) -> str | list[str]:
        if isinstance(instructions, str):
            return_str = True
        else:
            return_str = False
        instructions, inputs = validate_llm_inputs(instructions, inputs)
        responses: list[str] = []
        for pos in tqdm(
            range(0, len(instructions), self.batch_size),
            desc=f"Inference (bs={self.batch_size})",
        ):
            batch_instructions = instructions[pos : pos + self.batch_size]
            batch_inputs = inputs[pos : pos + self.batch_size]
            batch_responses = self._predict(batch_instructions, batch_inputs)
            responses.extend(batch_responses)
        if return_str:
            return responses[0]
        return responses
