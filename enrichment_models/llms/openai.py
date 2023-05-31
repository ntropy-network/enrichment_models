import os
import warnings

import openai
from tqdm import tqdm  # type: ignore

from enrichment_models.llms.llm import LLM
from enrichment_models.llms.utils import validate_llm_inputs


class ChatGPT(LLM):
    SYSTEM_PROMPT: str = (
        "You are a financial assistant. You have to enrich transactions."
    )

    def __init__(
        self,
        openai_api_key: str | None = None,
        model_name: str = "gpt-3.5-turbo",
        max_output_tokens: int = 128,
        temperature: int = 0,
        retry: int = 10,
        request_timeout: int = 15,  # in seconds
    ) -> None:
        # gpt4 doesn't support batch when writing this code
        # This class doesn't use batch, it does prediction sequentially per prompt.
        if openai_api_key is None:
            openai.api_key = os.getenv("OPENAI_API_KEY")
        else:
            openai.api_key = openai_api_key
        self.model_name = model_name
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.retry = retry
        self.request_timeout = request_timeout

    def predict(
        self, instructions: str | list[str], inputs: str | list[str]
    ) -> str | list[str]:
        if isinstance(instructions, str):
            return_str = True
        else:
            return_str = False
        instructions, inputs = validate_llm_inputs(instructions, inputs)
        prompts = [
            f"{instruction}\n{input}"
            for instruction, input in zip(instructions, inputs)
        ]
        results = [
            self._predict(prompt)
            for prompt in tqdm(prompts, total=len(prompts), desc="OpenAI Inference")
        ]
        if return_str:
            return results[0]
        return results

    def _predict(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        for i in range(self.retry + 1):
            try:
                results = openai.ChatCompletion.create(
                    model=self.model_name,
                    max_tokens=self.max_output_tokens,
                    temperature=self.temperature,
                    n=1,
                    messages=messages,
                    request_timeout=self.request_timeout,
                )
                outputs = [
                    choice["message"]["content"]
                    for choice in results["choices"]
                    if choice["finish_reason"] is None
                    or choice["finish_reason"] == "stop"
                ]
                if len(outputs) > 0:
                    return outputs[0]
                warnings.warn("OpenAI answered was empty")
                return (
                    ""  # Case if OpenAI API answers with empty list (shouldn't happen)
                )
            except openai.error.OpenAIError as e:
                print(f"Error {e}, retrying {self.retry - i} times...", flush=True)
                if i == self.retry:
                    raise RuntimeError("OpenAI service unavailable", e)
        return ""
