import json
import warnings
from pathlib import Path
from typing import Union


class LLMPrompter(object):
    """
    A dedicated helper to manage templates and prompt building."""

    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "alpaca_short", verbose: bool = False):
        self._verbose = verbose
        file_name = Path(__file__).parent / f"prompt_templates/{template_name}.json"
        if not file_name.exists():
            raise ValueError(f"Can't read {file_name}")
        with open(file_name) as fp:
            self.template = json.load(fp)
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    def generate_prompt(
        self,
        instruction: str,
        input: str | None = None,
        label: str | None = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(instruction=instruction)
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        if self.template["response_split"] is not None:
            try:
                return output.split(self.template["response_split"])[1].strip()
            except IndexError as e:
                warnings.warn(
                    f"output: {output}, not consistent with template. ({self.template['response_split']} not found)"
                )
        return output
