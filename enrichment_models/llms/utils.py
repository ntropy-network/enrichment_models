from transformers import GenerationConfig  # type: ignore
from transformers import LlamaForCausalLM, LlamaTokenizer


def validate_llm_inputs(
    instructions: str | list[str], inputs: str | list[str]
) -> tuple[list[str], list[str]]:
    if isinstance(instructions, str):
        instructions = [instructions]
    if isinstance(inputs, str):
        inputs = [inputs]
    assert len(instructions) == len(inputs), (
        "instructions and inputs should have same length:"
        f"instructions length: {len(instructions)}"
        f"inputs length: {len(inputs)}"
    )
    return instructions, inputs


# This util function is to handle the fact that LLAMA model don't have a padding token
def fix_special_tokens(
    model: LlamaForCausalLM, tokenizer: LlamaTokenizer, base_model: str
) -> tuple[LlamaForCausalLM, LlamaTokenizer]:
    if "llama" in base_model:
        model.config.pad_token_id = tokenizer.unk_token_id

    if "decapoda-research" in base_model:
        # unwind broken decapoda-research config
        model.config.pad_token_id = tokenizer.unk_token_id = 0
        model.config.bos_token_id = tokenizer.bos_token_id = 1
        model.config.eos_token_id = tokenizer.eos_token_id = 2
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = "left"  # Allow batched inference
    return model, tokenizer


# greedy search
DEFAULT_GENERATION_CONFIG: GenerationConfig = GenerationConfig(
    num_beams=1, max_new_tokens=32, do_sample=False
)
