import os

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer
)

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"


def merge_llm_with_lora(base_model_name, adapter_model_name, output_name, push_to_hub=False):
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        return_dict=True,
        torch_dtype=torch.bfloat16
    )

    model = PeftModel.from_pretrained(base_model, adapter_model_name)
    model = model.merge_and_unload()

    if "decapoda" in base_model_name.lower():
        tokenizer = LlamaTokenizer.from_pretrained(base_model_name)
        tokenizer.add_special_tokens(
            {
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
                "pad_token": DEFAULT_PAD_TOKEN,
            }
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=False)

    if push_to_hub:
        print(f"Saving to hub ...")
        model.push_to_hub(f"{base_model_name}-merged", use_temp_dir=False, private=True)
        tokenizer.push_to_hub(f"{base_model_name}-merged", use_temp_dir=False, private=True)
    else:
        output_name = os.path.join(output_name, "final_checkpoint-merged")
        model.save_pretrained(output_name)
        tokenizer.save_pretrained(output_name)
        print(f"Model saved to {output_name}")
