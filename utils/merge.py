import peft
import torch
from peft import PeftModel
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, HfArgumentParser

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"


def merge_llm_with_lora(base_model_name, adapter_model_name, output_name, push_to_hub=False):
    model = AutoModelForCausalLM.from_pretrained(base_model_name, return_dict=True,
                                                 torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    config = AutoConfig.from_pretrained(base_model_name)
    architecture = config.architectures[0]
    if "Llama" in architecture:
        print("Setting EOS, BOS, and UNK tokens for LLama tokenizer")
        tokenizer.add_special_tokens(
            {
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
                "pad_token": DEFAULT_PAD_TOKEN,
            }
        )

    # Load the Lora model
    model = PeftModel.from_pretrained(model, adapter_model_name)
    model.eval()

    key_list = [key for key, _ in model.base_model.model.named_modules() if "lora" not in key]
    for key in key_list:
        parent, target, target_name = model.base_model._get_submodules(key)
        if isinstance(target, peft.tuners.lora.Linear):
            bias = target.bias is not None
            new_module = torch.nn.Linear(target.in_features, target.out_features, bias=bias)
            model.base_model._replace_module(parent, target_name, new_module, target)

    model = model.base_model.model
    model.save_pretrained(f"{output_name}")

    if push_to_hub:
        model.push_to_hub(f"{output_name}", use_temp_dir=False)


assert adapter_model_name is not None, "please provide the name of the Adapter you would like to merge"
assert base_model_name is not None, "please provide the name of the Base model"
assert base_model_name is not None, "please provide the output name of the merged model"
