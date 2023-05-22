import os
import argparse

from tqdm import tqdm
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
    TrainingArguments,
    logging,
    set_seed
)
from trl import SFTTrainer
from trl.trainer import ConstantLengthDataset

from utils.merge import merge_llm_with_lora


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="")
    parser.add_argument("--dataset_name", type=str, default="./data/alpaca_gpt4_data.json")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--size_valid_set", type=int, default=4000)
    parser.add_argument("--streaming", action="store_true", default=False)
    parser.add_argument("--shuffle_buffer", type=int, default=5000)

    parser.add_argument("--seq_length", type=int, default=1024)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--eos_token_id", type=int, default=49152)

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", type=str, default=None)

    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")
    parser.add_argument("--num_warmup_steps", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--warmup_ratio", type=float, default=0.)

    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--no_bf16", action="store_false", default=True)
    parser.add_argument("--no_gradient_checkpointing", action="store_false", default=True)
    parser.add_argument("--seed", type=int, default=1103)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="./checkpoints/supervised_llama/")
    parser.add_argument("--log_freq", type=int, default=1)
    parser.add_argument("--eval_freq", type=int, default=1000)
    parser.add_argument("--save_freq", type=int, default=1000)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--run_name", type=str, default="llama-supervised-finetuned")
    parser.add_argument("--merge_lora", action="store_true", default=False)

    return parser.parse_args()


def chars_token_ratio(dataset, tokenizer, nb_examples=400):
    """
    Estimate the average number of characters per token in the dataset.
    """
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        text = prepare_sample_text(example)
        total_characters += len(text)
        if tokenizer.is_fast:
            total_tokens += len(tokenizer(text).tokens())
        else:
            total_tokens += len(tokenizer.tokenize(text))

    return total_characters / total_tokens


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def prepare_sample_text(data_point):
    """Prepare the text from a sample of the dataset."""
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{data_point["instruction"]}

### Input:
{data_point["input"]}

### Response:
{data_point["output"]}"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{data_point["instruction"]}

### Response:
{data_point["output"]}"""


def create_datasets(tokenizer, args):
    data_path = args.dataset_name
    data_kwargs = {
        "split": args.split,
        "num_proc": args.num_workers if not args.streaming else None,
        "streaming": args.streaming,
    }
    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        dataset = load_dataset("json", data_files=data_path, **data_kwargs)
    else:
        dataset = load_dataset(data_path, **data_kwargs)

    if args.streaming:
        print("Loading the dataset in streaming mode")
        valid_data = dataset.take(args.size_valid_set)
        train_data = dataset.skip(args.size_valid_set)
        train_data = train_data.shuffle(buffer_size=args.shuffle_buffer, seed=args.seed)
    else:
        dataset = dataset.train_test_split(test_size=0.1, seed=args.seed)
        train_data = dataset["train"]
        valid_data = dataset["test"]
        print(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}")

    chars_per_token = chars_token_ratio(train_data, tokenizer)
    print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")

    train_dataset = ConstantLengthDataset(
        tokenizer,
        train_data,
        formatting_func=prepare_sample_text,
        infinite=True,
        seq_length=args.seq_length,
        chars_per_token=chars_per_token,
    )
    valid_dataset = ConstantLengthDataset(
        tokenizer,
        valid_data,
        formatting_func=prepare_sample_text,
        infinite=False,
        seq_length=args.seq_length,
        chars_per_token=chars_per_token,
    )
    return train_dataset, valid_dataset


def run_training(args, train_data, val_data, tokenizer=None):
    print("Loading the model")

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    train_data.start_iteration = 0

    print("Starting main loop")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        dataloader_drop_last=True,
        evaluation_strategy="steps",
        max_steps=args.max_steps,
        eval_steps=args.eval_freq,
        save_steps=args.save_freq,
        logging_steps=args.log_freq,
        save_total_limit=args.save_total_limit,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.num_warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.no_gradient_checkpointing,
        fp16=args.fp16,
        bf16=args.no_bf16,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        run_name=args.run_name,
        report_to="wandb",
        ddp_find_unused_parameters=False if int(os.environ.get("WORLD_SIZE", 1)) != 1 else None,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        load_in_8bit=True,
        device_map={"": Accelerator().local_process_index},
    )

    if args.resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            args.resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                args.resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            args.resume_from_checkpoint = None

        if os.path.exists(checkpoint_name):
            import torch
            from peft import (
                get_peft_model,
                prepare_model_for_int8_training,
                set_peft_model_state_dict
            )
            print(f"Restarting from {checkpoint_name}")
            model = prepare_model_for_int8_training(model)
            model = get_peft_model(model, lora_config)

            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        peft_config=lora_config,
        max_seq_length=args.seq_length,
        packing=True,
    )

    print_trainable_parameters(model)

    print("Training...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    print("Saving last checkpoint of the model")
    final_model_path = os.path.join(args.output_dir, "final_checkpoint/")
    trainer.model.save_pretrained(final_model_path)

    if args.merge_lora:
        merge_llm_with_lora(args.base_model, final_model_path, args.output_dir)


def main(args):
    if "decapoda" in args.base_model.lower():
        tokenizer = LlamaTokenizer.from_pretrained(args.base_model)
        tokenizer.add_special_tokens(
            {
                "eos_token": "</s>",
                "bos_token": "</s>",
                "unk_token": "</s>",
                "pad_token": "</s>",
            }
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=False)
        if getattr(tokenizer, "pad_token", None) is None:
            tokenizer.pad_token = tokenizer.eos_token

    train_dataset, eval_dataset = create_datasets(tokenizer, args)
    run_training(args, train_dataset, eval_dataset, tokenizer)


if __name__ == "__main__":
    args = get_args()
    assert args.base_model != "", "Please provide the llama model path"

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    logging.set_verbosity_error()

    main(args)
