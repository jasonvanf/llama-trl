<h1 align="center">LLaMA-TRL</h1>
<p align="center">Fine-tuning LLaMA with PPO and LoRA</p>

- Implement PPO (Proximal Policy Optimization) with TRL (Transformer Reinforcement Learning)
- Implement LoRA (Low-Rank Adaption of Large Language Models) with PEFT (Parameter-Efficient Fine-Tuning)
- Collect instruction-following data from this repo [GPT-4-LLM](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM)


### Setup

Install dependencies

```
pip install -r requirements.txt
```

### How to use?

- **Step 1 - Supervised Fine-tuning**

```
torchrun --nnodes 1 --nproc_per_node 8 supervised_finetuning.py \
    --base_model 'decapoda-research/llama-7b-hf' \
    --dataset_name './data/alpaca_gpt4_data.json' \
    --streaming \
    --lr_scheduler_type 'cosine' \
    --learning_rate 1e-5 \
    --max_steps 4000 \
    --output_dir './checkpoints/supervised_llama/'
```

or full weight supervised fine-tuning with DeepSpeed stage-3 (offload)

```
pip install deepspeed
torchrun --nnodes 1 --nproc_per_node 8 supervised_finetuning_full_weight.py \
    --base_model 'decapoda-research/llama-7b-hf' \
    --dataset_name './data/alpaca_gpt4_data.json' \
    --streaming \
    --lr_scheduler_type 'cosine' \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --seq_length 1024 \
    --batch_size 4 \
    --gradient_accumulation_steps 8 \
    --eval_freq 2000 \
    --save_freq 2000 \
    --max_steps 4000 \
    --save_total_limit 1 \
    --deepspeed "./configs/default_offload_opt_param.json" \
    --run_name 'llama-7b-sft-full-weight' \
    --output_dir './checkpoints/supervised_llama_full_weight/'
```

- **Step 2 - Training Reward Model**

```
torchrun --nnodes 1 --nproc_per_node 8 training_reward_model.py \
    --model_name 'decapoda-research/llama-7b-hf' \
    --dataset_name './data/comparison_data.json' \
    --output_dir './checkpoints/training_reward_model/'
```

- **Step 3 - Tuning LM with PPO**

```
accelerate launch --multi_gpu --num_machines 1  --num_processes 8 \
    tuning_lm_with_rl.py \
    --log_with wandb \
    --model_name <LLAMA_FINETUNED_MODEL> \
    --reward_model_name <LLAMA_RM_MODEL> \
    --adafactor False \
    --tokenizer_name <LLAMA_TOKENIZER> \
    --save_freq 100 \
    --output_max_length 128 \
    --batch_size 8 \
    --gradient_accumulation_steps 8 \
    --batched_gen True \
    --ppo_epochs 4 \
    --learning_rate 1.4e-5 \
    --early_stopping True \
    --output_dir './checkpoints/tuning_llama_rl/'
```
