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
    --dataset_name './datasets/alpaca_gpt4_data.json' \
    --streaming --no_gradient_checkpointing \
    --learning_rate 1e-5 \
    --max_steps 5000 \
    --output_dir './checkpoints/supervised_llama/'
```

- **Step 2 - Training Reward Model**

```
torchrun --nnodes 1 --nproc_per_node 8 training_reward_model.py \
    --model_name=decapoda-research/llama-7b-hf \
    --dataset_name=./datasets/comparison_data_v2.json \
    --output_dir=./checkpoints/training_reward_model/
```

- **Step 3 - Tuning LM with PPO**

```
accelerate launch --multi_gpu --num_machines 1  --num_processes 8 \
    tuning_lm_with_rl.py \
    --log_with=wandb \
    --model_name=<LLAMA_FINETUNED_MODEL> \
    --reward_model_name=<LLAMA_RM_MODEL> \
    --adafactor=False \
    --tokenizer_name=<LLAMA_TOKENIZER> \
    --save_freq=100 \
    --output_max_length=128 \
    --batch_size=8 \
    --gradient_accumulation_steps=8 \
    --batched_gen=True \
    --ppo_epochs=4 \
    --seed=0 \
    --learning_rate=1.4e-5 \
    --early_stopping=True \
    --output_dir=./checkpoints/tuning_llama_rl/
```
