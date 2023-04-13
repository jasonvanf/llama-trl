<h1 align="center">LLaMA-TRL</h1>
<p align="center">Fine-tuning LLaMA with PPO and LoRA</p>

- Collect instruction-following data from this repo [GPT-4-LLM](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM)
- Implement PPO (Proximal Policy Optimization) with TRL (Transformer Reinforcement Learning)
- Implement LoRA (Low-Rank Adaption of Large Language Models) with PEFT (Parameter-Efficient Fine-Tuning)


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
    --output_dir ./supervised_llama
```

- **Step 2 - Training Reward Model**

```
torchrun --nnodes 1 --nproc_per_node 8 training_reward_model.py \
    --model_name 'decapoda-research/llama-7b-hf'
```

- **Step 3 - Tuning LM with PPO**
