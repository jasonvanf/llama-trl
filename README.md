<h1 align="center">LLaMA-TRL</h1>
<p align="center">Fine-tuning LLaMA with PPO and LoRA</p>

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
torchrun --nnodes 1 --nproc_per_node 8 ./supervised_finetuning.py \
    --base_model 'decapoda-research/llama-7b-hf' \
    --dataset_name './datasets/alpaca_gpt4_data.json' \
    --streaming --no_gradient_checkpointing \
    --learning_rate 1e-5 \
    --max_steps 5000 \
    --output_dir ./supervised_lm
```

- **Step 2 - Training Reward Model**

- **Step 3 - Tuning LM with PPO**
