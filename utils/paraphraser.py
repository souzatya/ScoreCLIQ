import torch
import pandas as pd
from trl import RLOOTrainer
from datasets import Dataset
from estimator import evaluate
from transformers import AutoTokenizer, AutoModelForCausalLM

def get_rewards(df, reward_model=None, device='cuda', checkpoint='checkpoints/bert_regressor_2_454e-05_mse_0_0601.pt', text_column='ItemStem', target_column='Difficulty'):
    rewards = []
    for i in range(len(df)):
        row = df.iloc[[i]]
        if reward_model is not None:
            _, pred, target, _, _ = evaluate(row, model=reward_model, device=device, text_column=text_column, target_column=target_column)
        _, pred, target, _, _ = evaluate(row, device=device, checkpoint=checkpoint, text_column=text_column, target_column=target_column)
        reward = -((pred[0] - target[0])**2)  # Higher reward for better alignment with true difficulty
        rewards.append(reward)
    return rewards


def reinforce(df, config, reward_model=None, device='cuda', model_repo='mistralai/Mistral-7B-Instruct-v0.3', bert_checkpoint='checkpoints/bert_regressor_2_454e-05_mse_0_0601.pt', text_column='ItemStem', pred_column='PredDifficulty', target_column='Difficulty'):
    # Load paraphraser model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_repo)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_repo,
        torch_dtype=torch.float16,
        device_map=device
    )

    # Preprocess Data
    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(lambda x: {
        "prompt" : [
            {"role": "user", "content": f"Original Item:\n {x[text_column]};\n\n Predicted Difficulty: {x[pred_column]}; True Difficulty: {x[target_column]}. Rewrite the Item to better align with the True Difficulty."},
        ]
    })

    def reward_fn(prompts, completions, **kwargs):
        # Convert completions (paraphrased items) to DataFrame for reward calculation
        df_samples = pd.DataFrame({
            text_column: completions,
            target_column: [p[0]['content'].split('True Difficulty: ')[1].split('. Rewrite')[0] for p in prompts]
        })
        if reward_model is not None:
            rewards = get_rewards(df_samples, reward_model, device=device, text_column=text_column, target_column=target_column)
        else:
            rewards = get_rewards(df_samples, device=device, checkpoint=bert_checkpoint, text_column=text_column, target_column=target_column)
        return rewards

    trainer = RLOOTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        tokenizer=tokenizer,
        reward_fn=reward_fn,
    )

    trainer.train()

    return trainer.model, trainer.tokenizer

def paraphrase(row, model=None, tokenizer=None, device='cuda', checkpoint='rloo-checkpoints/checkpoint-50', text_column='ItemStem', pred_column='PredDifficulty', target_column='Difficulty'):
    if model==None and tokenizer==None:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            torch_dtype=torch.float16,
            device_map=device
        )
        print(f"Checkpoint loaded from {checkpoint}")

    messages = [
        {"role": "user", "content": f"Original Item:\n {row[text_column]};\n\n Predicted Difficulty: {row[pred_column]}; True Difficulty: {row[target_column]}. Rewrite the Item to better align with the True Difficulty."},
    ]

    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=120,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

    # Extract only generated part
    generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    print(response)
    return response