import os
import yaml
import torch
import argparse
import pandas as pd
from trl import RLOOConfig

from utils.estimator import trainer, evaluate
from utils.paraphraser import reinforce, paraphrase
from utils.data_prep import set_input_format, train_val_split


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def parse_args():
    """Parse command-line arguments with config file defaults."""
    # First, parse only the config argument
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('--config', type=str, default='config/mistral_config.yaml', help='Path to config YAML file')
    pre_args, _ = pre_parser.parse_known_args()
    
    # Load defaults from config file
    config = load_config(pre_args.config)
    
    # Main parser with config-based defaults
    parser = argparse.ArgumentParser(description='ScoreCLIQ', parents=[pre_parser])
    
    # Device
    device_default = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    if device_default == 'cuda' and not torch.cuda.is_available():
        device_default = 'cpu'
    parser.add_argument('--device', type=str, default=device_default, help='Device to use')
    
    # Data
    data_cfg = config.get('data', {})
    parser.add_argument('--val_size', type=float, default=data_cfg.get('val_size', 0.3), help='Validation set size')
    parser.add_argument('--text_column', type=str, default=data_cfg.get('text_column', 'ItemStem'), help='Text column name')
    parser.add_argument('--target_column', type=str, default=data_cfg.get('target_column', 'Difficulty'), help='Target column name')
    parser.add_argument('--pred_column', type=str, default=data_cfg.get('pred_column', 'PredDifficulty'), help='Prediction column name')
    
    # Estimator
    est_cfg = config.get('estimator', {})
    parser.add_argument('--est_batch_size', type=int, default=est_cfg.get('batch_size', 8), help='Estimator batch size')
    parser.add_argument('--est_epochs', type=int, default=est_cfg.get('epochs', 50), help='Estimator epochs')
    parser.add_argument('--est_learning_rate', type=float, default=est_cfg.get('learning_rate', 2e-5), help='Estimator learning rate')
    parser.add_argument('--est_weight_decay', type=float, default=est_cfg.get('weight_decay', 0.01), help='Estimator weight decay')
    parser.add_argument('--est_model_repo', type=str, default=est_cfg.get('model_repo', 'bert-base-uncased'), help='Estimator model repository')
    parser.add_argument('--scheduler_warmup_steps', type=int, default=est_cfg.get('scheduler_warmup_steps', 0), help='Scheduler warmup steps')
    parser.add_argument('--est_output_dir', type=str, default=est_cfg.get('est_output_dir', './bert-checkpoints'), help='Estimator output directory')
    
    # Subset Selection
    subset_cfg = config.get('subset_selection', {})
    parser.add_argument('--t', type=float, default=subset_cfg.get('t', 0.04), help='Threshold for subset selection')
    
    # LLM Refinement (RLOO)
    rloo_cfg = config.get('rloo', {})
    parser.add_argument('--llm_model_repo', type=str, default=rloo_cfg.get('llm_model_repo', 'mistralai/Mistral-7B-Instruct-v0.3'), help='LLM model repository')
    parser.add_argument('--rloo_output_dir', type=str, default=rloo_cfg.get('output_dir', './rloo-checkpoints'), help='RLOO output directory')
    parser.add_argument('--rloo_batch_size', type=int, default=rloo_cfg.get('batch_size', 1), help='RLOO batch size')
    parser.add_argument('--rloo_grad_accum_steps', type=int, default=rloo_cfg.get('grad_accum_steps', 4), help='RLOO gradient accumulation steps')
    parser.add_argument('--rloo_epochs', type=int, default=rloo_cfg.get('epochs', 1), help='RLOO epochs')
    parser.add_argument('--rloo_learning_rate', type=float, default=rloo_cfg.get('learning_rate', 1e-6), help='RLOO learning rate')
    parser.add_argument('--rloo_logging_steps', type=int, default=rloo_cfg.get('logging_steps', 10), help='RLOO logging steps')
    parser.add_argument('--rloo_save_steps', type=int, default=rloo_cfg.get('save_steps', 50), help='RLOO save steps')
    parser.add_argument('--rloo_num_samples', type=int, default=rloo_cfg.get('num_samples', 4), help='RLOO number of samples')
    
    # Fine-tuning
    ft_cfg = config.get('finetuning', {})
    parser.add_argument('--lamda', type=float, default=ft_cfg.get('lamda', 0.99), help='Lambda for regularization')
    parser.add_argument('--ft_batch_size', type=int, default=ft_cfg.get('batch_size', 8), help='Fine-tuning batch size')
    parser.add_argument('--ft_epochs', type=int, default=ft_cfg.get('epochs', 50), help='Fine-tuning epochs')
    parser.add_argument('--ft_learning_rate', type=float, default=ft_cfg.get('learning_rate', 2e-5), help='Fine-tuning learning rate')
    parser.add_argument('--ft_weight_decay', type=float, default=ft_cfg.get('weight_decay', 0.01), help='Fine-tuning weight decay')
    parser.add_argument('--ft_model_repo', type=str, default=ft_cfg.get('model_repo', 'bert-base-uncased'), help='Fine-tuning model repository')
    parser.add_argument('--ft_scheduler_warmup_steps', type=int, default=ft_cfg.get('scheduler_warmup_steps', 0), help='Fine-tuning scheduler warmup steps')
    parser.add_argument('--ft_output_dir', type=str, default=ft_cfg.get('ft_output_dir', './checkpoints'), help='Fine-tuning output directory')
    
    return parser.parse_args()


args = parse_args()

# Assign args to variables
device = args.device
val_size = args.val_size
text_column = args.text_column
target_column = args.target_column
pred_column = args.pred_column
est_batch_size = args.est_batch_size
est_epochs = args.est_epochs
est_learning_rate = args.est_learning_rate
est_weight_decay = args.est_weight_decay
est_model_repo = args.est_model_repo
scheduler_warmup_steps = args.scheduler_warmup_steps
est_output_dir = args.est_output_dir
t = args.t
llm_model_repo = args.llm_model_repo
rloo_output_dir = args.rloo_output_dir
rloo_batch_size = args.rloo_batch_size
rloo_grad_accum_steps = args.rloo_grad_accum_steps
rloo_epochs = args.rloo_epochs
rloo_learning_rate = args.rloo_learning_rate
rloo_logging_steps = args.rloo_logging_steps
rloo_save_steps = args.rloo_save_steps
rloo_num_samples = args.rloo_num_samples
lamda = args.lamda
ft_batch_size = args.ft_batch_size
ft_epochs = args.ft_epochs
ft_learning_rate = args.ft_learning_rate
ft_weight_decay = args.ft_weight_decay
ft_model_repo = args.ft_model_repo
ft_scheduler_warmup_steps = args.ft_scheduler_warmup_steps
ft_output_dir = args.ft_output_dir

# Make directories for saving models and tokenizers
os.makedirs(est_output_dir, exist_ok=True)
os.makedirs(rloo_output_dir, exist_ok=True)
os.makedirs(ft_output_dir, exist_ok=True)

print(f"Device: {device}")
print(f"Batch Size: {est_batch_size}, Threshold: {t}")

###################
# 1. Loading Data #
###################

print("\n[1] Loading Data...")
train_df = pd.read_excel('data/train_final.xlsx')
test_df = pd.read_excel('data/test_final.xlsx')
gold_df = pd.read_excel('data/gold_final.xlsx')

print(f"Train data shape: {train_df.shape}")
print(f"Test data shape: {test_df.shape}")
print(f"Gold data shape: {gold_df.shape}")

test_df = test_df.merge(gold_df, on='ItemNum')
print(f"Test data after merge: {test_df.shape}")

#########################
# 2. Data Preprocessing #
#########################

print("\n[2] Preprocessing Data...")
train = set_input_format(train_df)
train, val = train_val_split(train, val_size=val_size)
test = set_input_format(test_df)

print(f"Train set shape: {train.shape}")
print(f"Validation set shape: {val.shape}")
print(f"Test set shape: {test.shape}")

####################################
# 3. Baseline Estimator - Training #
####################################

print("\n[3] Training Baseline Estimator (BERT)...")
model, tokenizer = trainer(
    train, 
    val, 
    est_output_dir,
    device=device, 
    batch_size=est_batch_size, 
    epochs=est_epochs, 
    learning_rate=est_learning_rate, 
    weight_decay=est_weight_decay, 
    model_repo=est_model_repo, 
    text_column=text_column, 
    target_column=target_column, 
    scheduler_warmup_steps=scheduler_warmup_steps
)
print("Baseline model training completed.")

#######################
# 4. Subset Selection #
#######################

print("\n[4] Evaluating baseline model and selecting subset...")
texts, preds, targets, _, _ = evaluate(
    pd.concat([train, val], ignore_index=True),
    model, 
    tokenizer, 
    device=device, 
    batch_size=est_batch_size,
    text_column=text_column, 
    target_column=target_column
)

results_df = pd.DataFrame({
    text_column: texts,
    pred_column: preds,
    target_column: targets
})

subset = results_df[abs(results_df[pred_column] - results_df[target_column]) > t].reset_index(drop=True)
print(f"Subset size (predictions diff > {t}): {len(subset)} / {len(results_df)} ({100*len(subset)/len(results_df):.2f}%)")

######################################
# 5. LLM-Guide Refinement - Training #
######################################

print("\n[5] Training LLM-guided refinement model (Mistral)...")
config = RLOOConfig(
    output_dir=rloo_output_dir,
    per_device_train_batch_size=rloo_batch_size,
    gradient_accumulation_steps=rloo_grad_accum_steps,
    num_train_epochs=rloo_epochs,
    learning_rate=rloo_learning_rate,
    logging_steps=rloo_logging_steps,
    save_steps=rloo_save_steps,
    num_samples=rloo_num_samples,
)

llm_model, llm_tokenizer = reinforce(
    subset,
    config,
    reward_model=model,
    device=device,
    model_repo=llm_model_repo,
    text_column=text_column,
    pred_column=pred_column,
    target_column=target_column
)
print("LLM model training completed.")

#######################################
# 6. Estimator Improvement - Training #
#######################################

print("\n[6] Generating paraphrases with LLM and improving estimator...")
subset_df = subset.copy()

for i in range(len(subset_df)):
    row = subset_df.iloc[[i]]
    paraphrased_item = paraphrase(
        row, 
        model=llm_model, 
        tokenizer=llm_tokenizer, 
        device=device, 
        text_column=text_column, 
        pred_column=pred_column, 
        target_column=target_column
    )
    subset_df.at[i, text_column] = paraphrased_item
    if (i + 1) % 10 == 0:
        print(f"  Paraphrased {i + 1}/{len(subset_df)} items")

print("Paraphrasing completed. Evaluating paraphrased items...")
texts, preds, targets, _, _ = evaluate(
    subset_df,
    model, 
    tokenizer, 
    device=device, 
    batch_size=est_batch_size,
    text_column=text_column, 
    target_column=target_column
)

subset_df = pd.DataFrame({
    text_column: texts,
    pred_column: preds,
    target_column: targets
})

L_reg = ((subset_df[pred_column] - subset_df[target_column])**2).mean()
print(f"Regularization loss from paraphrased subset: {L_reg:.6f}")

print("Fine-tuning baseline model with regularization...")
final_model, final_tokenizer = trainer(
    train, 
    val, 
    ft_output_dir,
    model,
    tokenizer,
    device=device, 
    batch_size=ft_batch_size, 
    epochs=ft_epochs, 
    learning_rate=ft_learning_rate, 
    weight_decay=ft_weight_decay, 
    lamda=lamda,
    l_reg=L_reg,
    model_repo=ft_model_repo, 
    text_column=text_column, 
    target_column=target_column, 
    scheduler_warmup_steps=ft_scheduler_warmup_steps
)
print("Final model training completed.")

###############################
# 7. Final Evaluation on Test #
###############################

print("\n[7] Evaluating final model on test set...")
texts, preds, targets, fmse, fmae = evaluate(
    test,
    final_model, 
    final_tokenizer, 
    device=device, 
    batch_size=est_batch_size,
    text_column=text_column, 
    target_column=target_column
)

print(f"\n{'='*50}")
print(f"Final Test MSE: {fmse:.4f} | MAE: {fmae:.4f}")
print(f"{'='*50}\n")