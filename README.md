# ScoreCLIQ

**A Dynamic LLM-Based Framework for Item Difficulty Estimation**

[![Paper](https://img.shields.io/badge/Paper-AIED%202025-blue)](https://doi.org/10.1007/978-3-031-99264-3_21)
[![GitHub](https://img.shields.io/badge/GitHub-ScoreCLIQ-green)](https://github.com/souzatya/ScoreCLIQ)

ScoreCLIQ is a modular framework for dynamic item difficulty estimation that integrates model feedback with paraphrastic refinement. Unlike traditional approaches that treat difficulty as a static property, ScoreCLIQ accounts for how subtle linguistic variations—such as stem rewording or distractor phrasing—can shift perceived complexity.

The framework achieves a **12.46% reduction in RMSE** over the previous state-of-the-art on the BEA 2024 shared task dataset.

## Key Results

| Method | RMSE ↓ |
|--------|--------|
| **ScoreCLIQ (Mistral-7B-Instruct)** | **0.246** |
| ScoreCLIQ (Llama-3.2-3B-Instruct) | 0.259 |
| ScoreCLIQ (Llama-3.2-1B-Instruct) | 0.272 |
| UnibucLLM (Previous SOTA) | 0.281 |
| EduTec | 0.299 |

## Features

- **BERT-based Difficulty Estimator**: Fine-tuned BERT encoder with regression head for difficulty prediction
- **LLM-guided Refinement**: Uses Mistral-7B (or Llama variants) fine-tuned via REINFORCE to generate difficulty-aware paraphrases
- **Subset Selection**: Automatically identifies high-error items where `|prediction - target| > threshold`
- **Consistency-based Regularization**: Improves the estimator by enforcing prediction consistency between original and paraphrased items
- **No Additional Annotations Required**: Feedback-driven loop operates without human supervision

## Installation

```bash
# Clone the repository
git clone https://github.com/souzatya/ScoreCLIQ.git
cd ScoreCLIQ

# Install dependencies
pip install -r requirements.txt
```

### Quick Start

```bash
python main.py --config config/mistral_config.yaml
```

### Requirements

- Python 3.8+
- PyTorch
- Transformers
- TRL (Transformer Reinforcement Learning)
- pandas
- scikit-learn
- PyYAML
- GPU with ~24GB VRAM (e.g., A100) for LLM training

## Project Structure

```
ScoreCLIQ/
├── main.py                 # Main training pipeline
├── config/
│   └── mistral_config.yaml # Default configuration
├── data/
│   ├── train_final.xlsx    # Training data
│   ├── test_final.xlsx     # Test data
│   └── gold_final.xlsx     # Gold standard labels
├── models/
│   └── BertRegressor.py    # BERT-based regression model
├── utils/
│   ├── data_prep.py        # Data preprocessing utilities
│   ├── estimator.py        # Training and evaluation functions
│   └── paraphraser.py      # LLM paraphrasing with RLOO
└── checkpoints/            # Saved model checkpoints
```

## Configuration

Configuration is managed via YAML files with command-line overrides. Default config: `config/mistral_config.yaml`

```yaml
device: cuda

data:
  val_size: 0.3
  text_column: ItemStem
  target_column: Difficulty
  pred_column: PredDifficulty

estimator:
  batch_size: 8
  epochs: 50
  learning_rate: 2e-5
  weight_decay: 0.01
  model_repo: bert-base-uncased
  scheduler_warmup_steps: 0
  est_output_dir: ./bert-checkpoints

subset_selection:
  t: 0.04

rloo:
  llm_model_repo: mistralai/Mistral-7B-Instruct-v0.3
  output_dir: ./rloo-checkpoints
  batch_size: 1
  grad_accum_steps: 4
  epochs: 1
  learning_rate: 1e-6
  logging_steps: 10
  save_steps: 50
  num_samples: 4

finetuning:
  lamda: 0.99
  batch_size: 8
  epochs: 50
  learning_rate: 2e-5
  weight_decay: 0.01
  model_repo: bert-base-uncased
  scheduler_warmup_steps: 0
  ft_output_dir: ./checkpoints
```

## Usage

### Basic Run

```bash
python main.py
```

### Custom Configuration

```bash
# Use a different config file
python main.py --config config/custom_config.yaml

# Override specific parameters
python main.py --est_epochs 100 --t 0.05 --device cpu

# Combine config file with overrides
python main.py --config config/custom.yaml --rloo_epochs 2 --lamda 0.95
```

### Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--config` | Path to YAML config file | `config/mistral_config.yaml` |
| `--device` | Device (cuda/cpu) | `cuda` |
| `--val_size` | Validation split ratio | `0.3` |
| `--est_epochs` | Estimator training epochs | `50` |
| `--est_learning_rate` | Estimator learning rate | `2e-5` |
| `--t` | Subset selection threshold | `0.04` |
| `--llm_model_repo` | LLM model repository | `mistralai/Mistral-7B-Instruct-v0.3` |
| `--rloo_epochs` | RLOO training epochs | `1` |
| `--lamda` | Regularization lambda | `0.99` |
| `--ft_epochs` | Fine-tuning epochs | `50` |

Run `python main.py --help` for full list of arguments.

## Methodology

ScoreCLIQ consists of three main stages:

### Stage 1: Baseline Estimation
A BERT encoder with regression head is trained using MSE loss:

$$\mathcal{L}_{base}(\theta) = \frac{1}{N} \sum_{i=1}^{N} (f_\theta(Q_i) - D_i)^2$$

After training, high-error items are selected: $\tilde{\mathcal{D}} = \{(Q_i, D_i) \in \mathcal{D} \mid |f_\theta(Q_i) - D_i| > t\}$

### Stage 2: LLM-Guided Refinement
An LLM (Mistral-7B) is fine-tuned via REINFORCE to paraphrase high-error items. The reward signal comes from the frozen estimator:

$$R(Q_i, Q_i^*) = -(f_\theta(Q_i^*) - D_i)^2$$

The LLM is prompted with: *"Original item: [Q]; Predicted difficulty: [value]; True difficulty: [value]; Rewrite the item to better align with the true difficulty."*

### Stage 3: Estimator Improvement
The estimator is refined using a consistency-based regularization loss:

$$\mathcal{L}_{final}(\theta) = \mathcal{L}_{base}(\theta) + \lambda \mathcal{L}_{reg}(\theta)$$

where $\mathcal{L}_{reg}$ enforces prediction consistency between original and paraphrased items.

## Pipeline Steps

1. **Data Loading**: Load training, test, and gold standard datasets
2. **Preprocessing**: Format items and split into 70% train / 30% validation
3. **Baseline Estimator Training**: Train BERT regressor on difficulty prediction
4. **Subset Selection**: Identify items where `|prediction - target| > t` (default t=0.04)
5. **LLM Refinement**: Train LLM via REINFORCE to generate difficulty-aware paraphrases
6. **Estimator Fine-tuning**: Re-train estimator with regularization (λ=0.99)
7. **Final Evaluation**: Evaluate on test set and report RMSE/MAE

## Data Format

Input Excel files should contain clinical multiple-choice questions with:

| Column | Description |
|--------|-------------|
| `ItemNum` | Unique item identifier |
| `ItemStem_Text` | Question stem text |
| `Answer__A` to `Answer__J` | Answer options |
| `Difficulty` | Ground-truth scalar difficulty score |

Items are concatenated into a single input: question stem + answer options.

## Model Architecture

### BertRegressor

```
BERT (bert-base-uncased)
    ↓
[CLS] token pooled output (768-dim)
    ↓
Dropout (0.1)
    ↓
Linear (768 → 1)
    ↓
Difficulty Score
```

## Dataset

Evaluated on the **BEA 2024 Shared Task dataset** containing multiple-choice clinical questions annotated with scalar difficulty scores:
- Training set: 466 questions (split 70/30 for train/val)
- Test set: 201 questions

Each item includes question stem, answer options, correct key, and ground-truth difficulty.

## Hyperparameter Sensitivity

| Threshold (t) | RMSE | | Lambda (λ) | RMSE |
|---------------|------|-|------------|------|
| 0.10 | 0.291 | | 1.00 | 0.249 |
| 0.08 | 0.282 | | **0.99** | **0.246** |
| 0.06 | 0.273 | | 0.98 | 0.246 |
| **0.04** | **0.246** | | 0.97 | 0.247 |
| 0.02 | 0.251 | | 0.96 | 0.250 |

## License

MIT License

## Citation

If you use ScoreCLIQ in your research, please cite:

```bibtex
@inproceedings{sarkar2025scorecliq,
  title={ScoreCLIQ: A Dynamic LLM-Based Framework for Item Difficulty Estimation},
  author={Sarkar, Soujatya and Ravikiran, Manikandan and Saluja, Rohit},
  booktitle={Artificial Intelligence in Education (AIED 2025)},
  series={Communications in Computer and Information Science},
  volume={2591},
  pages={169--176},
  year={2025},
  publisher={Springer Nature Switzerland},
  doi={10.1007/978-3-031-99264-3_21}
}
```

## Acknowledgments

This work is supported by and is part of [BharatGen](https://bharatgen.tech/), an Indian Government-funded initiative focused on developing multimodal large language models for Indian languages, also funded by the Department of Science & Technology (DST).

## Authors

- **Soujatya Sarkar** - Indian Institute of Technology Mandi
- **Manikandan Ravikiran** - Indian Institute of Technology Mandi  
- **Rohit Saluja** - Indian Institute of Technology Mandi & BharatGen Consortium

Contact: s23106@students.iitmandi.ac.in