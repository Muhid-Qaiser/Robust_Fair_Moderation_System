# Robust Fair Moderation System

This project implements a robust, fair, and adversarial-resistant moderation system for detecting toxic comments in online platforms. It is based on fine-tuning DistilBERT for binary toxicity classification, with additional components for fairness auditing, adversarial attack simulations (e.g., character-level evasion and label-flipping poisoning), and mitigation strategies. The system includes a multi-layered moderation pipeline that combines rule-based filtering, calibrated model predictions, and human-in-the-loop review.

## Project Overview

The project is divided into five parts, each addressing a key aspect of building a reliable moderation system:

- **Part 1**: Data preparation, model training, threshold optimization, and baseline evaluation.
- **Part 2**: Fairness audit on protected attributes (e.g., racial groups) using metrics like disparate impact and AIF360.
- **Part 3**: Adversarial attack simulations, including character-level evasion and label-flipping poisoning.
- **Part 4**: Mitigation strategies, such as reweighting, oversampling, and model calibration.
- **Part 5**: End-to-end pipeline integration and evaluation.

The final output is a `ModerationPipeline` class in `pipeline.py` that provides a structured decision-making process for moderating text.

## Features

- **Toxicity Detection**: Fine-tuned DistilBERT model for binary classification (toxic/non-toxic).
- **Fairness Auditing**: Evaluation of bias across subgroups using standard metrics and AIF360.
- **Adversarial Robustness**: Simulation and mitigation of evasion attacks and poisoning.
- **Multi-Layered Pipeline**: Rule-based input filtering, calibrated model predictions, and review thresholds.
- **Visualization**: Extensive plots for model performance, fairness, and attack impacts.

## Setup Instructions

### Prerequisites
- Python 3.8+
- GPU recommended for training (CUDA-compatible).
- Access to the Jigsaw Multilingual Toxic Comment Classification dataset (place it in a `Data/` folder as referenced in the notebooks).

### Installation
1. Clone or download this repository.
2. Install dependencies: pip install -r requirements.txt
3. Ensure the dataset paths in the notebooks match your local setup (e.g., update `train_path` in the code).

### Data Preparation
- Download the Jigsaw dataset from Kaggle or the original source.
- The notebooks expect CSV files with columns: `comment_text`, `toxic`, `black`, `white`, `muslim`, `jewish`, `other_sexual_orientation`.
- Run Part 1 to preprocess and split the data.

## Usage

### Training and Evaluation
1. Run the Jupyter notebooks in order (part1.ipynb to part5.ipynb).
2. Key outputs:
- Trained models saved in directories like `distilbert-toxic/`, `distilbert-oversampled/`, etc.
- Evaluation metrics, fairness scores, and attack success rates printed in the notebooks.

### Using the Moderation Pipeline
- Use `pipeline.py` for inference on new text.
- Example:
  ```python
  from pipeline import ModerationPipeline

  pipeline = ModerationPipeline(model_path="./distilbert-toxic-mitigated")
  # Fit calibrator with validation data (texts and labels)
  pipeline.fit_calibrator(val_texts, val_labels)
  
  result = pipeline.predict("Your text here")
  print(result)  # {'decision': 'block'/'allow'/'review', 'layer': '...', 'confidence': 0.XX}

## Key Thresholds and Configurations
Toxicity threshold: 0.5 for binarization.
Decision threshold: 0.4 (optimized for F1 score).
Pipeline thresholds: Block if confidence >= 0.6, allow if <= 0.4, review otherwise.

## Project Structure
robust-fair-moderation-system/
├── part1.ipynb          # Data prep, training, and baseline evaluation
├── part2.ipynb          # Fairness audit
├── part3.ipynb          # Adversarial attacks
├── part4.ipynb          # Mitigation strategies
├── part5.ipynb          # Pipeline integration
├── pipeline.py          # Moderation pipeline class
├── requirements.txt     # Python dependencies
├── ReadMe.md            # This file
├── distilbert-toxic/    # Baseline model checkpoints
├── distilbert-oversampled/  # Oversampled model
├── distilbert-reweighed/    # Reweighed model
├── distilbert-poisoned/     # Poisoned model
├── distilbert-toxic-mitigated/  # Final mitigated model
└── tmp_trainer/         # Temporary training files

## Results Summary
Baseline Model: F1 Macro ~0.82 at threshold 0.4.
Fairness: Disparate impact observed in racial subgroups; mitigated via reweighting.
Adversarial Attacks: Evasion attack ASR ~XX% (varies); poisoning degrades F1 by ~YY%.
Mitigation: Improved fairness and robustness through oversampling and calibration.
Contributing
This is an academic project. For improvements, focus on enhancing fairness metrics or adding more attack types.

## Contributing
This is an academic project. For improvements, focus on enhancing fairness metrics or adding more attack types.

### License
MIT.


### requirements.txt Content
Based on the imports in your notebooks and pipeline.py, here's the requirements.txt. I've included approximate versions for stability (based on common compatibility). Adjust as needed for your environment.

torch>=2.0.0
transformers>=4.21.0
scikit-learn>=1.3.0
pandas>=1.5.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
aif360>=0.5.0
fairlearn>=0.8.0
accelerate>=0.20.0
jupyter>=1.0.0
ipykernel>=6.0.0

If you need to update these (e.g., for specific versions or additional packages), let me know more details about your environment or any errors encountered during installation. If the project has specific version constraints from the assignment, provide them for refinement.



