# MT Exercise 2: RNNs and Language Modelling

This repository contains the setup, data preparation, and training scripts for training a recurrent neural network (RNN) language model using PyTorch, as well as the experimental setup for dropout parameter tuning.

## Setup Instructions

Ensure you have Python 3.8+ installed, along with `git` and `curl`. 

First, create the virtual environment and install the required dependencies (PyTorch, sacremoses, etc.):

```bash
./scripts/make_virtualenv.sh
source venvs/torch3/bin/activate
./scripts/install_packages.sh
```

## Part 1: Training a Language Model on RFCs

### 1. Data Preparation
For this exercise, the model is trained on a custom dataset of foundational Request for Comments (RFC) documents. 

To download and preprocess the dataset, run:
```bash
./bootstrap.sh
```
**What this does:**
* Downloads a random sample of RFCs from `rfc-editor.org`.
* Cleans the raw text (strips page breaks, diagrams, and TOC fill lines).
* Tokenizes the text using `sacremoses` and lowercases it.
* Applies a vocabulary cutoff (default top 5000 words), replacing rare words with `<unk>`.
* Splits the processed text into `train.txt` (80%), `valid.txt` (10%), and `test.txt` (10%).

### 2. Training the Model
To train the base model with the default hyperparameters (Embedding size: 200, Hidden size: 200, Dropout: 0.5), run:
```bash
./scripts/train.sh
```
This script wraps the PyTorch `main.py` script and outputs the trained model to the `models/` directory.

### 3. Text Generation
Once the model is trained, you can generate a text sample to evaluate its quality:
```bash
./scripts/generate.sh --checkpoint models/model_dp0.5_s42.pt --words 200 --out samples/sample.txt
```

---

## Part 2: Parameter Tuning (Dropout)

To evaluate the effect of dropout on model perplexity, `main.py` has been modified to export epoch-level training and validation metrics to a log file.

### 1. Running the Dropout Sweep
You can train multiple models with varying dropout rates (e.g., 0.0, 0.3, 0.6) by utilizing the `--dropout` and `--log` flags:

```bash
mkdir -p logs

# Example: Training with 0.0 dropout
./scripts/train.sh --dropout 0.0 --save models/model_dp0.0.pt > logs/train_dp0.0.log

# Example: Training with 0.3 dropout
./scripts/train.sh --dropout 0.3 --save models/model_dp0.3.pt > logs/train_dp0.3.log

# Example: Training with 0.6 dropout
./scripts/train.sh --dropout 0.6 --save models/model_dp0.6.pt > logs/train_dp0.6.log
```

### 2. Plotting Results
To generate line plots comparing the perplexities across epochs:
```bash
python3 scripts/plot_perplexities.py <path_to_log_or_csv> --out plots/dropout_comparison.png
```

---

## Documented Modifications

As required by the exercise, the following modifications were made to the given repository:

* **Deleted `scripts/download_data.sh`:** Removed the original Grimm fairy tale dataset script.
* **Added `bootstrap.sh`:** Created a new script in the root directory to download, clean, tokenize, apply a vocabulary cutoff, and split Request for Comments (RFC) documents to serve as the new training corpus.
* **Modified `tools/pytorch-examples/word_language_model/main.py`:** Added an `--export-log` argument and file I/O logic to write per-epoch training and validation perplexities directly to a structured CSV file.
* **Added `scripts/plot_perplexities.py`:** Created a Python script utilizing `matplotlib` to parse the exported logs and generate both the comparative text tables and the perplexity line charts required for the dropout parameter tuning analysis.
* **Modified `scripts/train.sh`:** Updated directory paths to point to the new RFC dataset. Added CLI flags (`--dropout`, `--save`, `--log`) to automate the parameter tuning sweep.
* **Modified `scripts/generate.sh`:** Added argument parsing (`--checkpoint`, `--words`, `--out`) to easily point the text generation script at the different models produced during the dropout sweep.