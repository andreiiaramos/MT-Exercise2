# MT Exercise 2: RNNs and Language Modelling

## Setup Instructions

Ensure you have Python 3.8+ installed, along with `git` and `curl`.

First, create the virtual environment and install the required dependencies (PyTorch, sacremoses, etc.):

```bash
./scripts/make_virtualenv.sh
source venvs/torch3/bin/activate
./scripts/install_packages.sh
```

---

## Part 1: Training a Language Model on RFCs

### 1. Dataset

For this exercise, the model is trained on a custom corpus of foundational Request for Comments (RFC) documents sourced from [rfc-editor.org](https://www.rfc-editor.org).

RFCs are the technical and procedural documents that define the standards of the Internet — covering protocols such as TCP, IP, HTTP, DNS, and many others. The corpus was assembled by downloading a random sample of RFCs, cleaning and tokenizing the raw text, applying a vocabulary cutoff of 5000 words, and splitting the result into training, validation, and test sets.

**Dataset attributes and expected influence on text generation:**

RFC documents have several distinctive properties that are expected to influence the behaviour of the language model:

- **Highly technical, domain-specific vocabulary.** RFCs use precise protocol terminology (e.g., *datagram*, *octet*, *acknowledgement*, *encapsulation*). With a vocabulary cutoff of 5000 words, a substantial portion of rare technical terms will be collapsed to `<unk>`, which will limit the model's ability to reproduce accurate protocol-level detail.
- **Formal, structured prose.** RFCs follow a rigid document structure: numbered sections, defined terminology blocks, requirement statements using normative language (MUST, SHOULD, MAY as defined by RFC 2119). The model may partially learn this register and produce output that superficially resembles standards prose.
- **Repetitive syntactic patterns.** RFC text reuses sentence templates heavily (e.g., *"The sender MUST... The receiver SHOULD..."*). This regularity makes the corpus relatively learnable for an LSTM, and the generated text is likely to echo these patterns even if the semantics are incoherent.
- **No narrative structure.** Unlike fiction or news corpora, RFCs have no story arc, dialogue, or temporal progression. The model will not learn any narrative coherence — generated text will consist of locally plausible sentences that do not build toward any larger meaning.

### 2. Data Preparation

To download and preprocess the dataset, run:

```bash
./bootstrap.sh
```

**What this does:**
- Downloads a random sample of RFCs from `rfc-editor.org`.
- Cleans the raw text (strips page breaks, ASCII diagrams, and table-of-contents fill lines).
- Tokenizes the text using `sacremoses` and lowercases it.
- Applies a vocabulary cutoff (top 5000 words), replacing rare words with `<unk>`.
- Splits the processed text into `train.txt` (80%), `valid.txt` (10%), and `test.txt` (10%).

### 3. Training the Model

To train the base model with the default hyperparameters (Embedding size: 200, Hidden size: 200, Dropout: 0.5, Epochs: 40), run:

```bash
./scripts/train.sh
```

This wraps the PyTorch `main.py` script and saves the trained model to the `models/` directory.

### 4. Text Generation

Once the model is trained, generate a sample to evaluate its quality:

```bash
./scripts/generate.sh --checkpoint models/model_dp0.5_s42.pt --words 200 --out samples/sample.txt
```

The generated text is saved to `samples/sample.txt`.

**Impressions of the generated sample:**

[*To be completed after training finishes. Describe here: does the output resemble RFC prose? Does it reproduce normative language patterns (MUST/SHOULD)? Is it locally coherent at the sentence level? Does vocabulary look plausible or dominated by `<unk>`?*]

---

## Part 2: Parameter Tuning — Dropout

To evaluate the effect of dropout on model perplexity, `tools/pytorch-examples/word_language_model/main.py` has been modified to accept an `--export-log` flag that writes per-epoch training and validation perplexities to a CSV file.

### 1. Training Models with Varying Dropout

Train six models with dropout values of 0.0, 0.1, 0.2, 0.3, 0.5, and 0.7. Run each command sequentially:

```bash
mkdir -p logs models

./scripts/train.sh --dropout 0.0 --epochs 40 --emsize 200 --nhid 200 --seed 42 \
  --save models/model_dp0.0_s42.pt --log logs/log_dp0.0.csv

./scripts/train.sh --dropout 0.1 --epochs 40 --emsize 200 --nhid 200 --seed 42 \
  --save models/model_dp0.1_s42.pt --log logs/log_dp0.1.csv

./scripts/train.sh --dropout 0.2 --epochs 40 --emsize 200 --nhid 200 --seed 42 \
  --save models/model_dp0.2_s42.pt --log logs/log_dp0.2.csv

./scripts/train.sh --dropout 0.3 --epochs 40 --emsize 200 --nhid 200 --seed 42 \
  --save models/model_dp0.3_s42.pt --log logs/log_dp0.3.csv

./scripts/train.sh --dropout 0.5 --epochs 40 --emsize 200 --nhid 200 --seed 42 \
  --save models/model_dp0.5_s42.pt --log logs/log_dp0.5.csv

./scripts/train.sh --dropout 0.7 --epochs 40 --emsize 200 --nhid 200 --seed 42 \
  --save models/model_dp0.7_s42.pt --log logs/log_dp0.7.csv
```

Each run produces a CSV log at `logs/log_dp<value>.csv` with columns `epoch`, `train_ppl`, `valid_ppl`, and a model checkpoint at `models/model_dp<value>_s42.pt`.

### 2. Plotting Results

To generate training and validation perplexity line charts across all dropout values:

```bash
mkdir -p plots

python3 scripts/plot_perplexities.py \
  logs/log_dp0.0.csv logs/log_dp0.1.csv logs/log_dp0.2.csv \
  logs/log_dp0.3.csv logs/log_dp0.5.csv logs/log_dp0.7.csv \
  --out-train plots/train_perplexity.png \
  --out-valid plots/valid_perplexity.png
```

### 3. Text Generation from Best and Worst Models

After identifying the models with the lowest and highest test perplexity, generate sample text from each:

```bash
# Best model (lowest test perplexity — replace <best> with actual dropout value)
./scripts/generate.sh --checkpoint models/model_dp<best>_s42.pt \
  --words 200 --out samples/sample_best.txt

# Worst model (highest test perplexity — replace <worst> with actual dropout value)
./scripts/generate.sh --checkpoint models/model_dp<worst>_s42.pt \
  --words 200 --out samples/sample_worst.txt
```

---

## Documented Modifications

The following changes were made relative to the base repository at [https://github.com/marcamsler1/mt-exercise-02](https://github.com/marcamsler1/mt-exercise-02):

- **Deleted `scripts/download_data.sh`:** Removed the original Brothers Grimm / Project Gutenberg dataset script.
- **Added `bootstrap.sh`:** New root-level script that downloads, cleans, tokenizes, applies vocabulary cutoff, and splits RFC documents into the train/valid/test corpus.
- **Modified `tools/pytorch-examples/word_language_model/main.py`:** Added `--export-log` argument and CSV file I/O logic to write per-epoch `train_ppl` and `valid_ppl` to a structured log file. Added epoch-level loss accumulation so `train()` returns the mean epoch loss. A copy of the modified `main.py` is committed to this repository.
- **Added `scripts/plot_perplexities.py`:** Python script using `matplotlib` that reads the CSV logs and produces comparative perplexity tables (stdout) and line charts (PNG files) for training and validation perplexity across dropout settings.
- **Modified `scripts/train.sh`:** Updated dataset paths to point to the RFC corpus. Added `--dropout`, `--epochs`, `--emsize`, `--nhid`, `--seed`, `--save`, and `--log` CLI flags.
- **Modified `scripts/generate.sh`:** Added `--checkpoint`, `--words`, and `--out` CLI argument parsing to support targeting specific model checkpoints from the dropout sweep.