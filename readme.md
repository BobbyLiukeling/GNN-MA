# GNN-MA: Soft Molecular Alignment with Cross-Graph Attention for Ligand-Based Virtual Screening

GNN-MA is a graph neural network framework for ligand similarity modeling and ligand-based virtual screening (LBVS). It supports cross-graph attention, node–edge collaborative convolution, and complete preprocessing → training → evaluation workflows on standard benchmarks such as DUD-E and LIT-PCBA.

Repository: https://github.com/BobbyLiukeling/GNN-MA

---

## 1. Environment Requirements

- OS: Linux / macOS / Windows (Linux recommended)
- Python: 3.10

### Installation (Recommended)

```bash
pip install -r requirements.txt
```

If using conda (recommended for RDKit):

```bash
conda create -n gnnma python=3.10 -y
conda activate gnnma
conda install -c conda-forge rdkit -y
pip install -r requirements.txt
```

---

## 2. Data Download

This project uses the public LBVS benchmark datasets:

- DUD-E: https://dude.docking.org/
- LIT-PCBA: http://drugdesign.unistra.fr/LIT-PCBA

Download the datasets from the official websites and place them under the expected folder structure (e.g., `data/DUD-E/`, `data/LIT-PCBA/`).

---

## 3. Quickstart

### Step 1: Encode molecules to NPZ format

```bash
python data/encoding.py
python data/encoding_LIT.py
```

### Step 2: Generate fixed train/val/test splits

```bash
python data/DUD-split.py
python data/LIT-split.py
```

### Step 3: Train the model

```bash
python train.py
```

---

## 4. Training Configuration (DUD-E, `train.py`)

### Core Hyperparameters

- Batch size: 32
- Optimizer: AdamW
- Learning rate: 1e-3
- Epochs: 20
- Random seed: 2025

### Ranking Loss Settings

- Warm-up epochs: 2 (BCE loss only, λ = 0)
- After warm-up: λ = 0.05
- Top-K hardest negatives: K = 10

### Split Policy

The model does not perform random pair-level splitting during training.

For each target, we read a pre-generated split directory:

```
split_811/
    candidates_train.txt
    candidates_val.txt
    candidates_test.txt
```

Training pairs are constructed as (crystal_ligand, candidate). Labels are inferred from the candidate path. This design prevents pair-level leakage and ensures that train/validation/test splits follow the predefined candidate split files.

---

## 5. Reproducibility Notes

- Fix the random seed (Python / NumPy / PyTorch) to ensure deterministic behavior.
- Record learning rate, batch size, λ, K, and split policy for every experiment.
- Store checkpoints, logs, and metric CSV files under a unified directory (e.g., `results/`).
- If reporting pooled and macro metrics, use a consistent aggregation script and document the command used.

---

## 6. Scope

This repository focuses on 2D molecular graph inputs (topology + atom/bond attributes). Soft-alignment visualization is provided as a qualitative inspection tool rather than a systematic interpretability analysis.

---

## 7. Contact

For questions, please open a GitHub issue or contact the authors via the email addresses listed in the manuscript.
