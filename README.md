# DualPrior Retrosynthesis Prediction

A deep learning framework for **dual-reactant retrosynthesis prediction** that predicts two reactants from a single product molecule. Built on a BLIP-2 style architecture with a Q-Former molecular encoder and a Transformer decoder, trained with dual-prior contrastive learning.

## Overview

This project implements a two-stage training pipeline for retrosynthesis:

1. **Stage 1 (Pretraining)**: Trains the Q-Former molecular encoder on SMILES-to-text alignment using CheBI, PubChem, and MolIns datasets.
2. **Stage 2 (Fine-tuning)**: Fine-tunes the decoder on USPTO-FULL retrosynthesis data with dual-prior contrastive loss (IMolCLR + ProGCL) and reactant augmentation.
3. **Inference**: Predicts dual reactants given a product molecule, with chemical rule-based filtering for refinement.

## Requirements

- Python 3.8+
- PyTorch 2.0.0 (CUDA 11.8)
- RDKit (via conda)

### Quick Install

```bash
# All Python dependencies including PyTorch (use extra-index-url for CUDA packages)
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118

# Install RDKit separately (conda required)
conda install -c conda-forge rdkit==2024.3.5
```

> **Note**: All commands below should be run from the project root directory (`github_demo/`).

## Directory Structure

```
github_demo/
├── config.py                  # Global configuration with defaults (all paths configurable)
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── run_train_stage1.sh        # Launch script for Stage 1
├── run_train_stage2.sh        # Launch script for Stage 2
├── run_inference.sh          # Launch script for inference
├── data/                      # Default data directory (configurable via --*-path)
│   └── dataset.py            # Dataset classes
├── scripts/
│   ├── train_stage1.py       # Stage 1: Q-Former pretraining (DDP)
│   ├── train_stage2.py       # Stage 2: Decoder fine-tuning (DDP)
│   ├── inference.py          # Inference & refinement pipeline
│   └── preprocess_pretrain_data.py  # Data preprocessing
├── models/
│   ├── model.py              # Core model definitions
│   └── model_ddp.py          # DDP-safe model wrapper
├── utils/
│   └── smiles_utils.py       # SMILES processing utilities
├── lavis/                    # Modified LAVIS framework
│   ├── models/               # BLIP-2 Q-Former, MRL encoder
│   └── common/               # Distributed utilities
├── weights/                  # Default weights directory (configurable)
├── logs/                     # Training logs
└── MolR/                     # Molecular representation learning
```

## Data Preparation

### Path Configuration

All data, weight, and vocabulary paths are defined in `config.py` with defaults relative to the project root (`./data/`, `./weights/`). You can override any path via CLI arguments:

```bash
# Override data paths for Stage 1
bash run_train_stage1.sh --pretrain-dataset-path /custom/path/to/data

# Override data paths for Stage 2
bash run_train_stage2.sh --ft-dataset-path /custom/path/to/uspto_multi

# Override inference inputs
bash run_inference.sh --input-csv /path/to/input.csv --weights /path/to/model.pth
```

### Stage 1 (Pretraining) Data

You can obtain the following datasets from the following URLs:

- ChEBI-20-MM: https://huggingface.co/datasets/liupf/ChEBI-20-MM
- Mol-Instructions: https://huggingface.co/datasets/zjunlp/Mol-Instructions
- PubChemSTM: https://github.com/chao1224/MoleculeSTM

Extract the downloaded datasets under `./data/Dataset/` so the directory layout looks like:

```
data/
├── Dataset/
│   ├── ChEBI-20-MM/
│   ├── PubChemSTM_data/raw/
│   └── Mol-Instructions/Molecule-oriented_Instructions/
├── uspto_multi/
│   └── USPTO_FULL_canonical_smiles_*.csv
└── ...
```

Then run the preprocessing script to generate the files used by `MyDataset_pretrain`:

```bash
# Default: reads from ./data/Dataset, writes to ./data/
python scripts/preprocess_pretrain_data.py

# Custom paths
python scripts/preprocess_pretrain_data.py \
  --source-dir /path/to/Dataset \
  --out-dir /path/to/output
```

Generated files (output dir: `./data/` by default):

```
chebi20mm_canonical_smiles_train.csv
chebi20mm_canonical_smiles_valid.csv
chebi20mm_canonical_smiles_test.csv
pubchem_canonical_smiles_smiles_to_name.csv
pubchem_canonical_smiles_smiles_to_text.csv
mol-ins_canonical_smiles_molecular_description_generation.json
mol-ins_canonical_smiles_property_prediction.json
```

### Stage 2 (Fine-tuning) Data

You can obtain the standardized USPTO dataset from here:
https://drive.google.com/drive/folders/1ScDvBXj5xtPOuSajcYf1A2mfB6xiDT7R?usp=drive_link

Place these files in your USPTO dataset directory (default: `./data/uspto_multi/`):

```
USPTO_FULL_canonical_smiles_train.csv
USPTO_FULL_canonical_smiles_valid.csv
USPTO_FULL_canonical_smiles_test.csv
```

Each CSV must contain columns: `Product`, `Reactants` (dot-separated dual reactants).

### Model Weights

Place pretrained model weights in a `weights/` directory or specify paths via command-line arguments:

```
weights/
├── best_model_192.pth                    # Stage 1 output / Stage 2 Q-Former
└── best_model12_*_acc2.pth              # Stage 2 best model
```

## Training

Defaults for all scripts live in `config.py`. The `run_*.sh` scripts are thin wrappers and do not duplicate defaults; override values by passing CLI arguments to the Python entrypoints.

### Stage 1: Q-Former Pretraining

Trains the Q-Former to align SMILES molecular representations with textual descriptions. Uses distributed data parallel (DDP) training.

```bash
# Default settings
bash run_train_stage1.sh

# Custom settings
bash run_train_stage1.sh --batch-size 16 --pretrain-dataset-path /path/to/data --lr 1e-4
```

Key hyperparameters:
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--batch-size` | 16 | Batch size per GPU |
| `--num-epochs` | 100 | Number of training epochs |
| `--lr` | 1e-4 | Initial learning rate |
| `--max-lr` | 1e-3 | Peak learning rate (cosine schedule) |
| `--min-lr` | 1e-8 | Minimum learning rate |
| `--pretrain-dataset-path` | ./data | Path to pretrain dataset files |
| `--load-pretrained` | false | Resume from checkpoint |
| `--model-path` | None | Pretrained model path |

### Stage 2: Decoder Fine-tuning

Fine-tunes the decoder with dual-prior contrastive learning on USPTO-FULL retrosynthesis data.

```bash
# Training mode
bash run_train_stage2.sh

# Validation mode
bash run_train_stage2.sh --mode valid

# Custom settings
bash run_train_stage2.sh --batch-size 512 --ft-dataset-path /path/to/uspto_multi --lr 1e-4
```

Key hyperparameters:
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--mode` | train | `train` or `valid` |
| `--batch-size` | 512 | Batch size per GPU |
| `--num-epochs` | 100 | Number of training epochs |
| `--lr` | 1e-4 | Initial learning rate |
| `--ft-dataset-path` | ./data/uspto_multi | Path to USPTO dataset |
| `--qformer-path` | ./weights/best_model_192.pth | Stage 1 model path |
| `--num-training-steps` | 38800 | Total training steps |

## Inference

Predicts dual reactants for a given product and applies chemical rule-based refinement.

### Input Format

The inference input CSV (`--input-csv`) must contain:
- Column `Product`: SMILES string of the target product molecule
- Column `Reactants`: optional ground-truth reactants (for evaluation; ignored during prediction)

The vocabulary CSV (`--vocab-csv`) provides reference reactant SMILES used during candidate retrieval. Format: CSV with a `Reactants` column.

### Usage

```bash
# Default settings (uses paths from config.py)
bash run_inference.sh

# Custom
bash run_inference.sh --input-csv /path/to/input.csv --output-csv ./results.csv --weights /path/to/model.pth
```

Key parameters:
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--input-csv` | `./data/all_top1_predictions_post_EditR.csv` | Input CSV with Product and Reactants columns |
| `--vocab-csv` | `./data/uspto_multi/USPTO_FULL_canonical_smiles_test.csv` | Reference vocabulary |
| `--output-csv` | ./refined_output.csv | Output predictions |
| `--weights` | `./weights/best_model12_mydec_openQ_dualprior_usptofull_i1_p1_oriProGCL_bz512_ddp_acc2.pth` | Fine-tuned model weights |
| `--batch-size` | 256 | Inference batch size |
| `--topn` | 10 | Top-N candidates for refinement |

## Model Architecture

```
Product SMILES
     ↓
[MRL Encoder] → Molecular Embedding
     ↓
[Q-Former] → Product Features (num_queries × d_model)
     ↓
[Linear Projection] + [Start Token]
     ↓
[Transformer Decoder] → Reactant 1 Embedding, Reactant 2 Embedding
     ↓
[Contrastive Loss]  ←  [Morgan FP Dual Prior Weights]
```

The model uses:
- **MRL (Molecular Representation Learning)** for encoding SMILES into embeddings
- **Q-Former** (from BLIP-2) for cross-modal alignment
- **Transformer Decoder** for autoregressive reactant embedding prediction
- **Dual Prior** combining IMolCLR (fingerprint similarity) and ProGCL weights

## Citation

If you use this code in your research, please cite the relevant papers for BLIP-2, MolR, and the USPTO dataset.
