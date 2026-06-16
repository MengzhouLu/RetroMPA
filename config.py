"""
Global configuration with sensible defaults.
Defaults live here; shell scripts do not duplicate them.
Override via command-line arguments in script entrypoints.

All paths default to project-relative locations under ./data/ and ./weights/.
Override via --*-path / --*-csv / --*-dir CLI arguments.
"""

import math
import os

# ==============================================================================
# Project Directories
# ==============================================================================
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_PROJECT_ROOT, "data")
WEIGHTS_DIR = os.path.join(_PROJECT_ROOT, "weights")

# ==============================================================================
# General Settings
# ==============================================================================
stage = "pretrain"
num_epochs = 100
num_gpus = 4
batch_size = 16
num_workers = 8
seed = 42
amp = True

# ==============================================================================
# Data Paths
# ==============================================================================
pretrain_dataset_path = DATA_DIR
ft_dataset_path = os.path.join(DATA_DIR, "uspto_multi")
reactant_dict_path = DATA_DIR

# ==============================================================================
# Stage-specific defaults
# ==============================================================================
# Stage 1 (pretrain) defaults
stage1_batch_size = 16
stage1_num_epochs = 100
stage1_num_workers = 8
stage1_init_lr = 1e-4
stage1_max_lr = 1e-3
stage1_min_lr = 1e-8
stage1_warmup_start_lr = 1e-6
stage1_weight_decay = 0.05
stage1_amp = True
stage1_load_pretrained = False
stage1_model_path = None
stage1_scaler_path = None
stage1_output_dir = "./weights"
stage1_log_dir = "./logs"

# Stage 2 (finetune) defaults
stage2_batch_size = 512
stage2_num_epochs = 100
stage2_num_workers = 8
stage2_init_lr = 1e-4
stage2_max_lr = 1e-4
stage2_min_lr = 1e-7
stage2_warmup_start_lr = 1e-6
stage2_weight_decay = 0.05
stage2_amp = True
stage2_num_training_steps = 38800
stage2_load_pretrained = False
stage2_model_path = None
stage2_output_dir = "./weights"
stage2_log_dir = "./tblogs"
stage2_mode = "train"

# Inference defaults
infer_input_csv = os.path.join(DATA_DIR, "all_top1_predictions_post_EditR.csv")
infer_vocab_csv = os.path.join(
    DATA_DIR, "uspto_multi", "USPTO_FULL_canonical_smiles_test.csv"
)
infer_output_csv = "./refined_output.csv"
infer_weights = os.path.join(
    WEIGHTS_DIR,
    "best_model12_mydec_openQ_dualprior_usptofull_i1_p1_oriProGCL_bz512_ddp_acc2.pth",
)
infer_batch_size = 256
infer_topn = 10
infer_augment_vocab = False
infer_match_workers = None

# ==============================================================================
# Pretrain (Stage 1) - Q-Former Settings
# ==============================================================================
load_pretrained = False
model_path = None
config = None
max_txt_len = 192
use_smiles_in_text = False
molecular_precision = 32

# ==============================================================================
# Optimizer Defaults
# ==============================================================================
init_lr = 1e-4
min_lr = 1e-8
max_lr = 1e-3
warmup_start_lr = 1e-6
betas = (0.9, 0.98)
weight_decay = 0.05
num_training_steps = 1127619 // batch_size // num_gpus * 16 // batch_size * 50
warmup_steps = int(0.1 * num_training_steps)
num_cycles = 40

# ==============================================================================
# Finetune (Stage 2) Settings
# ==============================================================================
load_pretrained_scaler = False
Qformer_path = os.path.join(WEIGHTS_DIR, "best_model_192.pth")
scaler_path = None
Decoder_path = None
d_qformer = 768
d_model = 1024
dec_voc_size = 1024
n_head = 8
max_len = 2
ffn_hidden = 2048
n_layers = 6
drop_prob = 0.1
freeze_Qformer = True
load_pretrained_decoder = False

vocab_dataset_path = os.path.join(DATA_DIR, "uspto50k_canonical_smiles_test.csv")


def override_from_args(args):
    """Override global config variables from an argparse namespace."""
    overrides = vars(args)
    for key, value in overrides.items():
        if value is None:
            continue
        if key in globals():
            globals()[key] = value
