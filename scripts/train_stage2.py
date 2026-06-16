"""
Stage 2 Fine-tuning Script (DDP).
Fine-tunes the dual-prior decoder on USPTO-FULL retrosynthesis data.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import gc
import logging
import math

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import config as cfg
from data.dataset import USPTOfull_Dataset_1f_dualprior_augSwap
from models.model_ddp import Mymodel_Mydecoder_openMrl_dualprior_ddp

USE_IMOLCLR = True
USE_PROGCL = True


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 2 Fine-tuning")
    parser.add_argument("--mode", choices=["train", "valid"], default=cfg.stage2_mode)
    parser.add_argument("--batch-size", type=int, default=cfg.stage2_batch_size)
    parser.add_argument("--num-epochs", type=int, default=cfg.stage2_num_epochs)
    parser.add_argument("--num-workers", type=int, default=cfg.stage2_num_workers)
    parser.add_argument("--lr", type=float, default=cfg.stage2_init_lr)
    parser.add_argument("--max-lr", type=float, default=cfg.stage2_max_lr)
    parser.add_argument("--min-lr", type=float, default=cfg.stage2_min_lr)
    parser.add_argument(
        "--warmup-start-lr", type=float, default=cfg.stage2_warmup_start_lr
    )
    parser.add_argument("--weight-decay", type=float, default=cfg.stage2_weight_decay)
    parser.add_argument("--seed", type=int, default=cfg.seed)
    parser.add_argument("--amp", action="store_true", default=cfg.stage2_amp)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument(
        "--num-training-steps", type=int, default=cfg.stage2_num_training_steps
    )
    parser.add_argument("--ft-dataset-path", type=str, default=cfg.ft_dataset_path)
    parser.add_argument(
        "--load-pretrained", action="store_true", default=cfg.stage2_load_pretrained
    )
    parser.add_argument("--model-path", type=str, default=cfg.stage2_model_path)
    parser.add_argument("--qformer-path", type=str, default=cfg.Qformer_path)
    parser.add_argument("--scaler-path", type=str, default=cfg.scaler_path)
    parser.add_argument("--output-dir", type=str, default=cfg.stage2_output_dir)
    parser.add_argument("--log-dir", type=str, default=cfg.stage2_log_dir)
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank for DDP")
    return parser.parse_args()


task = "mydec_openQ_dualprior_usptofull_i%d_p%d_oriProGCL_bz512_ddp" % (
    int(USE_IMOLCLR),
    int(USE_PROGCL),
)


def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()


def is_main_process():
    return (not is_dist_avail_and_initialized()) or dist.get_rank() == 0


def setup_logger(task_name, main_proc=True):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    if main_proc:
        file_handler = logging.FileHandler("fine_tune_%s.log" % task_name)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    else:
        null_handler = logging.NullHandler()
        logger.addHandler(null_handler)
    return logger


def init_distributed_mode():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        distributed = world_size > 1
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        distributed = False
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    if distributed:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
        dist.barrier()
    return distributed, rank, world_size, local_rank, device


def distributed_mean(value, device, world_size):
    tensor = torch.tensor(value, dtype=torch.float64, device=device)
    if is_dist_avail_and_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= world_size
    return tensor.item()


def load_scaler(load_pretrained, scaler_path):
    scaler = GradScaler(enabled=cfg.amp)
    if load_pretrained and scaler_path:
        scaler_weights = torch.load(scaler_path, map_location="cpu")
        scaler.load_state_dict(scaler_weights)
    return scaler


def load_model_fn(load_pretrained, model_path, Qformer_path, device, logger):
    model = Mymodel_Mydecoder_openMrl_dualprior_ddp(
        Qformer_path=Qformer_path,
        device=device,
        logger=logger,
        use_imolclr=USE_IMOLCLR,
        use_progcl=USE_PROGCL,
    )
    if load_pretrained and model_path:
        logger.info("Loading pretrained model from %s", model_path)
        model_weights = torch.load(model_path, map_location="cpu")
        model.load_state_dict(model_weights, strict=False)
    model.to(device)
    return model


def load_optimizer(model, init_lr=None, betas=None, weight_decay=None):
    init_lr = cfg.init_lr if init_lr is None else init_lr
    betas = cfg.betas if betas is None else betas
    weight_decay = cfg.weight_decay if weight_decay is None else weight_decay
    return optim.AdamW(
        model.parameters(), lr=init_lr, betas=betas, weight_decay=weight_decay
    )


class LinearWarmupCosineLRScheduler:
    """Linear warmup followed by cosine decay."""

    def __init__(
        self,
        optimizer,
        num_training_steps,
        min_lr,
        max_lr,
        warmup_steps,
        warmup_start_lr,
        num_cycles=0.5,
        **kwargs,
    ):
        self.optimizer = optimizer
        self.num_training_steps = num_training_steps
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr if warmup_start_lr >= 0 else min_lr
        self.num_cycles = num_cycles

    def step(self, cur_step):
        cosine_lr_schedule_with_warmup(
            current_step=cur_step,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.num_training_steps,
            min_lr=self.min_lr,
            max_lr=self.max_lr,
            warmup_start_lr=self.warmup_start_lr,
            optimizer=self.optimizer,
            num_cycles=self.num_cycles,
        )


def cosine_lr_schedule_with_warmup(
    optimizer,
    current_step,
    num_warmup_steps,
    num_training_steps,
    num_cycles,
    min_lr,
    max_lr,
    warmup_start_lr,
):
    if current_step < num_warmup_steps:
        lr = min(
            max_lr,
            warmup_start_lr
            + (max_lr - warmup_start_lr) * current_step / max(num_warmup_steps, 1),
        )
    else:
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        lr = max(
            min_lr,
            max_lr
            * 0.5
            * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)),
        )
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def load_scheduler(optimizer):
    return LinearWarmupCosineLRScheduler(
        optimizer=optimizer,
        num_training_steps=cfg.num_training_steps,
        min_lr=cfg.min_lr,
        max_lr=cfg.max_lr,
        warmup_steps=cfg.warmup_steps,
        warmup_start_lr=cfg.warmup_start_lr,
        num_cycles=cfg.num_cycles,
    )


def make_loader(dataset, batch_size, shuffle, num_workers, drop_last, sampler=None):
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle if sampler is None else False,
        "sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": True,
        "drop_last": drop_last,
        "persistent_workers": False,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
    return torch.utils.data.DataLoader(**loader_kwargs)


def build_dataloaders(distributed, rank, world_size, dataset_path):
    dataset_train = USPTOfull_Dataset_1f_dualprior_augSwap(
        dataset_path=dataset_path, split="train"
    )
    dataset_valid = USPTOfull_Dataset_1f_dualprior_augSwap(
        dataset_path=dataset_path, split="valid"
    )
    dataset_test = USPTOfull_Dataset_1f_dualprior_augSwap(
        dataset_path=dataset_path, split="test"
    )

    train_sampler = None
    if distributed:
        train_sampler = DistributedSampler(
            dataset_train,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
        )

    train_dataloader = make_loader(
        dataset=dataset_train,
        batch_size=cfg.batch_size,
        shuffle=(train_sampler is None),
        num_workers=cfg.num_workers,
        drop_last=True,
        sampler=train_sampler,
    )

    valid_sampler = None
    if distributed:
        valid_sampler = DistributedSampler(
            dataset_valid,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )

    valid_dataloader = make_loader(
        dataset=dataset_valid,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        drop_last=False,
        sampler=valid_sampler,
    )

    return train_dataloader, valid_dataloader


def build_valid_labels_and_reactants():
    valid_csv_path = os.path.join(
        cfg.ft_dataset_path, "USPTO_FULL_canonical_smiles_valid.csv"
    )
    data = pd.read_csv(valid_csv_path, header=0, usecols=["Reactants"])
    data = data[data["Reactants"].str.split(".").apply(len) == 2]
    labels = [row[0].split(".") for row in data.values.tolist()]
    df = pd.read_csv(valid_csv_path)
    reactants = df["Reactants"].str.split(".")
    exploded_reactants = reactants.explode()
    unique_reactants = set(exploded_reactants)
    unique_reactants_smiles = list(unique_reactants)
    return labels, unique_reactants_smiles, valid_csv_path


def run_valid_topk_metrics(valid_dataloader, model, logger, device):
    logger.info("Running validation with ACC/ACC2 @1/@3/@5 (distributed locally)...")
    labels, unique_reactants_smiles, valid_csv_path = build_valid_labels_and_reactants()
    total = len(labels)
    if total == 0:
        logger.error("No valid labels found in %s", valid_csv_path)
        return {
            "acc1": 0.0,
            "acc3": 0.0,
            "acc5": 0.0,
            "acc2_1": 0.0,
            "acc2_3": 0.0,
            "acc2_5": 0.0,
        }, 0

    base_model = model.module if hasattr(model, "module") else model
    base_model.eval()

    with torch.no_grad():
        if hasattr(base_model, "encode_molformer_smiles"):
            reactant_embeds = base_model.encode_molformer_smiles(
                unique_reactants_smiles, batch_size=512
            )
        else:
            reactant_embeds, _ = base_model.mrl.transform(
                unique_reactants_smiles, batch_size=512
            )
    lib_embeddings = F.normalize(reactant_embeds, dim=-1).to(device).float()
    processed_lib = (lib_embeddings, unique_reactants_smiles)

    if isinstance(valid_dataloader.sampler, DistributedSampler):
        rank_actual_indices = list(valid_dataloader.sampler)
    else:
        rank_actual_indices = list(range(total))

    local_seen = set()
    k_values = (1, 3, 5)
    local_acc_counts = {k: 0 for k in k_values}
    local_acc2_counts = {k: 0 for k in k_values}

    with torch.no_grad():
        data_iterator = iter(valid_dataloader)
        idx_iterator = iter(rank_actual_indices)
        for src, tgt1, tgt2, fp_r1, fp_r2 in tqdm(
            data_iterator, disable=not is_main_process()
        ):
            res1_batch, res2_batch = base_model.predict_reactants(
                src, tgt1, tgt2, dict_data=processed_lib, topK=50
            )
            batch_size_actual = len(src)
            for i in range(batch_size_actual):
                global_idx = next(idx_iterator)
                if global_idx in local_seen:
                    continue
                local_seen.add(global_idx)
                label1, label2 = labels[global_idx]
                pred1 = res1_batch[i]
                pred2 = res2_batch[i]
                for k in k_values:
                    if label1 in pred1[:k] and label2 in pred2[:k]:
                        local_acc_counts[k] += 1
                    if label2 in pred2[:k]:
                        local_acc2_counts[k] += 1

    stats_tensor = torch.tensor(
        [
            local_acc_counts[1],
            local_acc_counts[3],
            local_acc_counts[5],
            local_acc2_counts[1],
            local_acc2_counts[3],
            local_acc2_counts[5],
            len(local_seen),
        ],
        dtype=torch.float64,
        device=device,
    )

    if is_dist_avail_and_initialized():
        dist.all_reduce(stats_tensor, op=dist.ReduceOp.SUM)

    total_evaluated = stats_tensor[-1].item()
    if total_evaluated == 0:
        total_evaluated = 1

    metrics = {
        "acc1": stats_tensor[0].item() / total_evaluated,
        "acc3": stats_tensor[1].item() / total_evaluated,
        "acc5": stats_tensor[2].item() / total_evaluated,
        "acc2_1": stats_tensor[3].item() / total_evaluated,
        "acc2_3": stats_tensor[4].item() / total_evaluated,
        "acc2_5": stats_tensor[5].item() / total_evaluated,
    }
    return metrics, int(total_evaluated)


def train(
    train_dataloader,
    valid_dataloader,
    model,
    optimizer,
    scheduler,
    scaler,
    writer,
    device,
    world_size,
    logger,
):
    best_eval_loss = float("inf")
    best_eval_acc = 0
    cur_step = 0

    labels, unique_reactants_smiles, _ = build_valid_labels_and_reactants()
    total_labels = len(labels)
    if total_labels == 0 and is_main_process():
        logger.error("No valid labels found, validation will be skipped.")

    for epoch in range(cfg.num_epochs):
        if isinstance(train_dataloader.sampler, DistributedSampler):
            train_dataloader.sampler.set_epoch(epoch)

        model.train()
        total_loss = 0.0
        total_r2p_1_loss = 0.0
        total_r2p_2_loss = 0.0
        total_p2r_1_loss = 0.0
        total_p2r_2_loss = 0.0
        total_msl_pos_loss = 0.0
        total_msl_neg_loss = 0.0
        total_r1_loss = 0.0
        total_r2_loss = 0.0

        for src, tgt1, tgt2, fp_r1, fp_r2 in tqdm(
            train_dataloader, disable=not is_main_process()
        ):
            cur_step += 1
            optimizer.zero_grad(set_to_none=True)

            if cfg.amp:
                with torch.cuda.amp.autocast(enabled=True):
                    with torch.backends.cuda.sdp_kernel(enable_flash=False):
                        model_loss = model(
                            src, tgt1, tgt2, fp_tgt1=fp_r1, fp_tgt2=fp_r2
                        )
            else:
                model_loss = model(src, tgt1, tgt2, fp_tgt1=fp_r1, fp_tgt2=fp_r2)

            if cfg.amp:
                scaler.scale(model_loss.loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                model_loss.loss.backward()
                optimizer.step()

            scheduler.step(cur_step)

            total_loss += model_loss.loss.item()
            total_r2p_1_loss += model_loss.loss_r2p_1.item()
            total_r2p_2_loss += model_loss.loss_r2p_2.item()
            total_p2r_1_loss += model_loss.loss_p2r_1.item()
            total_p2r_2_loss += model_loss.loss_p2r_2.item()
            total_msl_pos_loss += model_loss.loss_msl_pos.item()
            total_msl_neg_loss += model_loss.loss_msl_neg.item()
            total_r1_loss += model_loss.loss_r1.item()
            total_r2_loss += model_loss.loss_r2.item()

            current_lr = optimizer.param_groups[0]["lr"]
            if writer is not None:
                if cur_step % 10 == 0:
                    writer.add_scalar("Loss/train", model_loss.loss.item(), cur_step)
                writer.add_scalar("Learning Rate", current_lr, cur_step)

            if cur_step % 388 == 0 and cur_step > 0:
                if is_dist_avail_and_initialized():
                    dist.barrier()

                base_model = model.module if hasattr(model, "module") else model
                base_model.eval()

                with torch.no_grad():
                    if hasattr(base_model, "encode_molformer_smiles"):
                        reactant_embeds = base_model.encode_molformer_smiles(
                            unique_reactants_smiles, batch_size=512
                        )
                    else:
                        reactant_embeds, _ = base_model.mrl.transform(
                            unique_reactants_smiles, batch_size=512
                        )
                    lib_embeddings = (
                        F.normalize(reactant_embeds, dim=-1).to(device).float()
                    )
                    processed_lib = (lib_embeddings, unique_reactants_smiles)

                if isinstance(valid_dataloader.sampler, DistributedSampler):
                    rank_actual_indices = list(valid_dataloader.sampler)
                else:
                    rank_actual_indices = list(range(total_labels))

                eval_loss = 0.0
                eval_r2p_1_loss = 0.0
                eval_r2p_2_loss = 0.0
                eval_p2r_1_loss = 0.0
                eval_p2r_2_loss = 0.0
                eval_msl_pos_loss = 0.0
                eval_msl_neg_loss = 0.0
                eval_r1_loss = 0.0
                eval_r2_loss = 0.0

                local_seen = set()
                local_acc1 = 0
                local_acc2 = 0
                local_acc_both = 0

                with torch.no_grad():
                    data_iterator = iter(valid_dataloader)
                    idx_iterator = iter(rank_actual_indices)
                    for src, tgt1, tgt2, fp_r1, fp_r2 in tqdm(
                        data_iterator, disable=not is_main_process()
                    ):
                        if cfg.amp:
                            with torch.cuda.amp.autocast(enabled=True):
                                with torch.backends.cuda.sdp_kernel(enable_flash=False):
                                    model_loss = base_model(
                                        src, tgt1, tgt2, fp_tgt1=fp_r1, fp_tgt2=fp_r2
                                    )
                        else:
                            model_loss = base_model(
                                src, tgt1, tgt2, fp_tgt1=fp_r1, fp_tgt2=fp_r2
                            )

                        eval_loss += model_loss.loss.item()
                        eval_r2p_1_loss += model_loss.loss_r2p_1.item()
                        eval_r2p_2_loss += model_loss.loss_r2p_2.item()
                        eval_p2r_1_loss += model_loss.loss_p2r_1.item()
                        eval_p2r_2_loss += model_loss.loss_p2r_2.item()
                        eval_msl_pos_loss += model_loss.loss_msl_pos.item()
                        eval_msl_neg_loss += model_loss.loss_msl_neg.item()
                        eval_r1_loss += model_loss.loss_r1.item()
                        eval_r2_loss += model_loss.loss_r2.item()

                        res1_batch, res2_batch = base_model.predict_reactants(
                            src, tgt1, tgt2, dict_data=processed_lib, topK=50
                        )

                        batch_size_actual = len(src)
                        for i in range(batch_size_actual):
                            global_idx = next(idx_iterator)
                            if global_idx in local_seen:
                                continue
                            local_seen.add(global_idx)
                            label = labels[global_idx]
                            pred1 = res1_batch[i]
                            pred2 = res2_batch[i]
                            if label[0] in pred1[:5]:
                                local_acc1 += 1
                                if label[1] in pred2[:5]:
                                    local_acc2 += 1
                                    local_acc_both += 1
                                    continue
                            if label[1] in pred2[:5]:
                                local_acc2 += 1

                num_batches_local = len(valid_dataloader)
                loss_tensors = torch.tensor(
                    [
                        eval_loss,
                        eval_r2p_1_loss,
                        eval_r2p_2_loss,
                        eval_p2r_1_loss,
                        eval_p2r_2_loss,
                        eval_msl_pos_loss,
                        eval_msl_neg_loss,
                        eval_r1_loss,
                        eval_r2_loss,
                        float(num_batches_local),
                    ],
                    dtype=torch.float64,
                    device=device,
                )
                if is_dist_avail_and_initialized():
                    dist.all_reduce(loss_tensors, op=dist.ReduceOp.SUM)
                total_batches = loss_tensors[-1].item()
                avg_eval_loss = loss_tensors[0].item() / total_batches
                avg_eval_r2p_1_loss = loss_tensors[1].item() / total_batches
                avg_eval_r2p_2_loss = loss_tensors[2].item() / total_batches
                avg_eval_p2r_1_loss = loss_tensors[3].item() / total_batches
                avg_eval_p2r_2_loss = loss_tensors[4].item() / total_batches
                avg_eval_msl_pos_loss = loss_tensors[5].item() / total_batches
                avg_eval_msl_neg_loss = loss_tensors[6].item() / total_batches
                avg_eval_r1_loss = loss_tensors[7].item() / total_batches
                avg_eval_r2_loss = loss_tensors[8].item() / total_batches

                acc_tensors = torch.tensor(
                    [local_acc1, local_acc2, local_acc_both, len(local_seen)],
                    dtype=torch.float64,
                    device=device,
                )
                if is_dist_avail_and_initialized():
                    dist.all_reduce(acc_tensors, op=dist.ReduceOp.SUM)

                total_labels_eval = acc_tensors[3].item()
                acc1 = (
                    acc_tensors[0].item() / total_labels_eval
                    if total_labels_eval > 0
                    else 0
                )
                acc2 = (
                    acc_tensors[1].item() / total_labels_eval
                    if total_labels_eval > 0
                    else 0
                )
                acc = (
                    acc_tensors[2].item() / total_labels_eval
                    if total_labels_eval > 0
                    else 0
                )

                if is_main_process():
                    logger.info(
                        "[Step %d] Eval Loss: %.4f, Acc1: %.4f, Acc2: %.4f, Acc: %.4f\n"
                        "R2P_1: %.4f, R2P_2: %.4f\nP2R_1: %.4f, P2R_2: %.4f\n"
                        "MSL(pos): %.4f, MSL(neg): %.4f\nR1: %.4f, R2: %.4f",
                        cur_step,
                        avg_eval_loss,
                        acc1,
                        acc2,
                        acc,
                        avg_eval_r2p_1_loss,
                        avg_eval_r2p_2_loss,
                        avg_eval_p2r_1_loss,
                        avg_eval_p2r_2_loss,
                        avg_eval_msl_pos_loss,
                        avg_eval_msl_neg_loss,
                        avg_eval_r1_loss,
                        avg_eval_r2_loss,
                    )

                    if avg_eval_loss < best_eval_loss:
                        logger.info(
                            "Loss improved %.4f -> %.4f", best_eval_loss, avg_eval_loss
                        )
                        best_eval_loss = avg_eval_loss
                        os.makedirs(cfg.stage2_output_dir, exist_ok=True)
                        torch.save(
                            base_model.state_dict(),
                            os.path.join(
                                cfg.stage2_output_dir, "best_model12_%s.pth" % task
                            ),
                        )
                    if acc2 > best_eval_acc:
                        logger.info("Acc2 improved %.4f -> %.4f", best_eval_acc, acc2)
                        best_eval_acc = acc2
                        os.makedirs(cfg.stage2_output_dir, exist_ok=True)
                        torch.save(
                            base_model.state_dict(),
                            os.path.join(
                                cfg.stage2_output_dir, "best_model12_%s_acc2.pth" % task
                            ),
                        )

                del processed_lib, reactant_embeds, lib_embeddings
                torch.cuda.empty_cache()
                gc.collect()
                if epoch % 5 == 0:
                    base_model.mrl._graph_cache.clear()

                model.train()
                if is_dist_avail_and_initialized():
                    dist.barrier()

        avg_train_loss = distributed_mean(
            total_loss / len(train_dataloader), device, world_size
        )
        avg_train_r2p_1_loss = distributed_mean(
            total_r2p_1_loss / len(train_dataloader), device, world_size
        )
        avg_train_r2p_2_loss = distributed_mean(
            total_r2p_2_loss / len(train_dataloader), device, world_size
        )
        avg_train_p2r_1_loss = distributed_mean(
            total_p2r_1_loss / len(train_dataloader), device, world_size
        )
        avg_train_p2r_2_loss = distributed_mean(
            total_p2r_2_loss / len(train_dataloader), device, world_size
        )
        avg_train_msl_pos_loss = distributed_mean(
            total_msl_pos_loss / len(train_dataloader), device, world_size
        )
        avg_train_msl_neg_loss = distributed_mean(
            total_msl_neg_loss / len(train_dataloader), device, world_size
        )
        avg_train_r1_loss = distributed_mean(
            total_r1_loss / len(train_dataloader), device, world_size
        )
        avg_train_r2_loss = distributed_mean(
            total_r2_loss / len(train_dataloader), device, world_size
        )

        if is_main_process():
            logger.info(
                "Epoch %d/%d\nTrain Loss: %.4f\n"
                "R2P_1: %.4f, R2P_2: %.4f\nP2R_1: %.4f, P2R_2: %.4f\n"
                "MSL(pos): %.4f, MSL(neg): %.4f\nR1: %.4f, R2: %.4f",
                epoch + 1,
                cfg.num_epochs,
                avg_train_loss,
                avg_train_r2p_1_loss,
                avg_train_r2p_2_loss,
                avg_train_p2r_1_loss,
                avg_train_p2r_2_loss,
                avg_train_msl_pos_loss,
                avg_train_msl_neg_loss,
                avg_train_r1_loss,
                avg_train_r2_loss,
            )
            if writer is not None:
                writer.add_scalar("Loss/epoch_train", avg_train_loss, epoch)

    if writer is not None:
        writer.close()


def main():
    args = parse_args()
    cfg.override_from_args(args)

    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    np.random.seed(cfg.seed)

    distributed, rank, world_size, local_rank, device = init_distributed_mode()
    logger = setup_logger(task, main_proc=(rank == 0))

    if is_main_process():
        print("gpu num:", torch.cuda.device_count())
        logger.info(
            "Distributed=%s | rank=%d | local_rank=%d | world_size=%d",
            distributed,
            rank,
            local_rank,
            world_size,
        )

    train_dataloader, valid_dataloader = build_dataloaders(
        distributed, rank, world_size, dataset_path=cfg.ft_dataset_path
    )

    model = load_model_fn(
        load_pretrained=cfg.load_pretrained,
        model_path=cfg.model_path,
        Qformer_path=cfg.Qformer_path,
        device=device,
        logger=logger,
    )

    if distributed:
        model = DDP(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            find_unused_parameters=True,
        )

    if args.mode == "valid":
        rank = dist.get_rank() if is_dist_avail_and_initialized() else 0
        metrics, total = run_valid_topk_metrics(valid_dataloader, model, logger, device)
        if is_main_process() and metrics is not None:
            logger.info(
                "[VALID] ACC@1: %.4f, ACC@3: %.4f, ACC@5: %.4f, "
                "ACC2@1: %.4f, ACC2@3: %.4f, ACC2@5: %.4f (N=%d)",
                metrics["acc1"],
                metrics["acc3"],
                metrics["acc5"],
                metrics["acc2_1"],
                metrics["acc2_3"],
                metrics["acc2_5"],
                total,
            )
        if is_dist_avail_and_initialized():
            dist.barrier()
            dist.destroy_process_group()
        return

    scaler = load_scaler(
        load_pretrained=cfg.load_pretrained_scaler, scaler_path=cfg.scaler_path
    )
    writer = (
        SummaryWriter(log_dir=os.path.join(args.log_dir, task))
        if is_main_process()
        else None
    )
    optimizer = load_optimizer(model)
    scheduler = load_scheduler(optimizer)

    train(
        train_dataloader,
        valid_dataloader,
        model,
        optimizer,
        scheduler,
        scaler,
        writer,
        device,
        world_size,
        logger,
    )

    if is_dist_avail_and_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
