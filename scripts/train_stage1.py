"""
Stage 1 Pretraining Script (DDP).
Trains the Q-Former + Decoder on SMILES-to-text data.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging
import math

import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
from torch.cuda.amp import GradScaler
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import config as cfg
from data.dataset import MyDataset_pretrain
from lavis.models.blip2_models.blip2_qformer import Blip2Qformer


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 1 Pretraining")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size per GPU")
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--max-lr", type=float, default=1e-3)
    parser.add_argument("--min-lr", type=float, default=1e-8)
    parser.add_argument("--warmup-start-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--amp", action="store_true", default=True, help="Use mixed precision"
    )
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--max-txt-len", type=int, default=192)
    parser.add_argument(
        "--pretrain-dataset-path",
        type=str,
        default=cfg.pretrain_dataset_path,
        help="Path to pretraining dataset files",
    )
    parser.add_argument(
        "--load-pretrained",
        action="store_true",
        default=False,
        help="Resume from checkpoint",
    )
    parser.add_argument(
        "--model-path", type=str, default=None, help="Pretrained model path"
    )
    parser.add_argument(
        "--scaler-path", type=str, default=None, help="Scaler checkpoint path"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=cfg.stage1_output_dir,
        help="Model save directory",
    )
    parser.add_argument(
        "--log-dir", type=str, default="./logs", help="TensorBoard log directory"
    )
    return parser.parse_args()


def setup_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler("Pretraining.log")
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


logger = setup_logger()


def reduce_mean(tensor, nprocs):
    """Average tensor across all GPUs."""
    if isinstance(tensor, float):
        tensor = torch.tensor(tensor).to(device)
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt


def load_scaler(load_pretrained=False, scaler_path=None):
    scaler = GradScaler()
    if load_pretrained and scaler_path:
        scaler_weights = torch.load(scaler_path)
        scaler.load_state_dict(scaler_weights)
        logger.info(f"Loaded scaler from {scaler_path}")
    return scaler


def load_model(load_pretrained=False, model_path=None, config=None, max_txt_len=None):
    max_txt_len = cfg.max_txt_len if max_txt_len is None else max_txt_len
    model = Blip2Qformer(
        max_txt_len=max_txt_len,
        molecular_precision=cfg.molecular_precision,
        device=device,
    )
    if load_pretrained and model_path:
        model_weights = torch.load(model_path)
        model.load_state_dict(model_weights)
        logger.info(f"Loaded model from {model_path}")
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
    """Cosine LR schedule with linear warmup."""
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


def train(
    train_sampler,
    train_dataloader,
    valid_dataloader,
    model,
    optimizer,
    scheduler,
    scaler,
    writer,
):
    rank = dist.get_rank()
    best_eval_loss = float("inf")
    cur_step = 0
    cur_epoch = 0
    cur_val_step = 0

    for epoch in range(cfg.num_epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        total_loss = 0
        total_loss_mtm = 0
        total_loss_mtc = 0
        total_loss_lm = 0
        i = 0

        for train_batch in tqdm(train_dataloader):
            optimizer.zero_grad()

            if cfg.amp:
                with torch.cuda.amp.autocast():
                    model_loss = model(train_batch)
            else:
                model_loss = model(train_batch)

            total_loss += model_loss.loss.item()
            total_loss_mtm += model_loss.loss_mtm.item()
            total_loss_mtc += model_loss.loss_mtc.item()
            total_loss_lm += model_loss.loss_lm.item()

            if cfg.amp:
                scaler.scale(model_loss.loss).backward()
                cur_step_loss = reduce_mean(
                    model_loss.loss.item(), dist.get_world_size()
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                model_loss.loss.backward()
                cur_step_loss = reduce_mean(
                    model_loss.loss.item(), dist.get_world_size()
                )
                optimizer.step()

            scheduler.step(cur_step)
            current_lr = optimizer.param_groups[0]["lr"]

            if rank == 0:
                writer.add_scalar(
                    "Loss/train",
                    cur_step_loss.item(),
                    epoch * len(train_dataloader) + i,
                )
                writer.add_scalar(
                    "Learning Rate", current_lr, epoch * len(train_dataloader) + i
                )

            i += 1
            cur_step += 1

            if cur_step % 3000 == 0:
                model.eval()
                eval_loss = 0
                eval_loss_mtm = 0
                eval_loss_mtc = 0
                eval_loss_lm = 0
                with torch.no_grad():
                    for valid_batch in tqdm(valid_dataloader):
                        if cfg.amp:
                            with torch.cuda.amp.autocast():
                                model_loss = model(valid_batch)
                        else:
                            model_loss = model(valid_batch)

                        eval_loss += model_loss.loss.item()
                        eval_loss_mtm += model_loss.loss_mtm.item()
                        eval_loss_mtc += model_loss.loss_mtc.item()
                        eval_loss_lm += model_loss.loss_lm.item()
                        cur_step_eval_loss = reduce_mean(
                            model_loss.loss.item(), dist.get_world_size()
                        )
                        if rank == 0:
                            writer.add_scalar(
                                "Loss/eval", cur_step_eval_loss.item(), cur_val_step
                            )
                        cur_val_step += 1

                avg_train_loss = total_loss / i
                avg_eval_loss = eval_loss / len(valid_dataloader)
                avg_train_loss_mtm = total_loss_mtm / i
                avg_train_loss_mtc = total_loss_mtc / i
                avg_train_loss_lm = total_loss_lm / i
                avg_eval_loss_mtm = eval_loss_mtm / len(valid_dataloader)
                avg_eval_loss_mtc = eval_loss_mtc / len(valid_dataloader)
                avg_eval_loss_lm = eval_loss_lm / len(valid_dataloader)

                if rank == 0:
                    writer.add_scalar("Loss/average_train", avg_train_loss, cur_step)
                    writer.add_scalar("Loss/average_eval", avg_eval_loss, cur_step)
                    logger.info(
                        f"Step {cur_step}, Average Loss: {avg_train_loss:.4f}, Eval Loss: {avg_eval_loss:.4f}, "
                        f"Avg Train MTM Loss: {avg_train_loss_mtm:.4f}, Avg Train MTC Loss: {avg_train_loss_mtc:.4f}, "
                        f"Avg Train LM Loss: {avg_train_loss_lm:.4f}, Avg Eval MTM Loss: {avg_eval_loss_mtm:.4f}, "
                        f"Avg Eval MTC Loss: {avg_eval_loss_mtc:.4f}, Avg Eval LM Loss: {avg_eval_loss_lm:.4f}"
                    )
                    if avg_eval_loss < best_eval_loss:
                        logger.info(
                            f"Validation loss decreased ({best_eval_loss:.4f} -> {avg_eval_loss:.4f}). Saving model!"
                        )
                        best_eval_loss = avg_eval_loss
                        os.makedirs(cfg.stage1_output_dir, exist_ok=True)
                        torch.save(
                            model.module.state_dict(),
                            os.path.join(cfg.stage1_output_dir, "best_model_192.pth"),
                        )
                        torch.save(
                            scaler.state_dict(),
                            os.path.join(cfg.stage1_output_dir, "scaler_192.pth"),
                        )
                model.train()

        cur_epoch += 1
    writer.close()


def main():
    args = parse_args()
    cfg.override_from_args(args)

    global device
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(cfg.seed)

    print("gpu num:", torch.cuda.device_count())
    torch.distributed.init_process_group(backend="nccl")
    print("world_size", torch.distributed.get_world_size())

    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    dataset_s2n_train = MyDataset_pretrain(
        task="smiles2name",
        device=device,
        dataset_path=cfg.pretrain_dataset_path,
        split="train",
        use_smiles_in_text=cfg.use_smiles_in_text,
    )
    dataset_s2n_valid = MyDataset_pretrain(
        task="smiles2name",
        device=device,
        dataset_path=cfg.pretrain_dataset_path,
        split="valid",
        use_smiles_in_text=cfg.use_smiles_in_text,
    )
    dataset_s2n_test = MyDataset_pretrain(
        task="smiles2name",
        device=device,
        dataset_path=cfg.pretrain_dataset_path,
        split="test",
        use_smiles_in_text=cfg.use_smiles_in_text,
    )
    dataset_s2d_train = MyDataset_pretrain(
        task="smiles2description",
        device=device,
        dataset_path=cfg.pretrain_dataset_path,
        split="train",
        use_smiles_in_text=cfg.use_smiles_in_text,
    )
    dataset_s2d_valid = MyDataset_pretrain(
        task="smiles2description",
        device=device,
        dataset_path=cfg.pretrain_dataset_path,
        split="valid",
        use_smiles_in_text=cfg.use_smiles_in_text,
    )
    dataset_s2d_test = MyDataset_pretrain(
        task="smiles2description",
        device=device,
        dataset_path=cfg.pretrain_dataset_path,
        split="test",
        use_smiles_in_text=cfg.use_smiles_in_text,
    )

    combined_dataset_train = ConcatDataset(
        [
            dataset_s2n_train,
            dataset_s2d_train,
        ]
    )
    combined_dataset_valid = ConcatDataset([dataset_s2n_valid, dataset_s2d_valid])

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        combined_dataset_train
    )
    train_dataloader = torch.utils.data.DataLoader(
        combined_dataset_train,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
        sampler=train_sampler,
        persistent_workers=True,
    )

    valid_sampler = torch.utils.data.distributed.DistributedSampler(
        combined_dataset_valid
    )
    valid_dataloader = torch.utils.data.DataLoader(
        combined_dataset_valid,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
        sampler=valid_sampler,
        persistent_workers=True,
    )

    scaler = load_scaler(
        load_pretrained=cfg.load_pretrained, scaler_path=cfg.scaler_path
    )
    writer = SummaryWriter(log_dir=args.log_dir)

    model = load_model(load_pretrained=cfg.load_pretrained, model_path=cfg.model_path)
    model = model.to(device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
        broadcast_buffers=False,
    )

    optimizer = load_optimizer(model)
    scheduler = load_scheduler(optimizer)

    if cfg.stage == "pretrain":
        train(
            train_sampler,
            train_dataloader,
            valid_dataloader,
            model,
            optimizer,
            scheduler,
            scaler,
            writer,
        )
    else:
        print("stage error")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
