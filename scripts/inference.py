"""
Inference / Refinement Pipeline.
Uses trained model to predict dual reactants and applies
chemical rule filtering for post-processing.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging
import multiprocessing as mp
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit import RDLogger
from tqdm import tqdm

import config as cfg
from config import Qformer_path
from models.model import Mymodel_Mydecoder_openMrl_dualprior

RDLogger.logger().setLevel(RDLogger.ERROR)

DEFAULT_INPUT_CSV = cfg.infer_input_csv
DEFAULT_VOCAB_CSV = cfg.infer_vocab_csv
DEFAULT_OUTPUT_CSV = cfg.infer_output_csv
DEFAULT_WEIGHTS = cfg.infer_weights


def _get_element_counts(mol):
    """Return element count dictionary {symbol: count} excluding hydrogen."""
    counts = Counter()
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() > 1:
            counts[atom.GetSymbol()] += 1
    return counts


def evaluate_candidate(args):
    """
    Score a candidate reactant using chemical heuristics.
    Args: (idx, product_smiles, known_reactant_smiles, candidate_reactant_smiles, model_rank)
    Returns: (idx, candidate_smiles, final_score, original_rank)
    """
    idx, src_smi, r1_smi, r2_smi, orig_rank = args

    base_score = 50.0 - orig_rank
    score = base_score

    if r1_smi == r2_smi or src_smi == r1_smi or src_smi == r2_smi:
        return (idx, r2_smi, score - 1000.0, orig_rank)

    mol_prod = Chem.MolFromSmiles(src_smi)
    mol_r1 = Chem.MolFromSmiles(r1_smi)
    mol_r2 = Chem.MolFromSmiles(r2_smi)

    if mol_prod is None or mol_r1 is None or mol_r2 is None:
        return (idx, r2_smi, score - 1000.0, orig_rank)

    can_p = Chem.MolToSmiles(mol_prod)
    can_r1 = Chem.MolToSmiles(mol_r1)
    can_r2 = Chem.MolToSmiles(mol_r2)
    if can_r1 == can_r2 or can_p == can_r1 or can_p == can_r2:
        return (idx, r2_smi, score - 1000.0, orig_rank)

    n_prod = mol_prod.GetNumHeavyAtoms()
    n_r1 = mol_r1.GetNumHeavyAtoms()
    n_r2 = mol_r2.GetNumHeavyAtoms()
    if n_prod > n_r1 + n_r2 + 2:
        score -= 1000.0

    prod_elem = _get_element_counts(mol_prod)
    r1_elem = _get_element_counts(mol_r1)
    r2_elem = _get_element_counts(mol_r2)
    total_react_elem = r1_elem + r2_elem

    ghost = set(prod_elem.keys()) - set(total_react_elem.keys())
    if len(ghost) > 0:
        score -= 1000.0

    mw_p = Descriptors.MolWt(mol_prod)
    mw_a = Descriptors.MolWt(mol_r1)
    mw_b = Descriptors.MolWt(mol_r2)
    if mw_p - (mw_a + mw_b) > 5.0:
        score -= 1000.0

    if score < 0:
        return (idx, r2_smi, score, orig_rank)

    p_rings = Descriptors.RingCount(mol_prod)
    r1_rings = Descriptors.RingCount(mol_r1)
    r2_rings = Descriptors.RingCount(mol_r2)
    d_ring = p_rings - (r1_rings + r2_rings)
    if d_ring == 0:
        score += 1.5
    elif d_ring in (-2, -1, 1):
        score += 0.5

    leaving = set(total_react_elem.keys()) - set(prod_elem.keys())
    good_lg = {"Cl", "Br", "B", "I", "O", "S"}
    if leaving.intersection(good_lg):
        score += 1.0

    is_light_heavy = (mw_a < 100 and mw_b >= 250) or (mw_b < 100 and mw_a >= 250)
    if is_light_heavy:
        score += 1.0

    return (idx, r2_smi, score, orig_rank)


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("refinement_pipeline_dual_top1")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        )
        logger.addHandler(handler)
    return logger


def canonicalize_smiles(smiles: str) -> Optional[str]:
    text = str(smiles).strip() if smiles is not None else ""
    if not text:
        return None
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def split_dual_reactants(reactants: str) -> Optional[Tuple[str, str]]:
    parts = [p.strip() for p in str(reactants).split(".") if p.strip()]
    if len(parts) != 2:
        return None
    return parts[0], parts[1]


def canonicalize_triplet(
    product: str, b1: str, b2: str
) -> Optional[Tuple[str, str, str]]:
    cp = canonicalize_smiles(product)
    c1 = canonicalize_smiles(b1)
    c2 = canonicalize_smiles(b2)
    if cp is None or c1 is None or c2 is None:
        return None
    return cp, c1, c2


def load_imp_model(
    weights_path: str, qformer_path: str, device: torch.device, logger: logging.Logger
):
    model = Mymodel_Mydecoder_openMrl_dualprior(
        Qformer_path=qformer_path,
        device=device,
        logger=logger,
        use_imolclr=True,
        use_progcl=True,
    )
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model


def build_reactant_dict(vocab_csv: str, model, batch_size: int = 256):
    df = pd.read_csv(vocab_csv, dtype=str, keep_default_na=False)
    if "Reactants" not in df.columns:
        raise ValueError(f"Missing required column 'Reactants' in {vocab_csv}")
    reactants = df["Reactants"].astype(str).str.split(".").explode().dropna().tolist()
    unique: List[str] = []
    seen = set()
    for x in tqdm(reactants, desc="Stage 2/6 | Canonicalize vocab reactants"):
        cx = canonicalize_smiles(x)
        if cx is None or cx in seen:
            continue
        seen.add(cx)
        unique.append(cx)
    with torch.no_grad():
        embeddings, _ = model.mrl.transform(unique, batch_size=batch_size)
    return {smiles: embeddings[i] for i, smiles in enumerate(unique)}


def _collect_input_reactants(input_csv: str) -> List[str]:
    df = pd.read_csv(input_csv, dtype=str, keep_default_na=False)
    if "Reactants" not in df.columns:
        raise ValueError(f"Missing required column 'Reactants' in {input_csv}")
    reactants = df["Reactants"].astype(str).str.split(".").explode().dropna().tolist()
    return _normalize_candidates(reactants)


def _normalize_candidates(candidates: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for candidate in candidates:
        cc = canonicalize_smiles(candidate)
        if cc is None or cc in seen:
            continue
        seen.add(cc)
        out.append(cc)
    return out


@torch.no_grad()
def infer_missing_side_topk(
    model,
    products: Sequence[str],
    known_sides: Sequence[str],
    aux_sides: Sequence[str],
    reactant_dict,
    topk: int,
    batch_size: int,
    logger: logging.Logger,
) -> List[List[str]]:
    results: List[List[str]] = []
    total = len(products)
    for start in tqdm(
        range(0, total, batch_size), desc="Stage 4/6 | Imp batch inference", leave=False
    ):
        end = min(start + batch_size, total)
        src = list(products[start:end])
        tgt1 = list(known_sides[start:end])
        tgt2 = list(aux_sides[start:end])
        try:
            _, pred_missing = model.predict_reactants(
                src, tgt1, tgt2, dict_data=reactant_dict, topK=topk
            )
            for row in pred_missing:
                results.append(_normalize_candidates(row[:topk]))
        except Exception as exc:
            logger.warning(
                "Batch inference failed, falling back per sample: %s", str(exc)
            )
            for p, k, a in zip(src, tgt1, tgt2):
                try:
                    _, row_pred = model.predict_reactants(
                        [p], [k], [a], dict_data=reactant_dict, topK=topk
                    )
                    results.append(_normalize_candidates(row_pred[0][:topk]))
                except Exception as row_exc:
                    logger.warning("Single-sample inference failed: %s", str(row_exc))
                    results.append([])
    return results


def batch_filter_candidates(
    products: Sequence[str],
    known_sides: Sequence[str],
    candidates_list_of_lists: List[List[str]],
    num_workers: int,
) -> List[List[str]]:
    """Filter and re-rank candidate lists using chemical rules in parallel."""
    tasks = []
    for idx, (src_smi, r1_smi, candidates) in enumerate(
        zip(products, known_sides, candidates_list_of_lists)
    ):
        for rank, r2_smi in enumerate(candidates):
            tasks.append((idx, src_smi, r1_smi, r2_smi, rank))

    filtered_results = {i: [] for i in range(len(products))}
    if tasks:
        with mp.Pool(processes=num_workers) as pool:
            for res in tqdm(
                pool.imap_unordered(evaluate_candidate, tasks),
                total=len(tasks),
                desc="Stage 4.5/6 | Filtering candidates",
                leave=False,
            ):
                idx, r2_smi, score, orig_rank = res
                filtered_results[idx].append((r2_smi, score, orig_rank))

    new_candidates_lists = []
    for i in range(len(products)):
        sorted_cands = sorted(filtered_results[i], key=lambda x: (-x[1], x[2]))
        new_candidates_lists.append([x[0] for x in sorted_cands])
    return new_candidates_lists


def refine_one_sample(b1: str, b2: str, b1_topk, b2_topk, topn: int = 10) -> str:
    """
    Unidirectional trust refinement:
    Since the prior guarantees if base model only gets one correct, it must be b1.
    We fully trust b1 and use the b2_topk predicted from b1 to verify/fix b2.
    """
    b2_slice = list(b2_topk[:topn]) if b2_topk is not None else []
    if b2 in b2_slice:
        return f"{b1}.{b2}"
    if b2_topk:
        return f"{b1}.{b2_topk[0]}"
    return f"{b1}.{b2}"


def run_pipeline(args: argparse.Namespace) -> None:
    logger = setup_logger()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with tqdm(total=1, desc="Stage 1/6 | Load input CSV") as pbar:
        base_df = pd.read_csv(args.input_csv, dtype=str, keep_default_na=False)
        pbar.update(1)
    if "Product" not in base_df.columns or "Reactants" not in base_df.columns:
        raise ValueError(
            f"Input CSV must contain 'Product' and 'Reactants' columns: {args.input_csv}"
        )

    total_rows = len(base_df)
    outputs_refined: List[str] = [""] * total_rows
    out_product: List[str] = [""] * total_rows
    out_b1: List[str] = [""] * total_rows
    out_b2: List[str] = [""] * total_rows
    out_base_model: List[str] = [""] * total_rows
    out_b2_improve: List[str] = [""] * total_rows
    out_b1_improve: List[str] = [""] * total_rows
    out_is_dual: List[bool] = [False] * total_rows

    stats: Dict[str, int] = {
        "total": total_rows,
        "dual": 0,
        "non_dual_passthrough": 0,
        "rdkit_failed": 0,
        "entered_imp": 0,
        "kept_base": 0,
        "replaced": 0,
        "imp_fail_fallback": 0,
    }

    queued_indices: List[int] = []
    queued_products: List[str] = []
    queued_b1: List[str] = []
    queued_b2: List[str] = []

    for idx, row in tqdm(
        base_df.iterrows(), total=total_rows, desc="Stage 3/6 | Preprocess rows"
    ):
        product_raw = row["Product"]
        reactants_raw = row["Reactants"]
        out_product[idx] = str(product_raw)
        out_base_model[idx] = str(reactants_raw)

        pair = split_dual_reactants(reactants_raw)
        if pair is None:
            outputs_refined[idx] = str(reactants_raw)
            stats["non_dual_passthrough"] += 1
            continue

        stats["dual"] += 1
        b1_raw, b2_raw = pair
        canon_triplet = canonicalize_triplet(product_raw, b1_raw, b2_raw)
        if canon_triplet is None:
            outputs_refined[idx] = str(reactants_raw)
            stats["rdkit_failed"] += 1
            continue

        p, b1, b2 = canon_triplet
        out_b1[idx] = b1
        out_b2[idx] = b2
        out_is_dual[idx] = True
        queued_indices.append(idx)
        queued_products.append(p)
        queued_b1.append(b1)
        queued_b2.append(b2)

    if queued_indices:
        with tqdm(total=2, desc="Stage 2/6 | Initialize Imp model") as pbar:
            model = load_imp_model(args.weights, args.qformer_path, device, logger)
            pbar.update(1)
            reactant_dict = build_reactant_dict(
                args.vocab_csv, model, batch_size=args.batch_size
            )
            if args.augment_vocab:
                input_reactants = _collect_input_reactants(args.input_csv)
                new_smiles = [s for s in input_reactants if s not in reactant_dict]
                if new_smiles:
                    with torch.no_grad():
                        embeddings, _ = model.mrl.transform(
                            new_smiles, batch_size=args.batch_size
                        )
                    for i, smi in enumerate(new_smiles):
                        reactant_dict[smi] = embeddings[i]
            pbar.update(1)

        stats["entered_imp"] = len(queued_indices)

        b2_topk_all = infer_missing_side_topk(
            model=model,
            products=queued_products,
            known_sides=queued_b1,
            aux_sides=queued_b2,
            reactant_dict=reactant_dict,
            topk=max(10, args.topn),
            batch_size=args.batch_size,
            logger=logger,
        )
        b1_topk_all = infer_missing_side_topk(
            model=model,
            products=queued_products,
            known_sides=queued_b2,
            aux_sides=queued_b1,
            reactant_dict=reactant_dict,
            topk=max(10, args.topn),
            batch_size=args.batch_size,
            logger=logger,
        )

        logger.info("Applying chemical rules filtering to b2 candidates...")
        b2_topk_all = batch_filter_candidates(
            products=queued_products,
            known_sides=queued_b1,
            candidates_list_of_lists=b2_topk_all,
            num_workers=args.match_workers,
        )
        logger.info("Applying chemical rules filtering to b1 candidates...")
        b1_topk_all = batch_filter_candidates(
            products=queued_products,
            known_sides=queued_b2,
            candidates_list_of_lists=b1_topk_all,
            num_workers=args.match_workers,
        )

        def match_one(i: int) -> Tuple[int, str, bool, bool, str, str]:
            idx = queued_indices[i]
            b1 = queued_b1[i]
            b2 = queued_b2[i]
            if i >= len(b1_topk_all) or i >= len(b2_topk_all):
                return idx, f"{b1}.{b2}", True, True, "", ""
            b1_topk_list = b1_topk_all[i]
            b2_topk_list = b2_topk_all[i]
            refined = refine_one_sample(
                b1, b2, b1_topk_list, b2_topk_list, topn=args.topn
            )
            return (
                idx,
                refined,
                (refined == f"{b1}.{b2}"),
                False,
                str(b1_topk_list),
                str(b2_topk_list),
            )

        total_match = len(queued_indices)
        if args.match_workers > 1 and total_match > 1:
            with ThreadPoolExecutor(max_workers=args.match_workers) as executor:
                matched_iter = executor.map(match_one, range(total_match))
                for idx, refined, is_kept, is_fallback, b1_str, b2_str in tqdm(
                    matched_iter,
                    total=total_match,
                    desc="Stage 5/6 | Apply refinement rule (parallel)",
                ):
                    outputs_refined[idx] = refined
                    out_b1_improve[idx] = b1_str
                    out_b2_improve[idx] = b2_str
                    if is_fallback:
                        stats["imp_fail_fallback"] += 1
                    if is_kept:
                        stats["kept_base"] += 1
                    else:
                        stats["replaced"] += 1
        else:
            for idx, refined, is_kept, is_fallback, b1_str, b2_str in tqdm(
                (match_one(i) for i in range(total_match)),
                total=total_match,
                desc="Stage 5/6 | Apply refinement rule",
            ):
                outputs_refined[idx] = refined
                out_b1_improve[idx] = b1_str
                out_b2_improve[idx] = b2_str
                if is_fallback:
                    stats["imp_fail_fallback"] += 1
                if is_kept:
                    stats["kept_base"] += 1
                else:
                    stats["replaced"] += 1

    out_df = pd.DataFrame(
        {
            "Product": out_product,
            "b1": out_b1,
            "b2": out_b2,
            "BaseModel_Top5_Reactants": out_base_model,
            "b2_improve_topk": out_b2_improve,
            "b1_improve_topk": out_b1_improve,
            "is_dual_reactant": out_is_dual,
            "Refined_Reactants": outputs_refined,
        }
    )

    if len(out_df) != total_rows:
        raise RuntimeError("Output length mismatch")
    with tqdm(total=1, desc="Stage 6/6 | Save output CSV") as pbar:
        out_df.to_csv(args.output_csv, index=False)
        pbar.update(1)

    logger.info("Saved refined output to %s", args.output_csv)
    logger.info(
        "Stats | total=%d dual=%d non_dual=%d rdkit_failed=%d entered_imp=%d kept_base=%d replaced=%d imp_fail_fallback=%d",
        stats["total"],
        stats["dual"],
        stats["non_dual_passthrough"],
        stats["rdkit_failed"],
        stats["entered_imp"],
        stats["kept_base"],
        stats["replaced"],
        stats["imp_fail_fallback"],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dual-reactant Top1 refinement pipeline with Chemical Rule Filtering"
    )
    parser.add_argument("--input-csv", type=str, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--vocab-csv", type=str, default=DEFAULT_VOCAB_CSV)
    parser.add_argument("--output-csv", type=str, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS)
    parser.add_argument("--qformer-path", type=str, default=Qformer_path)
    parser.add_argument("--batch-size", type=int, default=cfg.infer_batch_size)
    parser.add_argument("--topn", type=int, default=cfg.infer_topn)
    parser.add_argument(
        "--augment-vocab",
        action="store_true",
        default=cfg.infer_augment_vocab,
        help="Augment vocab with canonicalized reactants from input CSV",
    )
    parser.add_argument(
        "--match-workers",
        type=int,
        default=(
            cfg.infer_match_workers
            if cfg.infer_match_workers is not None
            else max(1, min(16, os.cpu_count() or 1))
        ),
        help="Worker threads/processes for filtering and matching",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_pipeline(parse_args())
