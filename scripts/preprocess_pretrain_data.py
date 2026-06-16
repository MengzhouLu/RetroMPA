#!/usr/bin/env python3
import argparse
import json
import os

import pandas as pd
from rdkit import Chem
import selfies as sf

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover

    def tqdm(iterable, **kwargs):
        return iterable


def canonicalize_smiles(smiles):
    if not isinstance(smiles, str) or not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def canonicalize_selfies(selfies, fallback_smiles=None):
    decoded = None
    if isinstance(selfies, str) and selfies:
        try:
            decoded = sf.decoder(selfies)
        except Exception:
            decoded = None
    if decoded:
        canonical = canonicalize_smiles(decoded)
        if canonical:
            return canonical, True
    if fallback_smiles:
        canonical = canonicalize_smiles(fallback_smiles)
        if canonical:
            return canonical, True
    return fallback_smiles, False


def require_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required file: {path}")


def should_write(path, overwrite):
    if os.path.exists(path) and not overwrite:
        print(f"Skip existing output: {path}")
        return False
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return True


def preprocess_chebi(source_dir, out_dir, overwrite):
    chebi_dir = os.path.join(source_dir, "ChEBI-20-MM")
    splits = {
        "train": "train",
        "validation": "valid",
        "test": "test",
    }
    for input_split, output_split in splits.items():
        input_csv = os.path.join(chebi_dir, f"{input_split}.csv")
        output_csv = os.path.join(
            out_dir, f"chebi20mm_canonical_smiles_{output_split}.csv"
        )
        require_file(input_csv)
        if not should_write(output_csv, overwrite):
            continue
        df = pd.read_csv(input_csv)
        invalid = 0
        if "SELFIES" in df.columns:
            for idx, row in tqdm(
                df.iterrows(), total=len(df), desc=f"ChEBI {input_split}"
            ):
                canonical, ok = canonicalize_selfies(
                    row.get("SELFIES"), row.get("SMILES")
                )
                if ok and canonical:
                    df.at[idx, "SMILES"] = canonical
                else:
                    invalid += 1
        else:
            for idx, row in tqdm(
                df.iterrows(), total=len(df), desc=f"ChEBI {input_split}"
            ):
                canonical = canonicalize_smiles(row.get("SMILES"))
                if canonical:
                    df.at[idx, "SMILES"] = canonical
                else:
                    invalid += 1
        df.to_csv(output_csv, index=False)
        print(f"Wrote {output_csv} (rows={len(df)}, invalid={invalid})")


def build_pubchem_table(cid2smiles_df, cid2items, value_key, desc):
    rows = []
    missing = 0
    invalid = 0
    for _, row in tqdm(cid2smiles_df.iterrows(), total=len(cid2smiles_df), desc=desc):
        cid = str(row["CID"])
        items = cid2items.get(cid)
        if not items:
            missing += 1
            continue
        canonical = canonicalize_smiles(row["SMILES"])
        if not canonical:
            invalid += 1
            continue
        for item in items:
            if item is None:
                continue
            rows.append({"SMILES": canonical, value_key: item})
    print(f"{desc}: rows={len(rows)}, missing_cid={missing}, invalid_smiles={invalid}")
    return pd.DataFrame(rows)


def preprocess_pubchem(source_dir, out_dir, overwrite):
    raw_dir = os.path.join(source_dir, "PubChemSTM_data", "raw")
    cid2smiles_path = os.path.join(raw_dir, "CID2SMILES.csv")
    cid2name_path = os.path.join(raw_dir, "CID2name.json")
    cid2text_path = os.path.join(raw_dir, "CID2text.json")
    require_file(cid2smiles_path)
    require_file(cid2name_path)
    require_file(cid2text_path)

    cid2smiles_df = pd.read_csv(cid2smiles_path, usecols=["CID", "SMILES"])
    cid2smiles_df["CID"] = cid2smiles_df["CID"].astype(str)

    with open(cid2name_path, "r", encoding="utf-8") as f:
        cid2name = json.load(f)
    with open(cid2text_path, "r", encoding="utf-8") as f:
        cid2text = json.load(f)

    name_output = os.path.join(out_dir, "pubchem_canonical_smiles_smiles_to_name.csv")
    text_output = os.path.join(out_dir, "pubchem_canonical_smiles_smiles_to_text.csv")

    if should_write(name_output, overwrite):
        name_df = build_pubchem_table(
            cid2smiles_df, cid2name, "Name", "PubChem smiles_to_name"
        )
        name_df.to_csv(name_output, index=False)
        print(f"Wrote {name_output} (rows={len(name_df)})")

    if should_write(text_output, overwrite):
        text_df = build_pubchem_table(
            cid2smiles_df, cid2text, "Text", "PubChem smiles_to_text"
        )
        text_df.to_csv(text_output, index=False)
        print(f"Wrote {text_output} (rows={len(text_df)})")


def preprocess_mol_instructions(source_dir, out_dir, overwrite):
    mol_dir = os.path.join(
        source_dir, "Mol-Instructions", "Molecule-oriented_Instructions"
    )
    files = {
        "molecular_description_generation.json": "mol-ins_canonical_smiles_molecular_description_generation.json",
        "property_prediction.json": "mol-ins_canonical_smiles_property_prediction.json",
    }
    for input_name, output_name in files.items():
        input_path = os.path.join(mol_dir, input_name)
        output_path = os.path.join(out_dir, output_name)
        require_file(input_path)
        if not should_write(output_path, overwrite):
            continue
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        processed = []
        invalid = 0
        for item in tqdm(data, desc=f"Mol-Ins {input_name}"):
            canonical, ok = canonicalize_selfies(item.get("input"))
            if not ok or not canonical:
                invalid += 1
                continue
            new_item = dict(item)
            new_item["input"] = canonical
            processed.append(new_item)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(processed, f, indent=2)
        print(f"Wrote {output_path} (rows={len(processed)}, invalid={invalid})")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess pretrain datasets for MyDataset_pretrain"
    )
    parser.add_argument(
        "--source-dir",
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data",
            "Dataset",
        ),
        help="Dataset root containing ChEBI-20-MM, PubChemSTM_data, Mol-Instructions",
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
        ),
        help="Output directory for generated files",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip writing outputs that already exist",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    overwrite = not args.skip_existing
    os.makedirs(args.out_dir, exist_ok=True)
    preprocess_chebi(args.source_dir, args.out_dir, overwrite)
    preprocess_pubchem(args.source_dir, args.out_dir, overwrite)
    preprocess_mol_instructions(args.source_dir, args.out_dir, overwrite)
    print("Preprocessing complete")


if __name__ == "__main__":
    main()
