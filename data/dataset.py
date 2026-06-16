import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import pandas as pd
import json
import pickle
import numpy as np
from lavis.models import mrl
import selfies as sf
# mrl_model,tokenizer = mrl.create_mrl(model_name='SELFormer',device=None)


import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


def get_morgan_fp_tensor(smiles, radius=2, nBits=2048):
    """Convert SMILES to Morgan fingerprint tensor (float32, 0/1 values)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return torch.zeros(nBits, dtype=torch.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    arr = np.zeros((nBits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)

    # Use from_numpy to avoid memory copy when creating Tensor, then convert to float32
    return torch.from_numpy(arr).float()


def get_sequence_embeddings(selfies, model, tokenizer, device):
    token = torch.tensor(
        [
            tokenizer.encode(
                selfies,
                add_special_tokens=True,
                max_length=512,
                padding=True,
                truncation=True,
            )
        ]
    ).to(device)
    output = model(token)

    sequence_out = output[0]
    return torch.mean(sequence_out[0], dim=0).tolist()


class MyDataset_pretrain(Dataset):
    def __init__(self, device, task, dataset_path, split, use_smiles_in_text=False):
        super(MyDataset_pretrain, self).__init__()
        assert task in ["smiles2name", "smiles2description"]
        self.split = split
        self.task = task
        self.dataset_path = dataset_path
        self.use_smiles_in_text = use_smiles_in_text
        print(
            f"Task: {self.task}, split: {self.split}, use_smiles_in_text: {self.use_smiles_in_text}"
        )
        self.smiles = []
        self.description = []
        self.iupacnames = []
        self.embeddings = []
        # self.mrl_model,self.tokenizer = mrl.create_mrl(model_name='SELFormer',device=device)
        # self.sf=sf

        if self.task == "smiles2name":  #
            assert self.split in ["train", "valid", "test"]
            self.data = pd.read_csv(
                f"{self.dataset_path}/chebi20mm_canonical_smiles_{split}.csv",
                usecols=["SMILES", "description", "iupacname"],
            )
            self.smiles = self.data["SMILES"].tolist()
            # Process DataFrame, fill missing values
            self.data["iupacname"].fillna("Unknown", inplace=True)
            # Then extract as list
            self.iupacnames = self.data["iupacname"].tolist()

            self.data = pd.read_csv(
                f"{self.dataset_path}/pubchem_canonical_smiles_smiles_to_name.csv",
                usecols=["SMILES", "Name"],
            )
            # Process DataFrame, fill missing values
            self.data["Name"].fillna("Unknown", inplace=True)
            train_data, test_data = train_test_split(
                self.data, test_size=0.2, random_state=42
            )
            valid_data, test_data = train_test_split(
                test_data, test_size=0.5, random_state=42
            )
            if self.split == "train":
                self.data = train_data
            elif self.split == "valid":
                self.data = valid_data
            elif self.split == "test":
                self.data = test_data  ### What about the other file
            self.smiles += self.data["SMILES"].tolist()
            self.iupacnames += self.data["Name"].tolist()
            # # Process in chunks
            # from tqdm import tqdm
            # unique_reactants_selfies=[]
            # for smile in tqdm(self.smiles):
            #     try:
            #         embed=self.sf.encoder(smile)
            #     except:
            #         embed=""
            #     unique_reactants_selfies.append(embed)

            # # unique_reactants_selfies = [sf.encoder(smile) for smile in self.smiles]
            # print(len(unique_reactants_selfies))
            # self.embeddings = np.array([get_sequence_embeddings(x, mrl_model, tokenizer,device) for x in tqdm(unique_reactants_selfies)])

        elif self.task == "smiles2description":
            assert self.split in ["train", "valid", "test"]

            self.data = pd.read_csv(
                f"{self.dataset_path}/chebi20mm_canonical_smiles_{split}.csv",
                usecols=["SMILES", "description", "iupacname"],
            )
            self.smiles = self.data["SMILES"].tolist()
            self.description = self.data["description"].tolist()

            self.data = pd.read_csv(
                f"{self.dataset_path}/pubchem_canonical_smiles_smiles_to_text.csv",
                usecols=["SMILES", "Text"],
            )
            train_data, test_data = train_test_split(
                self.data, test_size=0.2, random_state=42
            )
            valid_data, test_data = train_test_split(
                test_data, test_size=0.5, random_state=42
            )
            if self.split == "train":
                self.data = train_data
            elif self.split == "valid":
                self.data = valid_data
            elif self.split == "test":
                self.data = test_data
            self.smiles += self.data["SMILES"].tolist()
            self.description += self.data["Text"].tolist()

            with open(
                f"{self.dataset_path}/mol-ins_canonical_smiles_molecular_description_generation.json",
                "r",
            ) as f:
                self.data = json.load(f)
                # Create dict to store data by split
                split_data = {}
                # Classify items by split
                for item in self.data:
                    split_value = item["metadata"]["split"]
                    if split_value not in split_data:
                        split_data[
                            split_value
                        ] = []  # Init list if split key doesn't exist
                    split_data[split_value].append(item)
                # Access data by split via split_data dict
                # e.g., access 'train' split data
                self.data = split_data.get(f"{self.split}", [])
                self.smiles += [d["input"] for d in self.data]
                self.description += [d["output"] for d in self.data]
            with open(
                f"{self.dataset_path}/mol-ins_canonical_smiles_property_prediction.json",
                "r",
            ) as f:
                self.data = json.load(f)
                # Create dict to store data by split
                split_data = {}
                # Classify items by split
                for item in self.data:
                    split_value = item["metadata"]["split"]
                    if split_value not in split_data:
                        split_data[
                            split_value
                        ] = []  # Init list if split key doesn't exist
                    split_data[split_value].append(item)
                # Access data by split via split_data dict
                # e.g., access 'train' split data
                self.data = split_data.get(f"{self.split}", [])
                self.smiles += [d["input"] for d in self.data]
                self.description += [
                    d["instruction"] + " The answer is " + str(d["output"])
                    for d in self.data
                ]
            # # Process in chunks
            # from tqdm import tqdm
            # unique_reactants_selfies=[]
            # for smile in tqdm(self.smiles):
            #     try:
            #         embed=self.sf.encoder(smile)
            #     except:
            #         embed=""
            #     unique_reactants_selfies.append(embed)

            # # unique_reactants_selfies = [sf.encoder(smile) for smile in self.smiles]
            # print(len(unique_reactants_selfies))
            # self.embeddings = np.array([get_sequence_embeddings(x, mrl_model, tokenizer,device=device) for x in tqdm(unique_reactants_selfies)])

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):

        if self.use_smiles_in_text:
            sample = {
                "smiles": self.smiles[idx],
                "text_input": f"The smiles is {self.smiles[idx]} with the description that {self.description[idx]}"
                if self.task == "smiles2description"
                else f"The smiles is {self.smiles[idx]} and the name is {self.iupacnames[idx]}",
                # "embedding": self.embeddings[idx]
            }
        else:
            sample = {
                "smiles": self.smiles[idx],
                "text_input": self.description[idx]
                if self.task == "smiles2description"
                else self.iupacnames[idx],
                # "embedding": self.embeddings[idx]
            }

        return sample


class USPTOfull_Dataset_1f_dualprior_augSwap(Dataset):
    def __init__(
        self,
        dataset_path,
        split,
        fp_radius=2,
        fp_nBits=2048,
        augment_splits=("train", "valid"),
    ):
        super().__init__()
        self.split = split
        self.dataset_path = dataset_path
        self.augment_splits = set(augment_splits)

        # 1. Read base data
        data = pd.read_csv(
            f"{self.dataset_path}/USPTO_FULL_canonical_smiles_{split}.csv",
            usecols=["Product", "Reactants"],
        )
        data = data[data["Reactants"].str.split(".").apply(len) == 2]

        base_prod = data["Product"].tolist()
        base_r1 = data["Reactants"].apply(lambda x: x.split(".")[0]).tolist()
        base_r2 = data["Reactants"].apply(lambda x: x.split(".")[1]).tolist()
        self.r1_list = (
            base_r1 + base_r2 if self.split in self.augment_splits else base_r1
        )
        self.r2_list = (
            base_r2 + base_r1 if self.split in self.augment_splits else base_r2
        )
        del data

        # 2. Pre-compute fingerprints and convert to tensor array
        # Core improvement: use index lists and contiguous tensor storage instead of dicts
        unique_smiles = list(set(base_prod + base_r1 + base_r2))
        self.smi2idx = {smi: i for i, smi in enumerate(unique_smiles)}

        # Try to load or generate fingerprint tensor
        cache_file = os.path.join(
            self.dataset_path, f"fp_tensor_{split}_r{fp_radius}_b{fp_nBits}.pt"
        )
        if os.path.exists(cache_file):
            self.fp_tensor_array = torch.load(cache_file)
        else:
            print("Computing fingerprints into tensor array...")
            # Assuming get_morgan_fp_tensor function is defined
            fplist = [
                get_morgan_fp_tensor(smi, radius=fp_radius, nBits=fp_nBits)
                for smi in tqdm(unique_smiles)
            ]
            self.fp_tensor_array = torch.stack(fplist)
            torch.save(self.fp_tensor_array, cache_file)

        # 3. Data augmentation
        if self.split in self.augment_splits:
            self.products = base_prod + base_prod
            self.r1_indices = [self.smi2idx[s] for s in base_r1] + [
                self.smi2idx[s] for s in base_r2
            ]
            self.r2_indices = [self.smi2idx[s] for s in base_r2] + [
                self.smi2idx[s] for s in base_r1
            ]
        else:
            self.products = base_prod
            self.r1_indices = [self.smi2idx[s] for s in base_r1]
            self.r2_indices = [self.smi2idx[s] for s in base_r2]

    def __len__(self):
        return len(self.products)

    def __getitem__(self, idx):
        return (
            self.products[idx],
            self.r1_list[idx],  # Must return strings for model's subsequent encoding
            self.r2_list[idx],  # Must return strings for model's subsequent encoding
            self.fp_tensor_array[self.r1_indices[idx]],
            self.fp_tensor_array[self.r2_indices[idx]],
        )
