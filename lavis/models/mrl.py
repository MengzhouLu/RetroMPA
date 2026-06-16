import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

import pickle
import dgl
import torch
import pysmiles
import numpy as np
from MolR.src.model import GNN
from dgl.dataloading import GraphDataLoader
from MolR.src.data_processing import networkx_to_dgl
import logging

logging.getLogger("pysmiles").setLevel(logging.CRITICAL)  # Anything higher than warning
import torch.nn.functional as F
from lavis.models import mrl


# @torch.no_grad()
class GraphDataset(dgl.data.DGLDataset):
    def __init__(self, path_to_model, smiles_list, device):
        self.path = path_to_model
        self.smiles_list = smiles_list
        self.device = device
        self.parsed = []
        self.graphs = []
        super().__init__(name="graph_dataset")

    def process(self):
        with open(self.path + "/feature_enc.pkl", "rb") as f:
            feature_encoder = pickle.load(f)
        for i, smiles in enumerate(self.smiles_list):
            try:
                raw_graph = pysmiles.read_smiles(smiles, zero_order_bonds=False)
                dgl_graph = networkx_to_dgl(raw_graph, feature_encoder)
                self.graphs.append(dgl_graph)
                self.parsed.append(i)
            except:
                print("ERROR: No. %d smiles is not parsed successfully" % i)
        # print('the number of smiles successfully parsed: %d' % len(self.parsed))
        # print('the number of smiles failed to be parsed: %d' % (len(self.smiles_list) - len(self.parsed)))
        # Graphs stay on CPU, moved to device per-batch in transform()

    def __getitem__(self, i):
        return self.graphs[i]

    def __len__(self):
        return len(self.graphs)


class MolEFeaturizer(object):
    def __init__(self, path_to_model, device="cuda", precision=32):
        self.path_to_model = path_to_model
        self.device = device
        self.precision = precision
        with open(path_to_model + "/hparams.pkl", "rb") as f:
            hparams = pickle.load(f)
        self.mole = GNN(
            hparams["gnn"], hparams["layer"], hparams["feature_len"], hparams["dim"]
        )
        # self.mole =self.mole.eval()
        self.dim = hparams["dim"]
        self.num_features = self.dim
        if torch.cuda.is_available() and self.device is not None:
            self.mole.load_state_dict(
                torch.load(path_to_model + "/model.pt", map_location=self.device)
            )
            self.mole = self.mole.to(self.device)
        else:
            self.mole.load_state_dict(
                torch.load(
                    path_to_model + "/model.pt", map_location=torch.device("cpu")
                )
            )
            self.device = torch.device("cpu")
        self._graph_cache = {}
        self._feature_encoder = None

    def _get_feature_encoder(self):
        if self._feature_encoder is None:
            with open(self.path_to_model + "/feature_enc.pkl", "rb") as f:
                self._feature_encoder = pickle.load(f)
        return self._feature_encoder

    def _get_cached_graph(self, smiles):
        cached_graph = self._graph_cache.get(smiles)
        if cached_graph is not None:
            return cached_graph

        feature_encoder = self._get_feature_encoder()
        raw_graph = pysmiles.read_smiles(smiles, zero_order_bonds=False)
        dgl_graph = networkx_to_dgl(raw_graph, feature_encoder)
        self._graph_cache[smiles] = dgl_graph
        return dgl_graph

    # @torch.no_grad()
    def transform(self, smiles_list, batch_size=None):
        if batch_size is None:
            batch_size = 512

        parsed = []
        graphs = []
        for i, smiles in enumerate(smiles_list):
            try:
                graphs.append(self._get_cached_graph(smiles))
                parsed.append(i)
            except Exception:
                print("ERROR: No. %d smiles is not parsed successfully" % i)

        graph_worker_count = min(4, max(0, (os.cpu_count() or 1) // 8))
        if len(graphs) < 1024:
            graph_worker_count = 0

        dataloader = GraphDataLoader(
            graphs,
            batch_size=batch_size,
            shuffle=False,
            num_workers=graph_worker_count,
            pin_memory=torch.cuda.is_available() and self.device is not None,
            persistent_workers=False,
        )
        all_embeddings = torch.zeros(
            (len(smiles_list), self.dim),
            dtype=torch.float64 if self.precision == 64 else torch.float32,
        )
        flags = np.zeros(len(smiles_list), dtype=bool)
        res = []
        for graphs in dataloader:
            graphs = graphs.to(self.device)
            graph_embeddings = self.mole(graphs)
            res.append(graph_embeddings)
        if len(res) > 0:
            res = torch.cat(res, dim=0).to(all_embeddings.dtype).cpu()
            all_embeddings[parsed, :] = res
            flags[parsed] = True
        all_embeddings = all_embeddings.to(self.device)
        return all_embeddings, flags


# @torch.no_grad()
def create_mrl(model_name, device, precision=32):
    # local_rank = torch.distributed.get_rank()
    # torch.cuda.set_device(local_rank)
    # global device
    # device = torch.device("cuda", local_rank)

    if model_name == "MolR":
        device = device
        model = MolEFeaturizer(
            path_to_model=os.path.join(_PROJECT_ROOT, "MolR", "saved", "gcn_1024"),
            precision=precision,
            device=device,
        )
        print(f"mrl model loaded successfully on {device} device")

    elif model_name == "Unimol":
        return
        import numpy as np
        from unimol_tools import UniMolRepr

        # single smiles unimol representation
        clf = UniMolRepr(data_type="molecule", remove_hs=False)
        return clf

        smiles = "c1ccc(cc1)C2=NCC(=O)Nc3c2cc(cc3)[N+](=O)[O]"
        smiles_list = [
            "CC(=O)c1ccc2[nH]ccc2c1",
            "CC(C)(C)OC(=O)OC(=O)OC(C)(C)C",
            "CC(=O)c1ccc2c(ccn2C(=O)OC(C)(C)C)c1",
        ]
        unimol_repr = clf.get_repr(smiles_list, return_atomic_reprs=True)

        # CLS token repr
        print(np.array(unimol_repr["cls_repr"]).shape)
        print(np.array(unimol_repr["cls_repr"]))
        input()
        # atomic level repr, align with rdkit mol.GetAtoms()
        print(np.array(unimol_repr["atomic_reprs"]).shape)
    elif model_name == "SELFormer":
        return
        import os

        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["WANDB_DISABLED"] = "true"
        # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        import pandas as pd
        from transformers import RobertaTokenizer, RobertaModel, RobertaConfig

        model_name = os.path.join(
            _PROJECT_ROOT, "SELFormer1", "data", "pretrained_models", "SELFormer"
        )  # path of the pre-trained model
        config = RobertaConfig.from_pretrained(model_name)
        config.output_hidden_states = True
        tokenizer = RobertaTokenizer.from_pretrained(
            os.path.join(_PROJECT_ROOT, "SELFormer1", "data", "RobertaFastTokenizer")
        )
        model = RobertaModel.from_pretrained(model_name, config=config).to(device)
        model.eval()
        return model, tokenizer
    else:
        raise ValueError("Invalid model name")
        model = None
    return model


def example_usage():
    model = mrl.create_mrl(model_name="MolR", precision=64, device="cuda:0")
    import time

    start_time = time.time()

    embeddings, flags = model.transform(
        [
            "CC(=O)c1ccc2c(ccn2C(=O)OC(C)(C)C)c1",
            "CC(=O)c1ccc2[nH]ccc2c1",
            "CC(C)(C)OC(=O)OC(=O)OC(C)(C)C",
        ]
    )
    # embeddings=F.normalize(torch.from_numpy(embeddings).to(dtype=torch.float32),dim=-1)
    embeddings = F.normalize((embeddings).to(dtype=torch.float32), dim=-1)
    # import pickle
    # with open(os.path.join(_PROJECT_ROOT, 'reactants_dict_all_unique.pkl'), 'rb') as f:
    #     embeddings1=pickle.load(f)
    # value=embeddings1['COC(=O)c1nc2cc(NC(=O)c3ccccc3)ccc2[nH]1']
    # print(value)
    # print(F.normalize(torch.from_numpy(value).to(dtype=torch.float32),dim=-1))
    end_time = time.time()
    print("Time used:", end_time - start_time)
    print(embeddings)
    print(embeddings.shape)  # dim=1024


# example_usage()
