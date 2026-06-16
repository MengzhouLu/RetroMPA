import os
import pickle
import dgl
import torch
import pysmiles
import numpy as np

from MolR.model import GNN
from dgl.dataloading import GraphDataLoader
from MolR.data_processing import networkx_to_dgl


class GraphDataset(dgl.data.DGLDataset):
    def __init__(self, path_to_model, smiles_list, gpu):
        self.path = path_to_model
        self.smiles_list = smiles_list
        self.gpu = gpu
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
        print("the number of smiles successfully parsed: %d" % len(self.parsed))
        print(
            "the number of smiles failed to be parsed: %d"
            % (len(self.smiles_list) - len(self.parsed))
        )
        if torch.cuda.is_available() and self.gpu is not None:
            self.graphs = [graph.to("cuda:" + str(self.gpu)) for graph in self.graphs]

    def __getitem__(self, i):
        return self.graphs[i]

    def __len__(self):
        return len(self.graphs)


class MolEFeaturizer(object):
    def __init__(self, path_to_model, gpu=0, precision=32):
        self.path_to_model = path_to_model
        self.gpu = gpu
        self.precision = precision
        with open(path_to_model + "/hparams.pkl", "rb") as f:
            hparams = pickle.load(f)
        self.mole = GNN(
            hparams["gnn"], hparams["layer"], hparams["feature_len"], hparams["dim"]
        )
        self.dim = hparams["dim"]
        if torch.cuda.is_available() and gpu is not None:
            self.mole.load_state_dict(
                torch.load(path_to_model + "/model.pt", map_location="cuda:0")
            )
            self.mole = self.mole.cuda(gpu)
        else:
            self.mole.load_state_dict(
                torch.load(
                    path_to_model + "/model.pt", map_location=torch.device("cpu")
                )
            )

    def transform(self, smiles_list, batch_size=None):
        data = GraphDataset(self.path_to_model, smiles_list, self.gpu)
        dataloader = GraphDataLoader(
            data, batch_size=batch_size if batch_size is not None else len(smiles_list)
        )
        all_embeddings = np.zeros(
            (len(smiles_list), self.dim),
            dtype=np.float64 if self.precision == 64 else np.float32,
        )
        flags = np.zeros(len(smiles_list), dtype=bool)
        res = []
        with torch.no_grad():
            self.mole.eval()
            for graphs in dataloader:
                graph_embeddings = self.mole(graphs)
                res.append(graph_embeddings)
            res = torch.cat(res, dim=0).cpu().numpy()
        all_embeddings[data.parsed, :] = res
        flags[data.parsed] = True
        print("done\n")
        return all_embeddings, flags


def example_usage():
    model = MolEFeaturizer(
        path_to_model=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "saved",
            "gcn_1024",
        )
    )
    embeddings, flags = model.transform(
        [
            "C[C@H1]1[C@@H1]([C@H1]([C@H1]([C@@H1](O1)OC[C@@H1]2[C@H1]([C@@H1]([C@H1]([C@@H1](O2)OC3=C(OC4=CC(=CC(=C4C3=O)O)O)C5=CC(=C(C(=C5)OC)O)O)O)O)O)O)O)O",
            "COc1cc(-c2oc3cc(O)cc(O)c3c(=O)c2O[C@@H]2O[C@H](CO[C@@H]3O[C@@H](C)[C@H](O)[C@@H](O)[C@H]3O)[C@@H](O)[C@H](O)[C@H]2O)cc(O)c1O",
            "ccc",
        ]
    )
    print(embeddings)
    print(embeddings.shape)  # dim=1024
    print(embeddings.dtype)
    print(embeddings.astype(np.float16))
    print(flags)
    print(model.dim)


if __name__ == "__main__":
    example_usage()
