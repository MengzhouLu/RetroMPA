import os
import sys
from typing import List

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta
from torch.nn import TransformerDecoder, TransformerDecoderLayer

import config as cfg
from lavis.common.dist_utils import is_dist_avail_and_initialized
from lavis.models.blip2_models.blip2 import disabled_train
from lavis.models.blip2_models.blip2_qformer import Blip2Qformer
from lavis.models.blip_models.blip_outputs import BlipOutput

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer decoder inputs."""

    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].detach()
        return self.dropout(x)


class Mydecoder(nn.Module):
    def __init__(
        self,
        device,
        input_dim=1024,
        n_layers=6,
        n_heads=8,
        ff_dim=2048,
        dropout=0.1,
        activation="gelu",
        batch_first=True,
    ):
        super().__init__()
        self.batch_first = batch_first
        self.device = device
        self.decoder_layer = TransformerDecoderLayer(
            d_model=input_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation=activation,
            device=device,
            batch_first=batch_first,
        )
        self.decoder = TransformerDecoder(self.decoder_layer, n_layers)
        self.positional_encoding = PositionalEncoding(input_dim, dropout=dropout)

    def forward(self, encoder_output, tgt):
        tgt = self.positional_encoding(tgt).to(self.device)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            tgt.shape[1] if self.batch_first else tgt.shape[0], device=self.device
        )
        output = self.decoder(tgt, encoder_output, tgt_mask=tgt_mask)
        return output


class GatherLayer(torch.autograd.Function):
    """
    Gather tensors from all workers with support for backward propagation.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]


class Mymodel_Mydecoder_openMrl_dualprior(nn.Module):
    """Mymodel with iMolCLR + ProGCL dual-prior weighting for contrastive loss."""

    def __init__(
        self,
        Qformer_path,
        logger,
        device,
        d_qformer=768,
        d_model=1024,
        dec_voc_size=1024,
        freeze_Qformer=True,
        freeze_mrl=False,
        weight1=1,
        weight2=1,
        use_imolclr=True,
        use_progcl=True,
    ):
        super().__init__()

        self.Qformer_path = Qformer_path
        self.device = device
        self.Qformer = self.load_Qformer(self.Qformer_path)
        self.mrl = self.Qformer.molecular_encoder
        self.Decoder = Mydecoder(device=device)
        self.molecular_proj = nn.Linear(d_qformer, d_model)
        self.prelabel_mlp = nn.Linear(d_model, dec_voc_size)
        self.temp = nn.Parameter(0.07 * torch.ones([]))
        self.weight1 = weight1
        self.weight2 = weight2

        self.use_imolclr = use_imolclr
        self.use_progcl = use_progcl
        logger.info(
            f"dualprior: use_imolclr={self.use_imolclr}, use_progcl={self.use_progcl}"
        )

        if freeze_mrl:
            pass
        else:
            for name, param in self.mrl.mole.named_parameters():
                param.requires_grad = True
            self.mrl.mole = self.mrl.mole.train()
            self.mrl.mole.train(True)
            logger.info("train Mrl")

        logger.info("load Qformer from {}".format(self.Qformer_path))
        if freeze_Qformer:
            for name, param in self.Qformer.named_parameters():
                param.requires_grad = False
            self.Qformer = self.Qformer.eval()
            self.Qformer.train = disabled_train
            logger.info("freeze Qformer")
        else:
            for name, param in self.Qformer.named_parameters():
                param.requires_grad = True
            self.Qformer = self.Qformer.train()
            self.Qformer.train(True)
            logger.info("train Qformer")
        logger.info(f"self.weight1:{self.weight1},self.weight2:{self.weight2}")

    def batch_tanimoto_sim(self, fp_query, fp_gallery):
        fp_gallery = fp_gallery.to(fp_query.device)
        intersect = torch.matmul(fp_query, fp_gallery.transpose(0, 1))
        sum_q = fp_query.sum(dim=1, keepdim=True)
        sum_g = fp_gallery.sum(dim=1, keepdim=True).transpose(0, 1)
        union = sum_q + sum_g - intersect
        return intersect / (union + 1e-8)

    @torch.no_grad()
    def compute_progcl_weights(self, cosine_sim_matrix, max_iters=5):
        B, K = cosine_sim_matrix.shape
        x = (cosine_sim_matrix.detach().flatten() + 1.0) / 2.0
        x = torch.clamp(x, min=0.01, max=0.99).float()

        pi = torch.tensor([0.8, 0.2], device=x.device, dtype=torch.float32)
        mu = torch.tensor([0.3, 0.8], device=x.device, dtype=torch.float32)
        var = torch.tensor([0.05, 0.05], device=x.device, dtype=torch.float32)

        gamma = torch.zeros((x.size(0), 2), device=x.device, dtype=torch.float32)

        for _ in range(max_iters):
            temp0 = mu[0] * (1 - mu[0]) / var[0] - 1
            temp1 = mu[1] * (1 - mu[1]) / var[1] - 1

            a0, b0 = mu[0] * temp0, (1 - mu[0]) * temp0
            a1, b1 = mu[1] * temp1, (1 - mu[1]) * temp1

            pdf0 = torch.exp(Beta(a0, b0).log_prob(x))
            pdf1 = torch.exp(Beta(a1, b1).log_prob(x))

            p0 = pi[0] * pdf0
            p1 = pi[1] * pdf1
            total_p = p0 + p1 + 1e-8

            gamma[:, 0] = p0 / total_p
            gamma[:, 1] = p1 / total_p

            N_k = gamma.sum(dim=0)
            pi = N_k / x.size(0)
            mu = (gamma * x.unsqueeze(1)).sum(dim=0) / N_k
            var = (gamma * ((x.unsqueeze(1) - mu) ** 2)).sum(dim=0) / N_k
            var = torch.clamp(var, min=1e-4)
            max_bound = mu * (1 - mu) - 1e-4
            var = torch.minimum(var, max_bound)

        prob_true_negative = gamma[:, 0].reshape(B, K)
        return prob_true_negative.to(cosine_sim_matrix.dtype)

    def forward(
        self, src, tgt1, tgt2, fp_tgt1=None, fp_tgt2=None, return_embedding=False
    ):
        product_embedding, _ = self.Qformer.forward_molecular(src)

        product_embedding_mrl, _ = self.mrl.transform(src)
        reactant1_embedding_mrl, _ = self.mrl.transform(tgt1)
        reactant2_embedding_mrl, _ = self.mrl.transform(tgt2)

        product_embedding_mrl = F.normalize(product_embedding_mrl, dim=-1)
        reactant1_embedding_mrl = F.normalize(reactant1_embedding_mrl, dim=-1)
        reactant2_embedding_mrl = F.normalize(reactant2_embedding_mrl, dim=-1)

        product_embedding_mrl = F.normalize(product_embedding_mrl.unsqueeze(1), dim=-1)
        reactant1_embedding_input = F.normalize(
            reactant1_embedding_mrl.unsqueeze(1), dim=-1
        )

        product_embedding_proj = self.molecular_proj(product_embedding)

        start_token = product_embedding_mrl.to(self.current_device())
        reactants_embedding_mrl_input = torch.cat(
            [start_token, reactant1_embedding_input], dim=1
        )

        predict_reactants_embedding = self.Decoder(
            product_embedding_proj, reactants_embedding_mrl_input
        )
        predict_reactant1_embedding = F.normalize(
            predict_reactants_embedding[:, 0, :].squeeze(), dim=-1
        )
        predict_reactant2_embedding = F.normalize(
            predict_reactants_embedding[:, 1, :].squeeze(), dim=-1
        )

        reactant1_features_all = self.concat_all_gather(reactant1_embedding_mrl)
        reactant2_features_all = self.concat_all_gather(reactant2_embedding_mrl)
        predict_reactant1_features_all = self.concat_all_gather(
            predict_reactant1_embedding
        )
        predict_reactant2_features_all = self.concat_all_gather(
            predict_reactant2_embedding
        )

        similarity_r2p_1 = torch.matmul(
            reactant1_embedding_mrl, predict_reactant1_features_all.T
        )
        similarity_r2p_2 = torch.matmul(
            reactant2_embedding_mrl, predict_reactant2_features_all.T
        )
        similarity_p2r_1 = torch.matmul(
            predict_reactant1_embedding, reactant1_features_all.T
        )
        similarity_p2r_2 = torch.matmul(
            predict_reactant2_embedding, reactant2_features_all.T
        )

        rank = 0
        bs = product_embedding.shape[0]
        device = self.current_device()

        sim_r2p_1 = similarity_r2p_1 / self.temp
        sim_r2p_2 = similarity_r2p_2 / self.temp
        sim_p2r_1 = similarity_p2r_1 / self.temp
        sim_p2r_2 = similarity_p2r_2 / self.temp

        targets = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=int).to(
            device
        )

        def apply_dual_prior(logits, cosine_sim, fp_query, fp_gallery):
            weight_matrix = torch.ones_like(logits)

            if self.use_imolclr and fp_query is not None:
                fp_world = self.concat_all_gather(fp_gallery)
                tanimoto_sim = self.batch_tanimoto_sim(fp_query, fp_world)
                tanimoto_sim = tanimoto_sim.to(weight_matrix.device)
                weight_matrix = weight_matrix * (1.0 - tanimoto_sim)

            if self.use_progcl:
                bmm_weights = self.compute_progcl_weights(cosine_sim)
                bmm_weights = bmm_weights.to(weight_matrix.device)
                weight_matrix = weight_matrix * bmm_weights

            weight_matrix = torch.clamp(weight_matrix, min=1e-6)

            mask = torch.zeros_like(logits, dtype=torch.bool)
            labels_local = (
                torch.arange(logits.shape[0], dtype=torch.long, device=logits.device)
                + rank * logits.shape[0]
            )
            mask.scatter_(1, labels_local.unsqueeze(1), True)

            adjusted_logits = logits + torch.log(weight_matrix)
            final_logits = torch.where(mask, logits, adjusted_logits)
            return final_logits

        final_sim_r2p_1 = apply_dual_prior(
            sim_r2p_1, similarity_r2p_1, fp_tgt1, fp_tgt1
        )
        final_sim_r2p_2 = apply_dual_prior(
            sim_r2p_2, similarity_r2p_2, fp_tgt2, fp_tgt2
        )
        final_sim_p2r_1 = apply_dual_prior(
            sim_p2r_1, similarity_p2r_1, fp_tgt1, fp_tgt1
        )
        final_sim_p2r_2 = apply_dual_prior(
            sim_p2r_2, similarity_p2r_2, fp_tgt2, fp_tgt2
        )

        loss_r2p_1 = F.cross_entropy(final_sim_r2p_1, targets, label_smoothing=0.1)
        loss_r2p_2 = F.cross_entropy(final_sim_r2p_2, targets, label_smoothing=0.1)
        loss_p2r_1 = F.cross_entropy(final_sim_p2r_1, targets, label_smoothing=0.1)
        loss_p2r_2 = F.cross_entropy(final_sim_p2r_2, targets, label_smoothing=0.1)

        loss_rpc = (
            self.weight1 * loss_r2p_1
            + self.weight2 * loss_r2p_2
            + self.weight1 * loss_p2r_1
            + self.weight2 * loss_p2r_2
        ) / 4

        loss_r1 = torch.norm(
            reactant1_embedding_mrl - predict_reactant1_embedding, dim=1
        ).sum()
        loss_r2 = torch.norm(
            reactant2_embedding_mrl - predict_reactant2_embedding, dim=1
        ).sum()
        loss_mol_sim_pos = self.weight1 * loss_r1 + self.weight2 * loss_r2

        with torch.no_grad():
            sim_p2r_1[:, rank * bs : rank * bs + bs].fill_diagonal_(-10000)
            sim_p2r_2[:, rank * bs : rank * bs + bs].fill_diagonal_(-10000)
            weights_p2r_1 = F.softmax(sim_p2r_1, dim=1)
            weights_p2r_2 = F.softmax(sim_p2r_2, dim=1)

        reactant1_embeds_neg = []
        reactant2_embeds_neg = []
        reactant1_embedding_world = self.all_gather_with_grad(reactant1_embedding_mrl)
        reactant2_embedding_world = self.all_gather_with_grad(reactant2_embedding_mrl)
        for b in range(bs):
            neg1_idx = torch.multinomial(weights_p2r_1[b], 1).item()
            neg2_idx = torch.multinomial(weights_p2r_2[b], 1).item()
            reactant1_embeds_neg.append(reactant1_embedding_world[neg1_idx])
            reactant2_embeds_neg.append(reactant2_embedding_world[neg2_idx])
        reactant1_embeds_neg = torch.stack(reactant1_embeds_neg, dim=0)
        reactant2_embeds_neg = torch.stack(reactant2_embeds_neg, dim=0)

        loss_r1_neg = (
            self.weight1
            * torch.norm(
                reactant1_embeds_neg - predict_reactant1_embedding, dim=1
            ).sum()
        )
        loss_r2_neg = (
            self.weight2
            * torch.norm(
                reactant2_embeds_neg - predict_reactant2_embedding, dim=1
            ).sum()
        )
        loss_mol_sim_neg = loss_r1_neg + loss_r2_neg
        loss_mol_sim = (loss_mol_sim_pos - loss_mol_sim_neg) / 2

        if return_embedding:
            return (
                loss_rpc + loss_mol_sim,
                predict_reactant1_embedding,
                predict_reactant2_embedding,
                reactant1_embedding_mrl,
                reactant2_embedding_mrl,
            )

        return BlipOutput(
            loss=loss_rpc + loss_mol_sim,
            loss_1=loss_r2p_1 + loss_p2r_1 + loss_r1 - loss_r1_neg,
            loss_2=loss_r2p_2 + loss_p2r_2 + loss_r2 - loss_r2_neg,
            loss_r2p_1=loss_r2p_1,
            loss_r2p_2=loss_r2p_2,
            loss_p2r_1=loss_p2r_1,
            loss_p2r_2=loss_p2r_2,
            loss_msl_pos=loss_mol_sim_pos,
            loss_msl_neg=loss_mol_sim_neg,
            loss_r1=loss_r1,
            loss_r2=loss_r2,
        )

    def predict_reactant_embeddings(self, src, tgt1, tgt2):
        product_embedding, _ = self.Qformer.forward_molecular(src)

        product_embedding_mrl, _ = self.mrl.transform(src)
        reactant1_embedding_mrl, _ = self.mrl.transform(tgt1)
        reactant2_embedding_mrl, _ = self.mrl.transform(tgt2)

        product_embedding_mrl = F.normalize(product_embedding_mrl, dim=-1)
        reactant1_embedding_mrl = F.normalize(reactant1_embedding_mrl, dim=-1)
        reactant2_embedding_mrl = F.normalize(reactant2_embedding_mrl, dim=-1)

        product_embedding_mrl = F.normalize(product_embedding_mrl.unsqueeze(1), dim=-1)
        reactant1_embedding_input = F.normalize(
            reactant1_embedding_mrl.unsqueeze(1), dim=-1
        )

        product_embedding_proj = self.molecular_proj(product_embedding)

        start_token = product_embedding_mrl.to(self.current_device())
        reactants_embedding_mrl_input = torch.cat(
            [start_token, reactant1_embedding_input], dim=1
        )

        predict_reactants_embedding = self.Decoder(
            product_embedding_proj, reactants_embedding_mrl_input
        )
        predict_reactant1_embedding = predict_reactants_embedding[:, 0, :].squeeze()
        predict_reactant2_embedding = predict_reactants_embedding[:, 1, :].squeeze()
        return predict_reactant1_embedding, predict_reactant2_embedding

    @torch.no_grad()
    def predict_reactants(self, src, tgt1, tgt2, dict_data, topK=50):
        device = self.current_device()

        p1_emb, p2_emb = self.predict_reactant_embeddings(src, tgt1, tgt2)
        p1_emb = F.normalize(p1_emb, dim=-1).float()
        p2_emb = F.normalize(p2_emb, dim=-1).float()

        if isinstance(dict_data, dict):
            molecular_names = list(dict_data.keys())
            lib_embeddings = (
                F.normalize(
                    torch.stack([dict_data[name] for name in molecular_names]), dim=-1
                )
                .to(device)
                .float()
            )
        else:
            lib_embeddings, molecular_names = dict_data

        def get_topk_results(query_emb):
            cos_sim = torch.matmul(query_emb, lib_embeddings.T)
            dist = torch.cdist(query_emb, lib_embeddings, p=2)

            d_min = dist.min(dim=1, keepdim=True).values
            d_max = dist.max(dim=1, keepdim=True).values

            norm_dist = (dist - d_min) / (d_max - d_min + 1e-8)
            dist_sim = 1.0 - norm_dist

            combined_score = cos_sim + dist_sim

            _, topk_indices = torch.topk(combined_score, topK, dim=1, largest=True)

            batch_res = []
            indices_np = topk_indices.cpu().numpy()
            for row in indices_np:
                batch_res.append([molecular_names[idx] for idx in row])
            return batch_res

        res1 = get_topk_results(p1_emb)
        res2 = get_topk_results(p2_emb)

        return res1, res2

    def load_Qformer(self, path):
        Qformer = Blip2Qformer(
            max_txt_len=cfg.max_txt_len,
            molecular_precision=cfg.molecular_precision,
            device=self.device,
        )
        state_dict = torch.load(path, map_location=self.device)
        Qformer.load_state_dict(state_dict)
        Qformer.to(self.device)
        print(f"Qformer loaded on {self.device}")
        return Qformer

    @torch.no_grad()
    def concat_all_gather(self, tensor):
        if not is_dist_avail_and_initialized():
            return tensor
        if not tensor.is_cuda:
            tensor = tensor.cuda()
        tensor = tensor.contiguous()

        tensors_gather = [torch.ones_like(tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(tensors_gather, tensor, async_op=False)

        output = torch.cat(tensors_gather, dim=0)
        return output

    def all_gather_with_grad(self, tensors):
        world_size = 1
        if world_size == 1:
            return tensors.contiguous()
        tensors = tensors.contiguous()
        tensor_all = GatherLayer.apply(tensors)
        return torch.cat(tensor_all, dim=0)

    def current_device(self):
        return next(self.parameters()).device
