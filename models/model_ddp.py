import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.distributed as dist
import torch.nn.functional as F

from models.model import BlipOutput
from models.model import GatherLayer
from models.model import Mymodel_Mydecoder_openMrl_dualprior


def _is_dist_ready():
    return dist.is_available() and dist.is_initialized()


class Mymodel_Mydecoder_openMrl_dualprior_ddp(Mymodel_Mydecoder_openMrl_dualprior):
    """DDP-safe version of Mymodel_Mydecoder_openMrl_dualprior."""

    def _rank_world_size(self):
        if _is_dist_ready():
            return dist.get_rank(), dist.get_world_size()
        return 0, 1

    def all_gather_with_grad(self, tensors):
        """
        Performs all_gather on tensors while keeping grad graph connected.
        """
        _, world_size = self._rank_world_size()
        if world_size == 1:
            return tensors.contiguous()

        tensors = tensors.contiguous()
        tensor_all = GatherLayer.apply(tensors)
        return torch.cat(tensor_all, dim=0)

    def forward(
        self, src, tgt1, tgt2, fp_tgt1=None, fp_tgt2=None, return_embedding=False
    ):
        product_embedding, product_embedding_mrl = self.Qformer.forward_molecular(src)

        product_embedding_mrl, flags = self.mrl.transform(src)
        reactant1_embedding_mrl, flags = self.mrl.transform(tgt1)
        reactant2_embedding_mrl, flags = self.mrl.transform(tgt2)

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

        rank, _ = self._rank_world_size()
        bs = product_embedding.shape[0]
        device = self.current_device()

        sim_r2p_1 = similarity_r2p_1 / self.temp
        sim_r2p_2 = similarity_r2p_2 / self.temp
        sim_p2r_1 = similarity_p2r_1 / self.temp
        sim_p2r_2 = similarity_p2r_2 / self.temp

        targets = torch.arange(
            rank * bs, rank * bs + bs, dtype=torch.long, device=device
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
