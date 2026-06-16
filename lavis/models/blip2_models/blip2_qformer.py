"""
Copyright (c) 2023, salesforce.com, inc.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F


import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from lavis.common.registry import registry
from lavis.models.base_model import all_gather_with_grad, concat_all_gather
from lavis.models.blip2_models.blip2 import (
    Blip2Base,
    compute_sim_matrix,
    disabled_train,
)
from lavis.models.blip_models.blip_outputs import BlipOutput, BlipOutputFeatures


@registry.register_model("blip2")
@registry.register_model("blip2_feature_extractor")
class Blip2Qformer(Blip2Base):
    """
    BLIP2 first-stage model with Q-former and ViT.
    Supported model types:
        - pretrained: pretrained model with vit-g
        - pretrain_vitL: pretrained model with vit-large
        - coco: fintuned model on coco
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2", "pretrain")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain": "configs/models/blip2/blip2_pretrain.yaml",
        "pretrain_vitL": "configs/models/blip2/blip2_pretrain_vitL.yaml",
        "coco": "configs/models/blip2/blip2_coco.yaml",
    }

    def __init__(
        self,
        device,
        # vit_model="eva_clip_g",
        molecular_model="MolR",  ###
        # img_size=224,
        # drop_path_rate=0,
        # use_grad_checkpoint=False,
        # vit_precision="fp16",
        molecular_precision=32,  ### Note: fp16 may be too low precision
        # freeze_vit=True,
        freeze_molecular_model=False,  ###
        num_query_token=32,
        cross_attention_freq=2,
        embed_dim=256,
        max_txt_len=32,
    ):
        super().__init__()

        self.tokenizer = self.init_tokenizer()

        # self.visual_encoder, self.ln_vision = self.init_vision_encoder(
        #     vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        # )
        # self.molecular_encoder,self.molecular_tokenizer, self.ln_molecular = self.init_molecular_encoder(molecular_model,molecular_precision,device)###
        self.molecular_encoder, self.ln_molecular = self.init_molecular_encoder(
            molecular_model, molecular_precision, device
        )  ###
        # if freeze_vit:
        #     for name, param in self.visual_encoder.named_parameters():
        #         param.requires_grad = False
        #     self.visual_encoder = self.visual_encoder.eval()
        #     self.visual_encoder.train = disabled_train
        #     logging.info("freeze vision encoder")
        if freeze_molecular_model:  ###
            for name, param in self.molecular_encoder.mole.named_parameters():
                param.requires_grad = False
            self.molecular_encoder.mole = self.molecular_encoder.mole.eval()
            self.molecular_encoder.mole.train = disabled_train
            logging.info("freeze molecular encoder")
        else:
            for name, param in self.molecular_encoder.mole.named_parameters():
                param.requires_grad = True
            self.molecular_encoder.mole = self.molecular_encoder.mole.train()
            self.molecular_encoder.mole.train(True)
        # self.Qformer, self.query_tokens = self.init_Qformer(
        #     num_query_token, self.visual_encoder.num_features, cross_attention_freq
        # )

        self.Qformer, self.query_tokens = self.init_Qformer(  ###
            num_query_token, self.molecular_encoder.num_features, cross_attention_freq
        )
        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])

        # self.vision_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.molecular_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)  ###
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)

        self.itm_head = nn.Linear(self.Qformer.config.hidden_size, 2)

        self.temp = nn.Parameter(0.07 * torch.ones([]))

        self.max_txt_len = max_txt_len
        print("max_txt_len:", self.max_txt_len)

    def forward(self, samples):
        device = self.current_device()

        # image = samples["image"]
        smiles = samples[
            "smiles"
        ]  ### MOLR input: list of SMILES strings ['smiles','smiles2',...]
        text = samples["text_input"]

        # image_embeds = self.ln_vision(self.visual_encoder(image))
        # image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
        #     image.device
        # )

        # with torch.no_grad():
        embeddings, flags = self.molecular_encoder.transform(smiles)
        embeddings = F.normalize((embeddings), dim=-1).to(device)
        # embeddings_unimol_repr=self.molecular_encoder.get_repr(smiles)
        # embeddings=np.array(embeddings_unimol_repr['cls_repr'])

        # selfies_list=[]
        # for smile in smiles:
        #     try:
        #         selfies=sf.encoder(smile)
        #     except:
        #         selfies=smile
        #     selfies_list.append(selfies)

        # self.molecular_encoder.eval()
        # # selfies_list=['[O][=C][Branch2][Ring2][C][C][S][C][=N][C][=C][C][=C][C][=C][Ring1][=Branch1][C][=Branch1][C][=O][N][Ring1][O][C][=C][C][=C][Branch1][C][F][C][Branch1][C][Cl][=C][Ring1][Branch2][N][C][=C][C][=C][C][=Branch1][Ring2][=C][Ring1][=Branch1][C][C][C][Ring1][=Branch1]', '[C][O][C][=C][C][=C][Branch2][Branch1][#Branch2][N][C][=Branch1][C][=O][N][C][=C][C][=C][C][=Branch1][Ring2][=C][Ring1][=Branch1][C][C][=Branch1][C][=O][N][Branch1][#Branch1][C][Branch1][C][C][C][O][C][C][Branch1][C][C][C][Branch1][S][C][N][Branch1][C][C][C][=Branch1][C][=O][N][C][Branch1][C][C][C][O][Ring2][Ring1][=Branch2][C][=C][Ring2][Ring2][=Branch1]', '[C][O][C@@][Branch1][C][C][Branch1][Branch2][/C][=C][/C@@H1][Branch1][C][C][O][C][C][C][C@H1][C][=C][C][=Branch1][C][=O][C@H1][C][C][=Branch1][#Branch2][=C][C][C@H1][Branch1][C][O][C][Ring1][Branch2][C@H1][Ring1][=N][C][C][C@][Ring2][Ring1][Ring2][Ring1][P][C]', '[C][C][Branch2][Ring2][Ring1][N][C][=Branch1][C][=O][C][Branch1][#Branch2][C][C][=C][C][=C][C][=C][Ring1][=Branch1][N][C][=Branch1][C][=O][C][Branch1][C][N][C][C][=C][N][=C][NH1][Ring1][Branch1][C][=Branch1][C][=O][O]', '[C][C][=Branch1][C][=O][N][C][C][Branch2][Ring2][#Branch2][O][C][C][Branch1][C][O][C][Branch1][Ring1][C][O][O][C][Branch2][Ring1][Branch1][O][C][C][Branch1][Ring1][C][O][O][C][Branch1][C][O][C][Branch1][C][O][C][Ring1][#Branch2][O][C][Ring2][Ring1][Branch1][O][O][C][Branch1][Ring1][C][O][C][Branch2][Branch2][=N][O][C][O][C][Branch1][Ring1][C][O][C][Branch1][C][O][C][Branch2][#Branch1][Branch2][O][C][O][C][Branch1][Ring1][C][O][C][Branch2][Branch1][=C][O][C][O][C][Branch1][Ring1][C][O][C][Branch1][C][O][C][Branch2][Ring2][=Branch2][O][C][Branch1][=Branch1][C][=Branch1][C][=O][O][C][C][Branch1][C][O][C][Branch1][#Branch1][N][C][Branch1][C][C][=O][C][Branch1][O][C][Branch1][C][O][C][Branch1][C][O][C][O][O][Ring2][Ring1][Ring2][C][Ring2][Ring1][=C][O][C][Branch1][C][O][C][Ring2][Ring2][=Branch2][N][C][Branch1][C][C][=O][C][Ring2][Branch1][#Branch1][O][C][Ring2][#Branch1][Branch2][O]', '[C][O][C][=C][C][=C][C][=C][Ring1][=Branch1][C][=Branch1][C][=O][O][C][C][#C][C][S][C][=N][N][=C][Branch1][#C][C][=C][C][=C][C][=C][C][=C][C][=C][Ring1][#Branch2][Ring1][=Branch1][O][Ring1][#C]', '[C][C][C][=Branch1][C][=O][O][C][C][=Branch1][C][=O][C@@][Branch1][Branch2][O][C][=Branch1][C][=O][C][C][C@H1][Branch1][C][C][C][C@H1][C@H1][C@H1][Branch1][N][C@@H1][Branch1][C][O][C][C@@][Ring1][#Branch1][Ring1][S][C][C@@][Branch1][C][C][C][=C][C][=Branch1][C][=O][C][=C][Ring1][Branch2][C][C@H1][Ring1][P][Cl]', '[O][=C][Branch1][Ring1][C][O][C@@H1][Branch1][C][O][C@H1][Branch1][C][O][C@H1][Branch1][C][O][C@@H1][Branch1][C][O][C@@H1][Branch1][C][O][C][=Branch1][C][=O][O-1]', '[C][C][C][C][C][C][C][C][C][C][C][C][C][Branch1][C][O][C][C][C][C][Branch2][Ring2][Branch1][C][C][C][C][Branch2][Ring1][Branch2][C][C][C][C][C][C][C][C][Branch1][C][O][C][C][=C][C][Branch1][C][C][O][C][Ring1][=Branch1][=O][O][C][O][Ring2][Ring1][Branch2][O][Ring2][Ring1][=N]', '[C][O][C][C][N][Branch1][C][C][C][=Branch1][C][=O][C][=C][C][=C][Branch1][#Branch1][N][C][Branch1][C][C][=O][C][=C][Ring1][#Branch2][O][C][C][Branch1][C][C][N][Branch1][=C][C][=Branch1][C][=O][C][=C][C][=C][C][=C][Ring1][=Branch1][F][C][C][Ring2][Ring1][S][C]', '[C][O][C][=C][C][Branch2][Branch1][#Branch1][N][C][=N][C][=C][Branch1][C][F][C][Branch2][Ring2][#Branch1][N][C][=C][C][=C][C][=Branch1][Ring2][=N][Ring1][=Branch1][N][Branch1][O][C][O][P][=Branch1][C][=O][Branch1][C][O-1][O-1][C][=Branch1][C][=O][C][Branch1][C][C][Branch1][C][C][O][Ring1][S][=N][Ring2][Ring1][O][=C][C][Branch1][Ring1][O][C][=C][Ring2][Ring2][Ring2][O][C].[Na+1].[Na+1]', '[O][=C][Branch1][O][C][=C][C][=C][C][=C][C][=C][Ring1][=Branch1][O][C][O][C][Branch1][=Branch1][C][=Branch1][C][=O][O][C][Branch1][C][O][C][Branch1][C][O][C][Ring1][O][O]', '[N][S][=Branch1][C][=O][=Branch1][C][=O][C][=C][C][=C][N][=C][Branch1][S][N][C][=Branch1][C][=O][C][=C][C][=C][Branch1][C][Br][O][Ring1][=Branch1][S][C][Ring1][=C][=C][Ring2][Ring1][C]', '[C][C][=C][Branch1][C][N][N][=C][Branch2][Ring1][Branch1][C@H1][Branch1][#Branch1][C][C][Branch1][C][N][=O][N][C][C@H1][Branch1][C][N][C][Branch1][C][N][=O][N][=C][Ring2][Ring1][Ring1][C][=Branch1][C][=O][N][C@H1][Branch2][=Branch1][Branch2][C][=Branch1][C][=O][N][C@H1][Branch1][C][C][C@@H1][Branch1][C][O][C@H1][Branch1][C][C][C][=Branch1][C][=O][N][C@H1][Branch2][Ring2][=Branch2][C][=Branch1][C][=O][N][C][C][C][=N][C][Branch2][Ring1][#Branch1][C][=N][C][Branch1][=C][C][=Branch1][C][=O][N][C][C][C][S+1][Branch1][C][C][C][=C][S][Ring1][=C][=C][S][Ring2][Ring1][Ring1][C@@H1][Branch1][C][C][O][C@@H1][Branch2][Ring2][=N][O][C][O][C][Branch1][Ring1][C][O][C][Branch1][C][O][C][Branch1][C][O][C][Ring1][#Branch2][O][C][O][C][Branch1][Ring1][C][O][C][Branch1][C][O][C][Branch1][#Branch1][O][C][Branch1][C][N][=O][C][Ring1][=N][O][C][=C][N][=C][NH1][Ring1][Branch1]', '[C][C][=Branch1][C][=O][N][C@@H1][C@@H1][Branch2][Ring1][Ring2][O][C@@H1][O][C@@H1][Branch1][C][C][C@@H1][Branch1][C][O][C@@H1][Branch1][C][O][C@@H1][Ring1][=Branch2][O][C@H1][Branch2][O][#Branch1][O][C@@H1][O][C@H1][Branch1][Ring1][C][O][C@@H1][Branch2][=Branch2][=N][O][C@@H1][O][C@H1][Branch2][Ring2][=C][C][O][C@H1][O][C@H1][Branch1][Ring1][C][O][C@@H1][Branch1][C][O][C@H1][Branch1][C][O][C@@H1][Ring1][#Branch2][O][C@@H1][O][C@H1][Branch1][Ring1][C][O][C@@H1][Branch1][C][O][C@H1][Branch1][C][O][C@H1][Ring1][#Branch2][N][C][Branch1][C][C][=O][C@@H1][Branch1][C][O][C@H1][Branch2][Ring2][=N][O][C@H1][O][C@H1][Branch1][Ring1][C][O][C@@H1][Branch1][C][O][C@H1][Branch1][C][O][C@@H1][Ring1][#Branch2][O][C@@H1][O][C@H1][Branch1][Ring1][C][O][C@@H1][Branch1][C][O][C@H1][Branch1][C][O][C@H1][Ring1][#Branch2][N][C][Branch1][C][C][=O][C@@H1][Ring2][Branch1][N][O][C@@H1][O][C][C@@H1][Branch1][C][O][C@H1][Branch1][C][O][C@H1][Ring1][Branch2][O][C@H1][Branch1][C][O][C@H1][Ring2][=Branch1][S][N][C][Branch1][C][C][=O][C@@H1][Branch2][Ring1][Branch1][C][O][C@@H1][O][C@@H1][Branch1][C][C][C@@H1][Branch1][C][O][C@@H1][Branch1][C][O][C@@H1][Ring1][=Branch2][O][O][C@H1][Ring2][=Branch2][C][O]', '[C][C@][C][C][C@@H1][C][=C][C][C@H1][Branch1][C][O][C][C@H1][Branch1][Ring2][C][Ring1][Branch2][C][=Branch1][C][=O][C][=C][Ring1][=N][C@@H1][Ring1][P][C][C][C@@H1][Ring2][Ring1][Ring2][C@@][Branch1][C][C][Branch1][C][O][C][C][O]']
        # tokens=self.molecular_tokenizer(selfies_list, add_special_tokens=True, max_length=512, padding='max_length', truncation=True)
        # input_ids = torch.tensor(tokens['input_ids']).to(device)  # Get input_ids
        # attention_mask = torch.tensor(tokens['attention_mask']).to(device)  # Get attention_mask
        # with torch.no_grad():
        #     output = self.molecular_encoder(input_ids=input_ids, attention_mask=attention_mask)
        # sequence_out = output.last_hidden_state
        #     # print(sequence_out.shape)#torch.Size([bz, 512(seq_len), 768])
        #     # Use attention_mask to compute mean of non-padding tokens
        # mask = attention_mask.unsqueeze(-1)  # (batch_size, seq_len, 1)
        # # print(mask.shape)
        # weighted_sum = torch.sum(sequence_out * mask, dim=1)  # Weighted sum by mask, shape (batch_size, hidden_dim)
        # count_non_pads = mask.sum(dim=1)  # Count non-padding tokens per sequence, shape (batch_size, 1)

        # # Prevent division by zero
        # count_non_pads[count_non_pads == 0] = 1

        # # Compute mean
        # mean_embeddings = weighted_sum / count_non_pads  # Shape (batch_size, hidden_dim)

        # embeddings=torch.tensor(mean_embeddings.tolist()).to(torch.float32).to(device)  # Convert to list and return

        molecular_embeds = self.ln_molecular(embeddings)  # Layer Norm
        molecular_embeds = molecular_embeds.view(
            embeddings.size(0), 1, 1024
        )  # BZ,SeqLen,Dim

        molecular_atts = torch.ones(molecular_embeds.size()[:-1], dtype=torch.long).to(
            device
        )

        # query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        # query_output = self.Qformer.bert(
        #     query_embeds=query_tokens,
        #     encoder_hidden_states=image_embeds,
        #     encoder_attention_mask=image_atts,
        #     use_cache=True,
        #     return_dict=True,
        # )

        # image_feats = F.normalize(
        #     self.vision_proj(query_output.last_hidden_state), dim=-1
        # )

        query_tokens = self.query_tokens.expand(molecular_embeds.shape[0], -1, -1)  ###

        query_output = self.Qformer.bert(  ###
            query_embeds=query_tokens,
            encoder_hidden_states=molecular_embeds,
            encoder_attention_mask=molecular_atts,
            use_cache=True,
            return_dict=True,
        )

        molecular_feats = F.normalize(  ###
            self.molecular_proj(query_output.last_hidden_state), dim=-1
        )

        text_tokens = self.tokenizer(  # Returned text_tokens are padded vectors.
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,  # Max token length: truncate long, pad short
            return_tensors="pt",
        ).to(device)  ###

        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )
        text_feat = F.normalize(
            self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
        )

        ###============== Molecular-text Contrastive ===================###

        # image_feats_all = concat_all_gather(
        #     image_feats
        # )  # [batch_size*num_gpu, num_query_tokens, embed_dim]
        text_feat_all = concat_all_gather(text_feat)  # [batch_size*num_gpu, embed_dim]
        molecular_feats_all = concat_all_gather(molecular_feats)  ###

        # sim_q2t = torch.matmul(
        #     image_feats.unsqueeze(1), text_feat_all.unsqueeze(-1)
        # ).squeeze()
        # # [batch_size, batch_size*num_gpu, num_query_tokens]

        sim_q2t = torch.matmul(
            molecular_feats.unsqueeze(1), text_feat_all.unsqueeze(-1)
        ).squeeze()  ###

        # image-text similarity: aggregate across all query tokens
        # sim_i2t, _ = sim_q2t.max(-1)
        # sim_i2t = sim_i2t / self.temp
        sim_m2t, _ = sim_q2t.max(-1)  ###
        sim_m2t = sim_m2t / self.temp  ###

        # text-query similarity: [batch_size, batch_size*num_gpu, num_query_tokens]
        # sim_t2q = torch.matmul(
        #     text_feat.unsqueeze(1).unsqueeze(1), image_feats_all.permute(0, 2, 1)
        # ).squeeze()
        sim_t2q = torch.matmul(
            text_feat.unsqueeze(1).unsqueeze(1), molecular_feats_all.permute(0, 2, 1)
        ).squeeze()  ###

        # text-image similarity: aggregate across all query tokens
        # sim_t2i, _ = sim_t2q.max(-1)
        # sim_t2i = sim_t2i / self.temp  # [batch_size, batch_size*num_gpu]
        sim_t2m, _ = sim_t2q.max(-1)
        sim_t2m = sim_t2m / self.temp  # [batch_size, batch_size*num_gpu]

        rank = dist.get_rank()
        # bs = image.size(0)
        bs = len(smiles)  ###

        targets = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=int).to(
            device
        )  ###

        # if "image_id" in samples.keys(): #coco retrieval finetuning
        #     image_ids = samples["image_id"].view(-1,1)
        #     image_ids_all = concat_all_gather(image_ids)
        #     pos_idx = torch.eq(image_ids, image_ids_all.t()).float()
        #     sim_targets = pos_idx / pos_idx.sum(1,keepdim=True)
        #     sim_targets = 0.9 * sim_targets + 0.1 * torch.ones_like(sim_targets) / sim_targets.size(1)

        #     loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1)*sim_targets,dim=1).mean()
        #     loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1)*sim_targets,dim=1).mean()
        #     loss_itc = (loss_t2i+loss_i2t)/2
        # else:
        #     loss_itc = (
        #         F.cross_entropy(sim_i2t, targets, label_smoothing=0.1)
        #         + F.cross_entropy(sim_t2i, targets, label_smoothing=0.1)
        #     ) / 2

        loss_mtc = (
            F.cross_entropy(sim_m2t, targets, label_smoothing=0.1)
            + F.cross_entropy(sim_t2m, targets, label_smoothing=0.1)
        ) / 2  ###

        ###============== Molecular-text Matching ===================###
        text_input_ids_world = concat_all_gather(text_tokens.input_ids)
        text_attention_mask_world = concat_all_gather(text_tokens.attention_mask)
        # image_embeds_world = all_gather_with_grad(image_embeds)
        molecular_embeds_world = all_gather_with_grad(molecular_embeds)  ###

        # with torch.no_grad():
        #     if "image_id" in samples.keys():#coco retrieval finetuning
        #         mask = torch.eq(image_ids, image_ids_all.t())
        #         sim_t2i.masked_fill_(mask, -10000)
        #         sim_i2t.masked_fill_(mask, -10000)
        #     else:
        #         sim_t2i[:, rank * bs : rank * bs + bs].fill_diagonal_(-10000)#Fill diagonal with -10000. Setting diagonal similarity to a large negative value suppresses self-similarity during softmax, encouraging focus on other samples
        #         sim_i2t[:, rank * bs : rank * bs + bs].fill_diagonal_(-10000)

        #     weights_t2i = F.softmax(sim_t2i, dim=1)
        #     weights_i2t = F.softmax(sim_i2t, dim=1)

        with torch.no_grad():  ###
            sim_t2m[:, rank * bs : rank * bs + bs].fill_diagonal_(-10000)
            sim_m2t[:, rank * bs : rank * bs + bs].fill_diagonal_(-10000)

            weights_t2m = F.softmax(sim_t2m, dim=1)
            weights_m2t = F.softmax(sim_m2t, dim=1)

        # # select a negative image for each text
        # image_embeds_neg = []
        # for b in range(bs):
        #     neg_idx = torch.multinomial(weights_t2i[b], 1).item()
        #     image_embeds_neg.append(image_embeds_world[neg_idx])
        # image_embeds_neg = torch.stack(image_embeds_neg, dim=0)

        # select a negative molecular for each text
        molecular_embeds_neg = []  ###
        for b in range(bs):  ###
            neg_idx = torch.multinomial(weights_t2m[b], 1).item()
            molecular_embeds_neg.append(molecular_embeds_world[neg_idx])
        molecular_embeds_neg = torch.stack(molecular_embeds_neg, dim=0)  ###

        # # select a negative text for each image
        # text_ids_neg = []
        # text_atts_neg = []
        # for b in range(bs):
        #     neg_idx = torch.multinomial(weights_i2t[b], 1).item()
        #     text_ids_neg.append(text_input_ids_world[neg_idx])
        #     text_atts_neg.append(text_attention_mask_world[neg_idx])

        # text_ids_neg = torch.stack(text_ids_neg, dim=0)
        # text_atts_neg = torch.stack(text_atts_neg, dim=0)

        # select a negative text for each molecular
        text_ids_neg = []  ###
        text_atts_neg = []  ###
        for b in range(bs):  ###
            neg_idx = torch.multinomial(weights_m2t[b], 1).item()
            text_ids_neg.append(text_input_ids_world[neg_idx])
            text_atts_neg.append(text_attention_mask_world[neg_idx])

        text_ids_neg = torch.stack(text_ids_neg, dim=0)  ###
        text_atts_neg = torch.stack(text_atts_neg, dim=0)  ###

        text_ids_all = torch.cat(
            [text_tokens.input_ids, text_tokens.input_ids, text_ids_neg], dim=0
        )  # pos, pos, neg
        text_atts_all = torch.cat(
            [text_tokens.attention_mask, text_tokens.attention_mask, text_atts_neg],
            dim=0,
        )

        # query_tokens_itm = self.query_tokens.expand(text_ids_all.shape[0], -1, -1)
        # query_atts_itm = torch.ones(query_tokens_itm.size()[:-1], dtype=torch.long).to(
        #     image.device
        # )
        # attention_mask_all = torch.cat([query_atts_itm, text_atts_all], dim=1)

        query_tokens_mtm = self.query_tokens.expand(text_ids_all.shape[0], -1, -1)  ###
        query_atts_mtm = torch.ones(query_tokens_mtm.size()[:-1], dtype=torch.long).to(
            device  ###
        )
        attention_mask_all = torch.cat([query_atts_mtm, text_atts_all], dim=1)  ###

        # image_embeds_all = torch.cat(
        #     [image_embeds, image_embeds_neg, image_embeds], dim=0
        # )  # pos, neg, pos
        # image_atts_all = torch.ones(image_embeds_all.size()[:-1], dtype=torch.long).to(
        #     image.device
        # )

        molecular_embeds_all = torch.cat(  ###
            [molecular_embeds, molecular_embeds_neg, molecular_embeds], dim=0
        )  # pos, neg, pos
        molecular_atts_all = torch.ones(
            molecular_embeds_all.size()[:-1], dtype=torch.long
        ).to(
            device  ###
        )

        # output_itm = self.Qformer.bert(
        #     text_ids_all,
        #     query_embeds=query_tokens_itm,
        #     attention_mask=attention_mask_all,
        #     encoder_hidden_states=image_embeds_all,
        #     encoder_attention_mask=image_atts_all,
        #     return_dict=True,
        # )

        output_mtm = self.Qformer.bert(  ###
            text_ids_all,
            query_embeds=query_tokens_mtm,
            attention_mask=attention_mask_all,
            encoder_hidden_states=molecular_embeds_all,
            encoder_attention_mask=molecular_atts_all,
            return_dict=True,
        )

        # vl_embeddings = output_itm.last_hidden_state[:, : query_tokens_itm.size(1), :]
        # vl_output = self.itm_head(vl_embeddings)
        # logits = vl_output.mean(dim=1)
        vl_embeddings = output_mtm.last_hidden_state[
            :, : query_tokens_mtm.size(1), :
        ]  ###
        vl_output = self.itm_head(vl_embeddings)
        logits = vl_output.mean(dim=1)

        # itm_labels = torch.cat(
        #     [torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
        #     dim=0,
        # ).to(image.device)
        # loss_itm = F.cross_entropy(logits, itm_labels)

        mtm_labels = torch.cat(  ###
            [torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
            dim=0,
        ).to(device)
        loss_mtm = F.cross_entropy(logits, mtm_labels)

        ##================= Molecular-text Captioning ========================##
        decoder_input_ids = text_tokens.input_ids.clone()
        decoder_input_ids[:, 0] = self.tokenizer.bos_token_id
        labels = decoder_input_ids.masked_fill(
            decoder_input_ids == self.tokenizer.pad_token_id, -100
        )

        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            device  ###
        )
        attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)

        lm_output = self.Qformer(
            decoder_input_ids,
            attention_mask=attention_mask,
            past_key_values=query_output.past_key_values,
            return_dict=True,
            labels=labels,
        )

        loss_lm = lm_output.loss  # This is the ITG loss

        return BlipOutput(  ###
            loss=loss_mtc + loss_mtm + loss_lm,
            loss_mtc=loss_mtc,
            loss_mtm=loss_mtm,
            loss_lm=loss_lm,
        )

    def current_device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def generate(  # Image-to-text generation
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=3,
        max_length=30,
        min_length=10,
        top_p=0.9,
        repetition_penalty=1.0,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        image = samples["image"]
        image_embeds = self.ln_vision(self.visual_encoder(image))

        if not use_nucleus_sampling:
            image_embeds = image_embeds.repeat_interleave(num_beams, dim=0)
        else:
            num_beams = 1
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        model_kwargs = {
            "encoder_hidden_states": image_embeds,
            "encoder_attention_mask": image_atts,
        }

        input_ids = (
            torch.LongTensor(image.size(0), 1)
            .fill_(self.tokenizer.bos_token_id)
            .to(image.device)
        )
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        outputs = self.Qformer.generate(
            input_ids=input_ids,
            query_embeds=query_tokens,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            do_sample=use_nucleus_sampling,
            top_p=top_p,
            eos_token_id=self.tokenizer.sep_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            **model_kwargs,
        )
        captions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return captions

    def forward_image(self, image):
        image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        return query_output.last_hidden_state, image_embeds

    def forward_molecular(self, smiles):
        device = self.current_device()

        # selfies_list=[]
        # for smile in smiles:
        #     try:
        #         selfies=sf.encoder(smile)
        #     except:
        #         selfies=smile
        #     selfies_list.append(selfies)
        # tokens=self.molecular_tokenizer(selfies_list, add_special_tokens=True, max_length=512, padding='max_length', truncation=True)
        # input_ids = torch.tensor(tokens['input_ids']).to(device)  # Get input_ids
        # attention_mask = torch.tensor(tokens['attention_mask']).to(device)  # Get attention_mask

        # self.molecular_encoder.eval()
        # with torch.no_grad():
        #     output = self.molecular_encoder(input_ids=input_ids, attention_mask=attention_mask)
        # sequence_out = output.last_hidden_state
        # mask = attention_mask.unsqueeze(-1)  # (batch_size, seq_len, 1)
        # weighted_sum = torch.sum(sequence_out * mask, dim=1)  # Weighted sum by mask, shape (batch_size, hidden_dim)
        # count_non_pads = mask.sum(dim=1)  # Count non-padding tokens per sequence, shape (batch_size, 1)
        # # Prevent division by zero
        # count_non_pads[count_non_pads == 0] = 1
        # # Compute mean
        # mean_embeddings = weighted_sum / count_non_pads  # Shape (batch_size, hidden_dim)
        # embeddings=torch.tensor(mean_embeddings.tolist()).to(torch.float32).to(device)  # Convert to list and return

        with torch.no_grad():
            embeddings, flags = self.molecular_encoder.transform(smiles)
        embeddings = F.normalize((embeddings), dim=-1).to(device)
        #     # embeddings_unimol_repr=self.molecular_encoder.get_repr(smiles)
        #     # embeddings=np.array(embeddings_unimol_repr['cls_repr'])

        molecular_embeds = self.ln_molecular(embeddings)  # Layer Norm
        molecular_embeds = molecular_embeds.view(
            embeddings.size(0), 1, 1024
        )  # BZ,SeqLen,Dim
        molecular_atts = torch.ones(molecular_embeds.size()[:-1], dtype=torch.long).to(
            device
        )

        query_tokens = self.query_tokens.expand(molecular_embeds.shape[0], -1, -1)  ###

        query_output = self.Qformer.bert(  ###
            query_embeds=query_tokens,
            encoder_hidden_states=molecular_embeds,
            encoder_attention_mask=molecular_atts,
            return_dict=True,
        )
        return (
            query_output.last_hidden_state,
            embeddings,
        )  # bz,num_queries?,768,  bz,true_len,1024

    def forward_text(self, text_tokens):
        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )
        return text_output.last_hidden_state[:, 0, :]

    def compute_itm(self, image_inputs, text_ids, text_atts):
        image_atts = torch.ones(image_inputs.size()[:-1], dtype=torch.long).to(
            image_inputs.device
        )
        query_tokens = self.query_tokens.expand(image_inputs.shape[0], -1, -1)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            image_inputs.device
        )
        attention_mask = torch.cat([query_atts, text_atts], dim=1)
        output_itm = self.Qformer.bert(
            text_ids,
            query_embeds=query_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=image_inputs,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        vl_embeddings = output_itm.last_hidden_state[:, : query_tokens.size(1), :]
        itm_logit = self.itm_head(vl_embeddings)
        itm_logit = itm_logit[:, :, 1].mean(dim=1)
        return itm_logit

    @torch.no_grad()
    def extract_features(self, samples, mode="multimodal"):
        """
        Extract features for multimodal or unimodal samples.
        Args:
            samples (dict): A dictionary of samples, containing the following keys:
                - image (torch.Tensor): A tensor of shape (B, C, H, W) containing the image.
                    Raw images should be preprocessed before being passed to feature extractor.
                - text_input (list): A list of strings containing the text, length B.
            mode (str): The mode of feature extraction. Can be either "multimodal", "text" or "image".
                If "multimodal", return image features and multimodal features;
                if "text", return text features;
                if "image", return image features.
                Default: "multimodal".
        Returns:
            BlipOutputFeatures: A BlipOutputFeatures object containing the features.
                See lavis/models/blip_models/blip_outputs.py for more details.
        """
        image = samples.get("image")
        caption = samples.get("text_input")

        # assert mode is one of "image", "text", "multimodal"
        assert mode in [
            "image",
            "text",
            "multimodal",
        ], "mode must be one of 'image', 'text', 'multimodal'"

        # initalize output
        image_embeds, text_embeds, multimodal_embeds = None, None, None
        image_features, text_features = None, None

        if mode == "image":
            assert image is not None, (
                "Image is not provided for mode 'image' or 'multimodal'"
            )
            # return query features
            with self.maybe_autocast():
                image_embeds_frozen = self.ln_vision(self.visual_encoder(image))
            image_embeds_frozen = image_embeds_frozen.float()
            image_atts = torch.ones(
                image_embeds_frozen.size()[:-1], dtype=torch.long
            ).to(self.device)
            query_tokens = self.query_tokens.expand(
                image_embeds_frozen.shape[0], -1, -1
            )

            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds_frozen,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            image_embeds = query_output.last_hidden_state
            image_features = F.normalize(self.vision_proj(image_embeds), dim=-1)

        elif mode == "text":
            assert caption is not None, (
                "text input is None for mode 'text' or 'multimodal'"
            )

            # return text features
            text = self.tokenizer(caption, return_tensors="pt", padding=True).to(
                self.device
            )

            text_output = self.Qformer.bert(
                text.input_ids,
                attention_mask=text.attention_mask,
                return_dict=True,
            )
            text_embeds = text_output.last_hidden_state
            text_features = self.text_proj(text_embeds)
            text_features = F.normalize(text_features, dim=-1)

        elif mode == "multimodal":
            # return multimodel query features
            with self.maybe_autocast():
                image_embeds_frozen = self.ln_vision(self.visual_encoder(image))
            image_embeds_frozen = image_embeds_frozen.float()
            image_atts = torch.ones(
                image_embeds_frozen.size()[:-1], dtype=torch.long
            ).to(self.device)
            query_tokens = self.query_tokens.expand(
                image_embeds_frozen.shape[0], -1, -1
            )
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
                self.device
            )

            text = self.tokenizer(caption, return_tensors="pt", padding=True).to(
                self.device
            )
            attention_mask = torch.cat([query_atts, text.attention_mask], dim=1)

            output = self.Qformer.bert(
                text.input_ids,
                query_embeds=query_tokens,
                attention_mask=attention_mask,
                encoder_hidden_states=image_embeds_frozen,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            multimodal_embeds = output.last_hidden_state[:, : query_tokens.size(1), :]

        return BlipOutputFeatures(
            image_embeds=image_embeds,
            image_embeds_proj=image_features,
            text_embeds=text_embeds,
            text_embeds_proj=text_features,
            multimodal_embeds=multimodal_embeds,
        )

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        cross_attention_freq = cfg.get("cross_attention_freq", 2)

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        max_txt_len = cfg.get("max_txt_len", 32)

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            cross_attention_freq=cross_attention_freq,
            max_txt_len=max_txt_len,
        )
        model.load_checkpoint_from_config(cfg)

        return model

    def compute_sim_matrix(self, data_loader, task_cfg):
        """
        Compute similarity i2t, t2i matrix for the given data loader.
        """
        k_test = task_cfg.k_test

        return compute_sim_matrix(model=self, data_loader=data_loader, k_test=k_test)


import torch
import torch.nn as nn


class DenseInteraction(nn.Module):
    """
    FC interaction layer replacing attention
    Two linear layers with GeLU activation
    """

    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.dense1 = nn.Linear(hidden_size, intermediate_size)
        self.dense2 = nn.Linear(intermediate_size, hidden_size)
        self.activation = nn.GELU()

    def forward(self, hidden_states):
        # Keep dimension unchanged: [batch_size, seq_len, hidden_size]
        intermediate = self.dense1(hidden_states)
        intermediate = self.activation(intermediate)
        output = self.dense2(intermediate)
        return output


class BertLayer(nn.Module):
    """Modified BERT layer with FC interaction replacing attention"""

    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.dense_interaction = DenseInteraction(hidden_size, intermediate_size)
        self.linear_output = nn.Linear(hidden_size, hidden_size)
        self.layernorm1 = nn.LayerNorm(hidden_size)
        self.layernorm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states):
        # First sublayer: FC interaction + residual
        interaction_output = self.dense_interaction(hidden_states)
        interaction_output = self.dropout(interaction_output)
        hidden_states = self.layernorm1(hidden_states + interaction_output)

        # Second sublayer: FFN (original BERT structure)
        linear_output = self.linear_output(hidden_states)
        linear_output = self.dropout(linear_output)
        output = self.layernorm2(hidden_states + linear_output)
        return output


class BertEncoder(nn.Module):
    """BERT-like encoder based on FC interaction"""

    def __init__(self, num_layers, hidden_size, intermediate_size):
        super().__init__()
        self.layers = nn.ModuleList(
            [BertLayer(hidden_size, intermediate_size) for _ in range(num_layers)]
        )

    def forward(self, hidden_states):
        for layer_module in self.layers:
            hidden_states = layer_module(hidden_states)
        return hidden_states


class MLP(nn.Module):
    def __init__(self, input_dim=256, output_dim=768, hidden_dim=1024):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)  # Input dim 256 -> hidden dim 512
        x = self.relu(x)  # Apply ReLU activation
        x = self.fc2(x)  # Hidden dim 512 -> output dim 768
        return x


class Blip2Qformer_MLP(Blip2Base):
    """
    BLIP2 first-stage model with Q-former and ViT.
    Supported model types:
        - pretrained: pretrained model with vit-g
        - pretrain_vitL: pretrained model with vit-large
        - coco: fintuned model on coco
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2", "pretrain")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain": "configs/models/blip2/blip2_pretrain.yaml",
        "pretrain_vitL": "configs/models/blip2/blip2_pretrain_vitL.yaml",
        "coco": "configs/models/blip2/blip2_coco.yaml",
    }

    def __init__(
        self,
        device,
        # vit_model="eva_clip_g",
        molecular_model="MolR",  ###
        # img_size=224,
        # drop_path_rate=0,
        # use_grad_checkpoint=False,
        # vit_precision="fp16",
        molecular_precision=32,  ### Note: fp16 may be too low precision
        # freeze_vit=True,
        freeze_molecular_model=False,  ###
        num_query_token=32,
        cross_attention_freq=2,
        embed_dim=256,
        max_txt_len=32,
    ):
        super().__init__()
        self.testmodel = BertEncoder(
            num_layers=6, hidden_size=1024, intermediate_size=3072
        )
        self.tokenizer = self.init_tokenizer()

        # self.visual_encoder, self.ln_vision = self.init_vision_encoder(
        #     vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        # )
        # self.molecular_encoder,self.molecular_tokenizer, self.ln_molecular = self.init_molecular_encoder(molecular_model,molecular_precision,device)###
        self.molecular_encoder, self.ln_molecular = self.init_molecular_encoder(
            molecular_model, molecular_precision, device
        )  ###
        # if freeze_vit:
        #     for name, param in self.visual_encoder.named_parameters():
        #         param.requires_grad = False
        #     self.visual_encoder = self.visual_encoder.eval()
        #     self.visual_encoder.train = disabled_train
        #     logging.info("freeze vision encoder")
        if freeze_molecular_model:  ###
            for name, param in self.molecular_encoder.mole.named_parameters():
                param.requires_grad = False
            self.molecular_encoder.mole = self.molecular_encoder.mole.eval()
            self.molecular_encoder.mole.train = disabled_train
            logging.info("freeze molecular encoder")
        else:
            for name, param in self.molecular_encoder.mole.named_parameters():
                param.requires_grad = True
            self.molecular_encoder.mole = self.molecular_encoder.mole.train()
            self.molecular_encoder.mole.train(True)
        # self.Qformer, self.query_tokens = self.init_Qformer(
        #     num_query_token, self.visual_encoder.num_features, cross_attention_freq
        # )

        self.Qformer, self.query_tokens = self.init_Qformer(  ###
            num_query_token, self.molecular_encoder.num_features, cross_attention_freq
        )
        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])

        # self.vision_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.molecular_proj = nn.Linear(1024, embed_dim)  ###
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)

        self.itm_head = nn.Linear(self.Qformer.config.hidden_size, 2)

        self.temp = nn.Parameter(0.07 * torch.ones([]))

        self.mlp = MLP(input_dim=256, output_dim=768, hidden_dim=1024)
        self.max_txt_len = max_txt_len
        print("max_txt_len:", self.max_txt_len)

    def forward(self, samples):
        device = self.current_device()

        # image = samples["image"]
        smiles = samples[
            "smiles"
        ]  ### MOLR input: list of SMILES strings ['smiles','smiles2',...]
        text = samples["text_input"]

        # image_embeds = self.ln_vision(self.visual_encoder(image))
        # image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
        #     image.device
        # )

        # with torch.no_grad():
        embeddings, flags = self.molecular_encoder.transform(smiles)
        # embeddings=F.normalize(torch.from_numpy(embeddings),dim=-1).to(device)
        embeddings = F.normalize((embeddings), dim=-1).to(device)
        # embeddings_unimol_repr=self.molecular_encoder.get_repr(smiles)
        # embeddings=np.array(embeddings_unimol_repr['cls_repr'])

        # selfies_list=[]
        # for smile in smiles:
        #     try:
        #         selfies=sf.encoder(smile)
        #     except:
        #         selfies=smile
        #     selfies_list.append(selfies)

        # self.molecular_encoder.eval()
        # # selfies_list=['[O][=C][Branch2][Ring2][C][C][S][C][=N][C][=C][C][=C][C][=C][Ring1][=Branch1][C][=Branch1][C][=O][N][Ring1][O][C][=C][C][=C][Branch1][C][F][C][Branch1][C][Cl][=C][Ring1][Branch2][N][C][=C][C][=C][C][=Branch1][Ring2][=C][Ring1][=Branch1][C][C][C][Ring1][=Branch1]', '[C][O][C][=C][C][=C][Branch2][Branch1][#Branch2][N][C][=Branch1][C][=O][N][C][=C][C][=C][C][=Branch1][Ring2][=C][Ring1][=Branch1][C][C][=Branch1][C][=O][N][Branch1][#Branch1][C][Branch1][C][C][C][O][C][C][Branch1][C][C][C][Branch1][S][C][N][Branch1][C][C][C][=Branch1][C][=O][N][C][Branch1][C][C][C][O][Ring2][Ring1][=Branch2][C][=C][Ring2][Ring2][=Branch1]', '[C][O][C@@][Branch1][C][C][Branch1][Branch2][/C][=C][/C@@H1][Branch1][C][C][O][C][C][C][C@H1][C][=C][C][=Branch1][C][=O][C@H1][C][C][=Branch1][#Branch2][=C][C][C@H1][Branch1][C][O][C][Ring1][Branch2][C@H1][Ring1][=N][C][C][C@][Ring2][Ring1][Ring2][Ring1][P][C]', '[C][C][Branch2][Ring2][Ring1][N][C][=Branch1][C][=O][C][Branch1][#Branch2][C][C][=C][C][=C][C][=C][Ring1][=Branch1][N][C][=Branch1][C][=O][C][Branch1][C][N][C][C][=C][N][=C][NH1][Ring1][Branch1][C][=Branch1][C][=O][O]', '[C][C][=Branch1][C][=O][N][C][C][Branch2][Ring2][#Branch2][O][C][C][Branch1][C][O][C][Branch1][Ring1][C][O][O][C][Branch2][Ring1][Branch1][O][C][C][Branch1][Ring1][C][O][O][C][Branch1][C][O][C][Branch1][C][O][C][Ring1][#Branch2][O][C][Ring2][Ring1][Branch1][O][O][C][Branch1][Ring1][C][O][C][Branch2][Branch2][=N][O][C][O][C][Branch1][Ring1][C][O][C][Branch1][C][O][C][Branch2][#Branch1][Branch2][O][C][O][C][Branch1][Ring1][C][O][C][Branch2][Branch1][=C][O][C][O][C][Branch1][Ring1][C][O][C][Branch1][C][O][C][Branch2][Ring2][=Branch2][O][C][Branch1][=Branch1][C][=Branch1][C][=O][O][C][C][Branch1][C][O][C][Branch1][#Branch1][N][C][Branch1][C][C][=O][C][Branch1][O][C][Branch1][C][O][C][Branch1][C][O][C][O][O][Ring2][Ring1][Ring2][C][Ring2][Ring1][=C][O][C][Branch1][C][O][C][Ring2][Ring2][=Branch2][N][C][Branch1][C][C][=O][C][Ring2][Branch1][#Branch1][O][C][Ring2][#Branch1][Branch2][O]', '[C][O][C][=C][C][=C][C][=C][Ring1][=Branch1][C][=Branch1][C][=O][O][C][C][#C][C][S][C][=N][N][=C][Branch1][#C][C][=C][C][=C][C][=C][C][=C][C][=C][Ring1][#Branch2][Ring1][=Branch1][O][Ring1][#C]', '[C][C][C][=Branch1][C][=O][O][C][C][=Branch1][C][=O][C@@][Branch1][Branch2][O][C][=Branch1][C][=O][C][C][C@H1][Branch1][C][C][C][C@H1][C@H1][C@H1][Branch1][N][C@@H1][Branch1][C][O][C][C@@][Ring1][#Branch1][Ring1][S][C][C@@][Branch1][C][C][C][=C][C][=Branch1][C][=O][C][=C][Ring1][Branch2][C][C@H1][Ring1][P][Cl]', '[O][=C][Branch1][Ring1][C][O][C@@H1][Branch1][C][O][C@H1][Branch1][C][O][C@H1][Branch1][C][O][C@@H1][Branch1][C][O][C@@H1][Branch1][C][O][C][=Branch1][C][=O][O-1]', '[C][C][C][C][C][C][C][C][C][C][C][C][C][Branch1][C][O][C][C][C][C][Branch2][Ring2][Branch1][C][C][C][C][Branch2][Ring1][Branch2][C][C][C][C][C][C][C][C][Branch1][C][O][C][C][=C][C][Branch1][C][C][O][C][Ring1][=Branch1][=O][O][C][O][Ring2][Ring1][Branch2][O][Ring2][Ring1][=N]', '[C][O][C][C][N][Branch1][C][C][C][=Branch1][C][=O][C][=C][C][=C][Branch1][#Branch1][N][C][Branch1][C][C][=O][C][=C][Ring1][#Branch2][O][C][C][Branch1][C][C][N][Branch1][=C][C][=Branch1][C][=O][C][=C][C][=C][C][=C][Ring1][=Branch1][F][C][C][Ring2][Ring1][S][C]', '[C][O][C][=C][C][Branch2][Branch1][#Branch1][N][C][=N][C][=C][Branch1][C][F][C][Branch2][Ring2][#Branch1][N][C][=C][C][=C][C][=Branch1][Ring2][=N][Ring1][=Branch1][N][Branch1][O][C][O][P][=Branch1][C][=O][Branch1][C][O-1][O-1][C][=Branch1][C][=O][C][Branch1][C][C][Branch1][C][C][O][Ring1][S][=N][Ring2][Ring1][O][=C][C][Branch1][Ring1][O][C][=C][Ring2][Ring2][Ring2][O][C].[Na+1].[Na+1]', '[O][=C][Branch1][O][C][=C][C][=C][C][=C][C][=C][Ring1][=Branch1][O][C][O][C][Branch1][=Branch1][C][=Branch1][C][=O][O][C][Branch1][C][O][C][Branch1][C][O][C][Ring1][O][O]', '[N][S][=Branch1][C][=O][=Branch1][C][=O][C][=C][C][=C][N][=C][Branch1][S][N][C][=Branch1][C][=O][C][=C][C][=C][Branch1][C][Br][O][Ring1][=Branch1][S][C][Ring1][=C][=C][Ring2][Ring1][C]', '[C][C][=C][Branch1][C][N][N][=C][Branch2][Ring1][Branch1][C@H1][Branch1][#Branch1][C][C][Branch1][C][N][=O][N][C][C@H1][Branch1][C][N][C][Branch1][C][N][=O][N][=C][Ring2][Ring1][Ring1][C][=Branch1][C][=O][N][C@H1][Branch2][=Branch1][Branch2][C][=Branch1][C][=O][N][C@H1][Branch1][C][C][C@@H1][Branch1][C][O][C@H1][Branch1][C][C][C][=Branch1][C][=O][N][C@H1][Branch2][Ring2][=Branch2][C][=Branch1][C][=O][N][C][C][C][=N][C][Branch2][Ring1][#Branch1][C][=N][C][Branch1][=C][C][=Branch1][C][=O][N][C][C][C][S+1][Branch1][C][C][C][=C][S][Ring1][=C][=C][S][Ring2][Ring1][Ring1][C@@H1][Branch1][C][C][O][C@@H1][Branch2][Ring2][=N][O][C][O][C][Branch1][Ring1][C][O][C][Branch1][C][O][C][Branch1][C][O][C][Ring1][#Branch2][O][C][O][C][Branch1][Ring1][C][O][C][Branch1][C][O][C][Branch1][#Branch1][O][C][Branch1][C][N][=O][C][Ring1][=N][O][C][=C][N][=C][NH1][Ring1][Branch1]', '[C][C][=Branch1][C][=O][N][C@@H1][C@@H1][Branch2][Ring1][Ring2][O][C@@H1][O][C@@H1][Branch1][C][C][C@@H1][Branch1][C][O][C@@H1][Branch1][C][O][C@@H1][Ring1][=Branch2][O][C@H1][Branch2][O][#Branch1][O][C@@H1][O][C@H1][Branch1][Ring1][C][O][C@@H1][Branch2][=Branch2][=N][O][C@@H1][O][C@H1][Branch2][Ring2][=C][C][O][C@H1][O][C@H1][Branch1][Ring1][C][O][C@@H1][Branch1][C][O][C@H1][Branch1][C][O][C@@H1][Ring1][#Branch2][O][C@@H1][O][C@H1][Branch1][Ring1][C][O][C@@H1][Branch1][C][O][C@H1][Branch1][C][O][C@H1][Ring1][#Branch2][N][C][Branch1][C][C][=O][C@@H1][Branch1][C][O][C@H1][Branch2][Ring2][=N][O][C@H1][O][C@H1][Branch1][Ring1][C][O][C@@H1][Branch1][C][O][C@H1][Branch1][C][O][C@@H1][Ring1][#Branch2][O][C@@H1][O][C@H1][Branch1][Ring1][C][O][C@@H1][Branch1][C][O][C@H1][Branch1][C][O][C@H1][Ring1][#Branch2][N][C][Branch1][C][C][=O][C@@H1][Ring2][Branch1][N][O][C@@H1][O][C][C@@H1][Branch1][C][O][C@H1][Branch1][C][O][C@H1][Ring1][Branch2][O][C@H1][Branch1][C][O][C@H1][Ring2][=Branch1][S][N][C][Branch1][C][C][=O][C@@H1][Branch2][Ring1][Branch1][C][O][C@@H1][O][C@@H1][Branch1][C][C][C@@H1][Branch1][C][O][C@@H1][Branch1][C][O][C@@H1][Ring1][=Branch2][O][O][C@H1][Ring2][=Branch2][C][O]', '[C][C@][C][C][C@@H1][C][=C][C][C@H1][Branch1][C][O][C][C@H1][Branch1][Ring2][C][Ring1][Branch2][C][=Branch1][C][=O][C][=C][Ring1][=N][C@@H1][Ring1][P][C][C][C@@H1][Ring2][Ring1][Ring2][C@@][Branch1][C][C][Branch1][C][O][C][C][O]']
        # tokens=self.molecular_tokenizer(selfies_list, add_special_tokens=True, max_length=512, padding='max_length', truncation=True)
        # input_ids = torch.tensor(tokens['input_ids']).to(device)  # Get input_ids
        # attention_mask = torch.tensor(tokens['attention_mask']).to(device)  # Get attention_mask
        # with torch.no_grad():
        #     output = self.molecular_encoder(input_ids=input_ids, attention_mask=attention_mask)
        # sequence_out = output.last_hidden_state
        #     # print(sequence_out.shape)#torch.Size([bz, 512(seq_len), 768])
        #     # Use attention_mask to compute mean of non-padding tokens
        # mask = attention_mask.unsqueeze(-1)  # (batch_size, seq_len, 1)
        # # print(mask.shape)
        # weighted_sum = torch.sum(sequence_out * mask, dim=1)  # Weighted sum by mask, shape (batch_size, hidden_dim)
        # count_non_pads = mask.sum(dim=1)  # Count non-padding tokens per sequence, shape (batch_size, 1)

        # # Prevent division by zero
        # count_non_pads[count_non_pads == 0] = 1

        # # Compute mean
        # mean_embeddings = weighted_sum / count_non_pads  # Shape (batch_size, hidden_dim)

        # embeddings=torch.tensor(mean_embeddings.tolist()).to(torch.float32).to(device)  # Convert to list and return

        molecular_embeds = self.ln_molecular(embeddings)  # Layer Norm
        molecular_embeds = molecular_embeds.view(
            embeddings.size(0), 1, 1024
        )  # BZ,SeqLen,Dim

        molecular_atts = torch.ones(molecular_embeds.size()[:-1], dtype=torch.long).to(
            device
        )

        # query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        # query_output = self.Qformer.bert(
        #     query_embeds=query_tokens,
        #     encoder_hidden_states=image_embeds,
        #     encoder_attention_mask=image_atts,
        #     use_cache=True,
        #     return_dict=True,
        # )

        # image_feats = F.normalize(
        #     self.vision_proj(query_output.last_hidden_state), dim=-1
        # )

        query_tokens = self.query_tokens.expand(molecular_embeds.shape[0], -1, -1)  ###

        # query_output = self.Qformer.bert(###
        #     query_embeds=query_tokens,
        #     encoder_hidden_states=molecular_embeds,
        #     encoder_attention_mask=molecular_atts,
        #     use_cache=True,
        #     return_dict=True,
        # )
        query_output = self.testmodel(molecular_embeds)

        molecular_feats = F.normalize(  ###
            self.molecular_proj(query_output), dim=-1
        )

        text_tokens = self.tokenizer(  # Returned text_tokens are padded vectors.
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,  # Max token length: truncate long, pad short
            return_tensors="pt",
        ).to(device)  ###

        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )
        text_feat = F.normalize(
            self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
        )

        ###============== Molecular-text Contrastive ===================###

        # image_feats_all = concat_all_gather(
        #     image_feats
        # )  # [batch_size*num_gpu, num_query_tokens, embed_dim]
        text_feat_all = concat_all_gather(text_feat)  # [batch_size*num_gpu, embed_dim]
        molecular_feats_all = concat_all_gather(molecular_feats)  ###

        # sim_q2t = torch.matmul(
        #     image_feats.unsqueeze(1), text_feat_all.unsqueeze(-1)
        # ).squeeze()
        # # [batch_size, batch_size*num_gpu, num_query_tokens]

        sim_q2t = torch.matmul(
            molecular_feats.unsqueeze(1), text_feat_all.unsqueeze(-1)
        ).squeeze()  ###

        # image-text similarity: aggregate across all query tokens
        # sim_i2t, _ = sim_q2t.max(-1)
        # sim_i2t = sim_i2t / self.temp

        # sim_m2t, _ = sim_q2t.max(-1)###
        sim_m2t = sim_q2t / self.temp  ###

        # text-query similarity: [batch_size, batch_size*num_gpu, num_query_tokens]
        # sim_t2q = torch.matmul(
        #     text_feat.unsqueeze(1).unsqueeze(1), image_feats_all.permute(0, 2, 1)
        # ).squeeze()
        sim_t2q = torch.matmul(
            text_feat.unsqueeze(1).unsqueeze(1), molecular_feats_all.permute(0, 2, 1)
        ).squeeze()  ###

        # text-image similarity: aggregate across all query tokens
        # sim_t2i, _ = sim_t2q.max(-1)
        # sim_t2i = sim_t2i / self.temp  # [batch_size, batch_size*num_gpu]

        # sim_t2m, _ = sim_t2q.max(-1)
        sim_t2m = sim_t2q / self.temp  # [batch_size, batch_size*num_gpu]

        rank = dist.get_rank()
        # bs = image.size(0)
        bs = len(smiles)  ###

        targets = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=int).to(
            device
        )  ###

        # if "image_id" in samples.keys(): #coco retrieval finetuning
        #     image_ids = samples["image_id"].view(-1,1)
        #     image_ids_all = concat_all_gather(image_ids)
        #     pos_idx = torch.eq(image_ids, image_ids_all.t()).float()
        #     sim_targets = pos_idx / pos_idx.sum(1,keepdim=True)
        #     sim_targets = 0.9 * sim_targets + 0.1 * torch.ones_like(sim_targets) / sim_targets.size(1)

        #     loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1)*sim_targets,dim=1).mean()
        #     loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1)*sim_targets,dim=1).mean()
        #     loss_itc = (loss_t2i+loss_i2t)/2
        # else:
        #     loss_itc = (
        #         F.cross_entropy(sim_i2t, targets, label_smoothing=0.1)
        #         + F.cross_entropy(sim_t2i, targets, label_smoothing=0.1)
        #     ) / 2

        loss_mtc = (
            F.cross_entropy(sim_m2t, targets, label_smoothing=0.1)
            + F.cross_entropy(sim_t2m, targets, label_smoothing=0.1)
        ) / 2  ###

        ###============== Molecular-text Matching ===================###
        text_input_ids_world = concat_all_gather(text_tokens.input_ids)
        text_attention_mask_world = concat_all_gather(text_tokens.attention_mask)
        # image_embeds_world = all_gather_with_grad(image_embeds)
        molecular_embeds_world = all_gather_with_grad(molecular_embeds)  ###

        # with torch.no_grad():
        #     if "image_id" in samples.keys():#coco retrieval finetuning
        #         mask = torch.eq(image_ids, image_ids_all.t())
        #         sim_t2i.masked_fill_(mask, -10000)
        #         sim_i2t.masked_fill_(mask, -10000)
        #     else:
        #         sim_t2i[:, rank * bs : rank * bs + bs].fill_diagonal_(-10000)#Fill diagonal with -10000. Setting diagonal similarity to a large negative value suppresses self-similarity during softmax, encouraging focus on other samples
        #         sim_i2t[:, rank * bs : rank * bs + bs].fill_diagonal_(-10000)

        #     weights_t2i = F.softmax(sim_t2i, dim=1)
        #     weights_i2t = F.softmax(sim_i2t, dim=1)

        with torch.no_grad():  ###
            sim_t2m[:, rank * bs : rank * bs + bs].fill_diagonal_(-10000)
            sim_m2t[:, rank * bs : rank * bs + bs].fill_diagonal_(-10000)

            weights_t2m = F.softmax(sim_t2m, dim=1)
            weights_m2t = F.softmax(sim_m2t, dim=1)

        # # select a negative image for each text
        # image_embeds_neg = []
        # for b in range(bs):
        #     neg_idx = torch.multinomial(weights_t2i[b], 1).item()
        #     image_embeds_neg.append(image_embeds_world[neg_idx])
        # image_embeds_neg = torch.stack(image_embeds_neg, dim=0)

        # select a negative molecular for each text
        molecular_embeds_neg = []  ###
        for b in range(bs):  ###
            neg_idx = torch.multinomial(weights_t2m[b], 1).item()
            molecular_embeds_neg.append(molecular_embeds_world[neg_idx])
        molecular_embeds_neg = torch.stack(molecular_embeds_neg, dim=0)  ###

        # # select a negative text for each image
        # text_ids_neg = []
        # text_atts_neg = []
        # for b in range(bs):
        #     neg_idx = torch.multinomial(weights_i2t[b], 1).item()
        #     text_ids_neg.append(text_input_ids_world[neg_idx])
        #     text_atts_neg.append(text_attention_mask_world[neg_idx])

        # text_ids_neg = torch.stack(text_ids_neg, dim=0)
        # text_atts_neg = torch.stack(text_atts_neg, dim=0)

        # select a negative text for each molecular
        text_ids_neg = []  ###
        text_atts_neg = []  ###
        for b in range(bs):  ###
            neg_idx = torch.multinomial(weights_m2t[b], 1).item()
            text_ids_neg.append(text_input_ids_world[neg_idx])
            text_atts_neg.append(text_attention_mask_world[neg_idx])

        text_ids_neg = torch.stack(text_ids_neg, dim=0)  ###
        text_atts_neg = torch.stack(text_atts_neg, dim=0)  ###

        text_ids_all = torch.cat(
            [text_tokens.input_ids, text_tokens.input_ids, text_ids_neg], dim=0
        )  # pos, pos, neg
        text_atts_all = torch.cat(
            [text_tokens.attention_mask, text_tokens.attention_mask, text_atts_neg],
            dim=0,
        )

        # query_tokens_itm = self.query_tokens.expand(text_ids_all.shape[0], -1, -1)
        # query_atts_itm = torch.ones(query_tokens_itm.size()[:-1], dtype=torch.long).to(
        #     image.device
        # )
        # attention_mask_all = torch.cat([query_atts_itm, text_atts_all], dim=1)

        query_tokens_mtm = self.query_tokens.expand(text_ids_all.shape[0], -1, -1)  ###

        molecular_feats_mtm = (
            self.mlp(molecular_feats)
            .permute(1, 0, 2)
            .expand(text_ids_all.shape[0], -1, -1)
        )  ###

        query_atts_mtm = torch.ones(
            molecular_feats_mtm.size()[:-1], dtype=torch.long
        ).to(
            device  ###
        )
        attention_mask_all = torch.cat([query_atts_mtm, text_atts_all], dim=1)  ###

        # image_embeds_all = torch.cat(
        #     [image_embeds, image_embeds_neg, image_embeds], dim=0
        # )  # pos, neg, pos
        # image_atts_all = torch.ones(image_embeds_all.size()[:-1], dtype=torch.long).to(
        #     image.device
        # )

        molecular_embeds_all = torch.cat(  ###
            [molecular_embeds, molecular_embeds_neg, molecular_embeds], dim=0
        )  # pos, neg, pos
        molecular_atts_all = torch.ones(
            molecular_embeds_all.size()[:-1], dtype=torch.long
        ).to(
            device  ###
        )

        # output_itm = self.Qformer.bert(
        #     text_ids_all,
        #     query_embeds=query_tokens_itm,
        #     attention_mask=attention_mask_all,
        #     encoder_hidden_states=image_embeds_all,
        #     encoder_attention_mask=image_atts_all,
        #     return_dict=True,
        # )

        output_mtm = self.Qformer.bert(  ###
            text_ids_all,
            query_embeds=molecular_feats_mtm,
            attention_mask=attention_mask_all,
            encoder_hidden_states=molecular_embeds_all,
            encoder_attention_mask=molecular_atts_all,
            return_dict=True,
        )

        # vl_embeddings = output_itm.last_hidden_state[:, : query_tokens_itm.size(1), :]
        # vl_output = self.itm_head(vl_embeddings)
        # logits = vl_output.mean(dim=1)
        vl_embeddings = output_mtm.last_hidden_state[
            :, : query_tokens_mtm.size(1), :
        ]  ###
        vl_output = self.itm_head(vl_embeddings)
        logits = vl_output.mean(dim=1)

        # itm_labels = torch.cat(
        #     [torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
        #     dim=0,
        # ).to(image.device)
        # loss_itm = F.cross_entropy(logits, itm_labels)

        mtm_labels = torch.cat(  ###
            [torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
            dim=0,
        ).to(device)
        loss_mtm = F.cross_entropy(logits, mtm_labels)

        ##================= Molecular-text Captioning ========================##
        decoder_input_ids = text_tokens.input_ids.clone()
        decoder_input_ids[:, 0] = self.tokenizer.bos_token_id
        labels = decoder_input_ids.masked_fill(
            decoder_input_ids == self.tokenizer.pad_token_id, -100
        )

        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            device  ###
        )

        # attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)#BZ,224

        lm_output = self.Qformer(
            decoder_input_ids,
            attention_mask=text_tokens.attention_mask,
            # past_key_values=query_output.past_key_values,
            return_dict=True,
            labels=labels,
        )

        loss_lm = lm_output.loss  # This is the ITG loss

        return BlipOutput(  ###
            loss=loss_mtc + loss_mtm + loss_lm,
            loss_mtc=loss_mtc,
            loss_mtm=loss_mtm,
            loss_lm=loss_lm,
        )

    def current_device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def generate(  # Image-to-text generation
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=3,
        max_length=30,
        min_length=10,
        top_p=0.9,
        repetition_penalty=1.0,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        image = samples["image"]
        image_embeds = self.ln_vision(self.visual_encoder(image))

        if not use_nucleus_sampling:
            image_embeds = image_embeds.repeat_interleave(num_beams, dim=0)
        else:
            num_beams = 1
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        model_kwargs = {
            "encoder_hidden_states": image_embeds,
            "encoder_attention_mask": image_atts,
        }

        input_ids = (
            torch.LongTensor(image.size(0), 1)
            .fill_(self.tokenizer.bos_token_id)
            .to(image.device)
        )
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        outputs = self.Qformer.generate(
            input_ids=input_ids,
            query_embeds=query_tokens,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            do_sample=use_nucleus_sampling,
            top_p=top_p,
            eos_token_id=self.tokenizer.sep_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            **model_kwargs,
        )
        captions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return captions

    def forward_image(self, image):
        image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        return query_output.last_hidden_state, image_embeds

    def forward_molecular(self, smiles):
        device = self.current_device()

        # selfies_list=[]
        # for smile in smiles:
        #     try:
        #         selfies=sf.encoder(smile)
        #     except:
        #         selfies=smile
        #     selfies_list.append(selfies)
        # tokens=self.molecular_tokenizer(selfies_list, add_special_tokens=True, max_length=512, padding='max_length', truncation=True)
        # input_ids = torch.tensor(tokens['input_ids']).to(device)  # Get input_ids
        # attention_mask = torch.tensor(tokens['attention_mask']).to(device)  # Get attention_mask

        # self.molecular_encoder.eval()
        # with torch.no_grad():
        #     output = self.molecular_encoder(input_ids=input_ids, attention_mask=attention_mask)
        # sequence_out = output.last_hidden_state
        # mask = attention_mask.unsqueeze(-1)  # (batch_size, seq_len, 1)
        # weighted_sum = torch.sum(sequence_out * mask, dim=1)  # Weighted sum by mask, shape (batch_size, hidden_dim)
        # count_non_pads = mask.sum(dim=1)  # Count non-padding tokens per sequence, shape (batch_size, 1)
        # # Prevent division by zero
        # count_non_pads[count_non_pads == 0] = 1
        # # Compute mean
        # mean_embeddings = weighted_sum / count_non_pads  # Shape (batch_size, hidden_dim)
        # embeddings=torch.tensor(mean_embeddings.tolist()).to(torch.float32).to(device)  # Convert to list and return

        with torch.no_grad():
            embeddings, flags = self.molecular_encoder.transform(smiles)
        embeddings = F.normalize((embeddings), dim=-1).to(device)
        #     # embeddings_unimol_repr=self.molecular_encoder.get_repr(smiles)
        #     # embeddings=np.array(embeddings_unimol_repr['cls_repr'])

        molecular_embeds = self.ln_molecular(embeddings)  # Layer Norm
        molecular_embeds = molecular_embeds.view(
            embeddings.size(0), 1, 1024
        )  # BZ,SeqLen,Dim
        molecular_atts = torch.ones(molecular_embeds.size()[:-1], dtype=torch.long).to(
            device
        )

        query_tokens = self.query_tokens.expand(molecular_embeds.shape[0], -1, -1)  ###

        query_output = self.Qformer.bert(  ###
            query_embeds=query_tokens,
            encoder_hidden_states=molecular_embeds,
            encoder_attention_mask=molecular_atts,
            return_dict=True,
        )
        return (
            query_output.last_hidden_state,
            embeddings,
        )  # bz,num_queries?,768,  bz,true_len,1024

    def forward_text(self, text_tokens):
        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )
        return text_output.last_hidden_state[:, 0, :]

    def compute_itm(self, image_inputs, text_ids, text_atts):
        image_atts = torch.ones(image_inputs.size()[:-1], dtype=torch.long).to(
            image_inputs.device
        )
        query_tokens = self.query_tokens.expand(image_inputs.shape[0], -1, -1)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            image_inputs.device
        )
        attention_mask = torch.cat([query_atts, text_atts], dim=1)
        output_itm = self.Qformer.bert(
            text_ids,
            query_embeds=query_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=image_inputs,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        vl_embeddings = output_itm.last_hidden_state[:, : query_tokens.size(1), :]
        itm_logit = self.itm_head(vl_embeddings)
        itm_logit = itm_logit[:, :, 1].mean(dim=1)
        return itm_logit

    @torch.no_grad()
    def extract_features(self, samples, mode="multimodal"):
        """
        Extract features for multimodal or unimodal samples.
        Args:
            samples (dict): A dictionary of samples, containing the following keys:
                - image (torch.Tensor): A tensor of shape (B, C, H, W) containing the image.
                    Raw images should be preprocessed before being passed to feature extractor.
                - text_input (list): A list of strings containing the text, length B.
            mode (str): The mode of feature extraction. Can be either "multimodal", "text" or "image".
                If "multimodal", return image features and multimodal features;
                if "text", return text features;
                if "image", return image features.
                Default: "multimodal".
        Returns:
            BlipOutputFeatures: A BlipOutputFeatures object containing the features.
                See lavis/models/blip_models/blip_outputs.py for more details.
        """
        image = samples.get("image")
        caption = samples.get("text_input")

        # assert mode is one of "image", "text", "multimodal"
        assert mode in [
            "image",
            "text",
            "multimodal",
        ], "mode must be one of 'image', 'text', 'multimodal'"

        # initalize output
        image_embeds, text_embeds, multimodal_embeds = None, None, None
        image_features, text_features = None, None

        if mode == "image":
            assert image is not None, (
                "Image is not provided for mode 'image' or 'multimodal'"
            )
            # return query features
            with self.maybe_autocast():
                image_embeds_frozen = self.ln_vision(self.visual_encoder(image))
            image_embeds_frozen = image_embeds_frozen.float()
            image_atts = torch.ones(
                image_embeds_frozen.size()[:-1], dtype=torch.long
            ).to(self.device)
            query_tokens = self.query_tokens.expand(
                image_embeds_frozen.shape[0], -1, -1
            )

            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds_frozen,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            image_embeds = query_output.last_hidden_state
            image_features = F.normalize(self.vision_proj(image_embeds), dim=-1)

        elif mode == "text":
            assert caption is not None, (
                "text input is None for mode 'text' or 'multimodal'"
            )

            # return text features
            text = self.tokenizer(caption, return_tensors="pt", padding=True).to(
                self.device
            )

            text_output = self.Qformer.bert(
                text.input_ids,
                attention_mask=text.attention_mask,
                return_dict=True,
            )
            text_embeds = text_output.last_hidden_state
            text_features = self.text_proj(text_embeds)
            text_features = F.normalize(text_features, dim=-1)

        elif mode == "multimodal":
            # return multimodel query features
            with self.maybe_autocast():
                image_embeds_frozen = self.ln_vision(self.visual_encoder(image))
            image_embeds_frozen = image_embeds_frozen.float()
            image_atts = torch.ones(
                image_embeds_frozen.size()[:-1], dtype=torch.long
            ).to(self.device)
            query_tokens = self.query_tokens.expand(
                image_embeds_frozen.shape[0], -1, -1
            )
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
                self.device
            )

            text = self.tokenizer(caption, return_tensors="pt", padding=True).to(
                self.device
            )
            attention_mask = torch.cat([query_atts, text.attention_mask], dim=1)

            output = self.Qformer.bert(
                text.input_ids,
                query_embeds=query_tokens,
                attention_mask=attention_mask,
                encoder_hidden_states=image_embeds_frozen,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            multimodal_embeds = output.last_hidden_state[:, : query_tokens.size(1), :]

        return BlipOutputFeatures(
            image_embeds=image_embeds,
            image_embeds_proj=image_features,
            text_embeds=text_embeds,
            text_embeds_proj=text_features,
            multimodal_embeds=multimodal_embeds,
        )

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        cross_attention_freq = cfg.get("cross_attention_freq", 2)

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        max_txt_len = cfg.get("max_txt_len", 32)

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            cross_attention_freq=cross_attention_freq,
            max_txt_len=max_txt_len,
        )
        model.load_checkpoint_from_config(cfg)

        return model

    def compute_sim_matrix(self, data_loader, task_cfg):
        """
        Compute similarity i2t, t2i matrix for the given data loader.
        """
        k_test = task_cfg.k_test

        return compute_sim_matrix(model=self, data_loader=data_loader, k_test=k_test)


# @registry.register_model("blip2")
# @registry.register_model("blip2_feature_extractor")
# class Blip2Qformer(Blip2Base):
#     """
#     BLIP2 first-stage model with Q-former and ViT.
#     Supported model types:
#         - pretrained: pretrained model with vit-g
#         - pretrain_vitL: pretrained model with vit-large
#         - coco: fintuned model on coco
#     Usage:
#         >>> from lavis.models import load_model
#         >>> model = load_model("blip2", "pretrain")
#     """

#     PRETRAINED_MODEL_CONFIG_DICT = {
#         "pretrain": "configs/models/blip2/blip2_pretrain.yaml",
#         "pretrain_vitL": "configs/models/blip2/blip2_pretrain_vitL.yaml",
#         "coco": "configs/models/blip2/blip2_coco.yaml",
#     }

#     def __init__(
#         self,
#         vit_model="eva_clip_g",
#         img_size=224,
#         drop_path_rate=0,
#         use_grad_checkpoint=False,
#         vit_precision="fp16",
#         freeze_vit=True,
#         num_query_token=32,
#         cross_attention_freq=2,
#         embed_dim=256,
#         max_txt_len=32,
#     ):
#         super().__init__()

#         self.tokenizer = self.init_tokenizer()

#         self.visual_encoder, self.ln_vision = self.init_vision_encoder(
#             vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
#         )
#         if freeze_vit:
#             for name, param in self.visual_encoder.named_parameters():
#                 param.requires_grad = False
#             self.visual_encoder = self.visual_encoder.eval()
#             self.visual_encoder.train = disabled_train
#             logging.info("freeze vision encoder")

#         self.Qformer, self.query_tokens = self.init_Qformer(
#             num_query_token, self.visual_encoder.num_features, cross_attention_freq
#         )
#         self.Qformer.resize_token_embeddings(len(self.tokenizer))
#         state_dict = self.Qformer.state_dict()
#         for name, param in self.Qformer.named_parameters():
#             if "_query" in name:
#                 key_orig = name.replace("_query", "")
#                 param.data.copy_(state_dict[key_orig])

#         self.vision_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
#         self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)

#         self.itm_head = nn.Linear(self.Qformer.config.hidden_size, 2)

#         self.temp = nn.Parameter(0.07 * torch.ones([]))

#         self.max_txt_len = max_txt_len

#     def forward(self, samples):
#         image = samples["image"]
#         text = samples["text_input"]

#         image_embeds = self.ln_vision(self.visual_encoder(image))
#         image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
#             image.device
#         )

#         query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

#         query_output = self.Qformer.bert(
#             query_embeds=query_tokens,
#             encoder_hidden_states=image_embeds,
#             encoder_attention_mask=image_atts,
#             use_cache=True,
#             return_dict=True,
#         )

#         image_feats = F.normalize(
#             self.vision_proj(query_output.last_hidden_state), dim=-1
#         )

#         text_tokens = self.tokenizer(#Returned text_tokens are padded vectors.
#             text,
#             padding="max_length",
#             truncation=True,
#             max_length=self.max_txt_len,
#             return_tensors="pt",
#         ).to(image.device)
#         text_output = self.Qformer.bert(
#             text_tokens.input_ids,
#             attention_mask=text_tokens.attention_mask,
#             return_dict=True,
#         )
#         text_feat = F.normalize(
#             self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
#         )

#         ###============== Image-text Contrastive ===================###
#         image_feats_all = concat_all_gather(
#             image_feats
#         )  # [batch_size*num_gpu, num_query_tokens, embed_dim]
#         text_feat_all = concat_all_gather(text_feat)  # [batch_size*num_gpu, embed_dim]

#         sim_q2t = torch.matmul(
#             image_feats.unsqueeze(1), text_feat_all.unsqueeze(-1)#[batch_size, 1, num_query_tokens, embed_dim]   [batch_size*num_gpu, embed_dim, 1]  When using torch.matmul(), the last two dimensions are multiplied
#         ).squeeze()
#         # [batch_size, batch_size*num_gpu, num_query_tokens]

#         # image-text similarity: aggregate across all query tokens
#         sim_i2t, _ = sim_q2t.max(-1)
#         sim_i2t = sim_i2t / self.temp

#         # text-query similarity: [batch_size, batch_size*num_gpu, num_query_tokens]
#         sim_t2q = torch.matmul(
#             text_feat.unsqueeze(1).unsqueeze(1), image_feats_all.permute(0, 2, 1)
#         ).squeeze()

#         # text-image similarity: aggregate across all query tokens
#         sim_t2i, _ = sim_t2q.max(-1)
#         sim_t2i = sim_t2i / self.temp  # [batch_size, batch_size*num_gpu]

#         rank = dist.get_rank()
#         bs = image.size(0)
#         targets = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=int).to(
#             image.device
#         )

#         if "image_id" in samples.keys(): #coco retrieval finetuning
#             image_ids = samples["image_id"].view(-1,1)
#             image_ids_all = concat_all_gather(image_ids)
#             pos_idx = torch.eq(image_ids, image_ids_all.t()).float()
#             sim_targets = pos_idx / pos_idx.sum(1,keepdim=True)
#             sim_targets = 0.9 * sim_targets + 0.1 * torch.ones_like(sim_targets) / sim_targets.size(1)

#             loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1)*sim_targets,dim=1).mean()
#             loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1)*sim_targets,dim=1).mean()
#             loss_itc = (loss_t2i+loss_i2t)/2
#         else:
#             loss_itc = (
#                 F.cross_entropy(sim_i2t, targets, label_smoothing=0.1)
#                 + F.cross_entropy(sim_t2i, targets, label_smoothing=0.1)
#             ) / 2

#         ###============== Image-text Matching ===================###
#         text_input_ids_world = concat_all_gather(text_tokens.input_ids)
#         text_attention_mask_world = concat_all_gather(text_tokens.attention_mask)
#         image_embeds_world = all_gather_with_grad(image_embeds)
#         with torch.no_grad():
#             if "image_id" in samples.keys():
#                 mask = torch.eq(image_ids, image_ids_all.t())
#                 sim_t2i.masked_fill_(mask, -10000)
#                 sim_i2t.masked_fill_(mask, -10000)
#             else:
#                 sim_t2i[:, rank * bs : rank * bs + bs].fill_diagonal_(-10000)
#                 sim_i2t[:, rank * bs : rank * bs + bs].fill_diagonal_(-10000)

#             weights_t2i = F.softmax(sim_t2i, dim=1)
#             weights_i2t = F.softmax(sim_i2t, dim=1)

#         # select a negative image for each text
#         image_embeds_neg = []
#         for b in range(bs):
#             neg_idx = torch.multinomial(weights_t2i[b], 1).item()
#             image_embeds_neg.append(image_embeds_world[neg_idx])
#         image_embeds_neg = torch.stack(image_embeds_neg, dim=0)

#         # select a negative text for each image
#         text_ids_neg = []
#         text_atts_neg = []
#         for b in range(bs):
#             neg_idx = torch.multinomial(weights_i2t[b], 1).item()
#             text_ids_neg.append(text_input_ids_world[neg_idx])
#             text_atts_neg.append(text_attention_mask_world[neg_idx])

#         text_ids_neg = torch.stack(text_ids_neg, dim=0)
#         text_atts_neg = torch.stack(text_atts_neg, dim=0)

#         text_ids_all = torch.cat(
#             [text_tokens.input_ids, text_tokens.input_ids, text_ids_neg], dim=0
#         )  # pos, pos, neg
#         text_atts_all = torch.cat(
#             [text_tokens.attention_mask, text_tokens.attention_mask, text_atts_neg],
#             dim=0,
#         )

#         query_tokens_itm = self.query_tokens.expand(text_ids_all.shape[0], -1, -1)
#         query_atts_itm = torch.ones(query_tokens_itm.size()[:-1], dtype=torch.long).to(
#             image.device
#         )
#         attention_mask_all = torch.cat([query_atts_itm, text_atts_all], dim=1)

#         image_embeds_all = torch.cat(
#             [image_embeds, image_embeds_neg, image_embeds], dim=0
#         )  # pos, neg, pos
#         image_atts_all = torch.ones(image_embeds_all.size()[:-1], dtype=torch.long).to(
#             image.device
#         )

#         output_itm = self.Qformer.bert(
#             text_ids_all,
#             query_embeds=query_tokens_itm,
#             attention_mask=attention_mask_all,
#             encoder_hidden_states=image_embeds_all,
#             encoder_attention_mask=image_atts_all,
#             return_dict=True,
#         )

#         vl_embeddings = output_itm.last_hidden_state[:, : query_tokens_itm.size(1), :]
#         vl_output = self.itm_head(vl_embeddings)
#         logits = vl_output.mean(dim=1)

#         itm_labels = torch.cat(
#             [torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
#             dim=0,
#         ).to(image.device)
#         loss_itm = F.cross_entropy(logits, itm_labels)

#         ##================= Image Captioning ========================##
#         decoder_input_ids = text_tokens.input_ids.clone()
#         decoder_input_ids[:, 0] = self.tokenizer.bos_token_id
#         labels = decoder_input_ids.masked_fill(
#             decoder_input_ids == self.tokenizer.pad_token_id, -100
#         )

#         query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
#             image.device
#         )
#         attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)
#         lm_output = self.Qformer(
#             decoder_input_ids,
#             attention_mask=attention_mask,
#             past_key_values=query_output.past_key_values,
#             return_dict=True,
#             labels=labels,
#         )

#         loss_lm = lm_output.loss#This is the ITG loss

#         return BlipOutput(
#             loss=loss_itc + loss_itm + loss_lm,
#             loss_itc=loss_itc,
#             loss_itm=loss_itm,
#             loss_lm=loss_lm,
#         )

#     @torch.no_grad()
#     def generate(#Image-to-text generation
#         self,
#         samples,
#         use_nucleus_sampling=False,
#         num_beams=3,
#         max_length=30,
#         min_length=10,
#         top_p=0.9,
#         repetition_penalty=1.0,
#     ):
#         """
#         Args:
#             samples (dict): A dictionary containing the following keys:
#                 - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
#             use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
#             num_beams (int): Number of beams for beam search. 1 means no beam search.
#             max_length (int): The maximum length of the sequence to be generated.
#             min_length (int): The minimum length of the sequence to be generated.
#             top_p (float): The cumulative probability for nucleus sampling.
#             repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
#             num_captions (int): Number of captions to be generated for each image.
#         Returns:
#             captions (list): A list of strings of length batch_size * num_captions.
#         """
#         image = samples["image"]
#         image_embeds = self.ln_vision(self.visual_encoder(image))

#         if not use_nucleus_sampling:
#             image_embeds = image_embeds.repeat_interleave(num_beams, dim=0)
#         else:
#             num_beams = 1
#         image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
#             image.device
#         )

#         model_kwargs = {
#             "encoder_hidden_states": image_embeds,
#             "encoder_attention_mask": image_atts,
#         }

#         input_ids = (
#             torch.LongTensor(image.size(0), 1)
#             .fill_(self.tokenizer.bos_token_id)
#             .to(image.device)
#         )
#         query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

#         outputs = self.Qformer.generate(
#             input_ids=input_ids,
#             query_embeds=query_tokens,
#             max_length=max_length,
#             min_length=min_length,
#             num_beams=num_beams,
#             do_sample=use_nucleus_sampling,
#             top_p=top_p,
#             eos_token_id=self.tokenizer.sep_token_id,
#             pad_token_id=self.tokenizer.pad_token_id,
#             **model_kwargs
#         )
#         captions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
#         return captions

#     def forward_image(self, image):
#         image_embeds = self.ln_vision(self.visual_encoder(image))
#         image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
#             image.device
#         )

#         query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

#         query_output = self.Qformer.bert(
#             query_embeds=query_tokens,
#             encoder_hidden_states=image_embeds,
#             encoder_attention_mask=image_atts,
#             return_dict=True,
#         )
#         return query_output.last_hidden_state, image_embeds

#     def forward_text(self, text_tokens):
#         text_output = self.Qformer.bert(
#             text_tokens.input_ids,
#             attention_mask=text_tokens.attention_mask,
#             return_dict=True,
#         )
#         return text_output.last_hidden_state[:, 0, :]

#     def compute_itm(self, image_inputs, text_ids, text_atts):
#         image_atts = torch.ones(image_inputs.size()[:-1], dtype=torch.long).to(
#             image_inputs.device
#         )
#         query_tokens = self.query_tokens.expand(image_inputs.shape[0], -1, -1)
#         query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
#             image_inputs.device
#         )
#         attention_mask = torch.cat([query_atts, text_atts], dim=1)
#         output_itm = self.Qformer.bert(
#             text_ids,
#             query_embeds=query_tokens,
#             attention_mask=attention_mask,
#             encoder_hidden_states=image_inputs,
#             encoder_attention_mask=image_atts,
#             return_dict=True,
#         )
#         vl_embeddings = output_itm.last_hidden_state[:, : query_tokens.size(1), :]
#         itm_logit = self.itm_head(vl_embeddings)
#         itm_logit = itm_logit[:, :, 1].mean(dim=1)
#         return itm_logit

#     @torch.no_grad()
#     def extract_features(self, samples, mode="multimodal"):
#         """
#         Extract features for multimodal or unimodal samples.
#         Args:
#             samples (dict): A dictionary of samples, containing the following keys:
#                 - image (torch.Tensor): A tensor of shape (B, C, H, W) containing the image.
#                     Raw images should be preprocessed before being passed to feature extractor.
#                 - text_input (list): A list of strings containing the text, length B.
#             mode (str): The mode of feature extraction. Can be either "multimodal", "text" or "image".
#                 If "multimodal", return image features and multimodal features;
#                 if "text", return text features;
#                 if "image", return image features.
#                 Default: "multimodal".
#         Returns:
#             BlipOutputFeatures: A BlipOutputFeatures object containing the features.
#                 See lavis/models/blip_models/blip_outputs.py for more details.
#         """
#         image = samples.get("image")
#         caption = samples.get("text_input")

#         # assert mode is one of "image", "text", "multimodal"
#         assert mode in [
#             "image",
#             "text",
#             "multimodal",
#         ], "mode must be one of 'image', 'text', 'multimodal'"

#         # initalize output
#         image_embeds, text_embeds, multimodal_embeds = None, None, None
#         image_features, text_features = None, None

#         if mode == "image":
#             assert (
#                 image is not None
#             ), "Image is not provided for mode 'image' or 'multimodal'"
#             # return query features
#             with self.maybe_autocast():
#                 image_embeds_frozen = self.ln_vision(self.visual_encoder(image))
#             image_embeds_frozen = image_embeds_frozen.float()
#             image_atts = torch.ones(
#                 image_embeds_frozen.size()[:-1], dtype=torch.long
#             ).to(self.device)
#             query_tokens = self.query_tokens.expand(
#                 image_embeds_frozen.shape[0], -1, -1
#             )

#             query_output = self.Qformer.bert(
#                 query_embeds=query_tokens,
#                 encoder_hidden_states=image_embeds_frozen,
#                 encoder_attention_mask=image_atts,
#                 return_dict=True,
#             )
#             image_embeds = query_output.last_hidden_state
#             image_features = F.normalize(self.vision_proj(image_embeds), dim=-1)

#         elif mode == "text":
#             assert (
#                 caption is not None
#             ), "text input is None for mode 'text' or 'multimodal'"

#             # return text features
#             text = self.tokenizer(caption, return_tensors="pt", padding=True).to(
#                 self.device
#             )

#             text_output = self.Qformer.bert(
#                 text.input_ids,
#                 attention_mask=text.attention_mask,
#                 return_dict=True,
#             )
#             text_embeds = text_output.last_hidden_state
#             text_features = self.text_proj(text_embeds)
#             text_features = F.normalize(text_features, dim=-1)

#         elif mode == "multimodal":
#             # return multimodel query features
#             with self.maybe_autocast():
#                 image_embeds_frozen = self.ln_vision(self.visual_encoder(image))
#             image_embeds_frozen = image_embeds_frozen.float()
#             image_atts = torch.ones(
#                 image_embeds_frozen.size()[:-1], dtype=torch.long
#             ).to(self.device)
#             query_tokens = self.query_tokens.expand(
#                 image_embeds_frozen.shape[0], -1, -1
#             )
#             query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
#                 self.device
#             )

#             text = self.tokenizer(caption, return_tensors="pt", padding=True).to(
#                 self.device
#             )
#             attention_mask = torch.cat([query_atts, text.attention_mask], dim=1)

#             output = self.Qformer.bert(
#                 text.input_ids,
#                 query_embeds=query_tokens,
#                 attention_mask=attention_mask,
#                 encoder_hidden_states=image_embeds_frozen,
#                 encoder_attention_mask=image_atts,
#                 return_dict=True,
#             )

#             multimodal_embeds = output.last_hidden_state[:, : query_tokens.size(1), :]

#         return BlipOutputFeatures(
#             image_embeds=image_embeds,
#             image_embeds_proj=image_features,
#             text_embeds=text_embeds,
#             text_embeds_proj=text_features,
#             multimodal_embeds=multimodal_embeds,
#         )

#     @classmethod
#     def from_config(cls, cfg):
#         vit_model = cfg.get("vit_model", "eva_clip_g")
#         img_size = cfg.get("image_size")
#         num_query_token = cfg.get("num_query_token")
#         cross_attention_freq = cfg.get("cross_attention_freq", 2)

#         drop_path_rate = cfg.get("drop_path_rate", 0)
#         use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
#         vit_precision = cfg.get("vit_precision", "fp16")
#         freeze_vit = cfg.get("freeze_vit", True)

#         max_txt_len = cfg.get("max_txt_len", 32)

#         model = cls(
#             vit_model=vit_model,
#             img_size=img_size,
#             drop_path_rate=drop_path_rate,
#             use_grad_checkpoint=use_grad_checkpoint,
#             vit_precision=vit_precision,
#             freeze_vit=freeze_vit,
#             num_query_token=num_query_token,
#             cross_attention_freq=cross_attention_freq,
#             max_txt_len=max_txt_len,
#         )
#         model.load_checkpoint_from_config(cfg)

#         return model

#     def compute_sim_matrix(self, data_loader, task_cfg):
#         """
#         Compute similarity i2t, t2i matrix for the given data loader.
#         """
#         k_test = task_cfg.k_test

#         return compute_sim_matrix(model=self, data_loader=data_loader, k_test=k_test)
