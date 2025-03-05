import logging
import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F

from lavis.common.registry import registry
from lavis.models.blip2_models.blip2 import (
    Blip2Base,
    compute_sim_matrix,
    disabled_train,
)
from lavis.models.blip_models.blip_outputs import BlipOutputFeatures


@registry.register_model("blip2_cir_align_prompt")
class Blip2QformerCirAlignPrompt(Blip2Base):
    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain": "configs/models/blip2/blip2_pretrain.yaml",
        "pretrain_vitL": "configs/models/blip2/blip2_pretrain_vitL.yaml",
        "coco": "configs/models/blip2/blip2_coco.yaml",
    }

    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        cross_attention_freq=2,
        embed_dim=256,
        max_txt_len=128,
    ):
        super().__init__()

        self.tokenizer = self.init_tokenizer()

        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")

        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features, cross_attention_freq
        )
        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])

        self.vision_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.target_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)

        self.itm_head = nn.Linear(self.Qformer.config.hidden_size, 2)
        self.temp = nn.Parameter(0.07 * torch.ones([]))
        self.max_txt_len = 128  # 这里得改？32改为128
        self.token_alpha = nn.Parameter(1.0 * torch.ones([]))
        self.num_query_token = num_query_token

        # new tokens
        self.prompt_tokens = nn.Parameter(torch.zeros(1, num_query_token, self.Qformer.config.hidden_size))
        self.prompt_tokens.data.normal_(mean=0.0, std=self.Qformer.config.initializer_range)
        self.visual_tokens = nn.Parameter(torch.zeros(1, num_query_token, self.Qformer.config.hidden_size))
        self.visual_tokens.data.normal_(mean=0.0, std=self.Qformer.config.initializer_range)

    def prune_tokens(self, before_tokens, after_tokens, keep_after=30):
        # TODO: MIO module, expanded for Video domain (multi-frames similar to multi-images)
        pass

    def BBC_loss(self, feats1, feats2):
        prediction = 100 * feats1 @ feats2.T
        labels = torch.arange(0, feats1.size(0)).long().to(feats1.device)
        return F.cross_entropy(prediction, labels)

    def token_contrast_loss(self, token_feats, visual_token_feats):
        sim_f2t = torch.matmul(token_feats.unsqueeze(1).unsqueeze(1), visual_token_feats.permute(0, 2, 1)).squeeze()
        sim_matrix, _ = sim_f2t.max(-1)
        sim_matrix = sim_matrix / self.temp
        bs = token_feats.size(0)
        targets = torch.linspace(0, bs - 1, bs, dtype=int).to(token_feats.device)
        return F.cross_entropy(sim_matrix, targets)

    def forward(self, samples, mode="single"):
        if mode == "single":
            image = samples["image"]
            target = samples["target"]
            modifier = samples["text_input"]
            tar_cap = samples["tar_cap"]
            ref_cap = samples["ref_cap"]
            """reference-text fusion"""
            image_embeds = self.ln_vision(self.visual_encoder(image))  # [b, 257, 1408]
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

            # 传递覆写
            before_query_tokens = None
            if "fusion" in samples:
                query_tokens = samples["fusion"]
                before_query_tokens = query_tokens.clone().detach()  # query_tokens.clone()
            else:
                query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(self.device)

            tar_query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            tar_query_atts = torch.ones(tar_query_tokens.size()[:-1], dtype=torch.long).to(self.device)

            modifier_tokens = self.tokenizer(
                modifier,
                padding="max_length",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)

            ref_cap_tokens = self.tokenizer(
                ref_cap,
                padding="max_length",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)

            attention_mask = torch.cat([query_atts, modifier_tokens.attention_mask], dim=1)
            ref_cap_attn_mask = torch.cat([query_atts, ref_cap_tokens.attention_mask], dim=1)
            fusion_output = self.Qformer.bert(
                ref_cap_tokens.input_ids,  # modifier_tokens.input_ids,
                query_embeds=query_tokens,  # Qformer里query embeds和modifier_tokens会拼接
                attention_mask=ref_cap_attn_mask,  # attention_mask
                encoder_hidden_states=image_embeds,  # cross attention
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            text_output = self.Qformer.bert(
                modifier_tokens.input_ids,
                query_embeds=fusion_output.last_hidden_state[:, : query_tokens.size(1), :],  # [b, 32, 768]
                attention_mask=attention_mask,
                return_dict=True,
            )
            fusion_feats = F.normalize(self.text_proj(text_output.last_hidden_state[:, 32, :]), dim=-1)  # [b, 256]

            # 包含的是reference侧视觉-文本多模态信息，传递给下一轮
            query_fusion_tokens = fusion_output.last_hidden_state[:, : query_tokens.size(1), :]
            # 将两轮query token过KNN
            if before_query_tokens is not None:
                query_fusion_tokens = self.prune_tokens(before_query_tokens, query_fusion_tokens)

            """Fusion-target Contrastive Loss"""
            target_img_embeds = self.ln_vision(self.visual_encoder(target))
            target_img_atts = torch.ones(target_img_embeds.size()[:-1], dtype=torch.long).to(image.device)
            tar_cap_tokens = self.tokenizer(
                tar_cap,
                padding="max_length",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)
            tar_attention_mask = torch.cat([tar_query_atts, tar_cap_tokens.attention_mask], dim=1)
            tar_fusion_output = self.Qformer.bert(
                tar_cap_tokens.input_ids,
                query_embeds=tar_query_tokens,  # Qformer里query embeds和modifier_tokens会拼接
                attention_mask=tar_attention_mask,
                encoder_hidden_states=target_img_embeds,  # cross attention
                encoder_attention_mask=target_img_atts,
                return_dict=True,
            )
            tar_fusion_feats = F.normalize(self.target_proj(tar_fusion_output.last_hidden_state[:, 32, :]), dim=-1)

            # 融合特征与target caption特征的loss对齐
            tar_text_tokens = self.prompt_tokens.expand(image_embeds.shape[0], -1, -1)
            tar_text_output = self.Qformer.bert(
                tar_cap_tokens.input_ids,
                query_embeds=tar_text_tokens,
                attention_mask=tar_attention_mask,
                return_dict=True,
                no_img=True,
            )
            tar_cap_text_feat = F.normalize(self.text_proj(tar_text_output.last_hidden_state[:, 0, :]), dim=-1)

            mod_text_tokens = self.prompt_tokens.expand(
                image_embeds.shape[0], -1, -1)
            mod_text_output = self.Qformer.bert(
                modifier_tokens.input_ids,
                query_embeds=mod_text_tokens,
                attention_mask=attention_mask,
                return_dict=True,
                no_img=True,
            )
            mod_cap_feat = F.normalize(self.text_proj(mod_text_output.last_hidden_state[:, 0, :]), dim=-1)

            # TODO single turn 的loss权重调整
            return {
                "loss_fus2tar": self.BBC_loss(fusion_feats.to(image.device), tar_fusion_feats.to(image.device)),
                'loss_fus2cap': self.BBC_loss(fusion_feats.to(image.device), tar_cap_text_feat.to(image.device)),
                'loss_mod2fus': self.BBC_loss(mod_cap_feat.to(image.device), tar_fusion_feats.to(image.device)),
                'loss_mod2cap': self.BBC_loss(mod_cap_feat.to(image.device), tar_cap_text_feat.to(image.device))
            }, query_fusion_tokens

        elif mode == "fiq":
            image = samples["image"]
            target = samples["target"]
            modifier = samples["text_input"]
            tar_cap = samples["tar_cap"]
            ref_cap = samples["ref_cap"]
            """reference-text fusion"""
            image_embeds = self.ln_vision(self.visual_encoder(image))  # [b, 257, 1408]
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

            # 传递覆写
            before_query_tokens = None
            if "fusion" in samples:
                query_tokens = samples["fusion"]
                before_query_tokens = query_tokens.clone().detach()  # query_tokens.clone()
            else:
                query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(self.device)

            tar_query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            tar_query_atts = torch.ones(tar_query_tokens.size()[:-1], dtype=torch.long).to(self.device)

            modifier_tokens = self.tokenizer(
                modifier,
                padding="max_length",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)

            ref_cap_tokens = self.tokenizer(
                ref_cap,
                padding="max_length",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)

            attention_mask = torch.cat([query_atts, modifier_tokens.attention_mask], dim=1)
            ref_cap_attn_mask = torch.cat([query_atts, ref_cap_tokens.attention_mask], dim=1)
            fusion_output = self.Qformer.bert(
                modifier_tokens.input_ids,  # modifier_tokens.input_ids, ref_cap_tokens.input_ids
                query_embeds=query_tokens,  # Qformer里query embeds和modifier_tokens会拼接
                attention_mask=attention_mask,  # attention_mask, ref_cap_attn_mask
                encoder_hidden_states=image_embeds,  # cross attention
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            text_output = self.Qformer.bert(
                modifier_tokens.input_ids,
                query_embeds=fusion_output.last_hidden_state[:, : query_tokens.size(1), :],  # [b, 32, 768]
                attention_mask=attention_mask,
                return_dict=True,
            )
            fusion_feats = F.normalize(self.text_proj(text_output.last_hidden_state[:, 32, :]), dim=-1)  # [b, 256]

            # 包含的是reference侧视觉-文本多模态信息，传递给下一轮
            query_fusion_tokens = fusion_output.last_hidden_state[:, : query_tokens.size(1), :]
            # 将两轮query token过KNN
            if before_query_tokens is not None:
                query_fusion_tokens = self.prune_tokens(before_query_tokens, query_fusion_tokens)

            """Fusion-target Contrastive Loss"""
            taregt_embeds = self.ln_vision(self.visual_encoder(target))
            target_atts = torch.ones(taregt_embeds.size()[:-1], dtype=torch.long).to(image.device)
            target_output = self.Qformer.bert(
                query_embeds=tar_query_tokens,
                encoder_hidden_states=taregt_embeds,
                encoder_attention_mask=target_atts,
                use_cache=True,
                return_dict=True,
            )
            target_feats = F.normalize(self.target_proj(target_output.last_hidden_state), dim=-1)
            return {
                "loss_fus2tar": self.token_contrast_loss(fusion_feats.to(image.device), target_feats.to(image.device))
            }, query_fusion_tokens

        elif mode == "single_no_target":
            image = samples["image"]
            ref_cap = samples["ref_cap"]
            """reference-text fusion"""
            image_embeds = self.ln_vision(self.visual_encoder(image))  # [b, 257, 1408]
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
            # 传递覆写
            before_query_tokens = None
            if "fusion" in samples:
                query_tokens = samples["fusion"]
                before_query_tokens = query_tokens.clone().detach()  # query_tokens.clone()
            else:
                query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(self.device)
            ref_cap_tokens = self.tokenizer(
                ref_cap,
                padding="max_length",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)
            ref_cap_attn_mask = torch.cat([query_atts, ref_cap_tokens.attention_mask], dim=1)
            fusion_output = self.Qformer.bert(
                ref_cap_tokens.input_ids,  # modifier_tokens.input_ids,
                query_embeds=query_tokens,  # Qformer里query embeds和modifier_tokens会拼接
                attention_mask=ref_cap_attn_mask,  # attention_mask
                encoder_hidden_states=image_embeds,  # cross attention
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            # 包含的是reference侧视觉-文本多模态信息，传递给下一轮
            query_fusion_tokens = fusion_output.last_hidden_state[:, : query_tokens.size(1), :]  # [b, 32, 768] 映射?
            # 将两轮query token过KNN
            if before_query_tokens is not None:
                query_fusion_tokens = self.prune_tokens(before_query_tokens, query_fusion_tokens)
            return query_fusion_tokens

        else:  # mode == "multi_turn"
            image = samples["image"]  # tar2_img
            target = samples["target"]  # tar3_img
            modifier = samples["text_input"]
            tar_cap = samples["tar_cap"]
            ref_cap = samples["ref_cap"]
            """multi-turn reference-text fusion"""
            query_tokens = samples["fusion"]  # query tokens 传递覆写
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(self.device)
            tar_query_tokens = self.query_tokens.expand(query_tokens.shape[0], -1, -1)  # [b, 32, 768]
            tar_query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(self.device)

            # modified text tokens
            modifier_tokens = self.tokenizer(
                modifier,
                padding="max_length",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)
            attention_mask = torch.cat([query_atts, modifier_tokens.attention_mask], dim=1)
            """修改主要围绕这部分展开: 1. 引入caption; 2. 前query tokens的self-attention"""
            ref_cap_tokens = self.tokenizer(
                ref_cap,
                padding="max_length",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)

            multi_query_tokens = query_tokens
            multi_query_atts = torch.ones(multi_query_tokens.size()[:-1], dtype=torch.long).to(self.device)
            ref_cap_attn_mask = torch.cat([multi_query_atts, ref_cap_tokens.attention_mask], dim=1)

            # 是否引入modified text中最后一句
            fusion_output = self.Qformer.bert(
                ref_cap_tokens.input_ids,
                query_embeds=multi_query_tokens,
                attention_mask=ref_cap_attn_mask,
                return_dict=True,
            )
            text_output = self.Qformer.bert(
                modifier_tokens.input_ids,
                query_embeds=fusion_output.last_hidden_state[:, : query_tokens.size(1), :],
                attention_mask=attention_mask,
                return_dict=True,
            )
            fusion_feats = F.normalize(self.text_proj(text_output.last_hidden_state[:, 32, :]), dim=-1)  # [b, 32, 768]

            """Fusion-target Contrastive Loss"""
            target_img_embeds = self.ln_vision(self.visual_encoder(target))
            target_img_atts = torch.ones(target_img_embeds.size()[:-1], dtype=torch.long).to(image.device)
            tar_cap_tokens = self.tokenizer(
                tar_cap,
                padding="max_length",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)
            tar_attention_mask = torch.cat([tar_query_atts, tar_cap_tokens.attention_mask], dim=1)
            tar_fusion_output = self.Qformer.bert(
                tar_cap_tokens.input_ids,
                query_embeds=tar_query_tokens,  # Qformer里query embeds和modifier_tokens会拼接
                attention_mask=tar_attention_mask,
                encoder_hidden_states=target_img_embeds,  # cross attention
                encoder_attention_mask=target_img_atts,
                return_dict=True,
            )
            # # Multimodal - Multimodal
            tar_fusion_feats = F.normalize(self.target_proj(tar_fusion_output.last_hidden_state[:, 32, :]), dim=-1)
            # Multimodal - Text
            tar_text_tokens = self.prompt_tokens.expand(query_tokens.shape[0], -1, -1)
            tar_text_output = self.Qformer.bert(
                tar_cap_tokens.input_ids,
                query_embeds=tar_text_tokens,
                attention_mask=tar_attention_mask,
                return_dict=True,
                no_img=True,
            )
            tar_cap_text_feat = F.normalize(self.text_proj(tar_text_output.last_hidden_state[:, 0, :]), dim=-1)

            mod_text_tokens = self.prompt_tokens.expand(query_tokens.shape[0], -1, -1)
            mod_text_output = self.Qformer.bert(
                modifier_tokens.input_ids,
                query_embeds=mod_text_tokens,
                attention_mask=attention_mask,
                return_dict=True,
                no_img=True,
            )
            mod_cap_feat = F.normalize(self.text_proj(mod_text_output.last_hidden_state[:, 0, :]), dim=-1)

            # TODO single turn 的loss权重调整
            return {
                "loss_fus2tar": self.BBC_loss(fusion_feats.to(image.device), tar_fusion_feats.to(image.device)),
                'loss_fus2cap': self.BBC_loss(fusion_feats.to(image.device), tar_cap_text_feat.to(image.device)),
                'loss_mod2fus': self.BBC_loss(mod_cap_feat.to(image.device), tar_fusion_feats.to(image.device)),
                'loss_mod2cap': self.BBC_loss(mod_cap_feat.to(image.device), tar_cap_text_feat.to(image.device))
            }

    @torch.no_grad()
    def inference(self, samples, mode="single"):
        if mode == "single":
            image_embeds = samples["img_embeds"]
            target_feats = samples["target_feats"]
            modifier = samples["text_input"]
            ref_cap = samples["ref_cap"]
            query_tokens = samples["fusion"]
            """reference-text fusion"""
            if query_tokens is None:
                query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(self.device)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(self.device)

            modifier_tokens = self.tokenizer(
                modifier,
                padding="max_length",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(self.device)

            ref_cap_tokens = self.tokenizer(
                ref_cap,
                padding="max_length",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(self.device)

            attention_mask = torch.cat([query_atts, modifier_tokens.attention_mask], dim=1)
            ref_cap_attn_mask = torch.cat([query_atts, ref_cap_tokens.attention_mask], dim=1)
            fusion_output = self.Qformer.bert(
                ref_cap_tokens.input_ids,  # modifier_tokens.input_ids,
                query_embeds=query_tokens,  # Qformer里query embeds和modifier_tokens会拼接
                attention_mask=ref_cap_attn_mask,  # attention_mask
                encoder_hidden_states=image_embeds,  # cross attention
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            text_output = self.Qformer.bert(
                modifier_tokens.input_ids,
                query_embeds=fusion_output.last_hidden_state[:, : query_tokens.size(1), :],  # [b, 32, 768]
                attention_mask=attention_mask,
                return_dict=True,
            )
            fusion_feats = F.normalize(self.text_proj(text_output.last_hidden_state[:, 32, :]), dim=-1)  # [b, 256]
            sim_matrix = self.compute_distance_matrix(fusion_feats.to(self.device), target_feats)
            return sim_matrix

        elif mode == "fiq":
            image_embeds = samples["img_embeds"]
            target_feats = samples["target_feats"]
            modifier = samples["text_input"]
            ref_cap = samples["ref_cap"]
            query_tokens = samples["fusion"]
            """reference-text fusion"""
            if query_tokens is None:
                query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(self.device)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(self.device)

            modifier_tokens = self.tokenizer(
                modifier,
                padding="max_length",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(self.device)

            ref_cap_tokens = self.tokenizer(
                ref_cap,
                padding="max_length",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(self.device)

            attention_mask = torch.cat([query_atts, modifier_tokens.attention_mask], dim=1)
            ref_cap_attn_mask = torch.cat([query_atts, ref_cap_tokens.attention_mask], dim=1)
            fusion_output = self.Qformer.bert(
                modifier_tokens.input_ids,  # modifier_tokens.input_ids, ref_cap_tokens.input_ids
                query_embeds=query_tokens,  # Qformer里query embeds和 modifier_tokens会拼接
                attention_mask=attention_mask,  # attention_mask, ref_cap_attn_mask
                encoder_hidden_states=image_embeds,  # cross attention
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            text_output = self.Qformer.bert(
                modifier_tokens.input_ids,
                query_embeds=fusion_output.last_hidden_state[:, : query_tokens.size(1), :],  # [b, 32, 768]
                attention_mask=attention_mask,
                return_dict=True,
            )
            fusion_feats = F.normalize(self.text_proj(text_output.last_hidden_state[:, 32, :]), dim=-1)  # [b, 256]
            sim = torch.matmul(
                fusion_feats.unsqueeze(1).unsqueeze(1).to(target_feats.device),
                target_feats.permute(0, 2, 1),
            ).squeeze()
            sim_matrix, _ = sim.max(-1)
            return sim_matrix

        elif mode == "token":
            reference_embeds = samples["img_embeds"]
            ref_cap = samples["ref_cap"]

            image_atts = torch.ones(reference_embeds.size()[:-1], dtype=torch.long).to(reference_embeds.device)  # [b, 257]
            # 覆写机制 query tokens
            before_query_tokens = None
            if "fusion" in samples:
                query_tokens = samples["fusion"].to(reference_embeds.device)
                before_query_tokens = query_tokens.clone().detach()  # query_tokens.clone()
            else:
                query_tokens = self.query_tokens.expand(reference_embeds.shape[0], -1, -1)  # [b, 32, 768]
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(self.device)  # [b, 32]

            ref_cap_tokens = self.tokenizer(
                ref_cap,
                padding="max_length",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(reference_embeds.device)
            ref_cap_attn_mask = torch.cat([query_atts, ref_cap_tokens.attention_mask], dim=1)
            fusion_output = self.Qformer.bert(
                ref_cap_tokens.input_ids,  # modifier_tokens.input_ids,
                query_embeds=query_tokens,  # Qformer里query embeds和modifier_tokens会拼接
                attention_mask=ref_cap_attn_mask,  # attention_mask
                encoder_hidden_states=reference_embeds,  # cross attention
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            query_fusion_tokens = fusion_output.last_hidden_state[:, : query_tokens.size(1), :]  # [b ,32, 768]
            if before_query_tokens is not None:
                query_fusion_tokens = self.prune_tokens(before_query_tokens, query_fusion_tokens)
            return query_fusion_tokens

        elif mode == "target":
            tar_cap = samples["tar_cap"]
            target_embeds = samples["target_embeds"]

            query_tokens = self.query_tokens.expand(target_embeds.shape[0], -1, -1)  # [b, 32, 768]
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(target_embeds.device)  # [b, 32]
            """以下为target融合"""
            tar_cap_tokens = self.tokenizer(
                tar_cap,
                padding="max_length",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(target_embeds.device)

            target_embed_atts = torch.ones(target_embeds.size()[:-1], dtype=torch.long).to(target_embeds.device)
            tar_attention_mask = torch.cat([query_atts, tar_cap_tokens.attention_mask], dim=1)

            tar_fusion_output = self.Qformer.bert(
                tar_cap_tokens.input_ids,
                query_embeds=query_tokens,  # Qformer里query embeds和modifier_tokens会拼接
                attention_mask=tar_attention_mask,
                encoder_hidden_states=target_embeds,  # cross attention
                encoder_attention_mask=target_embed_atts,
                return_dict=True,
            )
            tar_fusion_feats = F.normalize(self.target_proj(tar_fusion_output.last_hidden_state[:, 32, :]), dim=-1)
            return tar_fusion_feats.cpu()

        else:  # mode == "multi
            target_feats = samples["target_feats"]
            modified_text = samples["modified_text"]
            ref_cap = samples["ref_cap"]

            # 覆写机制 query tokens
            query_tokens = samples["fusion"].to(self.device)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(self.device)  # [b, 32]

            # text tokens
            modifier_tokens = self.tokenizer(
                modified_text,
                padding="max_length",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(
                self.device
            )  # modifier_tokens.input_ids [b, 32]
            attention_mask = torch.cat([query_atts, modifier_tokens.attention_mask], dim=1)

            multi_query_tokens = query_tokens
            multi_query_atts = torch.ones(multi_query_tokens.size()[:-1], dtype=torch.long).to(self.device)
            # 修改主要围绕这部分展开：1. caption的引入；2. 前query tokens的self-attention
            ref_cap_tokens = self.tokenizer(
                ref_cap,
                padding="max_length",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(
                self.device
            )  # modifier_tokens.input_ids [b, 32]
            ref_cap_attn_mask = torch.cat([multi_query_atts, ref_cap_tokens.attention_mask], dim=1)
            fusion_output = self.Qformer.bert(
                ref_cap_tokens.input_ids,  # modifier_tokens.input_ids,
                query_embeds=multi_query_tokens,
                attention_mask=ref_cap_attn_mask,  # attention_mask=multi_attn_mask,
                return_dict=True,
            )  # last_hidden_state: [3, 160, 768]

            text_output = self.Qformer.bert(
                modifier_tokens.input_ids,
                query_embeds=fusion_output.last_hidden_state[:, : query_tokens.size(1), :],
                attention_mask=attention_mask,
                return_dict=True,
            )  # last_hidden_state: [3, 160, 768]
            fusion_feats = F.normalize(self.text_proj(text_output.last_hidden_state[:, 32, :]), dim=-1)  # [3, 256]
            sim_matrix = self.compute_distance_matrix(fusion_feats.to(self.device), target_feats)
            return sim_matrix

    @torch.no_grad()
    def extract_target_features(self, image):
        with self.maybe_autocast():
            image_embeds_frozen = self.ln_vision(self.visual_encoder(image))
        image_embeds_frozen = image_embeds_frozen.float()
        image_atts = torch.ones(image_embeds_frozen.size()[:-1], dtype=torch.long).to(self.device)
        query_tokens = self.query_tokens.expand(image_embeds_frozen.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds_frozen,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        image_embeds = query_output.last_hidden_state
        image_features = F.normalize(self.target_proj(image_embeds), dim=-1)  # return image_embeds
        return (
            image_features,
            image_embeds_frozen,
        )  # [batch, 32, 256], [batch, 257, 1408]

    def compute_distance_matrix(self, fusion_feats, target_feats, chunk_size=1024 * 4):
        device = fusion_feats.device  # cuda
        num_fusion = fusion_feats.size(0)
        num_target = target_feats.size(0)
        num_chunks = (num_target + chunk_size - 1) // chunk_size
        # 初始化距离矩阵
        distance_matrix = torch.zeros(num_fusion, num_target).cpu()
        # 分块计算
        for i in range(num_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, num_target)
            chunk_target_feats = target_feats[start:end, :].to(device)
            # 计算当前块与fusion_feats之间的距离矩阵
            chunk_distance = torch.matmul(fusion_feats, chunk_target_feats.T).detach().cpu()
            distance_matrix[:, start:end] = chunk_distance  # 更新总距离矩阵
        return distance_matrix

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
            assert image is not None, "Image is not provided for mode 'image' or 'multimodal'"
            # return query features
            with self.maybe_autocast():
                image_embeds_frozen = self.ln_vision(self.visual_encoder(image))
            image_embeds_frozen = image_embeds_frozen.float()
            image_atts = torch.ones(image_embeds_frozen.size()[:-1], dtype=torch.long).to(self.device)
            query_tokens = self.query_tokens.expand(image_embeds_frozen.shape[0], -1, -1)

            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds_frozen,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            image_embeds = query_output.last_hidden_state
            image_features = F.normalize(self.vision_proj(image_embeds), dim=-1)

        elif mode == "text":
            assert caption is not None, "text input is None for mode 'text' or 'multimodal'"

            # return text features
            text = self.tokenizer(caption, return_tensors="pt", padding=True).to(self.device)

            text_output = self.Qformer.bert(
                text.input_ids,
                attention_mask=text.attention_mask,
                return_dict=True,
            )
            text_embeds = text_output.last_hidden_state
            text_features = self.text_proj(text_embeds)
            text_features = F.normalize(text_features, dim=-1)

        elif mode == "multimodal":
            # return multimodal query features
            with self.maybe_autocast():
                image_embeds_frozen = self.ln_vision(self.visual_encoder(image))
            image_embeds_frozen = image_embeds_frozen.float()
            image_atts = torch.ones(image_embeds_frozen.size()[:-1], dtype=torch.long).to(self.device)
            query_tokens = self.query_tokens.expand(image_embeds_frozen.shape[0], -1, -1)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(self.device)

            text = self.tokenizer(caption, return_tensors="pt", padding=True).to(self.device)
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

    def forward_image(self, image):
        image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        return query_output.last_hidden_state, image_embeds

    def forward_text(self, text_tokens):
        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )
        return text_output.last_hidden_state[:, 0, :]

    def compute_itm(self, image_inputs, text_ids, text_atts):
        image_atts = torch.ones(image_inputs.size()[:-1], dtype=torch.long).to(image_inputs.device)
        query_tokens = self.query_tokens.expand(image_inputs.shape[0], -1, -1)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image_inputs.device)
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

    def compute_sim_matrix(self, data_loader, task_cfg):
        """
        Compute similarity i2t, t2i matrix for the given data loader.
        """
        k_test = task_cfg.k_test

        return compute_sim_matrix(model=self, data_loader=data_loader, k_test=k_test)

    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=3,
        max_length=30,
        min_length=10,
        top_p=0.9,
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
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        model_kwargs = {
            "encoder_hidden_states": image_embeds,
            "encoder_attention_mask": image_atts,
        }

        input_ids = torch.LongTensor(image.size(0), 1).fill_(self.tokenizer.bos_token_id).to(image.device)
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
            **model_kwargs
        )
        captions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return captions
