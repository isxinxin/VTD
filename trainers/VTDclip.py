import os.path as osp
from collections import OrderedDict

from math import sqrt

import math
import einops
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from trainers.vit import ViT

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from trainers.tcn_block import TemporalConvNet

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.ARCH
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    design_details = {"trainer": 'VTD_CLIP',
                      "vision_depth": cfg.TRAINER.VTD_CLIP.PROMPT_DEPTH_VISION,
                      "language_depth": cfg.TRAINER.VTD_CLIP.PROMPT_DEPTH_TEXT,
                      "vision_ctx": cfg.TRAINER.VTD_CLIP.N_CTX_VISION,
                      "language_ctx": cfg.TRAINER.VTD_CLIP.N_CTX_TEXT}
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, return_token=False):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        text_token = x @ self.text_projection.half()

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection.half()

        if return_token:
            return x, text_token
        else:
            return x


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.ln_1 = LayerNorm(d_model)

        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class TemporalModelling(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, dropout: float, attn_mask: torch.Tensor = None, ):
        super(TemporalModelling, self).__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads, dropout, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks((x))

class VLPromptLearner(nn.Module):   #prompt
    def __init__(self, cfg, classnames, clip_model, logger):
        super().__init__()
        dtype = clip_model.dtype
        self.use_prompt_stage = cfg.TRAINER.VTD_CLIP.PROMPT_MODEL  # no prompting
        ctx_init = cfg.TRAINER.VTD_CLIP.CTX_INIT
        self.ZS_evaluation = cfg.TRAINER.VTD_CLIP.ZS_EVAL               # no eval
        if self.ZS_evaluation:
            text_aug = f"{{}}"
            tokenized_prompts = torch.cat([clip.tokenize(text_aug.format(c), context_length=77) for c in classnames])
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype).cuda()
            self.register_buffer("complete_text_embeddings", embedding)
            self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        elif self.use_prompt_stage:
            n_cls = len(classnames)
            # Make sure Language depth >= 1
            assert cfg.TRAINER.VTD_CLIP.PROMPT_DEPTH_TEXT >= 1, "In VL prompting, Language prompt depth should be >=1" \
                                                        "\nPlease use VPT trainer if you want to learn only vision " \
                                                        "branch  "
            n_ctx = cfg.TRAINER.VTD_CLIP.N_CTX_TEXT
            ctx_dim = clip_model.ln_final.weight.shape[0]

            if ctx_init and (n_ctx) <= 4:
                # use given words to initialize context vectors
                ctx_init = ctx_init.replace("_", " ")
                n_ctx = n_ctx
                prompt = clip.tokenize(ctx_init)
                with torch.no_grad():
                    embedding = clip_model.token_embedding(prompt).type(dtype)
                ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
                prompt_prefix = ctx_init
            else:
                # random initialization
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
                nn.init.normal_(ctx_vectors, std=0.02)
                prompt_prefix = " ".join(["X"] * n_ctx)
            logger.info(f"V-L design")
            logger.info(f'Initial text context: "{prompt_prefix}"')
            logger.info(f"Number of context words (tokens) for Language prompting: {n_ctx}")
            logger.info(f"Number of context words (tokens) for Vision prompting: {cfg.TRAINER.VTD_CLIP.N_CTX_VISION}")
            self.ctx = nn.Parameter(ctx_vectors)

            classnames = [name.replace("_", " ") for name in classnames]
            prompts = [prompt_prefix + " " + name + "." for name in classnames]

            tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
            with torch.no_grad():
                embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

            # These token vectors will be saved when in save_model(),
            # but they should be ignored in load_model() as we want to use
            # those computed using the current class names
            self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
            self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS
            self.n_cls = n_cls
            self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        else:
        
            # No prompting
            ctx_init = ctx_init.replace("_", " ")
            prompt_prefix = ctx_init
            prompts = [prompt_prefix + " " + name + "." for name in classnames]
            tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
            with torch.no_grad():
                embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
            self.register_buffer("complete_text_embeddings", embedding)
            self.tokenized_prompts = tokenized_prompts  # torch.Tensor

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self):
        if self.ZS_evaluation:
            prompts = self.complete_text_embeddings
        else:
            ctx = self.ctx
            if ctx.dim() == 2:
                ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

            prefix = self.token_prefix
            suffix = self.token_suffix
            prompts = self.construct_prompts(ctx, prefix, suffix)

        return prompts

class MultiHeadAttention(nn.Module):  
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "Embedding dimension must be divisible by the number of heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        batch_size = query.size(0)

        q = self.q_proj(query)  # B, T, D
        k = self.k_proj(key)    # B, topk, D
        v = self.v_proj(value)  # B, topk, D

        q = q.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # B, H, T, D1
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # B, H, topk, D1
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # B, H, topk, D1

        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # B, H, T, topK
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention weights to value
        out = torch.matmul(attn_weights, v)  # B, H, T, D1

        # Reshape and concatenate heads
        out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.embed_dim)  # B, T, D

        # Final linear projection
        out = self.out_proj(out)

        return out

def VQ(image_features, text_features, weight):
    n_cls, D = text_features.size()
    z_expand = image_features.unsqueeze(2)  # (B, T, 1, 512)
    codebook_expand = text_features.unsqueeze(0).unsqueeze(0)  # (1, 1, class_num, 512)
    distances = torch.sum((z_expand - codebook_expand) ** 2, dim=3)  # (B, T, class_num)
    weight = weight.permute(1, 2, 0)  # (B, T, class_num)
    min_encoding_indices = torch.argmin(distances, dim=2)  # (B, T)
    frame_cls = F.one_hot(min_encoding_indices, num_classes=n_cls)  # (B, T, class_num)
    video_cls = frame_cls * weight  # (B, T, class_num)
    video_cls = video_cls.sum(dim=1)  # (B, class_num)
    topk = 1
    video_cls_topk = torch.topk(video_cls, topk, dim=1)[1]  # (B, topk)

    video_feat = text_features[video_cls_topk]  # (B, topk, D)


    return video_feat


class LRF(nn.Module):
    def __init__(self, cfg, input_dim, output_dim):
        super().__init__()  
        self.flatten = nn.Flatten()
        hidden_dim = 4*input_dim
        self.numF = cfg.DATA.NUM_FRAMES
        self.multihead_attn1 = MultiHeadAttention(512, 8)
        self.fc1 = nn.Linear(input_dim, hidden_dim)  
        self.fc2 = nn.Linear(hidden_dim, output_dim)  

    def forward(self, text, image):
        im_te = self.multihead_attn1(image, text, text)    #(1,D)
        x = self.flatten(im_te)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = torch.unsqueeze(x, dim=1)

        return x


def video_concept_spotting(vid_emb, text_emb):

    vid_emb = vid_emb / vid_emb.norm(dim=-1, keepdim=True)
    text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
    sims = torch.einsum('awd,btd->abwt', [text_emb, vid_emb.half()])
    att_weight_v = F.softmax(sims / 0.01, dim=-1)  # abwt
    att_weight_v = att_weight_v.mean(dim=-2)  # abt

    return att_weight_v

class VTDCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model, logger):
        super().__init__()
        self.prompt_learner = VLPromptLearner(cfg, classnames, clip_model, logger)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.numF = cfg.DATA.NUM_FRAMES
        mlp_factor: float = 4.0
        mlp_dropout: float = 0.5
        embed_dim = 512
        hidden_dim = 2048
        n_layers = 6
        num_heads = 8
        input_dim = 512
        output_dim = 512
        num_chans = [512] * (3) + [512]
        self.lrf = LRF(cfg, input_dim, output_dim)
        
    def forward(self, image):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()
        prompts = self.prompt_learner()
        image = image.cuda()
        text_features, text_tokens = self.text_encoder(prompts, tokenized_prompts, return_token=True)
        tfeatures = text_features.float()
        b, t, c, h, w = image.size()
        image = image.reshape(-1, c, h, w)
        # Now pass the image into CLIP visual encoder
        image_features = self.image_encoder(image)
        image = image_features.view(b, t, -1)

        tf = text_features.unsqueeze(1)  # cls token
        att_weight_v = video_concept_spotting(image, tf)  # n_cls, b, t
        tfeatures = VQ(image, tfeatures, att_weight_v)

        x = torch.split(image, 1, dim=1)
        x = list(x)

        video_feature = []
        for i in range(t):
            out = self.lrf(tfeatures, x[i])
            video_feature.append(out)
        video_feature = torch.cat(video_feature, dim=1)

        x = image + video_feature

        video_feature = x  # b, t, d

        video_feature = video_feature / video_feature.norm(dim=-1, keepdim=True)   # btd
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)   # ad  att_weight_v：abt

        logits = torch.einsum('btd,ad->abt', [video_feature.half(), text_features])
        weighted_sum = torch.sum(logits * att_weight_v, dim=2)
        sum_of_weights = torch.sum(att_weight_v, dim=2)
        logits = weighted_sum / sum_of_weights
        logits = logits.permute(1, 0)

        logits = logit_scale * logits


        return logits


def returnCLIP(config, logger=None,
               class_names=None):
    logger.info(f"Loading CLIP (backbone: {config.MODEL.ARCH})")
    clip_model = load_clip_to_cpu(config)

    logger.info("Building VTD-CLIP CLIP")
    model = VTDCLIP(config, class_names, clip_model, logger)

    if config.TRAINER.VTD_CLIP.PROMPT_MODEL:
        logger.info("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"
        for name, param in model.named_parameters():
            if "image_encoder" in name and "VPT" not in name:
                param.requires_grad_(False)
            elif "text_encoder" in name and "VPT" not in name:
                param.requires_grad_(False)
            else:
                param.requires_grad_(True)
    else:
        # Now need to control freezing of CLIP for fine-tuning
        train_complete_clip = config.TRAINER.VTD_CLIP.USE
        if train_complete_clip == "both":
            logger.info("Turning on gradients for COMPLETE VTD-CLIP model")
            for name, param in model.named_parameters():
                param.requires_grad_(True)
        elif train_complete_clip == "none":
            logger.info("Turning on gradients for COMPLETE VTD-CLIP model")
            for name, param in model.named_parameters():    #freeze CLIP    
                if "image_encoder" in name:  # replace by "text_encoder" incase you want to freeze text
                    param.requires_grad_(False) 
                elif "text_encoder" in name:  # replace by "text_encoder" incase you want to freeze text
                    param.requires_grad_(False)
                else:
                    param.requires_grad_(True)     
        else:
            if train_complete_clip == "image":
                logger.info("Turning on gradients for image side the VTD-CLIP model")
                for name, param in model.named_parameters():
                    if "image_encoder" in name:  # replace by "text_encoder" incase you want to freeze text
                        param.requires_grad_(True)
                    else:
                        param.requires_grad_(False)
            else:
                logger.info("Turning on gradients for TEXT side the VTD-CLIP model")
                for name, param in model.named_parameters():
                    if "text_encoder" in name:  # replace by "text_encoder" incase you want to freeze text
                        param.requires_grad_(True)
                    else:
                        param.requires_grad_(False)
    # Double check
    enabled = set()
    for name, param in model.named_parameters():
        if param.requires_grad:
            enabled.add(name)
            print(name, param.requires_grad)

    logger.info(f"Parameters to be updated: {enabled}")
    logger.info(f"Total learnable items: {len(enabled)}")
    model.float()
    return model
