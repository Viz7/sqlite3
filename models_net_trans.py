import torch
import torch.nn as nn
from models_trans import BasicTransformer, LinearTransformer, SparseTransformer
from timm.models.layers import trunc_normal_


def create_block(d_model=192, n_heads=8, d_head=24, dropout=0.1, map_name="elu+1", block_name="basic"):
    if block_name == "basic":
        return BasicTransformer(d_model, n_heads, d_head, dropout=dropout)
    elif block_name == "basic-gated":
        return BasicTransformer(d_model, n_heads, d_head, dropout=dropout, is_gated=True)
    elif block_name == "flash":
        return BasicTransformer(d_model, n_heads, d_head, dropout=dropout, 
                                use_flash_attention=True)
    elif block_name == "flash-gated":
        return BasicTransformer(d_model, n_heads, d_head, dropout=dropout, 
                                use_flash_attention=True, is_gated=True)
    elif block_name == "linear":
        return LinearTransformer(d_model, dropout=dropout, map_name=map_name)
    elif block_name == "sparse":
        return SparseTransformer(d_model, n_heads, dropout=dropout)
    else:
        raise NotImplementedError(f"Block {block_name} not implemented")
    

from timm.models.layers import trunc_normal_, lecun_normal_
import math
import time
from functools import partial


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def segm_init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.Conv2d, nn.Conv1d)):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


class StrideEmbed(nn.Module):
    def __init__(self, arr_length=1600, stride_size=4, in_chans=1, embed_dim=192):
        super().__init__()
        assert arr_length % stride_size == 0
        self.num_patches = arr_length // stride_size
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=stride_size, stride=stride_size)
        
    def forward(self, x):
        """
        x: [B, N]
        """
        return self.proj(x).transpose(1, 2) # [B, N, D]


class NetTransformer(nn.Module):
    def __init__(self, 
                 arr_length=1600,
                 stride_size=4,
                 in_chans=1,
                 embed_dim=192, depth=4, 
                 decoder_embed_dim=128, decoder_depth=2,
                 num_classes=1000,
                 n_heads=8, block_name="basic",
                 norm_pix_loss=False,
                 drop_rate=0.,
                 is_pretrain=False,
                 if_cls_token=True,
                 device=None, dtype=None,
                 **kwargs):
        super().__init__()

        factory_kwargs = {"device": device, "dtype": dtype}
        # add factory_kwargs into kwargs
        kwargs.update(factory_kwargs) 
        self.num_classes = num_classes
        self.d_model = self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.is_pretrain = is_pretrain

        # --------------------------------------------------------------------------
        # NetMamba encoder specifics
        self.patch_embed = StrideEmbed(arr_length=arr_length, stride_size=stride_size, embed_dim=embed_dim)
        self.num_patches = self.patch_embed.num_patches
        self.if_cls_token = if_cls_token
        if if_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.num_cls_token = 1
        else:
            self.num_cls_token = 0
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + self.num_cls_token, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        # Mamba blocks
        self.blocks = nn.ModuleList([
            create_block(d_model=embed_dim, n_heads=n_heads, d_head=embed_dim // n_heads, dropout=0.1, block_name=block_name)
            for _ in range(depth)])
        # --------------------------------------------------------------------------

        if is_pretrain:
            # --------------------------------------------------------------------------
            # NetMamba decoder specifics
            self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
            self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
            self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + self.num_cls_token, decoder_embed_dim))
            self.decoder_blocks = nn.ModuleList([
                create_block(d_model=decoder_embed_dim, n_heads=n_heads, d_head=decoder_embed_dim // n_heads, dropout=0.1, block_name=block_name)
                for _ in range(decoder_depth)])
            self.decoder_pred = nn.Linear(decoder_embed_dim, stride_size * in_chans, bias=True)  # decoder to stride
            # --------------------------------------------------------------------------
        else:
            # --------------------------------------------------------------------------
            # NetMamba classifier specifics
            self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
            # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss
        self.initialize_weights(depth)

    def initialize_weights(self, depth):
        self.patch_embed.apply(segm_init_weights)
        if not self.is_pretrain:
            self.head.apply(segm_init_weights)
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        if self.is_pretrain:
            trunc_normal_(self.decoder_pos_embed, std=.02)
            trunc_normal_(self.mask_token, std=.02)
        # initialize nn.Linear and nn.LayerNorm
        self.apply(partial(_init_weights, n_layer=depth,))
    
    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "dist_token", "cls_token_head", "cls_token_tail"}
    
    def stride_patchify(self, imgs, stride_size=4):
        """
        imgs: (N, 1, H, W)
        x: (N, L, patch_size**2 *1)
        """
        B, C, H, W = imgs.shape
        assert C == 1, "Input images should be grayscale"
        x = imgs.reshape(B, H*W // stride_size, stride_size)
        return x

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [B N D], sequence
        """
        B, N, D = x.shape  # batch, length, dim
        len_keep = int(N * (1 - mask_ratio))

        noise = torch.rand(B, N, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1) # ids_restore[i] = i-th noise element's rank

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D)) # x_masked are acctually non-masked elements

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([B, N], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio, if_mask=True,):
        """
        x: [B, 1, H, W]
        """
        # embed patches
        B, C, H, W = x.shape
        x = self.patch_embed(x.reshape(B, C, -1))

        # add pos embed w/o cls token
        if self.if_cls_token:
            x = x + self.pos_embed[:, :-1, :]
        else:
            x = x + self.pos_embed

        # masking: length -> length * mask_ratio
        if if_mask:
            x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        if self.if_cls_token:
            cls_token = self.cls_token + self.pos_embed[:, -1, :]
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((x, cls_tokens), dim=1)
        x = self.pos_drop(x)

        # apply Mamba blocks
        for blk in self.blocks:
            x = blk(x)
        if if_mask:
            return x, mask, ids_restore
        else:
            return x

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + self.num_cls_token - x.shape[1], 1)
        if self.if_cls_token:
            visible_tokens = x[:, :-1, :]
        else:
            visible_tokens = x
        x_ = torch.cat([visible_tokens, mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        if self.if_cls_token:
            x = torch.cat([x_, x[:, -1:, :]], dim=1)  # append cls token
        else:
            x = x_

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Mamba blocks
        for blk in self.decoder_blocks:
            x = blk(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        if self.if_cls_token:
            x = x[:, :-1, :]
        return x

    def forward_rec_loss(self, imgs, pred, mask):
        """
        imgs: [N, 1, H, W]
        pred: [N, L, p*p*1]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.stride_patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, mask_ratio=0.9, **kwargs):
        # imgs: [B, 1, H, W]
        B, C, H, W = imgs.shape
        assert C == 1, "Input images should be grayscale"
        if self.is_pretrain:
            latent, mask, ids_restore = self.forward_encoder(imgs, 
                    mask_ratio=mask_ratio,)
            pred = self.forward_decoder(latent, ids_restore)
            loss = self.forward_rec_loss(imgs, pred, mask)
            return loss, pred, mask
        else:
            x = self.forward_encoder(imgs, mask_ratio=mask_ratio, if_mask=False)
            if self.if_cls_token:
                return self.head(x[:, -1, :])
            else:
                return self.head(torch.mean(x, dim=1))


def net_bt_base_pretrain(**kwargs):
    model = NetTransformer(
        is_pretrain=True, embed_dim=192, depth=4,
        decoder_embed_dim=128, decoder_depth=2, **kwargs)
    return model

def net_bt_base_classifier(**kwargs):
    model = NetTransformer(
        is_pretrain=False, embed_dim=192, depth=4,
        **kwargs)
    return model

def net_bt_medium_pretrain(**kwargs):
    model = NetTransformer(
        is_pretrain=True, embed_dim=256, depth=4,
        decoder_embed_dim=128, decoder_depth=2, **kwargs)
    return model

def net_bt_meidum_classifier(**kwargs):
    model = NetTransformer(
        is_pretrain=False, embed_dim=256, depth=4,
        **kwargs)
    return model

def net_bgt_base_pretrain(**kwargs):
    model = NetTransformer(
        is_pretrain=True, embed_dim=192, depth=4,
        decoder_embed_dim=128, decoder_depth=2, block_name="basic-gated", **kwargs)
    return model

def net_bgt_base_classifier(**kwargs):
    model = NetTransformer(
        is_pretrain=False, embed_dim=192, depth=4, block_name="basic-gated",
        **kwargs)
    return model

def net_bgt_medium_pretrain(**kwargs):
    model = NetTransformer(
        is_pretrain=True, embed_dim=256, depth=4,
        decoder_embed_dim=128, decoder_depth=2, block_name="basic-gated", **kwargs)
    return model

def net_bgt_medium_classifier(**kwargs):
    model = NetTransformer(
        is_pretrain=False, embed_dim=256, depth=4, block_name="basic-gated",
        **kwargs)
    return model

def net_ft_base_pretrain(**kwargs):
    model = NetTransformer(
        is_pretrain=True, embed_dim=192, depth=4,
        decoder_embed_dim=128, decoder_depth=2, block_name="flash", **kwargs)
    return model

def net_ft_base_classifier(**kwargs):
    model = NetTransformer(
        is_pretrain=False, embed_dim=192, depth=4,
        block_name="flash", **kwargs)
    return model

def net_fgt_base_pretrain(**kwargs):
    model = NetTransformer(
        is_pretrain=True, embed_dim=192, depth=4,
        decoder_embed_dim=128, decoder_depth=2, block_name="flash-gated", **kwargs)
    return model

def net_fgt_base_classifier(**kwargs):
    model = NetTransformer(
        is_pretrain=False, embed_dim=192, depth=4,
        block_name="flash-gated", **kwargs)
    return model

def net_fgt_medium_pretrain(**kwargs):
    model = NetTransformer(
        is_pretrain=True, embed_dim=256, depth=4,
        decoder_embed_dim=128, decoder_depth=2, block_name="flash-gated", **kwargs)
    return model

def net_fgt_medium_classifier(**kwargs):
    model = NetTransformer(
        is_pretrain=False, embed_dim=256, depth=4,
        block_name="flash-gated", **kwargs)
    return model

def net_lt_base_pretrain(**kwargs):
    model = NetTransformer(
        is_pretrain=True, embed_dim=192, depth=4,
        decoder_embed_dim=128, decoder_depth=2, block_name="linear", **kwargs)
    return model

def net_lt_base_classifier(**kwargs):
    model = NetTransformer(
        is_pretrain=False, embed_dim=192, depth=4,
        block_name="linear", **kwargs)
    return model

def net_st_base_pretrain(**kwargs):
    model = NetTransformer(
        is_pretrain=True, embed_dim=192, depth=4,
        decoder_embed_dim=128, decoder_depth=2, block_name="sparse", 
        if_cls_token=False, **kwargs)
    return model

def net_st_base_classifier(**kwargs):
    model = NetTransformer(
        is_pretrain=False, embed_dim=192, depth=4,
        block_name="sparse", if_cls_token=False, **kwargs)
    return model
