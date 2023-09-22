import math
import logging
from functools import partial
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import to_2tuple

from lib.models.layers.patch_embed import PatchEmbed
from .utils import combine_tokens, recover_tokens
from .vit import VisionTransformer, Block
#from ..layers.attn_blocks import CEBlock

_logger = logging.getLogger(__name__)


class VisionTransformerMFF(VisionTransformer):

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='',
                 mff_loc=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """
        # super().__init__()
        super().__init__()
        if isinstance(img_size, tuple):
            self.img_size = img_size
        else:
            self.img_size = to_2tuple(img_size)
        self.patch_size = patch_size
        self.in_chans = in_chans

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        self.mff_loc = mff_loc
        proj_layers = [
            torch.nn.Linear(self.embed_dim,self.embed_dim)
            for _ in range(len(self.mff_loc)-1)
        ]
        self.proj_layers = torch.nn.ModuleList(proj_layers)
        self.proj_weights = torch.nn.Parameter(torch.ones(len(self.mff_loc)).view(-1,1,1,1))
        if len(self.mff_loc) == 1:
            self.proj_weights.requires_grad = False

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        blocks = []
        mff_index = 0
        
        for i in range(depth):
            
            if mff_loc is not None and i in mff_loc:
                #ce_keep_ratio_i = ce_keep_ratio[ce_index]
                mff_index += 1

            blocks.append(
                Block(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                    attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            )

        self.blocks = nn.Sequential(*blocks)
        self.norm = norm_layer(embed_dim)

        self.init_weights(weight_init)

    def forward_features(self, z, x):
        B, H, W = x.shape[0], x.shape[2], x.shape[3]

        x = self.patch_embed(x)
        z = self.patch_embed(z)

        if self.add_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            cls_tokens = cls_tokens + self.cls_pos_embed

        z += self.pos_embed_z
        x += self.pos_embed_x

        if self.add_sep_seg:
            x += self.search_segment_pos_embed
            z += self.template_segment_pos_embed

        x = combine_tokens(z, x, mode=self.cat_mode)
        if self.add_cls_token:
            x = torch.cat([cls_tokens, x], dim=1)

        x = self.pos_drop(x)

        res = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)

            if i in self.mff_loc:
                lens_z = self.pos_embed_z.shape[1]
                lens_x = self.pos_embed_x.shape[1]
                x_recover = recover_tokens(x, lens_z, lens_x, mode=self.cat_mode) #1,320,768

                if i != self.mff_loc[-1]:
                    proj_x = self.proj_layers[self.mff_loc.index(i)](x_recover)
                else:
                    proj_x = x_recover
                res.append(proj_x)

        res = torch.stack(res)
        proj_weights = F.softmax(self.proj_weights, dim=0)
        res = res * proj_weights
        res = res.sum(dim=0)
        
        #lens_z = self.pos_embed_z.shape[1]
        #lens_x = self.pos_embed_x.shape[1]
        #x = recover_tokens(x, lens_z, lens_x, mode=self.cat_mode)

        aux_dict = {"attn": None}
        return self.norm(res), aux_dict

    def forward(self, z, x, **kwargs):

        x, aux_dict = self.forward_features(z, x, )

        return x, aux_dict


def _create_vision_transformer(pretrained=False, **kwargs):
    model = VisionTransformerMFF(**kwargs)

    if pretrained:
        if 'npz' in pretrained:
            model.load_pretrained(pretrained, prefix='')
        else:
            checkpoint = torch.load(pretrained, map_location="cpu")
            #missing_keys, unexpected_keys = model.load_state_dict(checkpoint["model"], strict=False)
            
            state = checkpoint["state_dict"]
            new_stat = OrderedDict()
            new_stat = {key: value for key, value in state.items() if 'backbone' in key}
            new_dict = OrderedDict()
            for k,v in new_stat.items():
                parts = k.split('.')
                if len(parts) > 1 and parts[0] == 'backbone':
                    if parts[1] == 'layers':
                        parts[1] = 'blocks'
                    if parts[1] == 'ln1':
                        parts[1] = 'norm'
                    for i in range(1,len(parts)):

                        if parts[i] == 'ln1':
                            parts[i] = 'norm1'
                        if parts[i] == 'ln2':
                            parts[i] = 'norm2'
                        if parts[i] == 'projection':
                            parts[i] = 'proj'
                        if parts[i] == 'ffn':
                            parts[i] = 'mlp'
                        if i == 4 and len(parts) > 6:
                            if parts[4] == 'layers' and parts[5] == '0' and parts[6] == '0':
                                parts[4] = 'fc1'
                                del(parts[6])
                                del(parts[5])
                                new_k = '.'.join(parts[1:])
                                break
                        
                            if parts[4] == 'layers' and parts[5] == '1':
                                parts[4] = 'fc2'
                                del(parts[5])
                                new_k = '.'.join(parts[1:])
                                break
                
                new_k = '.'.join(parts[1:])
                new_dict[new_k] = v
            
            missing_keys, unexpected_keys = model.load_state_dict(new_dict, strict=False)
            print('Load pretrained model from: ' + pretrained)

    return model


def vit_base_patch16_224_mff(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
    return model


def vit_large_patch16_224_mff(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
    return model
