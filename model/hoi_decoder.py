# --------------------------------------------------------
# Modified from X-Decoder(https://github.com/microsoft/X-Decoder)
# --------------------------------------------------------

import torch
from torch import nn
from torch.nn import functional as F
from itertools import permutations

from timm.models.layers import trunc_normal_

import fvcore.nn.weight_init as weight_init

from .transformer_blocks import SelfAttentionLayer, CrossAttentionLayer, FFNLayer, MLP

from .position_encoding import PositionEmbeddingSine
from .lang_encoder import get_language_model
from datasets.hico_text_label import coco_panoptic_names, hico_object_classes, hico_verb_names, hico_seen_object_classes
from datasets.psg_text_label import psg_obj_names, psg_verb_names
from datasets.vcoco_text_label import vcoco_verb_names
from datasets.vrd_text_label import seen_object_names, full_object_names

class HOIDecoder(nn.Module):

    def __init__(
        self,
        dataset_name,
        is_eval,
        lang_encoder: nn.Module,
        in_channels,
        mask_classification=True,
        *,
        hidden_dim: int,
        dim_proj: int,
        num_queries: int,
        contxt_len: int,
        nheads: int,
        dim_feedforward: int,
        dec_layers: int,
        pre_norm: bool,
        mask_dim: int,
        task_switch: dict,
        use_triplet: bool, 
        enforce_input_project: bool,
        unseen_type: str,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            in_channels: channels of the input features
            mask_classification: whether to add mask classifier or not
            num_classes: number of classes
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
            nheads: number of heads
            dim_feedforward: feature dimension in feedforward network
            enc_layers: number of Transformer encoder layers
            dec_layers: number of Transformer decoder layers
            pre_norm: whether to use pre-LayerNorm or not
            mask_dim: mask feature dimension
            enforce_input_project: add input project 1x1 conv even if input
                channels and hidden dim is identical
        """
        super().__init__()
        assert mask_classification, "Only support mask classification model"
        self.mask_classification = mask_classification

        self.use_triplet = use_triplet

        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
        
        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.contxt_len = contxt_len
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.num_queries = num_queries
        # learnable query features
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        # learnable query p.e.
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        
        # level embedding (we always use 3 scales)
        self.num_feature_levels = 3
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()
        
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())

        self.task_switch = task_switch

        self.lang_encoder = lang_encoder
        self.dataset_name = dataset_name
        self.is_eval = is_eval

        self.unseen_type = unseen_type
        
        if self.task_switch['mask']:
            self.mask_embed_sub = MLP(hidden_dim, hidden_dim, mask_dim, 3)
            self.mask_embed_obj = MLP(hidden_dim, hidden_dim, mask_dim, 3)


        self.class_embed_obj = nn.Parameter(torch.empty(hidden_dim, dim_proj))
        self.class_embed_verb = nn.Parameter(torch.empty(hidden_dim, dim_proj))
        trunc_normal_(self.class_embed_obj, std=.02)
        trunc_normal_(self.class_embed_verb, std=.02)

        if self.use_triplet:
            self.class_embed_hoi = nn.Parameter(torch.empty(hidden_dim, dim_proj))
            trunc_normal_(self.class_embed_hoi, std=.02)

        if self.task_switch['psg'] or self.task_switch['vrd']:
            self.class_embed_sub = nn.Parameter(torch.empty(hidden_dim, dim_proj))
            trunc_normal_(self.class_embed_sub, std=.02)

        # register self_attn_mask to avoid information leakage, it includes interaction between object query, class query and caping query
        self_attn_mask = torch.zeros((1, num_queries, num_queries)).bool()
        self.register_buffer("self_attn_mask", self_attn_mask)

              
    def forward(self, x, mask_features, use_verb_temp, mask=None, target_queries=None, task='seg', extra={}):
        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []
        if self.dataset_name == 'coco_panoptic':
            self.lang_encoder.get_text_embeddings(coco_panoptic_names, use_verb_temp, name='coco_panoptic', is_eval=self.is_eval)
        elif self.dataset_name == 'hico':
            if self.unseen_type == 'uo' and not self.is_eval:
                self.lang_encoder.get_text_embeddings(hico_seen_object_classes, use_verb_temp, name='hico_obj', is_eval=self.is_eval, add_bgd=True)
            else:
                self.lang_encoder.get_text_embeddings(hico_object_classes, use_verb_temp, name='hico_obj', is_eval=self.is_eval, add_bgd=True)
            
            if self.unseen_type == 'uv' and not self.is_eval:
                self.lang_encoder.get_text_embeddings(hico_verb_names, use_verb_temp, name='hico_verb_seen', is_eval=self.is_eval, use_triplet=self.use_triplet)
            else:
                self.lang_encoder.get_text_embeddings(hico_verb_names, use_verb_temp, name='hico_verb', is_eval=self.is_eval, use_triplet=self.use_triplet)

        elif self.dataset_name == 'psg':
            self.lang_encoder.get_text_embeddings(psg_obj_names, use_verb_temp, name='psg_obj', is_eval=self.is_eval, add_bgd=True)
            self.lang_encoder.get_text_embeddings(psg_obj_names, use_verb_temp, name='psg_sub', is_eval=self.is_eval, add_bgd=True)
            self.lang_encoder.get_text_embeddings(psg_verb_names, use_verb_temp, name='psg_verb', is_eval=self.is_eval, add_bgd=True)
        
        elif self.dataset_name == 'vrd':
            if self.is_eval:
                self.lang_encoder.get_text_embeddings(full_object_names, use_verb_temp, name='vrd_obj', is_eval=self.is_eval, add_bgd=True)
                self.lang_encoder.get_text_embeddings(full_object_names, use_verb_temp, name='vrd_sub', is_eval=self.is_eval, add_bgd=True)
            else:
                self.lang_encoder.get_text_embeddings(seen_object_names, use_verb_temp, name='vrd_obj', is_eval=self.is_eval, add_bgd=True)
                self.lang_encoder.get_text_embeddings(seen_object_names, use_verb_temp, name='vrd_sub', is_eval=self.is_eval, add_bgd=True)
            self.lang_encoder.get_text_embeddings(psg_verb_names, use_verb_temp, name='vrd_verb', is_eval=self.is_eval)

        elif self.dataset_name == 'vcoco':
            self.lang_encoder.get_text_embeddings(vcoco_verb_names, use_verb_temp, name='vcoco_verb', is_eval=self.is_eval)
            if self.use_triplet:
                self.lang_encoder.get_text_embeddings(vcoco_verb_names, use_verb_temp, name='vcoco_hoi', is_eval=self.is_eval, use_triplet=self.use_triplet) # add_bgd=False
            self.lang_encoder.get_text_embeddings(hico_object_classes, use_verb_temp, name='vcoco_obj', is_eval=self.is_eval, add_bgd=True)
        elif self.dataset_name == 'hico+vcoco':
            if extra['source'] == 'hico':
                self.lang_encoder.get_text_embeddings(hico_object_classes, use_verb_temp, name='hico_obj', is_eval=self.is_eval, add_bgd=True)
                self.lang_encoder.get_text_embeddings(hico_verb_names, use_verb_temp, name='hico_verb', is_eval=self.is_eval, use_triplet=False)
            elif extra['source'] == 'vcoco':
                self.lang_encoder.get_text_embeddings(vcoco_verb_names, use_verb_temp, name='vcoco_verb', is_eval=self.is_eval)
                self.lang_encoder.get_text_embeddings(hico_object_classes, use_verb_temp, name='vcoco_obj', is_eval=self.is_eval, add_bgd=True)
                if self.use_triplet:
                    self.lang_encoder.get_text_embeddings(vcoco_verb_names, use_verb_temp, name='vcoco_hoi', is_eval=self.is_eval, use_triplet=self.use_triplet)
        elif self.dataset_name == 'hico+psg':
            if extra['source'] == 'hico':
                self.lang_encoder.get_text_embeddings(hico_object_classes, use_verb_temp, name='hico_obj', is_eval=self.is_eval, add_bgd=True)
                self.lang_encoder.get_text_embeddings(hico_verb_names, use_verb_temp, name='hico_verb', is_eval=self.is_eval, use_triplet=False)
            elif extra['source'] == 'psg':
                self.lang_encoder.get_text_embeddings(psg_obj_names, use_verb_temp, name='psg_obj', is_eval=self.is_eval, add_bgd=True)
                self.lang_encoder.get_text_embeddings(psg_obj_names, use_verb_temp, name='psg_sub', is_eval=self.is_eval, add_bgd=True)
                self.lang_encoder.get_text_embeddings(psg_verb_names, use_verb_temp, name='psg_verb', is_eval=self.is_eval, add_bgd=True)
        elif self.dataset_name == 'hico+vcoco+psg':
            if extra['source'] == 'hico':
                self.lang_encoder.get_text_embeddings(hico_object_classes, use_verb_temp, name='hico_obj', is_eval=self.is_eval, add_bgd=True)
                self.lang_encoder.get_text_embeddings(hico_verb_names, use_verb_temp, name='hico_verb', is_eval=self.is_eval, use_triplet=False)
            elif extra['source'] == 'psg':
                self.lang_encoder.get_text_embeddings(psg_obj_names, use_verb_temp, name='psg_obj', is_eval=self.is_eval, add_bgd=True)
                self.lang_encoder.get_text_embeddings(psg_obj_names, use_verb_temp, name='psg_sub', is_eval=self.is_eval, add_bgd=True)
                self.lang_encoder.get_text_embeddings(psg_verb_names, use_verb_temp, name='psg_verb', is_eval=self.is_eval, add_bgd=True)
            elif extra['source'] == 'vcoco':
                self.lang_encoder.get_text_embeddings(vcoco_verb_names, use_verb_temp, name='vcoco_verb', is_eval=self.is_eval)
                self.lang_encoder.get_text_embeddings(hico_object_classes, use_verb_temp, name='vcoco_obj', is_eval=self.is_eval, add_bgd=True)
                if self.use_triplet:
                    self.lang_encoder.get_text_embeddings(vcoco_verb_names, use_verb_temp, name='vcoco_hoi', is_eval=self.is_eval, use_triplet=self.use_triplet)

        
        # disable mask, it does not affect performance
        del mask
        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i], None).flatten(2))
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

            # flatten BxCxHxW to HWxBxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        _, bs, _ = src[0].shape # HWxBxC

        # QxBxC
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

        predictions_obj_class = []
        predictions_verb_class = []

        if self.task_switch['mask']:
            predictions_sub_mask = []
            predictions_obj_mask = []

        if self.use_triplet and extra['source']=='vcoco':
            predictions_hoi_class = []
        if self.task_switch['psg'] and (extra['source']=='psg' or extra['source']=='o365'):
            predictions_sub_class = []
        if self.task_switch['vrd']:
            predictions_sub_class = []
        if self.task_switch['grounding']:
            predictions_obj_class_emd = []  
            predictions_verb_class_emd = []   
            if self.task_switch['psg']:
                predictions_sub_class_emd = [] 
            if self.task_switch['vrd']:
                predictions_sub_class_emd = [] 

        self_tgt_mask = None
        if self.task_switch['grounding']:
            self_tgt_mask = self.self_attn_mask[:,:self.num_queries,:self.num_queries].repeat(output.shape[1]*self.num_heads, 1, 1)
            grounding_tokens = extra['grounding_tokens']
            _grounding_tokens = grounding_tokens.detach().clone()
            # initialize with negative attention at the beginning.
            pad_tgt_mask = torch.ones((1, self.num_queries + self.num_queries + len(grounding_tokens), self.num_queries + self.num_queries + len(grounding_tokens)), device=self_tgt_mask.device).bool().repeat(output.shape[1]*self.num_heads, 1, 1)
            pad_tgt_mask[:,:self.num_queries,:self.num_queries] = self_tgt_mask
            pad_tgt_mask[:,self.num_queries:,self.num_queries:] = False # grounding tokens could attend with eatch other
            self_tgt_mask = pad_tgt_mask
            output = torch.cat((output, output), dim=0)
            query_embed = torch.cat((query_embed, query_embed), dim=0) # also pad language embdding to fix embedding
        
        # prediction heads on learnable query features
        results = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[0], task=task, extra=extra)
        attn_mask = results["attn_mask"]
        if self.task_switch['psg'] and (extra['source']=='psg' or extra['source']=='o365'):
            predictions_sub_class.append(results["outputs_sub_class"])
        if self.task_switch['vrd']:
            predictions_sub_class.append(results["outputs_sub_class"])

        predictions_obj_class.append(results["outputs_obj_class"])
        predictions_verb_class.append(results["outputs_verb_class"])

        if self.use_triplet and extra['source']=='vcoco':
            predictions_hoi_class.append(results["outputs_hoi_class"])

        if self.task_switch['mask']:
            predictions_sub_mask.append(results["outputs_sub_mask"])
            predictions_obj_mask.append(results["outputs_obj_mask"])

        if self.task_switch['grounding']:
            predictions_obj_class_emd.append(results["outputs_obj_class_emd"])
            predictions_verb_class_emd.append(results["outputs_verb_class_emd"])
            if self.task_switch['psg'] or self.task_switch['vrd']:
                predictions_sub_class_emd.append(results["outputs_sub_class_emd"])

        
        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False

            # attention: cross-attention first
            output, avg_attn = self.transformer_cross_attention_layers[i](
                output, src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos[level_index], query_pos=query_embed
            )

            if self.task_switch['grounding']:
                output = torch.cat((output, _grounding_tokens), dim=0)
                query_embed = torch.cat((query_embed, grounding_tokens), dim=0)

            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=self_tgt_mask,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )
            
            # FFN
            output = self.transformer_ffn_layers[i](output)

            if self.task_switch['grounding']:
                _grounding_tokens = output[-len(_grounding_tokens):]
                output = output[:-len(_grounding_tokens)]
                query_embed = query_embed[:-len(_grounding_tokens)]

            results = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels], layer_id=i, task=task, extra=extra)
            attn_mask = results["attn_mask"]
            if self.task_switch['psg'] and (extra['source']=='psg' or extra['source']=='o365'):
                predictions_sub_class.append(results["outputs_sub_class"])
            if self.task_switch['vrd']:
                predictions_sub_class.append(results["outputs_sub_class"])
            predictions_obj_class.append(results["outputs_obj_class"])
            predictions_verb_class.append(results["outputs_verb_class"])

            if self.task_switch['mask']:
                predictions_sub_mask.append(results["outputs_sub_mask"])
                predictions_obj_mask.append(results["outputs_obj_mask"])
            
            if self.use_triplet and extra['source']=='vcoco':
                predictions_hoi_class.append(results["outputs_hoi_class"])
            if self.task_switch['grounding']:
                predictions_obj_class_emd.append(results["outputs_obj_class_emd"])
                predictions_verb_class_emd.append(results["outputs_verb_class_emd"])
                if self.task_switch['psg'] or self.task_switch['vrd']:
                    predictions_sub_class_emd.append(results["outputs_sub_class_emd"])

        assert len(predictions_obj_class) == self.num_layers + 1
        
        if self.task_switch['psg'] and (extra['source']=='psg' or extra['source']=='o365'):
            out = {
                'pred_sub_logits': predictions_sub_class[-1], # bx100x81
                'pred_obj_logits': predictions_obj_class[-1], # bx100x81
                'pred_verb_logits': predictions_verb_class[-1], # bx100x116
                'pred_sub_masks': predictions_sub_mask[-1], # bx100x160x160
                'pred_obj_masks': predictions_obj_mask[-1],
            }
            if self.task_switch['grounding']:
                out['pred_obj_class_emd'] = predictions_obj_class_emd[-1]
                out['pred_verb_class_emd'] = predictions_verb_class_emd[-1]
                out['pred_sub_class_emd'] = predictions_sub_class_emd[-1]
            
        elif self.task_switch['vrd']:
            out = {
                'pred_sub_logits': predictions_sub_class[-1], # bx100x81
                'pred_obj_logits': predictions_obj_class[-1], # bx100x81
                'pred_verb_logits': predictions_verb_class[-1], # bx100x116
                'pred_sub_masks': predictions_sub_mask[-1], # bx100x160x160
                'pred_obj_masks': predictions_obj_mask[-1]
            }
            if self.task_switch['grounding']:
                out['pred_obj_class_emd'] = predictions_obj_class_emd[-1]
                out['pred_verb_class_emd'] = predictions_verb_class_emd[-1]
                out['pred_sub_class_emd'] = predictions_sub_class_emd[-1]
        elif self.task_switch['hoi'] and (extra['source']=='hico' or extra['source']=='vcoco'):
            out = {
                'pred_obj_logits': predictions_obj_class[-1], # bx100x81
                'pred_verb_logits': predictions_verb_class[-1], # bx100x116
            }
            if self.task_switch['mask']:
                out['pred_sub_masks'] = predictions_sub_mask[-1]
                out['pred_obj_masks'] = predictions_obj_mask[-1]

            if self.task_switch['grounding']:
                out['pred_obj_class_emd'] = predictions_obj_class_emd[-1]
                out['pred_verb_class_emd'] = predictions_verb_class_emd[-1]
            
            if self.use_triplet and extra['source']=='vcoco':
                out['pred_hoi_logits'] = predictions_hoi_class[-1]
            
        return out


    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size, layer_id=-1, task='seg', extra={}):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)

        # compute class, mask and bbox.
        class_embed_obj = decoder_output @ self.class_embed_obj
        class_embed_verb = decoder_output @ self.class_embed_verb

        if self.use_triplet and extra['source']=='vcoco':
            class_embed_hoi = decoder_output @ self.class_embed_hoi

        
        if self.task_switch['psg'] and (extra['source']=='psg' or extra['source']=='o365'):
            class_embed_sub = decoder_output @ self.class_embed_sub
        if self.task_switch['vrd']:
            class_embed_sub = decoder_output @ self.class_embed_sub
        
        if self.task_switch['hoi'] and (extra['source']=='hico' or extra['source']=='vcoco'):
            if self.dataset_name == 'hico':
                
                outputs_obj_class = self.lang_encoder.compute_similarity(class_embed_obj, name='hico_obj')

                if self.unseen_type=='uv' and not self.is_eval:
                    outputs_verb_class = self.lang_encoder.compute_similarity(class_embed_verb, name='hico_verb_seen')
                else:
                    outputs_verb_class = self.lang_encoder.compute_similarity(class_embed_verb, name='hico_verb')
                if self.use_triplet:
                    outputs_hoi_class = self.lang_encoder.compute_similarity(class_embed_hoi, name='hico_hoi')
            elif self.dataset_name == 'vcoco':
                outputs_obj_class = self.lang_encoder.compute_similarity(class_embed_obj, name='vcoco_obj')
                outputs_verb_class = self.lang_encoder.compute_similarity(class_embed_verb, name='vcoco_verb')
                if self.use_triplet:
                    outputs_hoi_class = self.lang_encoder.compute_similarity(class_embed_hoi, name='vcoco_hoi')
            elif (self.dataset_name == 'hico+vcoco' or self.dataset_name == 'hico+psg' or self.dataset_name == 'hico+vcoco+psg') and extra['source']=='hico':
                outputs_obj_class = self.lang_encoder.compute_similarity(class_embed_obj, name='hico_obj')
                outputs_verb_class = self.lang_encoder.compute_similarity(class_embed_verb, name='hico_verb')
            elif (self.dataset_name == 'hico+vcoco' or self.dataset_name == 'hico+vcoco+psg') and extra['source']=='vcoco':
                outputs_obj_class = self.lang_encoder.compute_similarity(class_embed_obj, name='vcoco_obj')
                outputs_verb_class = self.lang_encoder.compute_similarity(class_embed_verb, name='vcoco_verb')
                if self.use_triplet:
                    outputs_hoi_class = self.lang_encoder.compute_similarity(class_embed_hoi, name='vcoco_hoi')
        if self.task_switch['psg'] and (extra['source']=='psg' or extra['source']=='o365'):
            if extra['source']=='psg':
                outputs_sub_class = self.lang_encoder.compute_similarity(class_embed_sub, name='psg_sub')
                outputs_obj_class = self.lang_encoder.compute_similarity(class_embed_obj, name='psg_obj')
                
                # outputs_verb_class = self.verb_classifier(decoder_output)
                outputs_verb_class = self.lang_encoder.compute_similarity(class_embed_verb, name='psg_verb')
        
        if self.task_switch['vrd']:
            outputs_sub_class = self.lang_encoder.compute_similarity(class_embed_sub, name='vrd_sub')
            outputs_obj_class = self.lang_encoder.compute_similarity(class_embed_obj, name='vrd_obj')
            outputs_verb_class = self.lang_encoder.compute_similarity(class_embed_verb, name='vrd_verb')

        if self.task_switch['mask']:
            mask_embed_sub = self.mask_embed_sub(decoder_output)
            outputs_mask_sub = torch.einsum("bqc,bchw->bqhw", mask_embed_sub, mask_features)

            mask_embed_obj = self.mask_embed_obj(decoder_output)
            outputs_mask_obj = torch.einsum("bqc,bchw->bqhw", mask_embed_obj, mask_features)

            # NOTE: prediction is of higher-resolution
            # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
            attn_mask_sub = F.interpolate(outputs_mask_sub, size=attn_mask_target_size, mode="bicubic", align_corners=False, antialias=True)
            attn_mask_obj = F.interpolate(outputs_mask_obj, size=attn_mask_target_size, mode="bicubic", align_corners=False, antialias=True)

            # must use bool type
            # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
            attn_mask_sub = (attn_mask_sub.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5).bool()
            attn_mask_obj = (attn_mask_obj.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5).bool()
            attn_mask = torch.logical_and(attn_mask_sub, attn_mask_obj)
            attn_mask = attn_mask.detach()
        else:
            attn_mask = torch.zeros((list(decoder_output.shape[:2]) + [attn_mask_target_size[0]*attn_mask_target_size[1]]), device=decoder_output.device).repeat(self.num_heads, 1, 1).bool()

        if self.task_switch['hoi'] and extra['source']!='psg':
            results = {
                "outputs_obj_class": outputs_obj_class,
                "outputs_verb_class": outputs_verb_class,
                "attn_mask": attn_mask,
            }
            if self.task_switch['mask']:
                results["outputs_sub_mask"] = outputs_mask_sub
                results["outputs_obj_mask"] = outputs_mask_obj

            if self.use_triplet and extra['source']=='vcoco':
                outputs_hoi_class = self.lang_encoder.compute_similarity(class_embed_hoi, name='vcoco_hoi')
                results["outputs_hoi_class"] = outputs_hoi_class
            if self.task_switch['grounding']:
                results["outputs_obj_class_emd"] = class_embed_obj
                results["outputs_verb_class_emd"] = class_embed_verb

        elif self.task_switch['psg'] and (extra['source']=='psg' or extra['source']=='o365'):
            results = {
                "outputs_sub_class": outputs_sub_class,
                "outputs_obj_class": outputs_obj_class,
                "outputs_verb_class": outputs_verb_class,
                "outputs_sub_mask": outputs_mask_sub,
                "outputs_obj_mask": outputs_mask_obj,
                "attn_mask": attn_mask,
            }
            if self.task_switch['grounding']:
                results["outputs_obj_class_emd"] = class_embed_obj
                results["outputs_verb_class_emd"] = class_embed_verb
                results["outputs_sub_class_emd"] = class_embed_sub
        elif self.task_switch['vrd']:
            results = {
                "outputs_sub_class": outputs_sub_class,
                "outputs_obj_class": outputs_obj_class,
                "outputs_verb_class": outputs_verb_class,
                "outputs_sub_mask": outputs_mask_sub,
                "outputs_obj_mask": outputs_mask_obj,
                "attn_mask": attn_mask,
            }
            if self.task_switch['grounding']:
                results["outputs_obj_class_emd"] = class_embed_obj
                results["outputs_verb_class_emd"] = class_embed_verb
                results["outputs_sub_class_emd"] = class_embed_sub
        return results

    @torch.jit.unused
    def _set_aux_loss(self, predictions_sub_class, predictions_obj_class, predictions_verb_class, predictions_sub_mask, predictions_obj_mask):
    
        if self.task_switch['psg']:
            return [
                {"pred_sub_logits": a, "pred_obj_logits": b, "pred_verb_logits": c, "pred_sub_masks": d, "pred_obj_masks": e}
                for a, b, c, d, e in zip(predictions_sub_class[:-1], predictions_obj_class[:-1], predictions_verb_class[:-1], predictions_sub_mask[:-1], predictions_obj_mask[:-1])
            ]

def build_hoi_decoder(args, matcher):
    task_switch = {'mask': False, 'box': False, 'psg': False, 'hoi': False, 'grounding': False, 'vrd': False}
    if args.psg:
        task_switch['psg'] = True
        task_switch['mask'] = True
    if args.vrd:
        task_switch['vrd'] = True
    if args.hoi:
        task_switch['hoi'] = True
        if args.use_mask:
            task_switch['mask'] = True
        if args.use_box:
            task_switch['box'] = True


    if args.flexible_grounding:
        task_switch['grounding'] = True
    
    flexible_decoder = HOIDecoder(dataset_name=args.dataset_file,
                                is_eval=args.eval,
                                lang_encoder=get_language_model(args).to(args.device),
                                in_channels=args.hoi_dec_in_channels, \
                                mask_classification=args.mask_classification, \
                                hidden_dim=args.hidden_dim, \
                                dim_proj=args.dim_proj, \
                                num_queries=args.num_queries, \
                                contxt_len=args.contxt_len, \
                                nheads=args.hoi_dec_nheads, \
                                dim_feedforward=args.transformer_dim_feedforward, \
                                dec_layers=args.hoi_dec_layers, \
                                pre_norm=args.transformer_pre_norm, \
                                mask_dim=args.mask_dim, \
                                task_switch=task_switch, \
                                use_triplet = args.use_triplet, \
                                enforce_input_project=False,
                                unseen_type=args.unseen_type)
    
    return flexible_decoder