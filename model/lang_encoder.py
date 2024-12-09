import torch
from torch import nn
from torch.nn import functional as F

from timm.models.layers import trunc_normal_
from transformers import CLIPTokenizer

from util.prompt_engineering import prompt_engineering, get_prompt_templates, get_prompt_templates_class

from collections import OrderedDict
import logging
import os
import numpy as np

from timm.models.layers import DropPath, trunc_normal_

from util.misc import is_main_process
from datasets.hico_text_label import hico_verb_text_label, hico_seen_verb_text_label
from datasets.vcoco_text_label import vcoco_verb_text_label, vcoco_hoi_text_label
from datasets.psg_text_label import psg_verb_text_label
from datasets.vrd_text_label import predicate_text_label

logger = logging.getLogger(__name__)

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        pdtype = x.dtype
        x = x.float()
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x.to(pdtype) + self.bias


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self,
                 d_model: int,
                 n_head: int,
                 attn_mask: torch.Tensor = None,
                 drop_path: float = 0.0):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def attention(self, x: torch.Tensor, key_padding_mask: torch.Tensor = None):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) \
            if self.attn_mask is not None else None


        return self.attn(
            x, x, x,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            attn_mask=self.attn_mask
        )[0]

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor = None):
        x = x + self.drop_path(self.attention(self.ln_1(x), key_padding_mask=key_padding_mask))
        x = x + self.drop_path(self.mlp(self.ln_2(x)))
        return x


class Transformer(nn.Module):
    def __init__(self,
                 context_length: int,
                 vocab_size: int,
                 width: int,
                 layers: int,
                 heads: int,
                 drop_path: float = 0.0,
                 autogressive: bool =True):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, width)

        self.context_length = context_length
        self.positional_embedding = nn.Parameter(
            torch.empty(self.context_length, width)
        )

        self.width = width
        self.layers = layers
        self.autogressive = autogressive
        attn_mask = self.build_attention_mask() if autogressive else None
        dpr = [x.item() for x in torch.linspace(0, drop_path, layers)]  # stochastic depth decay rule
        self.resblocks = nn.ModuleList(
            [
                ResidualAttentionBlock(width, heads, attn_mask, dpr[i])
                for i in range(layers)
            ]
        )

        self.ln_final = LayerNorm(width)

        trunc_normal_(self.positional_embedding, std=.02)
        trunc_normal_(self.token_embedding.weight, std=.02)
        self.apply(self._init_weights)

    @property
    def dim_out(self):
        return self.width

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            if is_main_process():
                logger.info('=> init weight of Linear/Conv2d from trunc norm')
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                if is_main_process():
                    logger.info('=> init bias of Linear/Conv2d to zeros')
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)


    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            'positional_embedding',
            'token_embedding',
        }

    def forward(self, input_ids, attention_mask=None):
        key_padding_mask = (attention_mask == 0) if (not self.autogressive and attention_mask is not None) else None
        x = self.token_embedding(input_ids)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        for block in self.resblocks:
            x = block(x, key_padding_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_final(x)

        return {'last_hidden_state': x}

class LanguageEncoder(nn.Module):

    def __init__(
        self,
        backbone,
        tokenizer,
        tokenizer_type,
        lang_encoder,
        lang_projection,
        max_token_num,
        use_triplet=False,
    ):
        super().__init__()
        # seg
        self.tokenizer = tokenizer
        self.tokenizer_type = tokenizer_type
        self.lang_encoder = lang_encoder
        self.lang_proj = lang_projection
        self.max_token_num = max_token_num
        self.backbone = backbone
       
        self.logit_scale_obj = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale_sub = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale_verb = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    
        if use_triplet:
            self.logit_scale_hoi = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
            
    def get_text_embeddings(self, class_names, use_verb_temp, name='default', is_eval=False, add_bgd=False, prompt=True, norm=True, use_triplet=False):
        if not is_eval:
            if prompt:
                # randomly sample one template
                if name=='coco_panoptic' or name=='hico_obj' or name=='psg_obj' or name=='psg_sub' or name=='vrd_sub' or name=='vrd_obj' or name=='o365_obj' or name=='o365_sub':
                    arbitary_concepts = [
                        prompt_engineering(name, class_names[label].replace('-other','').replace('-merged','').replace('-stuff',''), topk=10000, suffix='.') \
                        for label in range(len(class_names))
                    ]
                    if add_bgd:
                        arbitary_concepts.append("A background class.")

                elif name=='hico_verb':
                    if use_verb_temp:
                        arbitary_concepts = [
                            prompt_engineering(name, class_names[label], topk=10000, suffix='.') \
                            for label in range(len(class_names))
                        ]
                    else:
                        arbitary_concepts = [v[1] for v in hico_verb_text_label]

                elif name=='hico_verb_seen':
                    arbitary_concepts = hico_seen_verb_text_label
                elif name=='psg_verb':
                    arbitary_concepts = [v[1] for v in psg_verb_text_label]
                    if add_bgd:
                        arbitary_concepts.insert(0, 'A background class.')
                elif name=='vrd_verb':
                    arbitary_concepts = [v[1] for v in predicate_text_label]
                    arbitary_concepts.append("A background class.")

                elif name=='vcoco_verb':
                    arbitary_concepts = vcoco_verb_text_label
                elif name=='vcoco_hoi':
                    arbitary_concepts = [v for _, v in vcoco_hoi_text_label.items()]
                elif name=='vcoco_obj':
                    # arbitary_concepts = [v[1] for v in vcoco_obj_text_label]
                    arbitary_concepts = [
                        prompt_engineering(name, class_names[label].replace('-other','').replace('-merged','').replace('-stuff',''), topk=10000, suffix='.') \
                        for label in range(len(class_names))
                    ]
                    if add_bgd:
                        arbitary_concepts.append("A non-object in vcoco.")
                        arbitary_concepts.append("A background in vcoco.")

            else:
                arbitary_concepts = class_names
            
            input_ids = []
            attention_masks = []
            for txt in arbitary_concepts:
                tokens = self.tokenizer(
                    txt, padding='max_length', truncation=True, max_length=self.max_token_num, return_tensors='pt'
                )
                tokens['input_ids'].squeeze_()
                tokens['attention_mask'].squeeze_()

                input_ids.append(tokens['input_ids'])
                attention_masks.append(tokens['attention_mask'])

            arbitary_tokens = torch.stack(input_ids)
            arbitary_attention_masks = torch.stack(attention_masks)

            text_emb = self.forward_language((arbitary_tokens.cuda(), arbitary_attention_masks.cuda()), norm=norm)
            setattr(self, '{}_text_embeddings'.format(name), text_emb)
        else:
            with torch.no_grad():
                def extract_mean_emb(txts):
                    tokens = self.tokenizer(
                        txts, padding='max_length', truncation=True, max_length=self.max_token_num, return_tensors='pt'
                    )
                    clss_embedding = self.forward_language((tokens['input_ids'].cuda(), tokens['attention_mask'].cuda()), norm=norm)
                    clss_embedding = clss_embedding.mean(dim=0)
                    clss_embedding /= clss_embedding.norm()
                    return clss_embedding
                
                obj_templates = get_prompt_templates('hico_obj')
                clss_embeddings = []
                if prompt:
                    if name=='coco_panoptic' or name=='hico_obj' or name=='psg_obj' or name=='psg_sub' or name=='o365_obj' or name=='o365_sub' or name=='vrd_sub' or name=='vrd_obj':
                        for clss in class_names:
                            txts = [template.format(clss.replace('-other','').replace('-merged','').replace('-stuff','')) for template in obj_templates]
                            clss_embeddings.append(extract_mean_emb(txts))
                    elif name=='hico_verb':
                        if use_verb_temp:
                            for clss in class_names:
                                templates = get_prompt_templates_class(clss)
                                txts = [template.format(clss) for template in templates]
                                clss_embeddings.append(extract_mean_emb(txts))
                        else:
                            class_names = [v[1] for v in hico_verb_text_label]
                            for clss in class_names:
                                txts = [clss, clss]
                                clss_embeddings.append(extract_mean_emb(txts))
                    elif name=='psg_verb':
                        class_names = [v[1] for v in psg_verb_text_label]
                        for clss in class_names:
                            txts = [clss, clss]
                            clss_embeddings.append(extract_mean_emb(txts))
                    elif name=='vcoco_obj':
                        for clss in class_names:
                            txts = [template.format(clss.replace('-other','').replace('-merged','').replace('-stuff','')) for template in obj_templates]
                            clss_embeddings.append(extract_mean_emb(txts))
                    elif name=='vcoco_verb':
                        class_names = vcoco_verb_text_label
                        for clss in class_names:
                            txts = [clss, clss]
                            clss_embeddings.append(extract_mean_emb(txts))
                    elif name=='vcoco_hoi':
                        class_names = [v for _, v in vcoco_hoi_text_label.items()]
                        for clss in class_names:
                            txts = [clss, clss]
                            clss_embeddings.append(extract_mean_emb(txts))
                    elif name=='vrd_verb':
                        class_names = [v[1] for v in predicate_text_label]
                        for clss in class_names:
                            txts = [clss, clss]
                            clss_embeddings.append(extract_mean_emb(txts))     
                else:
                    clss_embeddings.append(extract_mean_emb(class_names))

                if add_bgd:
                    if name=='vcoco_obj':
                        txts1 = ["A non-object in vcoco."]
                        txts2 = ["A background in vcoco."]
                        clss_embeddings.append(extract_mean_emb(txts1))
                        clss_embeddings.append(extract_mean_emb(txts2))
                    elif name=='psg_obj' or name=='psg_sub':
                        txts = ["A background class."]
                        clss_embeddings.append(extract_mean_emb(txts))
                    elif name=='psg_verb':
                        txts = ["A background class."]
                        clss_embeddings.insert(0, extract_mean_emb(txts))
                    else:
                        txts = ["A background in hico."]
                        clss_embeddings.append(extract_mean_emb(txts))

                text_emb = torch.stack(clss_embeddings, dim=0)
                setattr(self, '{}_text_embeddings'.format(name), text_emb)
                

    def get_text_token_embeddings(self, txts, name='default', token=False, norm=False):
        if not token:
            tokens = self.tokenizer(
                txts, padding='max_length', truncation=True, max_length=self.max_token_num, return_tensors='pt'
            )
            tokens = {key: value.cuda() for key, value in tokens.items()}
        else:
            tokens = txts
        token_emb, class_emb = self.forward_language_token((tokens['input_ids'], tokens['attention_mask']), norm=norm)
        ret = {"tokens": tokens,
                "token_emb": token_emb,
                "class_emb": class_emb,}
        setattr(self, '{}_token_embeddings'.format(name), ret)
        return ret

    def forward_language(self, texts, norm=True):
        x = self.lang_encoder(*texts)
        x = x['last_hidden_state']

        if self.tokenizer_type == 'clip':
            x = x[torch.arange(x.size(0)), texts[0].argmax(dim=-1)]
        else:
            x = x[:, 0]

        x = x @ self.lang_proj
        if norm:
            x = x / (x.norm(dim=-1, keepdim=True) + 1e-7)
        return x
    
    def forward_language_token(self, texts, norm=False):
        x = self.lang_encoder(*texts)
        token_x = x['last_hidden_state']

        if self.tokenizer_type == 'clip':
            class_x = token_x[torch.arange(token_x.size(0)), texts[0].argmax(dim=-1)]
        else:
            class_x = token_x[:, 0]

        class_x = class_x @ self.lang_proj
        token_x = token_x @ self.lang_proj

        if norm:
            class_x = class_x / (class_x.norm(dim=-1, keepdim=True) + 1e-7)
            token_x = token_x / (token_x.norm(dim=-1, keepdim=True) + 1e-7)

        return token_x, class_x
    
    def compute_similarity(self, v_emb, name='default', fake=False):
        if fake:
            return None
        
        v_emb = v_emb / (v_emb.norm(dim=-1, keepdim=True) + 1e-7)
        t_emb = getattr(self, '{}_text_embeddings'.format(name))
    
        if name=='hico_obj' or name=='psg_obj' or name=='vcoco_obj' or name=='vrd_obj':
            output = self.logit_scale_obj.exp() * v_emb @ t_emb.unsqueeze(0).transpose(1, 2)
        elif name=='psg_sub' or name=='vrd_sub':
            output = self.logit_scale_sub.exp() * v_emb @ t_emb.unsqueeze(0).transpose(1, 2)
        elif name=='hico_verb' or name=='psg_verb' or name=='vcoco_verb' or name=='hico_verb_seen' or name=='vrd_verb':
            output = self.logit_scale_verb.exp() * v_emb @ t_emb.unsqueeze(0).transpose(1, 2)
        elif name=='hico_hoi' or name=='vcoco_hoi':
            output = self.logit_scale_hoi.exp() * v_emb @ t_emb.unsqueeze(0).transpose(1, 2)

        return output
    
def build_tokenizer():
    import os
    os.environ['CURL_CA_BUNDLE'] = ''
    tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')
    tokenizer.add_special_tokens({'cls_token': tokenizer.eos_token})
    return tokenizer

def build_transformer(tokenizer):
    transformer = Transformer(
        context_length=77,
        vocab_size=tokenizer.vocab_size,
        width=512,
        layers=12,
        heads=8,
        autogressive=True
    )
    return transformer

def get_language_model(args):
    tokenizer = build_tokenizer()
    transformer = build_transformer(tokenizer)
    lang_projection = nn.Parameter(torch.empty(512, 512))
    lang_encoder = LanguageEncoder(backbone=args.backbone, tokenizer=tokenizer, tokenizer_type='clip', lang_encoder=transformer, lang_projection=lang_projection, max_token_num=77, use_triplet=args.use_triplet)
    return lang_encoder