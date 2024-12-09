import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
from queue import Queue
from collections import defaultdict
from detectron2.layers import cat
from detectron2.utils.memory import retry_if_cuda_oom

from util.misc import (accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, Result)
from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou

from .hoi_matcher import build_matcher

import io
from PIL import Image
import util.box_ops as box_ops

try:
    from panopticapi.utils import id2rgb, rgb2id
except ImportError:
    pass

from detectron2.structures import ImageList

from bitarray import bitarray

import pycocotools.mask as mask_util

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


def bit_pack(mask):
    flat_mask = mask.flatten()
    a = bitarray(list(flat_mask))
    return a

def vl_similarity(image_feat, text_feat, temperature=1):
    # Only support single GPU for now.
    logits = torch.matmul(image_feat, text_feat.t())
    logits = temperature.exp().clamp(max=100) * logits
    return logits

def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(
    dice_loss
)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(
    sigmoid_ce_loss
)  # type: torch.jit.ScriptModule



class HOI(nn.Module):
    def __init__(self, backbone, pixel_decoder, hoi_decoder, args=None) -> None:
        super().__init__()
        self.backbone = backbone
        self.pixel_decoder = pixel_decoder
        self.hoi_decoder = hoi_decoder
        self.device = args.device
        self.size_divisibility = self.backbone.size_divisibility
        self.use_verb_temp = args.use_verb_temp
        self.flexible_grounding = args.flexible_grounding
        self.num_queries = args.num_queries
        self.flexible_eval_task = args.flexible_eval_task
        self.dataset = args.dataset_file

    def forward(self, batched_inputs, source=None, flexible_eval=False, use_gpt_emb=False):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = ImageList.from_tensors(images, self.size_divisibility)

        extra = {}
        if self.dataset == 'hico+vcoco+psg' or self.dataset == 'hico+vcoco' or self.dataset == 'hico+psg' or self.dataset == 'psg' or self.dataset=='hico' or self.dataset=='vrd':
            extra['source'] = source
        elif self.dataset == 'vcoco':
            extra['source'] = batched_inputs[0]['source']
        
        if self.flexible_grounding:
            if self.dataset == 'psg' or self.dataset == 'vrd':
                targets = self.prepare_targets_psg(batched_inputs, images)
            elif self.dataset == 'hico':
                targets = self.prepare_targets(batched_inputs, images)
            grounding_tokens = [x['grounding_query_embs'] for x in targets] # need to pad for more than one grounding token
            grounding_tokens = nn.utils.rnn.pad_sequence(grounding_tokens)
            extra['grounding_tokens'] = grounding_tokens

        features = self.backbone(images.tensor)
        mask_features, transformer_encoder_features, multi_scale_features = self.pixel_decoder(features)
        
        outputs = self.hoi_decoder(multi_scale_features, mask_features, self.use_verb_temp, extra=extra)

        # only for flexible model
        if self.flexible_grounding or flexible_eval:
            _outputs = {}
            for key, value in outputs.items():
                if key == 'pred_obj_logits':
                    _outputs[key] = value[:,:self.num_queries]
                    _outputs['pred_gobjs'] = value[:,self.num_queries:2*self.num_queries]
                if key == 'pred_sub_logits':
                    _outputs[key] = value[:,:self.num_queries]
                    _outputs['pred_gsubs'] = value[:,self.num_queries:2*self.num_queries]
                elif key == 'pred_verb_logits':
                    _outputs[key] = value[:,:self.num_queries]
                    _outputs['pred_gverbs'] = value[:,self.num_queries:2*self.num_queries]
                elif key == 'pred_sub_masks':
                    _outputs[key] = value[:,:self.num_queries]
                    _outputs['pred_gsub_masks'] = value[:,self.num_queries:2*self.num_queries]
                elif key == 'pred_obj_masks':
                    _outputs[key] = value[:,:self.num_queries]
                    _outputs['pred_gobj_masks'] = value[:,self.num_queries:2*self.num_queries]
                elif key == 'pred_obj_class_emd':
                    _outputs['pred_gobj_texts'] = value[:,self.num_queries:2*self.num_queries]
                elif key == 'pred_sub_class_emd':
                    _outputs['pred_gsub_texts'] = value[:,self.num_queries:2*self.num_queries]
                elif key == 'pred_verb_class_emd':
                    _outputs['pred_gverb_texts'] = value[:,self.num_queries:2*self.num_queries]
                elif key == 'aux_outputs':
                    _outputs[key] = []
                    for i in range(len(value)):
                        _outputs[key] += [{}]
                        for _key, _value in value[i].items():
                            if _key == 'pred_obj_logits':
                                _outputs[key][i][_key] = _value[:,:self.num_queries]
                                _outputs[key][i]['pred_gobjs'] = _value[:,self.num_queries:2*self.num_queries]
                            elif _key == 'pred_verb_logits':
                                _outputs[key][i][_key] = _value[:,:self.num_queries]
                                _outputs[key][i]['pred_gverbs'] = _value[:,self.num_queries:2*self.num_queries]
                            elif _key == 'pred_sub_masks':
                                _outputs[key][i][_key] = _value[:,:self.num_queries]
                                _outputs[key][i]['pred_gsub_masks'] = _value[:,self.num_queries:2*self.num_queries]
                            elif _key == 'pred_obj_masks':
                                _outputs[key][i][_key] = _value[:,:self.num_queries]
                                _outputs[key][i]['pred_gobj_masks'] = _value[:,self.num_queries:2*self.num_queries]
                            elif key == 'pred_obj_class_emd':
                                _outputs[key][i]['pred_gobj_texts'] = _value[:,self.num_queries:2*self.num_queries]
                            elif key == 'pred_verb_class_emd':
                                _outputs[key][i]['pred_gverb_texts'] = _value[:,self.num_queries:2*self.num_queries]

            outputs = _outputs  
        if not use_gpt_emb:
            extra = {'lang_logit_sub': self.hoi_decoder.lang_encoder.logit_scale_sub, 'lang_logit_obj': self.hoi_decoder.lang_encoder.logit_scale_obj, 'lang_logit_verb': self.hoi_decoder.lang_encoder.logit_scale_verb}          
        # extra = {'lang_logit_sub': self.hoi_decoder.lang_encoder.logit_scale_sub, 'lang_logit_obj': self.hoi_decoder.lang_encoder.logit_scale_obj} 
        if self.flexible_grounding:
            return outputs, extra, targets
        else:
            return outputs, extra
    
    def evaluate_grounding_vrd(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = ImageList.from_tensors(images, self.size_divisibility)

        extra = {}

        sub_mask_pred_results = []
        obj_mask_pred_results = []
        pred_gverb_results = []
        pred_gobj_results = []
        pred_gsub_results = []
        matching_scores = []
        for idx, batch_per_image in enumerate(batched_inputs):
            grd_texts = batch_per_image['grounding']['gtext']
            # grd_texts = [x[0] for x in grd_texts]

            gtext = self.hoi_decoder.lang_encoder.get_text_token_embeddings(grd_texts, name='grounding', token=False, norm=False)
            token_emb = gtext['token_emb']
            tokens = gtext['tokens']
            query_emb = token_emb[tokens['attention_mask'].bool()]
            extra['grounding_tokens'] = query_emb[:,None]

            features = self.backbone(images.tensor[idx][None,:])
            mask_features, transformer_encoder_features, multi_scale_features = self.pixel_decoder(features)
            outputs = self.hoi_decoder(multi_scale_features, mask_features, self.use_verb_temp, extra=extra)

            pred_gsub_masks = outputs['pred_sub_masks'][0][self.num_queries:2*self.num_queries]
            pred_gobj_masks = outputs['pred_obj_masks'][0][self.num_queries:2*self.num_queries]
            
            if self.flexible_eval_task=='pred_verb':
                pred_gverb_logits = outputs['pred_verb_logits'][0][self.num_queries:2*self.num_queries]
            if self.flexible_eval_task=='pred_obj':
                pred_gobj_logits = outputs['pred_obj_logits'][0][self.num_queries:2*self.num_queries]
            if self.flexible_eval_task=='pred_sub':
                pred_gsub_logits = outputs['pred_sub_logits'][0][self.num_queries:2*self.num_queries]

            v_emb_sub = outputs['pred_sub_class_emd'][0][self.num_queries:2*self.num_queries]
            v_emb_obj = outputs['pred_obj_class_emd'][0][self.num_queries:2*self.num_queries]
            v_emb_verb = outputs['pred_verb_class_emd'][0][self.num_queries:2*self.num_queries]

            v_emb_obj = v_emb_obj / (v_emb_obj.norm(dim=-1, keepdim=True) + 1e-7) 
            v_emb_verb = v_emb_verb / (v_emb_verb.norm(dim=-1, keepdim=True) + 1e-7) 
            v_emb_sub = v_emb_sub / (v_emb_sub.norm(dim=-1, keepdim=True) + 1e-7) 
            
            t_emb = gtext['class_emb']
            t_emb = t_emb / (t_emb.norm(dim=-1, keepdim=True) + 1e-7)
                       

            temperature_obj = self.hoi_decoder.lang_encoder.logit_scale_obj
            temperature_sub = self.hoi_decoder.lang_encoder.logit_scale_sub
            temperature_verb = self.hoi_decoder.lang_encoder.logit_scale_verb
            if self.flexible_eval_task=='pred_obj':
                out_prob = vl_similarity(v_emb_verb, t_emb, temperature=temperature_verb) \
                        + vl_similarity(v_emb_sub, t_emb, temperature=temperature_sub)
            elif self.flexible_eval_task=='pred_sub':
                out_prob = vl_similarity(v_emb_verb, t_emb, temperature=temperature_verb) \
                        + vl_similarity(v_emb_obj, t_emb, temperature=temperature_obj)
            elif self.flexible_eval_task=='pred_sub_obj':
                out_prob = vl_similarity(v_emb_verb, t_emb, temperature=temperature_verb) 
            matching_score, matched_id = torch.topk(out_prob[:, 0], 10)
            matching_scores += [matching_score]
            sub_mask_pred_results += [pred_gsub_masks[matched_id,:,:]]
            obj_mask_pred_results += [pred_gobj_masks[matched_id,:,:]]
            if self.flexible_eval_task=='pred_verb':
                pred_gverb_results += [pred_gverb_logits[matched_id]]
            if self.flexible_eval_task=='pred_obj':
                pred_gobj_results += [pred_gobj_logits[matched_id]]
            if self.flexible_eval_task=='pred_sub':
                pred_gsub_results += [pred_gsub_logits[matched_id]]

        for i in range(len(sub_mask_pred_results)):
            # upsample masks
            sub_mask_pred_results[i] = F.interpolate(
                sub_mask_pred_results[i][None,],
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bicubic",
                align_corners=False,
                antialias=True
            )[0]

            obj_mask_pred_results[i] = F.interpolate(
                obj_mask_pred_results[i][None,],
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bicubic",
                align_corners=False,
                antialias=True
            )[0]

        del outputs

        image_size = images.image_sizes # before align to the same size
        target_sizes = torch.stack([sample["orig_size"] for sample in batched_inputs], dim=0)
        
        bs = target_sizes.shape[0]
        re_sub_masks, re_obj_masks = retry_if_cuda_oom(self.resize_mask_to_original)(bs, sub_mask_pred_results, image_size, target_sizes, obj_mask_pred_results)
        
        results = []
        for index in range(len(re_sub_masks)):
            result = {}
            sm, om =  re_sub_masks[index], re_obj_masks[index]
            sm = (sm > 0).float()
            om = (om > 0).float()

            m = torch.cat((sm, om))
            
            # compress
            m = [mask_util.encode(np.array(m[i].to('cpu'), order="F", dtype="uint8")) for i in range(m.shape[0])]

            result = {'masks': m}

            ids = torch.arange(len(m))

            matching_s = matching_scores[index]


            if self.flexible_eval_task=='pred_obj':
                gobj_prob = F.softmax(pred_gobj_results[index], -1)
                gobj_scores, gobj_labels = gobj_prob[..., :-1].max(-1)
                o = gobj_labels
                matching_s = matching_s*gobj_scores
                result.update({'gobjs': o})
            else:
                result.update({'gobjs': None})

            if self.flexible_eval_task=='pred_sub':
                gsub_prob = F.softmax(pred_gsub_results[index], -1)
                gsub_scores, gsub_labels = gsub_prob[..., :-1].max(-1)
                o = gsub_labels
                matching_s = matching_s*gsub_scores
                result.update({'gsubs': o})
            else:
                result.update({'gsubs': None})
            result.update({'matching_s': matching_s, 'sub_ids': ids[:ids.shape[0] // 2],
                                    'obj_ids': ids[ids.shape[0] // 2:]})


            results.append(result)

        return results

    def evaluate_grounding_psg(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = ImageList.from_tensors(images, self.size_divisibility)

        extra = {}
        extra.update({'source': batched_inputs[0]['source']})

        sub_mask_pred_results = []
        obj_mask_pred_results = []
        pred_gverb_results = []
        pred_gobj_results = []
        pred_gsub_results = []
        matching_scores = []
        for idx, batch_per_image in enumerate(batched_inputs):
            grd_texts = batch_per_image['grounding']['gtext']
            # grd_texts = [x[0] for x in grd_texts]

            gtext = self.hoi_decoder.lang_encoder.get_text_token_embeddings(grd_texts, name='grounding', token=False, norm=False)
            token_emb = gtext['token_emb']
            tokens = gtext['tokens']
            query_emb = token_emb[tokens['attention_mask'].bool()]
            extra['grounding_tokens'] = query_emb[:,None]

            features = self.backbone(images.tensor[idx][None,:])
            mask_features, transformer_encoder_features, multi_scale_features = self.pixel_decoder(features)
            outputs = self.hoi_decoder(multi_scale_features, mask_features, self.use_verb_temp, extra=extra)

            pred_gsub_masks = outputs['pred_sub_masks'][0][self.num_queries:2*self.num_queries]
            pred_gobj_masks = outputs['pred_obj_masks'][0][self.num_queries:2*self.num_queries]
            
            if self.flexible_eval_task=='pred_verb':
                pred_gverb_logits = outputs['pred_verb_logits'][0][self.num_queries:2*self.num_queries]
            if self.flexible_eval_task=='pred_obj':
                pred_gobj_logits = outputs['pred_obj_logits'][0][self.num_queries:2*self.num_queries]
            if self.flexible_eval_task=='pred_sub':
                pred_gsub_logits = outputs['pred_sub_logits'][0][self.num_queries:2*self.num_queries]

            v_emb_obj = outputs['pred_obj_class_emd'][0][self.num_queries:2*self.num_queries]
            v_emb_verb = outputs['pred_verb_class_emd'][0][self.num_queries:2*self.num_queries]
            v_emb_sub = outputs['pred_sub_class_emd'][0][self.num_queries:2*self.num_queries]


            v_emb_obj = v_emb_obj / (v_emb_obj.norm(dim=-1, keepdim=True) + 1e-7) 
            v_emb_sub = v_emb_sub / (v_emb_sub.norm(dim=-1, keepdim=True) + 1e-7) 
            v_emb_verb = v_emb_verb / (v_emb_verb.norm(dim=-1, keepdim=True) + 1e-7) 
            
            t_emb = gtext['class_emb']
            t_emb = t_emb / (t_emb.norm(dim=-1, keepdim=True) + 1e-7)
            # v_emb = v_emb / (v_emb.norm(dim=-1, keepdim=True) + 1e-7)            

            temperature_obj = self.hoi_decoder.lang_encoder.logit_scale_obj
            temperature_verb = self.hoi_decoder.lang_encoder.logit_scale_verb
            temperature_sub = self.hoi_decoder.lang_encoder.logit_scale_sub
            if self.flexible_eval_task=='pred_verb':
                out_prob = vl_similarity(v_emb_obj, t_emb, temperature=temperature_obj) \
                        + vl_similarity(v_emb_sub, t_emb, temperature=temperature_sub)
            elif self.flexible_eval_task=='pred_obj':
                out_prob = vl_similarity(v_emb_verb, t_emb, temperature=temperature_verb) \
                        + vl_similarity(v_emb_sub, t_emb, temperature=temperature_sub)
            elif self.flexible_eval_task=='pred_sub':
                out_prob = vl_similarity(v_emb_verb, t_emb, temperature=temperature_verb) \
                        + vl_similarity(v_emb_obj, t_emb, temperature=temperature_obj)
            
            matching_score, matched_id = torch.topk(out_prob[:, 0], 10)
            matching_scores += [matching_score]
            sub_mask_pred_results += [pred_gsub_masks[matched_id,:,:]]
            obj_mask_pred_results += [pred_gobj_masks[matched_id,:,:]]
            if self.flexible_eval_task=='pred_verb':
                pred_gverb_results += [pred_gverb_logits[matched_id]]
            if self.flexible_eval_task=='pred_obj':
                pred_gobj_results += [pred_gobj_logits[matched_id]]
            if self.flexible_eval_task=='pred_sub':
                pred_gsub_results += [pred_gsub_logits[matched_id]]

        for i in range(len(sub_mask_pred_results)):
            # upsample masks
            sub_mask_pred_results[i] = F.interpolate(
                sub_mask_pred_results[i][None,],
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bicubic",
                align_corners=False,
                antialias=True
            )[0]

            obj_mask_pred_results[i] = F.interpolate(
                obj_mask_pred_results[i][None,],
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bicubic",
                align_corners=False,
                antialias=True
            )[0]

        del outputs

        image_size = images.image_sizes # before align to the same size
        target_sizes = torch.stack([sample["orig_size"] for sample in batched_inputs], dim=0)
        
        bs = target_sizes.shape[0]
        re_sub_masks, re_obj_masks = retry_if_cuda_oom(self.resize_mask_to_original)(bs, sub_mask_pred_results, image_size, target_sizes, obj_mask_pred_results)
        
        results = []
        for index in range(len(re_sub_masks)):
            result = {}
            sm, om =  re_sub_masks[index], re_obj_masks[index]
            sm = (sm > 0).float()
            om = (om > 0).float()

            m = torch.cat((sm, om))
            
            # compress
            m = [mask_util.encode(np.array(m[i].to('cpu'), order="F", dtype="uint8")) for i in range(m.shape[0])]
            
            result = {'masks': m}

            ids = torch.arange(len(m))

            matching_s = matching_scores[index]

            if self.flexible_eval_task=='pred_verb':
                v = (pred_gverb_results[index]>0).int()
                verb_scores = pred_gverb_results[index].sigmoid()
                matching_s = verb_scores * matching_s.unsqueeze(1) # 10x116
                result.update({'gverbs': v})
            else:
                # v = torch.full_like(pred_gverb_results[index], self.subject_category_id)
                result.update({'gverbs': None})
            if self.flexible_eval_task=='pred_obj':
                gobj_prob = F.softmax(pred_gobj_results[index], -1)
                gobj_scores, gobj_labels = gobj_prob[..., :-1].max(-1)
                o = gobj_labels
                matching_s = matching_s*gobj_scores
                result.update({'gobjs': o})
            else:
                result.update({'gobjs': None})

            if self.flexible_eval_task=='pred_sub':
                gsub_prob = F.softmax(pred_gsub_results[index], -1)
                gsub_scores, gsub_labels = gsub_prob[..., :-1].max(-1)
                s = gsub_labels
                matching_s = matching_s*gsub_scores
                result.update({'gsubs': s})
            else:
                result.update({'gsubs': None})
            result.update({'matching_s': matching_s, 'sub_ids': ids[:ids.shape[0] // 2],
                                    'obj_ids': ids[ids.shape[0] // 2:]})


            results.append(result)

        return results
    
    def evaluate_grounding(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = ImageList.from_tensors(images, self.size_divisibility)

        extra = {}
        extra.update({'source': batched_inputs[0]['source']})

        sub_mask_pred_results = []
        obj_mask_pred_results = []
        pred_gverb_results = []
        pred_gobj_results = []
        matching_scores = []
        for idx, batch_per_image in enumerate(batched_inputs):
            grd_texts = batch_per_image['grounding']['gtext']

            gtext = self.hoi_decoder.lang_encoder.get_text_token_embeddings(grd_texts, name='grounding', token=False, norm=False)
            token_emb = gtext['token_emb']
            tokens = gtext['tokens']
            query_emb = token_emb[tokens['attention_mask'].bool()]
            extra['grounding_tokens'] = query_emb[:,None]

            features = self.backbone(images.tensor[idx][None,:])
            mask_features, transformer_encoder_features, multi_scale_features = self.pixel_decoder(features)
            outputs = self.hoi_decoder(multi_scale_features, mask_features, self.use_verb_temp, extra=extra)

            pred_gsub_masks = outputs['pred_sub_masks'][0][self.num_queries:2*self.num_queries]
            pred_gobj_masks = outputs['pred_obj_masks'][0][self.num_queries:2*self.num_queries]
            
            if self.flexible_eval_task=='pred_verb':
                pred_gverb_logits = outputs['pred_verb_logits'][0][self.num_queries:2*self.num_queries]
            if self.flexible_eval_task=='pred_obj':
                pred_gobj_logits = outputs['pred_obj_logits'][0][self.num_queries:2*self.num_queries]

            v_emb_obj = outputs['pred_obj_class_emd'][0][self.num_queries:2*self.num_queries]
            v_emb_verb = outputs['pred_verb_class_emd'][0][self.num_queries:2*self.num_queries]

            v_emb_obj = v_emb_obj / (v_emb_obj.norm(dim=-1, keepdim=True) + 1e-7) 
            v_emb_verb = v_emb_verb / (v_emb_verb.norm(dim=-1, keepdim=True) + 1e-7) 
            
            t_emb = gtext['class_emb']
            t_emb = t_emb / (t_emb.norm(dim=-1, keepdim=True) + 1e-7)
            # v_emb = v_emb / (v_emb.norm(dim=-1, keepdim=True) + 1e-7)            

            temperature_obj = self.hoi_decoder.lang_encoder.logit_scale_obj
            temperature_verb = self.hoi_decoder.lang_encoder.logit_scale_verb
            if self.flexible_eval_task=='pred_verb':
                out_prob = vl_similarity(v_emb_obj, t_emb, temperature=temperature_obj)
            elif self.flexible_eval_task=='pred_obj':
                out_prob = vl_similarity(v_emb_verb, t_emb, temperature=temperature_verb)
            elif self.flexible_eval_task=='pred_sub':
                out_prob = vl_similarity(v_emb_verb, t_emb, temperature=temperature_verb) \
                        + vl_similarity(v_emb_obj, t_emb, temperature=temperature_obj)
            
            matching_score, matched_id = torch.topk(out_prob[:, 0], 10)
            matching_scores += [matching_score]
            sub_mask_pred_results += [pred_gsub_masks[matched_id,:,:]]
            obj_mask_pred_results += [pred_gobj_masks[matched_id,:,:]]
            if self.flexible_eval_task=='pred_verb':
                pred_gverb_results += [pred_gverb_logits[matched_id]]
            if self.flexible_eval_task=='pred_obj':
                pred_gobj_results += [pred_gobj_logits[matched_id]]

        for i in range(len(sub_mask_pred_results)):
            # upsample masks
            sub_mask_pred_results[i] = F.interpolate(
                sub_mask_pred_results[i][None,],
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bicubic",
                align_corners=False,
                antialias=True
            )[0]

            obj_mask_pred_results[i] = F.interpolate(
                obj_mask_pred_results[i][None,],
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bicubic",
                align_corners=False,
                antialias=True
            )[0]

        del outputs

        image_size = images.image_sizes # before align to the same size
        target_sizes = torch.stack([sample["orig_size"] for sample in batched_inputs], dim=0)
        
        bs = target_sizes.shape[0]
        re_sub_masks, re_obj_masks = retry_if_cuda_oom(self.resize_mask_to_original)(bs, sub_mask_pred_results, image_size, target_sizes, obj_mask_pred_results)
        
        results = []
        for index in range(len(re_sub_masks)):
            result = {}
            sm, om =  re_sub_masks[index], re_obj_masks[index]
            sm = (sm > 0).float()
            om = (om > 0).float()

            m = torch.cat((sm, om))
            
            # compress
            m = [mask_util.encode(np.array(m[i].to('cpu'), order="F", dtype="uint8")) for i in range(m.shape[0])]
          
            result = {'masks': m}

            ids = torch.arange(len(m))

            matching_s = matching_scores[index]

            if self.flexible_eval_task=='pred_verb':
                v = (pred_gverb_results[index]>0).int()
                verb_scores = pred_gverb_results[index].sigmoid()
                matching_s = verb_scores * matching_s.unsqueeze(1) # 10x116
                result.update({'gverbs': v})
            else:
                # v = torch.full_like(pred_gverb_results[index], self.subject_category_id)
                result.update({'gverbs': None})
            if self.flexible_eval_task=='pred_obj':
                gobj_prob = F.softmax(pred_gobj_results[index], -1)
                gobj_scores, gobj_labels = gobj_prob[..., :-1].max(-1)
                o = gobj_labels
                matching_s = matching_s*gobj_scores
                result.update({'gobjs': o})
            else:
                result.update({'gobjs': None})
            result.update({'matching_s': matching_s, 'sub_ids': ids[:ids.shape[0] // 2],
                                    'obj_ids': ids[ids.shape[0] // 2:]})


            results.append(result)
       
        return results
    
    def resize_mask_to_original(self, bs, out_sub_masks, image_size, target_sizes, out_obj_masks):
        re_sub_masks = []
        re_obj_masks = []
        for b in range(bs):
            sub_mask = out_sub_masks[b][:, :image_size[b][0], :image_size[b][1]].expand(1, -1, -1, -1)
            sub_mask = F.interpolate(
                sub_mask,
                size=tuple(target_sizes[b].tolist()),
                mode="bicubic",
                align_corners=False,
                antialias=True
            ).squeeze(0)
            obj_mask = out_obj_masks[b][:, :image_size[b][0], :image_size[b][1]].expand(1, -1, -1, -1)
            obj_mask = F.interpolate(
                obj_mask,
                size=tuple(target_sizes[b].tolist()),
                mode="bicubic",
                align_corners=False,
                antialias=True
            ).squeeze(0)
            re_sub_masks.append(sub_mask)
            re_obj_masks.append(obj_mask)
        return re_sub_masks, re_obj_masks
    
    def prepare_targets(self, batched_inputs, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        # batched_inputs = [{k: v.to(self.device) for k, v in sample.items() if k != 'filename' and k!='grounding'} for sample in batched_inputs]
        new_targets = []
        for idx, batch_per_image in enumerate(batched_inputs):
            target_per_image = {k: v.to(self.device) for k, v in batch_per_image.items() if k != 'filename' and k!='grounding' and k!='source' and k!='text_prompt'}
            gt_sub_masks = target_per_image['sub_masks']
            gt_obj_masks = target_per_image['obj_masks']
            # pad gt
            padded_sub_masks = torch.zeros((gt_sub_masks.shape[0], h_pad, w_pad), dtype=gt_sub_masks.dtype, device=gt_sub_masks.device)
            padded_sub_masks[:, : gt_sub_masks.shape[1], : gt_sub_masks.shape[2]] = gt_sub_masks

            padded_obj_masks = torch.zeros((gt_obj_masks.shape[0], h_pad, w_pad), dtype=gt_obj_masks.dtype, device=gt_obj_masks.device)
            padded_obj_masks[:, : gt_obj_masks.shape[1], : gt_obj_masks.shape[2]] = gt_obj_masks

            target_dict = {
                'obj_labels': target_per_image['obj_labels'],
                'verb_labels': target_per_image['verb_labels'],
                'sub_masks': padded_sub_masks,
                'obj_masks': padded_obj_masks,
            }
            grd_sub_mask = batch_per_image['grounding']['sub_gmasks']
            grd_obj_mask = batch_per_image['grounding']['obj_gmasks']
            grd_text = batch_per_image['grounding']['gtext']
            grd_hash = batch_per_image['grounding']['ghash']
            
            padded_sub_mask = torch.zeros((grd_sub_mask.shape[0], h_pad, w_pad), dtype=grd_sub_mask.dtype, device=grd_sub_mask.device)
            padded_sub_mask[:, : grd_sub_mask.shape[1], : grd_sub_mask.shape[2]] = grd_sub_mask

            padded_obj_mask = torch.zeros((grd_obj_mask.shape[0], h_pad, w_pad), dtype=grd_obj_mask.dtype, device=grd_obj_mask.device)
            padded_obj_mask[:, : grd_obj_mask.shape[1], : grd_obj_mask.shape[2]] = grd_obj_mask
            
            all_prompt = batch_per_image['text_prompt']
            all_pt_feats = []
            for pt in all_prompt:
                all_pt_feats.append(self.hoi_decoder.lang_encoder.get_text_token_embeddings(pt, name='grounding', token=False, norm=False))
            
            gtext = self.hoi_decoder.lang_encoder.get_text_token_embeddings(grd_text, name='grounding', token=False, norm=False)
            token_emb = gtext['token_emb']
            tokens = gtext['tokens']
            
            unique_hash_id = np.unique(grd_hash, return_index=True)[1]
            selected_mask = np.zeros(len(grd_hash)).astype(bool)
            selected_mask[unique_hash_id] = True

            selected_token_emb = token_emb[selected_mask]
            selected_attn_mask = tokens['attention_mask'][selected_mask]
            query_emb = selected_token_emb[selected_attn_mask.bool()]
            
            class_idx = tokens['attention_mask'].sum(dim=-1) - 1
            class_idx = torch.stack((torch.arange(len(class_idx), device=class_idx.device), class_idx)).tolist()
            class_emb = token_emb[class_idx]
            
            target_dict['grounding_sub_mask'] = padded_sub_mask
            target_dict['grounding_obj_mask'] = padded_obj_mask
            target_dict['grounding_query_embs'] = query_emb
            target_dict['grounding_class_embs'] = class_emb
            target_dict['grounding_verb_class'] = batch_per_image['grounding']['gverb_classes'].to(self.device)
            target_dict['grounding_obj_class'] = batch_per_image['grounding']['gobj_classes'].to(self.device)
            target_dict['grounding_hash'] = grd_hash
            target_dict['grounding_task'] = batch_per_image['grounding']['gtask']
            new_targets.append(target_dict)
        return new_targets
    
    def prepare_targets_psg(self, batched_inputs, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        # batched_inputs = [{k: v.to(self.device) for k, v in sample.items() if k != 'filename' and k!='grounding'} for sample in batched_inputs]
        new_targets = []
        for idx, batch_per_image in enumerate(batched_inputs):
            target_per_image = {k: v.to(self.device) for k, v in batch_per_image.items() if k != 'filename' and k!='grounding' and k!='source' and k!='id'}
            gt_sub_masks = target_per_image['sub_masks']
            gt_obj_masks = target_per_image['obj_masks']
            # pad gt
            padded_sub_masks = torch.zeros((gt_sub_masks.shape[0], h_pad, w_pad), dtype=gt_sub_masks.dtype, device=gt_sub_masks.device)
            padded_sub_masks[:, : gt_sub_masks.shape[1], : gt_sub_masks.shape[2]] = gt_sub_masks

            padded_obj_masks = torch.zeros((gt_obj_masks.shape[0], h_pad, w_pad), dtype=gt_obj_masks.dtype, device=gt_obj_masks.device)
            padded_obj_masks[:, : gt_obj_masks.shape[1], : gt_obj_masks.shape[2]] = gt_obj_masks

            target_dict = {
                'obj_labels': target_per_image['obj_labels'],
                'sub_labels': target_per_image['sub_labels'],
                'verb_labels': target_per_image['verb_labels'],
                'sub_masks': padded_sub_masks,
                'obj_masks': padded_obj_masks,
            }
            grd_sub_mask = batch_per_image['grounding']['sub_gmasks']
            grd_obj_mask = batch_per_image['grounding']['obj_gmasks']
            grd_text = batch_per_image['grounding']['gtext']
            grd_hash = batch_per_image['grounding']['ghash']
            
            padded_sub_mask = torch.zeros((grd_sub_mask.shape[0], h_pad, w_pad), dtype=grd_sub_mask.dtype, device=grd_sub_mask.device)
            padded_sub_mask[:, : grd_sub_mask.shape[1], : grd_sub_mask.shape[2]] = grd_sub_mask

            padded_obj_mask = torch.zeros((grd_obj_mask.shape[0], h_pad, w_pad), dtype=grd_obj_mask.dtype, device=grd_obj_mask.device)
            padded_obj_mask[:, : grd_obj_mask.shape[1], : grd_obj_mask.shape[2]] = grd_obj_mask
            
            gtext = self.hoi_decoder.lang_encoder.get_text_token_embeddings(grd_text, name='grounding', token=False, norm=False)
            token_emb = gtext['token_emb']
            tokens = gtext['tokens']
            
            unique_hash_id = np.unique(grd_hash, return_index=True)[1]
            selected_mask = np.zeros(len(grd_hash)).astype(bool)
            selected_mask[unique_hash_id] = True

            selected_token_emb = token_emb[selected_mask]
            selected_attn_mask = tokens['attention_mask'][selected_mask]
            query_emb = selected_token_emb[selected_attn_mask.bool()]
            
            class_idx = tokens['attention_mask'].sum(dim=-1) - 1
            class_idx = torch.stack((torch.arange(len(class_idx), device=class_idx.device), class_idx)).tolist()
            class_emb = token_emb[class_idx]
            
            target_dict['grounding_sub_mask'] = padded_sub_mask
            target_dict['grounding_obj_mask'] = padded_obj_mask
            target_dict['grounding_query_embs'] = query_emb
            target_dict['grounding_class_embs'] = class_emb
            target_dict['grounding_verb_class'] = batch_per_image['grounding']['gverb_classes'].to(self.device)
            target_dict['grounding_obj_class'] = batch_per_image['grounding']['gobj_classes'].to(self.device)
            target_dict['grounding_sub_class'] = batch_per_image['grounding']['gsub_classes'].to(self.device)
            target_dict['grounding_hash'] = grd_hash
            target_dict['grounding_task'] = batch_per_image['grounding']['gtask']
            new_targets.append(target_dict)
        return new_targets

def point_sample(input, point_coords, **kwargs):
    """
    A wrapper around :function:`torch.nn.functional.grid_sample` to support 3D point_coords tensors.
    Unlike :function:`torch.nn.functional.grid_sample` it assumes `point_coords` to lie inside
    [0, 1] x [0, 1] square.

    Args:
        input (Tensor): A tensor of shape (N, C, H, W) that contains features map on a H x W grid.
        point_coords (Tensor): A tensor of shape (N, P, 2) or (N, Hgrid, Wgrid, 2) that contains
        [0, 1] x [0, 1] normalized point coordinates.

    Returns:
        output (Tensor): A tensor of shape (N, C, P) or (N, C, Hgrid, Wgrid) that contains
            features for points in `point_coords`. The features are obtained via bilinear
            interplation from `input` the same way as :function:`torch.nn.functional.grid_sample`.
    """
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)
    output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs)
    if add_dim:
        output = output.squeeze(3)
    return output

def get_uncertain_point_coords_with_randomness(
    coarse_logits, uncertainty_func, num_points, oversample_ratio, importance_sample_ratio
):
    """
    Sample points in [0, 1] x [0, 1] coordinate space based on their uncertainty. The unceratinties
        are calculated for each point using 'uncertainty_func' function that takes point's logit
        prediction as input.
    See PointRend paper for details.

    Args:
        coarse_logits (Tensor): A tensor of shape (N, C, Hmask, Wmask) or (N, 1, Hmask, Wmask) for
            class-specific or class-agnostic prediction.
        uncertainty_func: A function that takes a Tensor of shape (N, C, P) or (N, 1, P) that
            contains logit predictions for P points and returns their uncertainties as a Tensor of
            shape (N, 1, P).
        num_points (int): The number of points P to sample.
        oversample_ratio (int): Oversampling parameter.
        importance_sample_ratio (float): Ratio of points that are sampled via importnace sampling.

    Returns:
        point_coords (Tensor): A tensor of shape (N, P, 2) that contains the coordinates of P
            sampled points.
    """
    assert oversample_ratio >= 1
    assert importance_sample_ratio <= 1 and importance_sample_ratio >= 0
    num_boxes = coarse_logits.shape[0]
    num_sampled = int(num_points * oversample_ratio)
    point_coords = torch.rand(num_boxes, num_sampled, 2, device=coarse_logits.device, dtype=coarse_logits.dtype)
    point_logits = point_sample(coarse_logits, point_coords, align_corners=False)
    # It is crucial to calculate uncertainty based on the sampled prediction value for the points.
    # Calculating uncertainties of the coarse predictions first and sampling them for points leads
    # to incorrect results.
    # To illustrate this: assume uncertainty_func(logits)=-abs(logits), a sampled point between
    # two coarse predictions with -1 and 1 logits has 0 logits, and therefore 0 uncertainty value.
    # However, if we calculate uncertainties for the coarse predictions first,
    # both will have -1 uncertainty, and the sampled point will get -1 uncertainty.
    point_uncertainties = uncertainty_func(point_logits)
    num_uncertain_points = int(importance_sample_ratio * num_points)
    num_random_points = num_points - num_uncertain_points
    idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
    shift = num_sampled * torch.arange(num_boxes, dtype=torch.long, device=coarse_logits.device)
    idx += shift[:, None]
    point_coords = point_coords.view(-1, 2)[idx.view(-1), :].view(
        num_boxes, num_uncertain_points, 2
    )
    if num_random_points > 0:
        point_coords = cat(
            [
                point_coords,
                torch.rand(num_boxes, num_random_points, 2, device=coarse_logits.device),
            ],
            dim=1,
        )
    return point_coords

def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


class ShapeSpec:
    """
    A simple structure that contains basic shape specification about a tensor.
    It is often used as the auxiliary inputs/outputs of models,
    to complement the lack of shape inference ability among pytorch modules.
    """

    channels = None
    height = None
    width = None
    stride = None


class SetCriterionHOI(nn.Module):

    def __init__(self, num_obj_classes, num_queries, num_verb_classes, num_hoi_classes, matcher, weight_dict, eos_coef, losses, args):
        super().__init__()

        if args.unseen_type == 'uo':
            self.num_obj_classes = num_obj_classes -12
        else:
            self.num_obj_classes = num_obj_classes
        if args.unseen_type == 'uv':
            self.num_verb_classes = num_verb_classes-20
        else:
            self.num_verb_classes = num_verb_classes

        
        self.num_hoi_classes = num_hoi_classes
        self.num_queries = num_queries
    
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        self.num_points = args.num_points
        self.oversample_ratio = args.oversample_ratio
        self.importance_sample_ratio = args.importance_sample_ratio
        self.device = args.device
        self.grounding_weight = args.grounding_weight

        if (args.psg and num_obj_classes==133) or args.vrd or num_obj_classes==365:
            subject_empty_weight = torch.ones(self.num_obj_classes + 1)
            subject_empty_weight[-1] = self.eos_coef
            self.register_buffer('subject_empty_weight', subject_empty_weight)

        object_empty_weight = torch.ones(self.num_obj_classes + 1)
        object_empty_weight[-1] = self.eos_coef
        self.register_buffer('object_empty_weight', object_empty_weight)

        if args.psg and num_obj_classes==133:
            verb_empty_weight = torch.ones(self.num_verb_classes + 1)
            #NOTE set background class as the first indice for relations as they are 1-based
            verb_empty_weight[0] = self.eos_coef
            self.register_buffer('verb_empty_weight', verb_empty_weight)
        else:
            verb_empty_weight = torch.ones(self.num_verb_classes + 1)
            verb_empty_weight[-1] = self.eos_coef
            self.register_buffer('verb_empty_weight', verb_empty_weight)

        if num_obj_classes==133 or num_obj_classes==365:
            self.task_switch = {'mask': True, 'psg': args.psg, 'vrd': args.vrd, 'hoi': False, 'grounding': args.flexible_grounding}
        elif num_obj_classes==80:
            self.task_switch = {'mask': False, 'box': False, 'psg': False, 'vrd': args.vrd, 'hoi': args.hoi, 'grounding': args.flexible_grounding}
            if args.use_mask:
                self.task_switch['mask'] = True
            if args.use_box:
                self.task_switch['box'] = True


    def loss_grounding(self, outputs, targets, indices, layer_id, extra):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_gsub_masks" in outputs
        assert "pred_gobj_masks" in outputs
        assert "pred_gobjs" in outputs
        assert "pred_gverbs" in outputs
        assert "pred_gobj_texts" in outputs
        assert "pred_gverb_texts" in outputs
        
       
        losses = {}

        valid_b = []
        for idx, (i, j) in enumerate(indices):
            if len(i)!=0:
                valid_b.append(idx)
        
        if len(valid_b)==0:
            loss = outputs['pred_gsub_masks'].sum() * 0.0
            return {"loss_grounding_sub_bce_0": loss, "loss_grounding_sub_dice_0": loss, "loss_grounding_obj_bce_0": loss, "loss_grounding_obj_dice_0": loss, "loss_grounding_ce_0": loss, "loss_grounding_verb_bce_0": loss, "loss_grounding_obj_ce_0": loss}

        tasks = [targets[idx]['grounding_task'] for idx in valid_b]
    
        targets = [targets[idx] for idx in valid_b]
        outputs = {k: v[valid_b] for k, v in outputs.items()}

        use_mask_emb = False
        if 'sub_mask_scl_emb' in outputs:
            sub_mask_scl_emb = outputs['sub_mask_scl_emb']   
            obj_mask_scl_emb = outputs['obj_mask_scl_emb']  
            use_mask_emb = True
        
        pred_logits = []
        for b in range(len(targets)):
            num_gts = len(targets[b]['grounding_obj_mask'])
            t_emb = torch.cat([targets[b]['grounding_class_embs'] for _ in range(num_gts)], dim=0)
            v_emb = outputs["pred_gobj_texts"][b]
            v_emb_verb = outputs["pred_gverb_texts"][b]

            if use_mask_emb:
                v_emb = sub_mask_scl_emb[b] + v_emb
                v_emb = obj_mask_scl_emb[b] + v_emb
                v_emb_verb = sub_mask_scl_emb[b] + v_emb_verb
                v_emb_verb = obj_mask_scl_emb[b] + v_emb_verb

            t_emb = t_emb / (t_emb.norm(dim=-1, keepdim=True) + 1e-7)
            v_emb = v_emb / (v_emb.norm(dim=-1, keepdim=True) + 1e-7)  

            v_emb_verb = v_emb_verb / (v_emb_verb.norm(dim=-1, keepdim=True) + 1e-7)    

            out_prob = vl_similarity(v_emb, t_emb, temperature=extra['lang_logit_obj'])
            out_prob_verb = vl_similarity(v_emb_verb, t_emb, temperature=extra['lang_logit_verb'])
            if tasks[b] == 'pred_verb':
                pred_logits += [out_prob]
                # src_verb_logits = outputs['pred_gverbs'][b]
            if tasks[b] == 'pred_obj':
                pred_logits += [out_prob_verb]

            if tasks[b] == 'pred_sub':
                pred_logits += [out_prob+out_prob_verb]  
            outputs['pred_logits'] = pred_logits

        indices = self.matcher(outputs, targets, mode='grounding', extra={'temperature':extra['lang_logit_obj']})
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        loss_verb_ce = outputs['pred_gsub_masks'].sum() * 0.0
        loss_obj_ce = outputs['pred_gsub_masks'].sum() * 0.0
        for b in range(len(targets)):
            if tasks[b] == 'pred_verb':
                src_verb_logits = outputs['pred_gverbs'][b]
                target_verb_classes_o = targets[b]['grounding_verb_class'][indices[b][1]]
                target_verb_classes = torch.zeros_like(src_verb_logits, device=src_verb_logits.device)
                target_verb_classes[indices[b][0]] = target_verb_classes_o

                src_verb_logits = src_verb_logits.sigmoid()
                loss_verb_ce = self._neg_loss(src_verb_logits, target_verb_classes, weights=self.verb_empty_weight, alpha=0.5) + loss_verb_ce

            elif tasks[b] == 'pred_obj':
                src_obj_logits = outputs['pred_gobjs'][b]
                target_classes_o = targets[b]['grounding_obj_class'][indices[b][1]]
                target_classes = torch.full(src_obj_logits.shape[:1], self.num_obj_classes, dtype=torch.int64, device=src_obj_logits.device)
                target_classes[indices[b][0]] = target_classes_o
                loss_obj_ce = F.cross_entropy(src_obj_logits, target_classes, self.object_empty_weight) + loss_obj_ce


        losses.update({"loss_grounding_verb_bce_0": loss_verb_ce})
        losses.update({'loss_grounding_obj_ce_0': loss_obj_ce})
       


        src_sub_masks = outputs["pred_gsub_masks"]
        src_sub_masks = src_sub_masks[src_idx]

        src_obj_masks = outputs["pred_gobj_masks"]
        src_obj_masks = src_obj_masks[src_idx]

        target_sub_masks = torch.cat([t['sub_masks'][i] for t, (_, i) in zip(targets, indices)], dim=0).to(src_sub_masks)
        target_obj_masks = torch.cat([t['obj_masks'][i] for t, (_, i) in zip(targets, indices)], dim=0).to(src_obj_masks)
        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        src_sub_masks = src_sub_masks[:, None]
        src_obj_masks = src_obj_masks[:, None]
        target_sub_masks = target_sub_masks[:, None].to(torch.float)
        target_obj_masks = target_obj_masks[:, None].to(torch.float)
        
        with torch.no_grad():
            # sample point_coords
            point_sub_coords = get_uncertain_point_coords_with_randomness(
                                src_sub_masks,
                                lambda logits: calculate_uncertainty(logits),
                                self.num_points,
                                self.oversample_ratio,
                                self.importance_sample_ratio,
                            ).type(src_sub_masks.dtype)
            # get gt labels
            point_sub_labels = point_sample(
                target_sub_masks,
                point_sub_coords,
                align_corners=False,
            ).squeeze(1)

            # sample point_coords
            point_obj_coords = get_uncertain_point_coords_with_randomness(
                                src_obj_masks,
                                lambda logits: calculate_uncertainty(logits),
                                self.num_points,
                                self.oversample_ratio,
                                self.importance_sample_ratio,
                            ).type(src_obj_masks.dtype)
            # get gt labels
            point_obj_labels = point_sample(
                target_obj_masks,
                point_obj_coords,
                align_corners=False,
            ).squeeze(1)

        point_sub_logits = point_sample(
                src_sub_masks,
                point_sub_coords,
                align_corners=False,
            ).squeeze(1)
        point_obj_logits = point_sample(
                src_obj_masks,
                point_obj_coords,
                align_corners=False,
            ).squeeze(1)

        losses.update({"loss_grounding_sub_bce_0": sigmoid_ce_loss_jit(point_sub_logits, point_sub_labels, len(src_sub_masks))})
        losses.update({"loss_grounding_sub_dice_0": dice_loss_jit(point_sub_logits, point_sub_labels, len(src_sub_masks))})
        losses.update({"loss_grounding_obj_bce_0": sigmoid_ce_loss_jit(point_obj_logits, point_obj_labels, len(src_obj_masks))})
        losses.update({"loss_grounding_obj_dice_0": dice_loss_jit(point_obj_logits, point_obj_labels, len(src_obj_masks))})

        # compute t2i loss
        loss_grd_ce = 0
        for b in range(len(targets)):
            task = targets[b]['grounding_task']
            pred_logit = outputs["pred_logits"][b]
            gt_logit = torch.zeros_like(pred_logit)
            select_idx = torch.stack((indices[b][0], indices[b][1])).tolist()
            gt_logit[select_idx] = 1

            num_gts = len(targets[b]['grounding_obj_mask'])
            t_hash = torch.tensor([i for i in range(num_gts)], device=gt_logit.device)
            hash_table = torch.zeros((len(t_hash), len(t_hash)), device=gt_logit.device)
            for idx in range(0, len(hash_table)):
                hash_table[idx][t_hash==t_hash[idx]] = 1
            hash_table = hash_table / hash_table.sum(-1, keepdim=True)
            gt_logit = gt_logit @ hash_table
            loss_grd_ce += self.grounding_weight*torch.sum(-gt_logit.t() * F.log_softmax(pred_logit.t(), dim=-1), dim=-1).mean()
        loss_grd_ce = loss_grd_ce / len(indices)
        losses.update({"loss_grounding_ce_0": loss_grd_ce})
        del src_sub_masks
        del target_sub_masks
        del src_obj_masks
        del target_obj_masks
        return losses
    
    def loss_grounding_only(self, outputs, targets, indices, layer_id, extra):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_gsub_masks" in outputs
        assert "pred_gobj_masks" in outputs
        assert "pred_gobjs" in outputs
        assert "pred_gverbs" in outputs
        assert "pred_gobj_texts" in outputs
        assert "pred_gverb_texts" in outputs
        
        losses = {}

        valid_b = []
        for idx, (i, j) in enumerate(indices):
            if len(i)!=0:
                valid_b.append(idx)
        
        if len(valid_b)==0:
            loss = outputs['pred_gsub_masks'].sum() * 0.0
            return {"loss_grounding_sub_bce_0": loss, "loss_grounding_sub_dice_0": loss, "loss_grounding_obj_bce_0": loss, "loss_grounding_obj_dice_0": loss, "loss_grounding_ce_0": loss, "loss_grounding_verb_bce_0": loss, "loss_grounding_obj_ce_0": loss}

        tasks = [targets[idx]['grounding_task'] for idx in valid_b]
        
        targets = [targets[idx] for idx in valid_b]
        outputs = {k: v[valid_b] for k, v in outputs.items()}

        use_mask_emb = False
        if 'sub_mask_scl_emb' in outputs:
            sub_mask_scl_emb = outputs['sub_mask_scl_emb']   
            obj_mask_scl_emb = outputs['obj_mask_scl_emb']  
            use_mask_emb = True

        pred_logits = []
        for b in range(len(targets)):
            num_gts = len(targets[b]['grounding_obj_mask'])
            t_emb = torch.cat([targets[b]['grounding_class_embs'] for _ in range(num_gts)], dim=0)
            v_emb = outputs["pred_gobj_texts"][b]
            v_emb_verb = outputs["pred_gverb_texts"][b]

            if use_mask_emb:
                v_emb = sub_mask_scl_emb[b] + v_emb
                v_emb = obj_mask_scl_emb[b] + v_emb
                v_emb_verb = sub_mask_scl_emb[b] + v_emb_verb
                v_emb_verb = obj_mask_scl_emb[b] + v_emb_verb

            t_emb = t_emb / (t_emb.norm(dim=-1, keepdim=True) + 1e-7)
            v_emb = v_emb / (v_emb.norm(dim=-1, keepdim=True) + 1e-7)  

            v_emb_verb = v_emb_verb / (v_emb_verb.norm(dim=-1, keepdim=True) + 1e-7)    

            out_prob = vl_similarity(v_emb, t_emb, temperature=extra['lang_logit_obj'])
            out_prob_verb = vl_similarity(v_emb_verb, t_emb, temperature=extra['lang_logit_verb'])
            if tasks[b] == 'pred_verb':
                pred_logits += [out_prob]
                
            if tasks[b] == 'pred_obj':
                pred_logits += [out_prob_verb]
                
            if tasks[b] == 'pred_sub':
                pred_logits += [out_prob+out_prob_verb]  
            outputs['pred_logits'] = pred_logits

        indices = self.matcher(outputs, targets, mode='grounding', extra={'temperature':extra['lang_logit_obj']})
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        loss_verb_ce = outputs['pred_gsub_masks'].sum() * 0.0
        loss_obj_ce = outputs['pred_gsub_masks'].sum() * 0.0
        for b in range(len(targets)):
            if tasks[b] == 'pred_verb':
                src_verb_logits = outputs['pred_gverbs'][b]
                target_verb_classes_o = targets[b]['grounding_verb_class'][indices[b][1]]
                target_verb_classes = torch.zeros_like(src_verb_logits, device=src_verb_logits.device)
                target_verb_classes[indices[b][0]] = target_verb_classes_o

                src_verb_logits = src_verb_logits.sigmoid()
                loss_verb_ce = self._neg_loss(src_verb_logits, target_verb_classes, weights=self.verb_empty_weight, alpha=0.5) + loss_verb_ce

            elif tasks[b] == 'pred_obj':
                src_obj_logits = outputs['pred_gobjs'][b]
                target_classes_o = targets[b]['grounding_obj_class'][indices[b][1]]
                target_classes = torch.full(src_obj_logits.shape[:1], self.num_obj_classes, dtype=torch.int64, device=src_obj_logits.device)
                target_classes[indices[b][0]] = target_classes_o
                loss_obj_ce = F.cross_entropy(src_obj_logits, target_classes, self.object_empty_weight) + loss_obj_ce


        losses.update({"loss_grounding_verb_bce_0": loss_verb_ce})
        losses.update({'loss_grounding_obj_ce_0': loss_obj_ce})
       
        src_sub_masks = outputs["pred_gsub_masks"]
        src_sub_masks = src_sub_masks[src_idx]

        src_obj_masks = outputs["pred_gobj_masks"]
        src_obj_masks = src_obj_masks[src_idx]

        target_sub_masks = torch.cat([t['sub_masks'][i] for t, (_, i) in zip(targets, indices)], dim=0).to(src_sub_masks)
        target_obj_masks = torch.cat([t['obj_masks'][i] for t, (_, i) in zip(targets, indices)], dim=0).to(src_obj_masks)
        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        src_sub_masks = src_sub_masks[:, None]
        src_obj_masks = src_obj_masks[:, None]
        target_sub_masks = target_sub_masks[:, None].to(torch.float)
        target_obj_masks = target_obj_masks[:, None].to(torch.float)
        
        with torch.no_grad():
            # sample point_coords
            point_sub_coords = get_uncertain_point_coords_with_randomness(
                                src_sub_masks,
                                lambda logits: calculate_uncertainty(logits),
                                self.num_points,
                                self.oversample_ratio,
                                self.importance_sample_ratio,
                            ).type(src_sub_masks.dtype)
            # get gt labels
            point_sub_labels = point_sample(
                target_sub_masks,
                point_sub_coords,
                align_corners=False,
            ).squeeze(1)

            # sample point_coords
            point_obj_coords = get_uncertain_point_coords_with_randomness(
                                src_obj_masks,
                                lambda logits: calculate_uncertainty(logits),
                                self.num_points,
                                self.oversample_ratio,
                                self.importance_sample_ratio,
                            ).type(src_obj_masks.dtype)
            # get gt labels
            point_obj_labels = point_sample(
                target_obj_masks,
                point_obj_coords,
                align_corners=False,
            ).squeeze(1)

        point_sub_logits = point_sample(
                src_sub_masks,
                point_sub_coords,
                align_corners=False,
            ).squeeze(1)
        point_obj_logits = point_sample(
                src_obj_masks,
                point_obj_coords,
                align_corners=False,
            ).squeeze(1)

        losses.update({"loss_grounding_sub_bce_0": sigmoid_ce_loss_jit(point_sub_logits, point_sub_labels, len(src_sub_masks))})
        losses.update({"loss_grounding_sub_dice_0": dice_loss_jit(point_sub_logits, point_sub_labels, len(src_sub_masks))})
        losses.update({"loss_grounding_obj_bce_0": sigmoid_ce_loss_jit(point_obj_logits, point_obj_labels, len(src_obj_masks))})
        losses.update({"loss_grounding_obj_dice_0": dice_loss_jit(point_obj_logits, point_obj_labels, len(src_obj_masks))})

        # compute t2i loss
        loss_grd_ce = 0
        for b in range(len(targets)):
            task = targets[b]['grounding_task']
            pred_logit = outputs["pred_logits"][b]
            gt_logit = torch.zeros_like(pred_logit)
            select_idx = torch.stack((indices[b][0], indices[b][1])).tolist()
            gt_logit[select_idx] = 1

            num_gts = len(targets[b]['grounding_obj_mask'])
            t_hash = torch.tensor([i for i in range(num_gts)], device=gt_logit.device)
            hash_table = torch.zeros((len(t_hash), len(t_hash)), device=gt_logit.device)
            for idx in range(0, len(hash_table)):
                hash_table[idx][t_hash==t_hash[idx]] = 1
            hash_table = hash_table / hash_table.sum(-1, keepdim=True)
            gt_logit = gt_logit @ hash_table
            loss_grd_ce += self.grounding_weight*torch.sum(-gt_logit.t() * F.log_softmax(pred_logit.t(), dim=-1), dim=-1).mean()
        loss_grd_ce = loss_grd_ce / len(indices)
        losses.update({"loss_grounding_ce_0": loss_grd_ce})
        del src_sub_masks
        del target_sub_masks
        del src_obj_masks
        del target_obj_masks
        return losses
    
    def loss_grounding_psg(self, outputs, targets, indices, layer_id, extra):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_gsub_masks" in outputs
        assert "pred_gobj_masks" in outputs
        assert "pred_gobjs" in outputs
        assert "pred_gverbs" in outputs
        assert "pred_gsubs" in outputs
        assert "pred_gobj_texts" in outputs
        assert "pred_gverb_texts" in outputs
        assert "pred_gsub_texts" in outputs
        
        # if layer_id >= self.top_x_layers['grounding']:
        #     return {"loss_grounding_bce_0": 0, "loss_grounding_dice_0": 0, "loss_grounding_ce_0": 0}
        
        losses = {}

        # print(indices)
        valid_b = []
        for idx, (i, j) in enumerate(indices):
            if len(i)!=0:
                valid_b.append(idx)
        
        if len(valid_b)==0:
            loss = outputs['pred_gsub_masks'].sum() * 0.0
            return {"loss_grounding_sub_bce_0": loss, "loss_grounding_sub_dice_0": loss, "loss_grounding_obj_bce_0": loss, "loss_grounding_obj_dice_0": loss, "loss_grounding_ce_0": loss, "loss_grounding_verb_bce_0": loss, "loss_grounding_obj_ce_0": loss}

        
        tasks = [targets[idx]['grounding_task'] for idx in valid_b]
        
        targets = [targets[idx] for idx in valid_b]
        outputs = {k: v[valid_b] for k, v in outputs.items()}

        # print(task)
        pred_logits = []
        for b in range(len(targets)):
            num_gts = len(targets[b]['grounding_obj_mask'])
            t_emb = torch.cat([targets[b]['grounding_class_embs'] for _ in range(num_gts)], dim=0)
            v_emb_obj = outputs["pred_gobj_texts"][b]
            v_emb_verb = outputs["pred_gverb_texts"][b]
            v_emb_sub = outputs["pred_gsub_texts"][b]

            t_emb = t_emb / (t_emb.norm(dim=-1, keepdim=True) + 1e-7)

            v_emb_obj = v_emb_obj / (v_emb_obj.norm(dim=-1, keepdim=True) + 1e-7)  

            v_emb_verb = v_emb_verb / (v_emb_verb.norm(dim=-1, keepdim=True) + 1e-7)   
            v_emb_sub = v_emb_sub / (v_emb_sub.norm(dim=-1, keepdim=True) + 1e-7)  

            out_prob_obj = vl_similarity(v_emb_obj, t_emb, temperature=extra['lang_logit_obj'])
            out_prob_sub = vl_similarity(v_emb_sub, t_emb, temperature=extra['lang_logit_sub'])
            out_prob_verb = vl_similarity(v_emb_verb, t_emb, temperature=extra['lang_logit_verb'])
            if tasks[b] == 'pred_verb':
                pred_logits += [out_prob_sub+out_prob_obj]
                # src_verb_logits = outputs['pred_gverbs'][b]
            if tasks[b] == 'pred_obj':
                pred_logits += [out_prob_sub+out_prob_verb]
                # src_obj_logits = outputs['pred_gobjs'][b]
            if tasks[b] == 'pred_sub':
                pred_logits += [out_prob_obj+out_prob_verb]  
            outputs['pred_logits'] = pred_logits

        
        indices = self.matcher(outputs, targets, mode='grounding', extra={'temperature':extra['lang_logit_obj']})
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        loss_verb_ce = outputs['pred_gsub_masks'].sum() * 0.0
        loss_obj_ce = outputs['pred_gsub_masks'].sum() * 0.0
        loss_sub_ce = outputs['pred_gsub_masks'].sum() * 0.0
        for b in range(len(targets)):
            if tasks[b] == 'pred_verb':
                src_verb_logits = outputs['pred_gverbs'][b]
                target_verb_classes_o = targets[b]['grounding_verb_class'][indices[b][1]]
                target_verb_classes = torch.full(src_verb_logits.shape[:1], 0, dtype=torch.int64, device=src_verb_logits.device)
                target_verb_classes[indices[b][0]] = target_verb_classes_o

                src_verb_logits = src_verb_logits.sigmoid()
                loss_verb_ce = F.cross_entropy(src_verb_logits, target_verb_classes, self.verb_empty_weight) + loss_verb_ce

            elif tasks[b] == 'pred_obj':
                src_obj_logits = outputs['pred_gobjs'][b]
                target_classes_o = targets[b]['grounding_obj_class'][indices[b][1]]
                target_classes = torch.full(src_obj_logits.shape[:1], self.num_obj_classes, dtype=torch.int64, device=src_obj_logits.device)
                target_classes[indices[b][0]] = target_classes_o
                loss_obj_ce = F.cross_entropy(src_obj_logits, target_classes, self.object_empty_weight) + loss_obj_ce

            elif tasks[b] == 'pred_sub':
                src_sub_logits = outputs['pred_gsubs'][b]
                target_classes_o = targets[b]['grounding_sub_class'][indices[b][1]]
                target_classes = torch.full(src_sub_logits.shape[:1], self.num_obj_classes, dtype=torch.int64, device=src_sub_logits.device)
                target_classes[indices[b][0]] = target_classes_o
                loss_sub_ce = F.cross_entropy(src_sub_logits, target_classes, self.subject_empty_weight) + loss_sub_ce


        losses.update({"loss_grounding_verb_bce_0": loss_verb_ce})
        losses.update({'loss_grounding_obj_ce_0': loss_obj_ce})
        losses.update({'loss_grounding_sub_ce_0': loss_sub_ce})
       


        src_sub_masks = outputs["pred_gsub_masks"]
        src_sub_masks = src_sub_masks[src_idx]

        src_obj_masks = outputs["pred_gobj_masks"]
        src_obj_masks = src_obj_masks[src_idx]

        target_sub_masks = torch.cat([t['sub_masks'][i] for t, (_, i) in zip(targets, indices)], dim=0).to(src_sub_masks)
        target_obj_masks = torch.cat([t['obj_masks'][i] for t, (_, i) in zip(targets, indices)], dim=0).to(src_obj_masks)
        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        src_sub_masks = src_sub_masks[:, None]
        src_obj_masks = src_obj_masks[:, None]
        target_sub_masks = target_sub_masks[:, None].to(torch.float)
        target_obj_masks = target_obj_masks[:, None].to(torch.float)
        
        with torch.no_grad():
            # sample point_coords
            point_sub_coords = get_uncertain_point_coords_with_randomness(
                                src_sub_masks,
                                lambda logits: calculate_uncertainty(logits),
                                self.num_points,
                                self.oversample_ratio,
                                self.importance_sample_ratio,
                            ).type(src_sub_masks.dtype)
            # get gt labels
            point_sub_labels = point_sample(
                target_sub_masks,
                point_sub_coords,
                align_corners=False,
            ).squeeze(1)

            # sample point_coords
            point_obj_coords = get_uncertain_point_coords_with_randomness(
                                src_obj_masks,
                                lambda logits: calculate_uncertainty(logits),
                                self.num_points,
                                self.oversample_ratio,
                                self.importance_sample_ratio,
                            ).type(src_obj_masks.dtype)
            # get gt labels
            point_obj_labels = point_sample(
                target_obj_masks,
                point_obj_coords,
                align_corners=False,
            ).squeeze(1)

        point_sub_logits = point_sample(
                src_sub_masks,
                point_sub_coords,
                align_corners=False,
            ).squeeze(1)
        point_obj_logits = point_sample(
                src_obj_masks,
                point_obj_coords,
                align_corners=False,
            ).squeeze(1)

        losses.update({"loss_grounding_sub_bce_0": sigmoid_ce_loss_jit(point_sub_logits, point_sub_labels, len(src_sub_masks))})
        losses.update({"loss_grounding_sub_dice_0": dice_loss_jit(point_sub_logits, point_sub_labels, len(src_sub_masks))})
        losses.update({"loss_grounding_obj_bce_0": sigmoid_ce_loss_jit(point_obj_logits, point_obj_labels, len(src_obj_masks))})
        losses.update({"loss_grounding_obj_dice_0": dice_loss_jit(point_obj_logits, point_obj_labels, len(src_obj_masks))})

        # compute t2i loss
        loss_grd_ce = 0
        for b in range(len(targets)):
            task = targets[b]['grounding_task']
            pred_logit = outputs["pred_logits"][b]
            gt_logit = torch.zeros_like(pred_logit)
            select_idx = torch.stack((indices[b][0], indices[b][1])).tolist()
            gt_logit[select_idx] = 1

            num_gts = len(targets[b]['grounding_obj_mask'])
            t_hash = torch.tensor([i for i in range(num_gts)], device=gt_logit.device)
            hash_table = torch.zeros((len(t_hash), len(t_hash)), device=gt_logit.device)
            for idx in range(0, len(hash_table)):
                hash_table[idx][t_hash==t_hash[idx]] = 1
            hash_table = hash_table / hash_table.sum(-1, keepdim=True)
            gt_logit = gt_logit @ hash_table
            loss_grd_ce += self.grounding_weight*torch.sum(-gt_logit.t() * F.log_softmax(pred_logit.t(), dim=-1), dim=-1).mean()
        loss_grd_ce = loss_grd_ce / len(indices)
        losses.update({"loss_grounding_ce_0": loss_grd_ce})
        del src_sub_masks
        del target_sub_masks
        del src_obj_masks
        del target_obj_masks
        return losses

    def loss_grounding_vrd(self, outputs, targets, indices, layer_id, extra):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_gsub_masks" in outputs
        assert "pred_gobj_masks" in outputs
        assert "pred_gobjs" in outputs
        assert "pred_gverbs" in outputs
        assert "pred_gsubs" in outputs
        assert "pred_gobj_texts" in outputs
        assert "pred_gverb_texts" in outputs
        assert "pred_gsub_texts" in outputs
        
        
        losses = {}

        valid_b = []
        for idx, (i, j) in enumerate(indices):
            if len(i)!=0:
                valid_b.append(idx)
        
        if len(valid_b)==0:
            loss = outputs['pred_gsub_masks'].sum() * 0.0
            return {"loss_grounding_sub_bce_0": loss, "loss_grounding_sub_dice_0": loss, "loss_grounding_obj_bce_0": loss, "loss_grounding_obj_dice_0": loss, "loss_grounding_ce_0": loss, "loss_grounding_verb_bce_0": loss, "loss_grounding_obj_ce_0": loss}

        tasks = [targets[idx]['grounding_task'] for idx in valid_b]
        
        targets = [targets[idx] for idx in valid_b]
        outputs = {k: v[valid_b] for k, v in outputs.items()}

        pred_logits = []
        for b in range(len(targets)):
            num_gts = len(targets[b]['grounding_obj_mask'])
            t_emb = torch.cat([targets[b]['grounding_class_embs'] for _ in range(num_gts)], dim=0)
            v_emb_obj = outputs["pred_gobj_texts"][b]
            v_emb_verb = outputs["pred_gverb_texts"][b]
            v_emb_sub = outputs["pred_gsub_texts"][b]

            t_emb = t_emb / (t_emb.norm(dim=-1, keepdim=True) + 1e-7)

            v_emb_obj = v_emb_obj / (v_emb_obj.norm(dim=-1, keepdim=True) + 1e-7)  

            v_emb_verb = v_emb_verb / (v_emb_verb.norm(dim=-1, keepdim=True) + 1e-7)   
            v_emb_sub = v_emb_sub / (v_emb_sub.norm(dim=-1, keepdim=True) + 1e-7)  

            out_prob_obj = vl_similarity(v_emb_obj, t_emb, temperature=extra['lang_logit_obj'])
            out_prob_sub = vl_similarity(v_emb_sub, t_emb, temperature=extra['lang_logit_sub'])
            out_prob_verb = vl_similarity(v_emb_verb, t_emb, temperature=extra['lang_logit_verb'])
            
            if tasks[b] == 'pred_obj':
                pred_logits += [out_prob_sub+out_prob_verb]
                # src_obj_logits = outputs['pred_gobjs'][b]
            if tasks[b] == 'pred_sub':
                pred_logits += [out_prob_obj+out_prob_verb]  
            if tasks[b] == 'pred_sub_obj':
                pred_logits += [out_prob_verb]  
            outputs['pred_logits'] = pred_logits

        
        indices = self.matcher(outputs, targets, mode='grounding', extra={'temperature':extra['lang_logit_obj']})
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        loss_verb_ce = outputs['pred_gsub_masks'].sum() * 0.0
        loss_obj_ce = outputs['pred_gsub_masks'].sum() * 0.0
        loss_sub_ce = outputs['pred_gsub_masks'].sum() * 0.0
        for b in range(len(targets)):
            if tasks[b] == 'pred_verb':
                src_verb_logits = outputs['pred_gverbs'][b]
                target_verb_classes_o = targets[b]['grounding_verb_class'][indices[b][1]]
                target_verb_classes = torch.zeros_like(src_verb_logits, device=src_verb_logits.device)
                target_verb_classes[indices[b][0]] = target_verb_classes_o

                src_verb_logits = src_verb_logits.sigmoid()
                loss_verb_ce = self._neg_loss(src_verb_logits, target_verb_classes, weights=self.verb_empty_weight, alpha=0.5) + loss_verb_ce


            elif tasks[b] == 'pred_obj':
                src_obj_logits = outputs['pred_gobjs'][b]
                target_classes_o = targets[b]['grounding_obj_class'][indices[b][1]]
                target_classes = torch.full(src_obj_logits.shape[:1], self.num_obj_classes, dtype=torch.int64, device=src_obj_logits.device)
                target_classes[indices[b][0]] = target_classes_o
                loss_obj_ce = F.cross_entropy(src_obj_logits, target_classes, self.object_empty_weight) + loss_obj_ce

            elif tasks[b] == 'pred_sub':
                src_sub_logits = outputs['pred_gsubs'][b]
                target_classes_o = targets[b]['grounding_sub_class'][indices[b][1]]
                target_classes = torch.full(src_sub_logits.shape[:1], self.num_obj_classes, dtype=torch.int64, device=src_sub_logits.device)
                target_classes[indices[b][0]] = target_classes_o
                loss_sub_ce = F.cross_entropy(src_sub_logits, target_classes, self.subject_empty_weight) + loss_sub_ce

            elif tasks[b] == 'pred_sub_obj':
                src_obj_logits = outputs['pred_gobjs'][b]
                target_classes_o = targets[b]['grounding_obj_class'][indices[b][1]]
                target_classes = torch.full(src_obj_logits.shape[:1], self.num_obj_classes, dtype=torch.int64, device=src_obj_logits.device)
                target_classes[indices[b][0]] = target_classes_o
                loss_obj_ce = F.cross_entropy(src_obj_logits, target_classes, self.object_empty_weight) + loss_obj_ce

                src_sub_logits = outputs['pred_gsubs'][b]
                target_classes_o_sub = targets[b]['grounding_sub_class'][indices[b][1]]
                target_classes_sub = torch.full(src_sub_logits.shape[:1], self.num_obj_classes, dtype=torch.int64, device=src_sub_logits.device)
                target_classes_sub[indices[b][0]] = target_classes_o_sub
                loss_sub_ce = F.cross_entropy(src_sub_logits, target_classes_sub, self.subject_empty_weight) + loss_sub_ce


        losses.update({"loss_grounding_verb_bce_0": loss_verb_ce})
        losses.update({'loss_grounding_obj_ce_0': loss_obj_ce})
        losses.update({'loss_grounding_sub_ce_0': loss_sub_ce})
       


        src_sub_masks = outputs["pred_gsub_masks"]
        src_sub_masks = src_sub_masks[src_idx]

        src_obj_masks = outputs["pred_gobj_masks"]
        src_obj_masks = src_obj_masks[src_idx]

        target_sub_masks = torch.cat([t['sub_masks'][i] for t, (_, i) in zip(targets, indices)], dim=0).to(src_sub_masks)
        target_obj_masks = torch.cat([t['obj_masks'][i] for t, (_, i) in zip(targets, indices)], dim=0).to(src_obj_masks)
        
        # N x 1 x H x W
        src_sub_masks = src_sub_masks[:, None]
        src_obj_masks = src_obj_masks[:, None]
        target_sub_masks = target_sub_masks[:, None].to(torch.float)
        target_obj_masks = target_obj_masks[:, None].to(torch.float)
        
        with torch.no_grad():
            # sample point_coords
            point_sub_coords = get_uncertain_point_coords_with_randomness(
                                src_sub_masks,
                                lambda logits: calculate_uncertainty(logits),
                                self.num_points,
                                self.oversample_ratio,
                                self.importance_sample_ratio,
                            ).type(src_sub_masks.dtype)
            # get gt labels
            point_sub_labels = point_sample(
                target_sub_masks,
                point_sub_coords,
                align_corners=False,
            ).squeeze(1)

            # sample point_coords
            point_obj_coords = get_uncertain_point_coords_with_randomness(
                                src_obj_masks,
                                lambda logits: calculate_uncertainty(logits),
                                self.num_points,
                                self.oversample_ratio,
                                self.importance_sample_ratio,
                            ).type(src_obj_masks.dtype)
            # get gt labels
            point_obj_labels = point_sample(
                target_obj_masks,
                point_obj_coords,
                align_corners=False,
            ).squeeze(1)

        point_sub_logits = point_sample(
                src_sub_masks,
                point_sub_coords,
                align_corners=False,
            ).squeeze(1)
        point_obj_logits = point_sample(
                src_obj_masks,
                point_obj_coords,
                align_corners=False,
            ).squeeze(1)

        losses.update({"loss_grounding_sub_bce_0": sigmoid_ce_loss_jit(point_sub_logits, point_sub_labels, len(src_sub_masks))})
        losses.update({"loss_grounding_sub_dice_0": dice_loss_jit(point_sub_logits, point_sub_labels, len(src_sub_masks))})
        losses.update({"loss_grounding_obj_bce_0": sigmoid_ce_loss_jit(point_obj_logits, point_obj_labels, len(src_obj_masks))})
        losses.update({"loss_grounding_obj_dice_0": dice_loss_jit(point_obj_logits, point_obj_labels, len(src_obj_masks))})

        # compute t2i loss
        loss_grd_ce = 0
        for b in range(len(targets)):
            task = targets[b]['grounding_task']
            pred_logit = outputs["pred_logits"][b]
            gt_logit = torch.zeros_like(pred_logit)
            select_idx = torch.stack((indices[b][0], indices[b][1])).tolist()
            gt_logit[select_idx] = 1

            num_gts = len(targets[b]['grounding_obj_mask'])
            t_hash = torch.tensor([i for i in range(num_gts)], device=gt_logit.device)
            hash_table = torch.zeros((len(t_hash), len(t_hash)), device=gt_logit.device)
            for idx in range(0, len(hash_table)):
                hash_table[idx][t_hash==t_hash[idx]] = 1
            hash_table = hash_table / hash_table.sum(-1, keepdim=True)
            gt_logit = gt_logit @ hash_table
            loss_grd_ce += self.grounding_weight*torch.sum(-gt_logit.t() * F.log_softmax(pred_logit.t(), dim=-1), dim=-1).mean()
        loss_grd_ce = loss_grd_ce / len(indices)
        losses.update({"loss_grounding_ce_0": loss_grd_ce})
        del src_sub_masks
        del target_sub_masks
        del src_obj_masks
        del target_obj_masks
        return losses

    def loss_obj_labels(self, outputs, targets, indices, num_interactions, log=True):
        assert 'pred_obj_logits' in outputs
        src_logits = outputs['pred_obj_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['obj_labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_obj_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_obj_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.object_empty_weight)
        losses = {'loss_obj_ce': loss_obj_ce}

        if log:
            losses['obj_class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def loss_sub_labels(self, outputs, targets, indices, num_interactions, log=True):
        assert 'pred_sub_logits' in outputs
        src_logits = outputs['pred_sub_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['sub_labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_obj_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_sub_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.subject_empty_weight)
        losses = {'loss_sub_ce': loss_sub_ce}

        if log:
            losses['sub_class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def loss_rel_labels_vrd(self, outputs, targets, indices, num_interactions, log=True):
        assert 'pred_verb_logits' in outputs
        src_logits = outputs['pred_verb_logits']

        idx = self._get_src_permutation_idx(indices)
        
        target_classes_o = torch.cat([t['verb_labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_verb_classes,
                                    dtype=torch.int64, device=src_logits.device) ### 0-based
        target_classes[idx] = target_classes_o

        loss_verb_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.verb_empty_weight)
        losses = {'loss_verb_ce': loss_verb_ce}

        if log:
            losses['verb_class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses
    
    def loss_rel_labels(self, outputs, targets, indices, num_interactions, log=True):
        assert 'pred_verb_logits' in outputs
        src_logits = outputs['pred_verb_logits']

        idx = self._get_src_permutation_idx(indices)
        
        target_classes_o = torch.cat([t['verb_labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], 0,
                                    dtype=torch.int64, device=src_logits.device) ### 1-based, class 0 as background
        target_classes[idx] = target_classes_o

        loss_verb_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.verb_empty_weight)
        losses = {'loss_verb_ce': loss_verb_ce}

        if log:
            losses['verb_class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_obj_cardinality(self, outputs, targets, indices, num_interactions):
        pred_logits = outputs['pred_obj_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v['obj_labels']) for v in targets], device=device)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'obj_cardinality_error': card_err}
        return losses
    
    @torch.no_grad()
    def loss_sub_cardinality(self, outputs, targets, indices, num_interactions):
        pred_logits = outputs['pred_sub_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v['sub_labels']) for v in targets], device=device)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'sub_cardinality_error': card_err}
        return losses

    def loss_verb_labels(self, outputs, targets, indices, num_interactions):
        assert 'pred_verb_logits' in outputs
        src_logits = outputs['pred_verb_logits']

        idx = self._get_src_permutation_idx(indices)
  
        target_classes_o = torch.cat([t['verb_labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.zeros_like(src_logits)
        target_classes[idx] = target_classes_o

        src_logits = src_logits.sigmoid()
        loss_verb_ce = self._neg_loss(src_logits, target_classes, weights=self.verb_empty_weight, alpha=0.5)

        losses = {'loss_verb_ce': loss_verb_ce}
        return losses
    
    def loss_hoi_labels(self, outputs, targets, indices, num_interactions):
        assert 'pred_hoi_logits' in outputs
        src_logits = outputs['pred_hoi_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['hoi_labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.zeros_like(src_logits)
        target_classes[idx] = target_classes_o

        src_logits = src_logits.sigmoid()
        loss_hoi_ce = self._neg_loss(src_logits, target_classes, weights=None, alpha=0.5)

        losses = {'loss_hoi_ce': loss_hoi_ce}
        return losses

    def loss_sub_obj_boxes(self, outputs, targets, indices, num_interactions):
        assert 'pred_sub_boxes' in outputs and 'pred_obj_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_sub_boxes = outputs['pred_sub_boxes'][idx]
        src_obj_boxes = outputs['pred_obj_boxes'][idx]
        target_sub_boxes = torch.cat([t['sub_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_obj_boxes = torch.cat([t['obj_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        exist_obj_boxes = (target_obj_boxes != 0).any(dim=1)

        losses = {}
        if src_sub_boxes.shape[0] == 0:
            losses['loss_sub_bbox'] = src_sub_boxes.sum()
            losses['loss_obj_bbox'] = src_obj_boxes.sum()
            losses['loss_sub_giou'] = src_sub_boxes.sum()
            losses['loss_obj_giou'] = src_obj_boxes.sum()
        else:
            loss_sub_bbox = F.l1_loss(src_sub_boxes, target_sub_boxes, reduction='none')
            loss_obj_bbox = F.l1_loss(src_obj_boxes, target_obj_boxes, reduction='none')
            losses['loss_sub_bbox'] = loss_sub_bbox.sum() / num_interactions
            losses['loss_obj_bbox'] = (loss_obj_bbox * exist_obj_boxes.unsqueeze(1)).sum() / (exist_obj_boxes.sum() + 1e-4)
            loss_sub_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_sub_boxes),
                                                               box_cxcywh_to_xyxy(target_sub_boxes)))
            loss_obj_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_obj_boxes),
                                                               box_cxcywh_to_xyxy(target_obj_boxes)))
            losses['loss_sub_giou'] = loss_sub_giou.sum() / num_interactions
            losses['loss_obj_giou'] = (loss_obj_giou * exist_obj_boxes).sum() / (exist_obj_boxes.sum() + 1e-4)
        return losses

    def loss_sub_obj_masks(self, outputs, targets, indices, num_interactions):
        assert 'pred_sub_masks' in outputs and 'pred_obj_masks' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_sub_masks = outputs['pred_sub_masks'][idx]
        src_obj_masks = outputs['pred_obj_masks'][idx]
        target_sub_masks = torch.cat([t['sub_masks'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_obj_masks = torch.cat([t['obj_masks'][i] for t, (_, i) in zip(targets, indices)], dim=0)


        src_sub_masks = src_sub_masks[:, None]
        src_obj_masks = src_obj_masks[:, None]
        target_sub_masks = target_sub_masks[:, None].to(torch.float)
        target_obj_masks = target_obj_masks[:, None].to(torch.float)

        losses = {}
        if src_sub_masks.shape[0] == 0:
            losses['loss_sub_mask'] = src_sub_masks.sum()
            losses['loss_obj_mask'] = src_obj_masks.sum()
            losses['loss_sub_dice'] = src_sub_masks.sum()
            losses['loss_obj_dice'] = src_obj_masks.sum()
        else:
            with torch.no_grad():
                # sample point_coords
                point_sub_coords = get_uncertain_point_coords_with_randomness(
                                    src_sub_masks,
                                    lambda logits: calculate_uncertainty(logits),
                                    self.num_points,
                                    self.oversample_ratio,
                                    self.importance_sample_ratio,
                                ).type(src_sub_masks.dtype)
                # get gt labels
                point_sub_labels = point_sample(
                    target_sub_masks,
                    point_sub_coords,
                    align_corners=False,
                ).squeeze(1)

                # sample point_coords
                point_obj_coords = get_uncertain_point_coords_with_randomness(
                                    src_obj_masks,
                                    lambda logits: calculate_uncertainty(logits),
                                    self.num_points,
                                    self.oversample_ratio,
                                    self.importance_sample_ratio,
                                ).type(src_obj_masks.dtype)
                # get gt labels
                point_obj_labels = point_sample(
                    target_obj_masks,
                    point_obj_coords,
                    align_corners=False,
                ).squeeze(1)

            point_sub_logits = point_sample(
                    src_sub_masks,
                    point_sub_coords,
                    align_corners=False,
                ).squeeze(1)
            point_obj_logits = point_sample(
                    src_obj_masks,
                    point_obj_coords,
                    align_corners=False,
                ).squeeze(1)

            losses['loss_sub_mask'] = sigmoid_ce_loss_jit(point_sub_logits, point_sub_labels, num_interactions)
            losses['loss_obj_mask'] = sigmoid_ce_loss_jit(point_obj_logits, point_obj_labels, num_interactions)
            losses['loss_sub_dice'] = dice_loss_jit(point_sub_logits, point_sub_labels, num_interactions)
            losses['loss_obj_dice'] = dice_loss_jit(point_obj_logits, point_obj_labels, num_interactions)

        del src_sub_masks
        del src_obj_masks
        del target_sub_masks
        del target_obj_masks
        return losses

    def _neg_loss(self, pred, gt, weights=None, alpha=0.25):
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        loss = 0

        pos_loss = alpha * torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
        if weights is not None:
            pos_loss = pos_loss * weights[:-1]

        neg_loss = (1 - alpha) * torch.log(1 - pred) * torch.pow(pred, 2) * neg_inds

        num_pos  = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
        return loss

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num, **kwargs):
        
        loss_map = {
            'sub_labels': self.loss_sub_labels,
            'obj_labels': self.loss_obj_labels,
            'obj_cardinality': self.loss_obj_cardinality,
            'verb_labels': self.loss_verb_labels,
            'hoi_labels': self.loss_hoi_labels,
            'sub_obj_masks': self.loss_sub_obj_masks,
            'sub_obj_boxes': self.loss_sub_obj_boxes,
            'grounding': self.loss_grounding,
            'grounding_psg': self.loss_grounding_psg,
            'grounding_vrd': self.loss_grounding_vrd,
            'rel_labels': self.loss_rel_labels,
            'rel_labels_vrd': self.loss_rel_labels_vrd,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num, **kwargs)

    def forward(self, outputs, targets, extra=None):
        
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        if extra['source'] == 'psg' or extra['source'] == 'vrd' or extra['source'] == 'o365':
            indices = self.matcher(outputs_without_aux, targets, mode='psg')
        else:
            indices = self.matcher(outputs_without_aux, targets)

        num_interactions = sum(len(t['obj_labels']) for t in targets)
        num_interactions = torch.as_tensor([num_interactions], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_interactions)
        num_interactions = torch.clamp(num_interactions / get_world_size(), min=1).item()

        losses = {}
        for loss in self.losses:
            if loss == 'grounding' or loss == 'grounding_psg' or loss == 'grounding_vrd':
                losses.update(self.get_loss(loss, outputs, targets, indices, num_interactions, extra=extra))
            else:
                losses.update(self.get_loss(loss, outputs, targets, indices, num_interactions))
            
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets, mode='psg')
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'obj_labels':
                        kwargs = {'log': False}
                    if loss == 'sub_labels':
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_interactions, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses



class PostProcessHOI(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.subject_category_id = args.subject_category_id
        self.use_matching = args.use_matching
        self.use_triplet= args.use_triplet

    def resize_mask_to_original(self, bs, out_sub_masks, image_size, target_sizes, out_obj_masks):
        re_sub_masks = []
        re_obj_masks = []
        for b in range(bs):
            sub_mask = out_sub_masks[b][:, :image_size[b][0], :image_size[b][1]].expand(1, -1, -1, -1)
            sub_mask = F.interpolate(
                sub_mask,
                size=tuple(target_sizes[b].tolist()),
                mode="bicubic",
                align_corners=False,
                antialias=True
            ).squeeze(0)
            obj_mask = out_obj_masks[b][:, :image_size[b][0], :image_size[b][1]].expand(1, -1, -1, -1)
            obj_mask = F.interpolate(
                obj_mask,
                size=tuple(target_sizes[b].tolist()),
                mode="bicubic",
                align_corners=False,
                antialias=True
            ).squeeze(0)
            re_sub_masks.append(sub_mask)
            re_obj_masks.append(obj_mask)
        return re_sub_masks, re_obj_masks
    
    @torch.no_grad()
    def forward(self, outputs, target_sizes, image_shape, image_size):
        out_obj_logits = outputs['pred_obj_logits']
        out_verb_logits = outputs['pred_verb_logits']
        out_sub_masks = outputs['pred_sub_masks']
        out_obj_masks = outputs['pred_obj_masks']

        if self.use_triplet:
            out_hoi_logits = outputs['pred_hoi_logits']

        assert len(out_obj_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        obj_prob = F.softmax(out_obj_logits, -1)
        obj_scores, obj_labels = obj_prob[..., :-1].max(-1)

        verb_scores = out_verb_logits.sigmoid()
        if self.use_triplet:
            hoi_scores = out_hoi_logits.sigmoid()
            obj_scores = out_obj_logits.sigmoid()

        if self.use_matching:
            out_matching_logits = outputs['pred_matching_logits']
            matching_scores = F.softmax(out_matching_logits, -1)[..., 1]

        
        out_sub_masks = F.interpolate(
            out_sub_masks,
            size=(image_shape[0], image_shape[1]),
            mode="bilinear",
            align_corners=False,
            antialias=True
        )

        out_obj_masks = F.interpolate(
            out_obj_masks,
            size=(image_shape[0], image_shape[1]),
            mode="bilinear",
            align_corners=False,
            antialias=True
        )
        
        del outputs
        
        bs = target_sizes.shape[0]
        re_sub_masks, re_obj_masks = retry_if_cuda_oom(self.resize_mask_to_original)(bs, out_sub_masks, image_size, target_sizes, out_obj_masks)

        results = []
        for index in range(len(obj_scores)):
            os, ol, vs, sm, om =  obj_scores[index], obj_labels[index], verb_scores[index], re_sub_masks[index], re_obj_masks[index]
            sl = torch.full_like(ol, self.subject_category_id)
            l = torch.cat((sl, ol))
            sm = (sm > 0).float()
            om = (om > 0).float()

            m = torch.cat((sm, om))
            # compress
            m = [mask_util.encode(np.array(m[i].to('cpu'), order="F", dtype="uint8")) for i in range(m.shape[0])]
            results.append({'labels': l.to('cpu'), 'masks': m})
            if self.use_triplet:
                hs = hoi_scores[index]
                # hois = hois * os.unsqueeze(1)
            else:
                vs = vs * os.unsqueeze(1)

            ids = torch.arange(len(m))

            if self.use_triplet:
                results[-1].update({'hoi_scores': hs.to('cpu'), 'obj_scores': os.to('cpu'), 'sub_ids': ids[:ids.shape[0] // 2],
                                'obj_ids': ids[ids.shape[0] // 2:]})
            else:
                results[-1].update({'verb_scores': vs.to('cpu'), 'sub_ids': ids[:ids.shape[0] // 2],
                                    'obj_ids': ids[ids.shape[0] // 2:]})

        return results

def dedup_triplets(triplets_ids, s_binary_masks, o_binary_masks, keep_tri):
    while len(triplets_ids) > 1:
        base_s_mask = s_binary_masks[triplets_ids[0]].unsqueeze(0)
        base_o_mask = o_binary_masks[triplets_ids[0]].unsqueeze(0)
        other_s_masks = s_binary_masks[triplets_ids[1:]]
        other_o_masks = o_binary_masks[triplets_ids[1:]]
        # calculate ious
        s_ious = base_s_mask.mm(other_s_masks.transpose(0,1))/((base_s_mask+other_s_masks)>0).sum(-1)
        o_ious = base_o_mask.mm(other_o_masks.transpose(0,1))/((base_o_mask+other_o_masks)>0).sum(-1)
        ids_left = []
        for s_iou, o_iou, other_id in zip(s_ious[0],o_ious[0],triplets_ids[1:]):
            if (s_iou>0.5) & (o_iou>0.5):
                keep_tri[other_id] = False
            else:
                ids_left.append(other_id)
        triplets_ids = ids_left
    return keep_tri

class PostProcessPSG(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.use_matching = args.use_matching
        self.num_relations = args.num_verb_classes

    def resize_mask_to_original(self, out_sub_masks, image_size, target_sizes, out_obj_masks):
        
        sub_mask = out_sub_masks[0][:, :image_size[0], :image_size[1]].expand(1, -1, -1, -1)
        sub_mask = F.interpolate(
            sub_mask,
            size=tuple(target_sizes.tolist()),
            mode="bicubic",
            align_corners=False,
            antialias=True
        ).squeeze(0)
        obj_mask = out_obj_masks[0][:, :image_size[0], :image_size[1]].expand(1, -1, -1, -1)
        obj_mask = F.interpolate(
            obj_mask,
            size=tuple(target_sizes.tolist()),
            mode="bicubic",
            align_corners=False,
            antialias=True
        ).squeeze(0)
    
        return sub_mask, obj_mask
    
    @torch.no_grad()
    def forward(self, outputs, target_sizes, image_shape, image_size):
        bsize = outputs['pred_sub_logits'].shape[0]
        out_sub_logits = outputs['pred_sub_logits']
        out_obj_logits = outputs['pred_obj_logits']
        out_verb_logits = outputs['pred_verb_logits']
        out_sub_masks = outputs['pred_sub_masks']
        out_obj_masks = outputs['pred_obj_masks']

        assert len(out_obj_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        results = []
        # for each single pred
        for i in range(bsize):

            sub_prob = F.softmax(out_sub_logits[i], -1)
            sub_scores, sub_labels = sub_prob[..., :-1].max(-1)
            obj_prob = F.softmax(out_obj_logits[i], -1)
            obj_scores, obj_labels = obj_prob[..., :-1].max(-1)

            r_lgs = F.softmax(out_verb_logits[i], dim=-1) # 100x57
            r_logits = r_lgs[..., 1:] # 100x56
            r_scores, r_indexes = r_logits.reshape(-1).topk(100) # 100
            r_labels = r_indexes % self.num_relations + 1 # 100
            triplet_index = r_indexes // self.num_relations # 100

            s_scores = sub_scores[triplet_index]
            s_labels = sub_labels[triplet_index] + 1

            o_scores = obj_scores[triplet_index]
            o_labels = obj_labels[triplet_index] + 1

            r_dists = r_lgs.reshape(
                    -1, self.num_relations +
                    1)[triplet_index]  #### NOTE: to match the evaluation in vg

            o_mask_pred = out_obj_masks[i][triplet_index]
            s_mask_pred = out_sub_masks[i][triplet_index]
            s_mask_pred = F.interpolate(
                s_mask_pred.unsqueeze(1),
                size=(image_shape[0], image_shape[1]),
                mode="bilinear",
                align_corners=False,
                antialias=True
            ).squeeze(1)

            o_mask_pred = F.interpolate(
                o_mask_pred.unsqueeze(1),
                size=(image_shape[0], image_shape[1]),
                mode="bilinear",
                align_corners=False,
                antialias=True
            ).squeeze(1)
            s_mask_pred, o_mask_pred = retry_if_cuda_oom(self.resize_mask_to_original)(s_mask_pred.unsqueeze(0), image_size[i], target_sizes[i], o_mask_pred.unsqueeze(0))
            
            s_mask_pred = torch.sigmoid(s_mask_pred) > 0.85
            o_mask_pred = torch.sigmoid(o_mask_pred) > 0.85

            ### triplets deduplicate####
            relation_classes = defaultdict(lambda: [])
            for k, (s_l,o_l,r_l) in enumerate(zip(s_labels,o_labels,r_labels)):
                relation_classes[(s_l.item(),o_l.item(),r_l.item())].append(k)
            s_binary_masks = s_mask_pred.to(torch.float).flatten(1)
            o_binary_masks = o_mask_pred.to(torch.float).flatten(1)

            keep_tri = torch.ones_like(r_labels,dtype=torch.bool)
            for triplets_ids in relation_classes.values():
                if len(triplets_ids)>1:
                    keep_tri = dedup_triplets(triplets_ids, s_binary_masks, o_binary_masks, keep_tri)

            s_labels = s_labels[keep_tri] 
            o_labels = o_labels[keep_tri]
            s_mask_pred = s_mask_pred[keep_tri]
            o_mask_pred = o_mask_pred[keep_tri]

            o_scores = o_scores[keep_tri]
            s_scores = s_scores[keep_tri]

            complete_labels = torch.cat((s_labels, o_labels), 0) # 194
            output_masks = torch.cat((s_mask_pred, o_mask_pred), 0) # [194, 480, 640]
            r_scores = r_scores[keep_tri]
            r_labels = r_labels[keep_tri]
            r_dists = r_dists[keep_tri]
            rel_pairs = torch.arange(keep_tri.sum()*2,
                            dtype=torch.int).reshape(2, -1).T # need
            complete_r_labels = r_labels # need
            complete_r_dists = r_dists # need
            complete_obj_scores = torch.cat((s_scores, o_scores), 0) # 194

            # save to Result
            labels = complete_labels.detach().cpu().numpy() # 148
            complete_o_scores = complete_obj_scores.detach().cpu().numpy() # 194
            rel_pairs = rel_pairs.detach().cpu().numpy() # 74x2
            complete_r_labels = complete_r_labels.detach().cpu().numpy() # 74
            complete_r_dists = complete_r_dists.detach().cpu().numpy() # 74x57
            masks = output_masks.detach().cpu().numpy() # 148x480x640

            compressed_masks = [mask_util.encode(np.array(output_masks[i].to('cpu'), order="F", dtype="uint8")) for i in range(output_masks.shape[0])]
            
            assert masks.shape[1]==target_sizes[i][0]
            assert masks.shape[2]==target_sizes[i][1]
            
            results.append(
                Result(o_scores=complete_o_scores,
                        labels=labels, # 148
                        rel_pair_idxes=rel_pairs, # 74x2
                        rel_dists=complete_r_dists, # 74x57
                        rel_labels=complete_r_labels, # 74
                        masks=compressed_masks) # 148x480x640
            )

       
        
        del outputs

        return results



class PostProcessPanoptic(nn.Module):
    """This class converts the output of the model to the final panoptic result, in the format expected by the
    coco panoptic API """

    def __init__(self, is_thing_map, threshold=0.85):
        """
        Parameters:
           is_thing_map: This is a whose keys are the class ids, and the values a boolean indicating whether
                          the class is  a thing (True) or a stuff (False) class
           threshold: confidence threshold: segments with confidence lower than this will be deleted
        """
        super().__init__()
        self.threshold = threshold
        self.is_thing_map = is_thing_map

    def forward(self, outputs, processed_sizes, target_sizes=None):
        """ This function computes the panoptic prediction from the model's predictions.
        Parameters:
            outputs: This is a dict coming directly from the model. See the model doc for the content.
            processed_sizes: This is a list of tuples (or torch tensors) of sizes of the images that were passed to the
                             model, ie the size after data augmentation but before batching.
            target_sizes: This is a list of tuples (or torch tensors) corresponding to the requested final size
                          of each prediction. If left to None, it will default to the processed_sizes
            """
        if target_sizes is None:
            target_sizes = processed_sizes
        assert len(processed_sizes) == len(target_sizes)

        out_logits, raw_masks = outputs["pred_obj_logits"], outputs["pred_obj_masks"]
        assert len(out_logits) == len(raw_masks) == len(target_sizes)
        preds = []

        def to_tuple(tup):
            if isinstance(tup, tuple):
                return tup
            return tuple(tup.cpu().tolist())

        for cur_logits, cur_masks, size, target_size in zip(
            out_logits, raw_masks, processed_sizes, target_sizes
        ):
            # we filter empty queries and detection below threshold
            scores, labels = cur_logits.softmax(-1).max(-1)
            keep = labels.ne(outputs["pred_obj_logits"].shape[-1] - 1) & (scores > self.threshold)
            cur_scores, cur_classes = cur_logits.softmax(-1).max(-1)
            cur_scores = cur_scores[keep]
            cur_classes = cur_classes[keep]
            cur_masks = cur_masks[keep]
            cur_masks = interpolate(cur_masks[:, None], to_tuple(size), mode="bilinear").squeeze(1)
            # cur_boxes = box_ops.box_cxcywh_to_xyxy(cur_boxes[keep])

            h, w = cur_masks.shape[-2:]
            # assert len(cur_boxes) == len(cur_classes)

            # It may be that we have several predicted masks for the same stuff class.
            # In the following, we track the list of masks ids for each stuff class (they are merged later on)
            cur_masks = cur_masks.flatten(1)
            stuff_equiv_classes = defaultdict(lambda: [])
            for k, label in enumerate(cur_classes):
                if not self.is_thing_map[label.item()]:
                    stuff_equiv_classes[label.item()].append(k)

            def get_ids_area(masks, scores, dedup=False):
                # This helper function creates the final panoptic segmentation image
                # It also returns the area of the masks that appears on the image

                m_id = masks.transpose(0, 1).softmax(-1)

                if m_id.shape[-1] == 0:
                    # We didn't detect any mask :(
                    m_id = torch.zeros((h, w), dtype=torch.long, device=m_id.device)
                else:
                    m_id = m_id.argmax(-1).view(h, w)

                if dedup:
                    # Merge the masks corresponding to the same stuff class
                    for equiv in stuff_equiv_classes.values():
                        if len(equiv) > 1:
                            for eq_id in equiv:
                                m_id.masked_fill_(m_id.eq(eq_id), equiv[0])

                final_h, final_w = to_tuple(target_size)

                seg_img = Image.fromarray(id2rgb(m_id.view(h, w).cpu().numpy()))
                seg_img = seg_img.resize(size=(final_w, final_h), resample=Image.NEAREST)

                np_seg_img = (
                    torch.ByteTensor(torch.ByteStorage.from_buffer(seg_img.tobytes())).view(final_h, final_w, 3).numpy()
                )
                m_id = torch.from_numpy(rgb2id(np_seg_img))

                area = []
                for i in range(len(scores)):
                    area.append(m_id.eq(i).sum().item())
                return area, seg_img

            area, seg_img = get_ids_area(cur_masks, cur_scores, dedup=True)
            if cur_classes.numel() > 0:
                # We know filter empty masks as long as we find some
                while True:
                    filtered_small = torch.as_tensor(
                        [area[i] <= 4 for i, c in enumerate(cur_classes)], dtype=torch.bool, device=keep.device
                    )
                    if filtered_small.any().item():
                        cur_scores = cur_scores[~filtered_small]
                        cur_classes = cur_classes[~filtered_small]
                        cur_masks = cur_masks[~filtered_small]
                        area, seg_img = get_ids_area(cur_masks, cur_scores)
                    else:
                        break

            else:
                cur_classes = torch.ones(1, dtype=torch.long, device=cur_classes.device)

            segments_info = []
            for i, a in enumerate(area):
                cat = cur_classes[i].item()
                segments_info.append({"id": i, "isthing": self.is_thing_map[cat], "category_id": cat, "area": a})
            del cur_classes

            with io.BytesIO() as out:
                seg_img.save(out, format="PNG")
                predictions = {"png_string": out.getvalue(), "segments_info": segments_info}
            preds.append(predictions)
        return preds


def build(args):
    device = torch.device(args.device)
    if args.backbone == 'focall':
        from .focal import build_backbone
        backbone = build_backbone()

    from .pixel_decoder import build_pixel_decoder
    if args.use_gpt_emb:
        from .hoi_decoder_wo_lang import build_hoi_decoder 
    else:
        from .hoi_decoder import build_hoi_decoder
    pixel_decoder = build_pixel_decoder(args)

    matcher = build_matcher(args)
    hoi_decoder = build_hoi_decoder(args, matcher)
    model = HOI(backbone, pixel_decoder=pixel_decoder, hoi_decoder=hoi_decoder, args=args)
    

    weight_dict = {}
    weight_dict['loss_obj_ce'] = args.obj_loss_coef
    weight_dict['loss_verb_ce'] = args.verb_loss_coef
    weight_dict['loss_sub_mask'] = args.mask_bce_loss_coef
    weight_dict['loss_obj_mask'] = args.mask_bce_loss_coef
    weight_dict['loss_sub_dice'] = args.mask_dice_loss_coef
    weight_dict['loss_obj_dice'] = args.mask_dice_loss_coef

    weight_dict['loss_sub_bbox'] = args.box_reg_loss_coef
    weight_dict['loss_obj_bbox'] = args.box_reg_loss_coef
    weight_dict['loss_obj_giou'] = args.box_giou_loss_coef
    weight_dict['loss_sub_giou'] = args.box_giou_loss_coef


    if args.psg or args.vrd:
         weight_dict['loss_sub_ce'] = args.obj_loss_coef
    if args.use_triplet:
        weight_dict['loss_hoi_ce'] = args.tri_loss_coef

    if args.flexible_grounding:
        # weight_dict['loss_grd'] = args.grd_loss_coef
        weight_dict['loss_grounding_ce_0'] = args.grd_loss_coef
        weight_dict['loss_grounding_sub_dice_0'] = 1.0
        weight_dict['loss_grounding_sub_bce_0'] = 1.0
        weight_dict['loss_grounding_obj_dice_0'] = 1.0
        weight_dict['loss_grounding_obj_bce_0'] = 1.0
        weight_dict['loss_grounding_verb_bce_0'] = 1.0
        weight_dict['loss_grounding_obj_ce_0'] = 1.0
        weight_dict['loss_grounding_sub_ce_0'] = 1.0
    
    if args.psg and args.dataset_file != 'hico+vcoco+psg':
        losses = ['sub_labels', 'obj_labels', 'rel_labels', 'sub_obj_masks', 'obj_cardinality']
    elif args.hoi:
        losses = ['obj_labels', 'verb_labels', 'obj_cardinality']
        if args.use_mask:
            losses.append('sub_obj_masks')
        if args.use_box:
            losses.append('sub_obj_boxes')
        if args.only_grd_loss:
            losses = []
        
    elif args.vrd:
        losses = ['sub_labels', 'obj_labels', 'rel_labels_vrd', 'sub_obj_masks', 'obj_cardinality']
    if args.use_triplet and args.dataset_file != 'hico+vcoco' and args.dataset_file != 'hico+vcoco+psg':
        losses.append('hoi_labels')
        losses.remove('verb_labels')
    if args.flexible_grounding:
        if args.hoi:
            losses.append('grounding')
        elif args.psg:
            losses.append('grounding_psg')
        elif args.vrd:
            losses.append('grounding_vrd')

    if args.dataset_file == 'hico+vcoco+psg':
        losses_hico = ['obj_labels', 'verb_labels', 'sub_obj_masks', 'obj_cardinality']
        losses_psg = ['sub_labels', 'obj_labels', 'rel_labels', 'sub_obj_masks', 'obj_cardinality']
        losses_vcoco = ['obj_labels', 'hoi_labels', 'sub_obj_masks', 'obj_cardinality']

        matcher_hoi = build_matcher(args, name="hico")
        matcher_psg = build_matcher(args, name="psg")
        criterion_hico = SetCriterionHOI(args.num_obj_classes_hico, args.num_queries, args.num_verb_classes_hico, args.num_hoi_classes, matcher=matcher_hoi,
                                    weight_dict=weight_dict, eos_coef=args.eos_coef_hoi, losses=losses_hico,
                                    args=args)
        criterion_vcoco = SetCriterionHOI(args.num_obj_classes_vcoco, args.num_queries, args.num_verb_classes_vcoco, args.num_hoi_classes, matcher=matcher_hoi,
                                    weight_dict=weight_dict, eos_coef=args.eos_coef_hoi, losses=losses_vcoco,
                                    args=args)

        criterion_psg = SetCriterionHOI(args.num_obj_classes_psg, args.num_queries, args.num_verb_classes_psg, args.num_hoi_classes, matcher=matcher_psg,
                                    weight_dict=weight_dict, eos_coef=args.eos_coef_psg, losses=losses_psg,
                                    args=args)
        
        criterion_hico.to(device)
        criterion_vcoco.to(device)
        criterion_psg.to(device)


    elif args.dataset_file == 'hico+psg':
        losses_hico = ['obj_labels', 'verb_labels', 'sub_obj_masks', 'obj_cardinality']
        losses_psg = ['sub_labels', 'obj_labels', 'rel_labels', 'sub_obj_masks', 'obj_cardinality']

        matcher_hico = build_matcher(args, name="hico")
        matcher_psg = build_matcher(args, name="psg")
        criterion_hico = SetCriterionHOI(args.num_obj_classes_hico, args.num_queries, args.num_verb_classes_hico, args.num_hoi_classes, matcher=matcher_hico,
                                    weight_dict=weight_dict, eos_coef=args.eos_coef_hoi, losses=losses_hico,
                                    args=args)

        criterion_psg = SetCriterionHOI(args.num_obj_classes_psg, args.num_queries, args.num_verb_classes_psg, args.num_hoi_classes, matcher=matcher_psg,
                                    weight_dict=weight_dict, eos_coef=args.eos_coef_psg, losses=losses_psg,
                                    args=args)
        
        criterion_hico.to(device)
        criterion_psg.to(device)

    elif args.dataset_file == 'hico+vcoco':
        losses_hico = losses
        if args.use_triplet:
            losses_vcoco = ['obj_labels', 'hoi_labels', 'sub_obj_masks', 'obj_cardinality']
        else:
            losses_vcoco = losses
        criterion_hico = SetCriterionHOI(args.num_obj_classes_hico, args.num_queries, args.num_verb_classes_hico, args.num_hoi_classes, matcher=matcher,
                                    weight_dict=weight_dict, eos_coef=args.eos_coef, losses=losses_hico,
                                    args=args)
        criterion_vcoco = SetCriterionHOI(args.num_obj_classes_vcoco, args.num_queries, args.num_verb_classes_vcoco, args.num_hoi_classes, matcher=matcher,
                                    weight_dict=weight_dict, eos_coef=args.eos_coef, losses=losses_vcoco,
                                    args=args)
        
        criterion_hico.to(device)
        criterion_vcoco.to(device)
    else:

        criterion = SetCriterionHOI(args.num_obj_classes, args.num_queries, args.num_verb_classes, args.num_hoi_classes, matcher=matcher,
                                    weight_dict=weight_dict, eos_coef=args.eos_coef, losses=losses,
                                    args=args)

        criterion.to(device)
    postprocessors = {'hoi': PostProcessHOI(args), 'psg': PostProcessPSG(args)}

    if args.dataset_file == 'hico+vcoco+psg':
        return model, (criterion_hico, criterion_vcoco, criterion_psg), postprocessors
    elif args.dataset_file == 'hico+psg':
        return model, (criterion_hico, criterion_psg), postprocessors
    elif args.dataset_file == 'hico+vcoco':
        return model, (criterion_hico, criterion_vcoco), postprocessors
    else:
        return model, criterion, postprocessors
    


