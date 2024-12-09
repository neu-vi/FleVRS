import math
import sys
from typing import Iterable
import numpy as np
import copy
import itertools
import random

import torch

import util.misc as utils
from datasets.flexible_hico_eval import HICOEvaluator, Flexible_HICOEvaluator
from datasets.vcoco_eval import VCOCOEvaluator
from detectron2.structures import ImageList
import pycocotools.mask as mask_util
from util.optim import adjust_learning_rate


def compute_iou(mask1, mask2):
    # Convert boolean masks to integers (True becomes 1, False becomes 0)
    mask1 = mask1.astype(np.int64)
    mask2 = mask2.astype(np.int64)

    # Compute the intersection and union of true values
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)

    # Calculate the IoU
    iou = np.sum(intersection) / np.sum(union)

    return iou

def train_one_epoch_three(model: torch.nn.Module, criterion1: torch.nn.Module, criterion2: torch.nn.Module, criterion3: torch.nn.Module,
                    data_loader1: Iterable, data_loader2: Iterable, data_loader3: Iterable, iterators, optimizer: torch.optim.Optimizer, 
                    device: torch.device, epoch: int, max_norm: float = 0, lr_scheduler=None, flexible_grounding=False):
    model.train()
    criterion1.train()
    criterion2.train()
    criterion3.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    metric_logger.add_meter('obj_class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 1
    iter_num = 0

    weights = {
    'hico': 0.4,  # Adjust these weights as necessary
    'vcoco': 0.2,
    'psg': 0.4
    }

    dataloaders = {
        'hico': data_loader1,
        'vcoco': data_loader2,
        'psg': data_loader3
    }

    for _ in metric_logger.log_every(data_loader3, print_freq, header):

        dataset_choice = random.choices(list(iterators.keys()), list(weights.values()))[0]
        iterator = iterators[dataset_choice]

        try:
            samples = next(iterator)
        except StopIteration:
            iterators[dataset_choice] = iter(dataloaders[dataset_choice])
            samples = next(iterators[dataset_choice])
        
        source = samples[0]['source']
        samples = [{k: v.to(device) for k, v in sample.items() if k != 'filename' and k!='source'} for sample in samples]
        

        # profiler for CPU memory
        outputs, extra = model(samples, source=source)
        targets = samples

        iter_num += 1
        extra.update({'source': source})
        if source=='hico':
            loss_dict = criterion1(outputs, targets, extra)
            weight_dict = criterion1.weight_dict
        elif source=='psg':
            
            loss_dict = criterion3(outputs, targets, extra)
            weight_dict = criterion3.weight_dict
        elif source=='vcoco':
            loss_dict = criterion2(outputs, targets, extra)
            weight_dict = criterion2.weight_dict    

        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if lr_scheduler is not None:
            lr_scheduler.step()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        
        metric_logger.update(obj_class_error=loss_dict_reduced['obj_class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer, 
                    device: torch.device, epoch: int, max_norm: float = 0, flexible_grounding=False, args=None):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter("lr_backbone", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    if hasattr(criterion, 'loss_labels'):
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    else:
        metric_logger.add_meter('obj_class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    if args.psg:
        metric_logger.add_meter('verb_class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
        metric_logger.add_meter('sub_class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)

    print_freq = 10
    
    num_training_steps = int(len(data_loader) * args.epochs)
    for i, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        curr_step = epoch * len(data_loader) + i
        source = samples[0]['source']
        if not flexible_grounding and not args.query_based:
            
            samples = [{k: v.to(device) for k, v in sample.items() if k != 'filename' and k!='source' and k!='image_id'} for sample in samples]
            
        # profiler for CPU memory
        # with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True) as prof:
        if flexible_grounding:
            outputs, extra, targets = model(samples, source=source)
        else:
            outputs, extra = model(samples, source=source, query_based=args.query_based, use_gpt_emb=args.use_gpt_emb)
            targets = samples

        extra.update({'source': source})
        loss_dict = criterion(outputs, targets, extra)
        weight_dict = criterion.weight_dict
        
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        if args.gradient_strategy == "gradient_accumulation":
            # first iteration
            if (i + 1) % args.cumulative_iters == 1:
                accumulation_losses = losses
            # intermediate iteration
            elif (i + 1) % args.cumulative_iters != 0:
                accumulation_losses += losses
            else:
                accumulation_losses += losses
                optimizer.zero_grad()
                accumulation_losses.backward()
                if args.clip_max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_max_norm)
                optimizer.step()
                
        
        elif args.gradient_strategy == "vanilla":
            # print('vanilla')
            optimizer.zero_grad()
            losses.backward()
            if args.clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_max_norm)
            optimizer.step()


        if args.schedule is not None:
            adjust_learning_rate(
                optimizer,
                epoch,
                curr_step,
                num_training_steps=num_training_steps,
                args=args,
            )
        

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        if hasattr(criterion, 'loss_labels'):
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
        else:
            metric_logger.update(obj_class_error=loss_dict_reduced['obj_class_error'])
        if args.psg:
            metric_logger.update(verb_class_error=loss_dict_reduced['verb_class_error'])
            metric_logger.update(sub_class_error=loss_dict_reduced['sub_class_error'])
        metric_logger.update(lr=optimizer.param_groups[1]["lr"])
        metric_logger.update(lr_backbone=optimizer.param_groups[0]["lr"])


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def encode_mask(results):
    compressed_results = []
    for res in results:
        masks = [mask_util.encode(np.array(res['masks'][i], order="F", dtype="uint8")) for i in range(res['masks'].shape[0])]
        compressed_results.append({
            'labels': res['labels'],
            'masks': masks,
            'verb_scores': res['verb_scores'],
            'sub_ids': res['sub_ids'],
            'obj_ids': res['obj_ids']
        })
    return compressed_results

def process_gts(targets):
    compressed_gts = []
    for target in targets:
        masks = [mask_util.encode(np.array(target['masks'][i], order="F", dtype="uint8")) for i in range(target['masks'].shape[0])]
        compressed_gts.append({
            'orig_size': target['orig_size'],
            'size': target['size'],
            'filename': target['filename'],
            'masks': masks,
            'labels': target['labels'],
            'id': target['id'],
            'hois': target['hois']
        })
    return compressed_gts


def process_gts_flexible(targets):
    compressed_gts = []
    for target in targets:    
        masks = [mask_util.encode(np.array(target['masks'][i], order="F", dtype="uint8")) for i in range(target['masks'].shape[0])]
        compressed_gts.append({
            'orig_size': target['orig_size'],
            'size': target['size'],
            'filename': target['filename'],
            'masks': masks,
            'labels': target['labels'],
            'prompt': target['grounding']['gtext'],
            'p_verb': target['grounding']['p_verb'],
            'p_obj': target['grounding']['p_obj'],
            'ghois': target['grounding']['ghois'],
        })
    return compressed_gts

def process_gts_flexible_psg(targets):
    compressed_gts = []
    for target in targets:    
        masks = [mask_util.encode(np.array(target['masks'][i], order="F", dtype="uint8")) for i in range(target['masks'].shape[0])]
        compressed_gts.append({
            'orig_size': target['orig_size'],
            'size': target['size'],
            'filename': target['filename'],
            'masks': masks,
            'labels': target['labels'],
            'prompt': target['grounding']['gtext'],
            'p_verb': target['grounding']['p_verb'],
            'p_obj': target['grounding']['p_obj'],
            'p_sub': target['grounding']['p_sub'],
            'gsops': target['grounding']['gsops'],
        })
    return compressed_gts

def process_gts_psg(targets):
    compressed_gts = []
    for target in targets:
        masks = [mask_util.encode(np.array(target['masks'][i], order="F", dtype="uint8")) for i in range(target['masks'].shape[0])]
        compressed_gts.append({
            'orig_size': target['orig_size'],
            'size': target['size'],
            'filename': target['filename'],
            'masks': masks,
            'labels': target['labels'],
            'id': target['id'],
            'sops': target['sops']
        })
    return compressed_gts

def process_gts_vrd(targets):
    compressed_gts = []
    for target in targets:
        masks = [mask_util.encode(np.array(target['masks'][i], order="F", dtype="uint8")) for i in range(target['masks'].shape[0])]
        compressed_gts.append({
            'orig_size': target['orig_size'],
            'size': target['size'],
            'filename': target['filename'],
            'masks': masks,
            'labels': target['labels'],
            'id': target['id'],
            'sops': target['sops'],
            'p_verb': target['grounding']['p_verb'],
            'p_obj': target['grounding']['p_obj'],
            'p_sub': target['grounding']['p_sub'],
            'gsops': target['grounding']['gsops'],
            'gtext': target['grounding']['gtext']
        })
    return compressed_gts

       
@torch.no_grad()
def evaluate_generic_hoi(dataset_file, model, postprocessors, data_loader, device, args):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    preds = []
    gts = []
    indices = []
    cnt = 0
    # cnt_for_save = 1
    flexible_eval = False
    if args.flexible_eval_task=='generic_eval':
        flexible_eval = True

    
    for samples in metric_logger.log_every(data_loader, 10, header):
        cnt += 1
        # if cnt > 10:
        #     break
        source = samples[0]['source']
        images = [x["image"].to(device) for x in samples]
        images = ImageList.from_tensors(images, size_divisibility=1)
        outputs, extra = model(samples, source=source, flexible_eval=flexible_eval)
        # outputs = model(samples) # swin
        orig_target_sizes = torch.stack([sample["orig_size"] for sample in samples], dim=0)
        image_shape = images.tensor.shape[-2:] # after align to the same size 
        image_size = images.image_sizes # before align to the same size 
        if args.dataset_file == 'hico' or args.dataset_file == 'vcoco':
            if args.use_box:
                results = postprocessors['hoi_box'](outputs, orig_target_sizes, image_shape, image_size)
            else:
                results = postprocessors['hoi'](outputs, orig_target_sizes, image_shape, image_size)
            # compressed_results = encode_mask(results)
                compressed_gts = process_gts(samples)
            # ori_preds = list(itertools.chain.from_iterable(utils.all_gather(results)))
            # ori_gts = list(itertools.chain.from_iterable(utils.all_gather(copy.deepcopy(targets))))
            # pro_preds = process_original_preds(ori_gts, ori_preds, data_loader.dataset.correct_mat)
            # preds.extend(pro_preds)

        preds.extend(list(itertools.chain.from_iterable(utils.all_gather(results))))
        if args.use_mask:
            gts.extend(list(itertools.chain.from_iterable(utils.all_gather(copy.deepcopy(compressed_gts)))))
        if args.use_box:
            targets = [{k: v for k, v in sample.items() if k != 'image' and k!='source'} for sample in samples]
            gts.extend(list(itertools.chain.from_iterable(utils.all_gather(copy.deepcopy(targets)))))

        # if cnt == 60:
        #     with open(os.path.join(args.output_dir, 'gts_60.pkl'), 'wb') as f: 
        #         pickle.dump(gts, f)
        #     with open(os.path.join(args.output_dir, 'preds_60.pkl'), 'wb') as f: 
        #         pickle.dump(preds, f)

        # if cnt == 5:
        #     break

    metric_logger.synchronize_between_processes()

    img_ids = [img_gts['id'] for img_gts in gts]
    _, indices = np.unique(img_ids, return_index=True)
    preds = [img_preds for i, img_preds in enumerate(preds) if i in indices]
    gts = [img_gts for i, img_gts in enumerate(gts) if i in indices]
    # with open('/work/vig/fangruiz/work/flexible_hoi/CDN/configs/exps/preds/pro_preds_compressed_masks.pkl', 'wb') as f:
    #     pickle.dump(preds, f)
    # import pickle
    # with open(os.path.join(args.output_dir, 'gt.pkl'), 'wb') as f: 
    #     pickle.dump(gts, f)
    # with open(os.path.join(args.output_dir, 'preds.pkl'), 'wb') as f: 
    #     pickle.dump(preds, f)

    # import ipdb; ipdb.set_trace()   
    if args.dataset_file == 'vcoco':
        evaluator = VCOCOEvaluator(preds, gts, data_loader.dataset.correct_mat, args=args)
    elif args.dataset_file == 'hico':
        if args.use_mask:
            evaluator = HICOEvaluator(preds, gts, data_loader.dataset.rare_triplets,
                                  data_loader.dataset.non_rare_triplets, data_loader.dataset.correct_mat, args=args)
    stats = evaluator.evaluate()

    return stats

@torch.no_grad()
def evaluate_flexible_hoi(dataset_file, model, postprocessors, data_loader, device, args):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    preds = []
    gts = []
    indices = []
    cnt = 0
    # cnt_for_save = 1
    iou_sub = []
    iou_obj = []

    data_loader.dataset.flexible_task = args.flexible_eval_task
    for samples in metric_logger.log_every(data_loader, 10, header):
        cnt += 1
        # if cnt > 6:
        #     break
        samples = [sample for sample in samples if sample['grounding'] is not None]
        # if samples[0]['grounding'] is None:
        #     continue
        images = [x["image"].to(device) for x in samples]
        images = ImageList.from_tensors(images, size_divisibility=32)
        if len(images)==0:
            continue      
        results = model.evaluate_grounding(samples)
        # orig_target_sizes = torch.stack([sample["orig_size"] for sample in samples], dim=0)
        # image_shape = images.tensor.shape[-2:] # after align to the same size 
        # image_size = images.image_sizes # before align to the same size 
        # if args.dataset_file == 'hico':
        #     results = postprocessors['hoi'](outputs, orig_target_sizes, image_shape, image_size)
            # compressed_results = encode_mask(results)
        compressed_gts = process_gts_flexible(samples)
            # ori_preds = list(itertools.chain.from_iterable(utils.all_gather(results)))
            # ori_gts = list(itertools.chain.from_iterable(utils.all_gather(copy.deepcopy(targets))))
            # pro_preds = process_original_preds(ori_gts, ori_preds, data_loader.dataset.correct_mat)
            # preds.extend(pro_preds)
        # iou_sub.append(mask_util.iou([results[0]['sub_mask']], [compressed_gts[0]['sub_mask']], [0]))
        # iou_obj.append(mask_util.iou([results[0]['obj_mask']], [compressed_gts[0]['obj_mask']], [0]))
        
        # with open('/work/vig/fangruiz/work/flexible_hoi/code/outputs/flexible_hico/pred_sub/preds.pkl', 'wb') as f:
        #     pickle.dump(preds, f)



        preds.extend(list(itertools.chain.from_iterable(utils.all_gather(results))))
        gts.extend(list(itertools.chain.from_iterable(utils.all_gather(copy.deepcopy(compressed_gts)))))
        # if cnt == 5:
        #     break

    metric_logger.synchronize_between_processes()

    

    evaluator = Flexible_HICOEvaluator(preds, gts, data_loader.dataset.correct_mat, args=args)
    # elif dataset_file == 'vcoco':
    #     evaluator = VCOCOEvaluator(preds, gts, data_loader.dataset.correct_mat, use_nms_filter=args.use_nms_filter)

    stats = evaluator.evaluate()

    return stats


@torch.no_grad()
def evaluate_flexible_vrd(dataset_file, model, postprocessors, data_loader, device, args):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    preds = []
    gts = []
    cnt = 0


    data_loader.dataset.flexible_task = args.flexible_eval_task
    for samples in metric_logger.log_every(data_loader, 10, header):
        cnt += 1
        # if cnt ==3:
        #     break
        samples = [sample for sample in samples if sample['grounding'] is not None]

        images = [x["image"].to(device) for x in samples]
        images = ImageList.from_tensors(images, size_divisibility=32)
        if len(images)==0:
            continue      
        results = model.module.evaluate_grounding_vrd(samples)

        compressed_gts = process_gts_vrd(samples)
        


        preds.extend(list(itertools.chain.from_iterable(utils.all_gather(results))))
        gts.extend(list(itertools.chain.from_iterable(utils.all_gather(copy.deepcopy(compressed_gts)))))


    metric_logger.synchronize_between_processes()

    save_preds = []
    save_gts = []
    for gt, pred in zip(gts,preds):
        pred = {k: v.to('cpu').numpy() if k != 'masks' and v is not None else v for k, v in pred.items()}
        gt = {k: v.to('cpu').numpy() if k != 'filename' and k != 'prompt' and k != 'masks' and k!='id' and k!='gtext' else v for k, v in gt.items()}
        save_preds.append(pred)
        save_gts.append(gt)
    return 


@torch.no_grad()
def evaluate_psg(dataset_file, model, postprocessors, data_loader, device, args):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    preds = []
    gts = []
    indices = []
    cnt = 0
    for samples in metric_logger.log_every(data_loader, 10, header):
        cnt += 1

        source = samples[0]['source']
        
        images = [x["image"].to(device) for x in samples]
        images = ImageList.from_tensors(images, size_divisibility=32)
        outputs, extra = model(samples, source=source)
        orig_target_sizes = torch.stack([sample["orig_size"] for sample in samples], dim=0)
        image_shape = images.tensor.shape[-2:] # after align to the same size 
        image_size = images.image_sizes # before align to the same size 
        if args.dataset_file == 'psg':
            results = postprocessors['psg'](outputs, orig_target_sizes, image_shape, image_size)
            compressed_gts = process_gts_psg(samples)

        preds.extend(list(itertools.chain.from_iterable(utils.all_gather(results))))
        gts.extend(list(itertools.chain.from_iterable(utils.all_gather(copy.deepcopy(compressed_gts)))))
        
    metric_logger.synchronize_between_processes()

    stats = data_loader.dataset.evaluate(preds)
    return stats



