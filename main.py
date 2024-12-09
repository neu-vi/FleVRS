import argparse
import time
import datetime
import random
from pathlib import Path
import json

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler

import util.misc as utils
from datasets import build_dataset
from engine import train_one_epoch, train_one_epoch_two_stage, train_one_epoch_multi, evaluate_generic_hoi, evaluate_coco, evaluate_psg, evaluate_flexible_hoi, train_one_epoch_three, evaluate_flexible_psg
from models import build_model
import os
from detectron2.data.build import trivial_batch_collator



def get_args_parser():
    parser = argparse.ArgumentParser('Flexible HOI parameters', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    # parser.add_argument('--weight_decay_zero', default=0.0, type=float)
    parser.add_argument('--epochs', default=90, type=int)
    #nparser.add_argument('--lr_drop', default=[60], nargs='+', type=int)
    parser.add_argument('--lr_drop', default=60, type=int)
    parser.add_argument('--save_model_epoch', default=5, type=int)
    
    parser.add_argument('--skip_mask_cls_init', action='store_true')

    parser.add_argument('--group_weight_decay', action='store_true')

    parser.add_argument('--gradient_strategy', default='vanilla', type=str, help='vanilla/gradient_accumulation')
    parser.add_argument('--clip_max_norm', default=0.01, type=float, help='gradient clipping max norm')
    parser.add_argument('--cumulative_iters', default=4, type=int)

    parser.add_argument("--schedule", default = None, type=str, choices=("step", "multistep", "step_with_warmup"))
    parser.add_argument('--num_warmup_steps', default=500, type=int)

    parser.add_argument('--hilo_data_aug', action='store_true')
    

    # * Backbone
    parser.add_argument('--backbone', default='convnext_tiny', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--pretrained_backbone', default=True,
                        help="whether use pretrained weights")                    
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # pixel decoder 
    parser.add_argument('--in_features', default=["res2", "res3", "res4", "res5"], 
                        help="multi-scale features from backbone")
    parser.add_argument('--in_strides', default=[4, 8, 16, 32], nargs='+', 
                        help="multi-scale strides")
    parser.add_argument('--in_channels', default=[96, 192, 384, 768], nargs='+', type=int,
                        help="multi-scale channels")
    parser.add_argument('--transformer_dropout', default=0.0, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--transformer_nheads', default=8, type=int,
                        help="number of head in the transformer")
    parser.add_argument('--transformer_dim_feedforward', default=2048, type=int,
                        help="dim in the transformer")
    parser.add_argument('--transformer_enc_layers', default=6, type=int,
                        help="enc layers in the transformer")
    parser.add_argument('--deformable_transformer_encoder_in_features', default=["res3", "res4", "res5"])
    parser.add_argument('--transformer_pre_norm', action='store_true')
    parser.add_argument('--conv_dim', default=512, type=int)
    parser.add_argument('--mask_dim', default=512, type=int)
    parser.add_argument('--common_stride', default=4, type=int)
    parser.add_argument('--mask_on', default=True)
    parser.add_argument('--norm', default='GN')
                        
    # HOI decoder
    parser.add_argument('--hoi_dec_in_channels', default=512, type=int)
    parser.add_argument('--mask_classification', default=True)
    parser.add_argument('--hidden_dim', default=512, type=int)
    parser.add_argument('--dim_proj', default=512, type=int)
    parser.add_argument('--num_queries', default=100, type=int)
    parser.add_argument('--contxt_len', default=77, type=int)
    parser.add_argument('--hoi_dec_nheads', default=8, type=int)
    parser.add_argument('--hoi_dec_layers', default=9, type=int)
    parser.add_argument('--hoi_dec_dim_feedforward', default=2048, type=int,
                        help="dim in the hoi decoder")
    parser.add_argument('--hoi_dec_mask_dim', default=256, type=int)

    # HOI
    parser.add_argument('--num_obj_classes', type=int, default=80,
                        help="Number of object classes")
    parser.add_argument('--num_verb_classes', type=int, default=117,
                        help="Number of verb classes")
    
    parser.add_argument('--num_obj_classes_hico', type=int, default=80,
                        help="Number of object classes")
    parser.add_argument('--num_verb_classes_hico', type=int, default=117,
                        help="Number of verb classes")
    
    parser.add_argument('--num_obj_classes_vcoco', type=int, default=81,
                        help="Number of object classes")
    parser.add_argument('--num_verb_classes_vcoco', type=int, default=29,
                        help="Number of verb classes")
    
    parser.add_argument('--num_obj_classes_psg', type=int, default=133,
                        help="Number of object classes")
    parser.add_argument('--num_verb_classes_psg', type=int, default=56,
                        help="Number of verb classes")
    
    parser.add_argument('--num_hoi_classes', type=int, default=520,
                        help="Number of hoi classes")
    parser.add_argument('--pretrained', type=str, default='',
                        help='Pretrained model path')
    parser.add_argument('--subject_category_id', default=0, type=int)
    parser.add_argument('--verb_loss_type', type=str, default='focal',
                        help='Loss type for the verb classification')
    
    parser.add_argument('--use_mask', action='store_true')
    parser.add_argument('--use_box', action='store_true')
    parser.add_argument('--remove_no_interaction', action='store_true')
    parser.add_argument('--use_gpt_emb', action='store_true')

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    parser.add_argument('--use_matching', action='store_true',
                        help="Use obj/sub matching 2class loss in first decoder, default not use")

    # * Matcher
    parser.add_argument('--set_cost_obj_class', default=1, type=float,
                        help="Object Class coefficient in the matching cost")
    parser.add_argument('--set_cost_verb_class', default=1, type=float,
                        help="verb Class coefficient in the matching cost")
    parser.add_argument('--set_cost_hoi_class', default=1, type=float,
                        help="hoi Class coefficient in the matching cost")
    parser.add_argument('--set_cost_mask', default=2.5, type=float,
                        help="mask coefficient in the matching cost")
    parser.add_argument('--set_cost_dice', default=1, type=float,
                        help="dice coefficient in the matching cost")
    parser.add_argument('--set_cost_reg', default=2.5, type=float,
                        help="box L1 coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=1, type=float,
                        help="box giou coefficient in the matching cost")
    parser.add_argument('--num_points', default=6400, type=int,
                        help="number of points for point sample")
    
    # * Loss coefficients
    parser.add_argument('--box_reg_loss_coef', default=2.5, type=float)
    parser.add_argument('--box_giou_loss_coef', default=1, type=float)
    parser.add_argument('--mask_dice_loss_coef', default=1, type=float)
    parser.add_argument('--mask_bce_loss_coef', default=2, type=float)
    parser.add_argument('--obj_loss_coef', default=1, type=float)
    parser.add_argument('--verb_loss_coef', default=2, type=float)
    parser.add_argument('--tri_loss_coef', default=2, type=float)
    parser.add_argument('--grd_loss_coef', default=2, type=float)
    parser.add_argument('--grounding_weight', default=2, type=float)
    
    parser.add_argument('--alpha', default=0.5, type=float, help='focal loss alpha')
    parser.add_argument('--matching_loss_coef', default=1, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")
    parser.add_argument('--eos_coef_psg', default=0.02, type=float,
                        help="Relative classification weight of the no-object class")
    parser.add_argument('--eos_coef_hoi', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    parser.add_argument('--oversample_ratio', default=3.0, type=float, help='get_uncertain_point_coords param')
    parser.add_argument('--importance_sample_ratio', default=0.75, type=float, help='get_uncertain_point_coords param')
    parser.add_argument('--use_triplet', action='store_true')
    # parser.add_argument('--use_triplet_vcoco', action='store_true')


    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--hoi_path', type=str)
    parser.add_argument('--vcoco_path', type=str)
    parser.add_argument('--input_img_size', type=int, default=640)
    parser.add_argument('--sam_model', type=str)
    parser.add_argument('--psg_folder', type=str)
    parser.add_argument('--correct_mat_path', type=str)
    parser.add_argument('--flexible_test_set', type=str)
    parser.add_argument('--verb_emb_path', type=str)
    parser.add_argument('--obj_emb_path', type=str)


    # test coco
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--coco_masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--test_epoch', default=-1, type=int,
                        help='test_epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=0, type=int)

    # distributed training parameters
    parser.add_argument('--use_distributed', action='store_true', help='use distributed training/testing')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--local_rank', type=int, help='local rank')

    # decoupling training parameters
    parser.add_argument('--freeze_lang_encoder', action='store_true')
    parser.add_argument('--use_verb_temp', action='store_true')


    # hoi eval parameters
    parser.add_argument('--use_nms_filter', action='store_true', help='Use pair nms filter, default not use')
    parser.add_argument('--thres_nms', default=0.7, type=float)
    parser.add_argument('--nms_alpha', default=1.0, type=float)
    parser.add_argument('--nms_beta', default=0.5, type=float)
    parser.add_argument('--json_file', default='results.json', type=str)
    parser.add_argument('--exclude_filenames_path', default='/work/vig/fangruiz/work/flexible_hoi/code/outputs/exclude_test_filename.pkl', type=str)

    # task switch
    parser.add_argument('--hoi', action='store_true')
    parser.add_argument('--psg', action='store_true')
    parser.add_argument('--vrd', action='store_true')
    parser.add_argument('--flexible_grounding', action='store_true')
    parser.add_argument('--add_mask2matching', action='store_true', help='add mask embedding to matching pred logits')

    parser.add_argument('--flexible_eval_task', default=None, type=str)
    parser.add_argument('--unseen_type', default=None, type=str)

    # model switch
    parser.add_argument('--two_stage', action='store_true')

    # ablation
    parser.add_argument('--only_grd_loss', action='store_true')
    parser.add_argument('--only_cls_loss', action='store_true')
    parser.add_argument('--query_based', action='store_true')
    parser.add_argument('--random_vis_model', action='store_true')

    return parser


def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    output_dir = Path(args.output_dir)

    print(args)
    args_dict = vars(args)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # import ipdb; ipdb.set_trace()   
    with open(os.path.join(output_dir, 'args.json'), 'w') as f:
        json.dump(args_dict, f, indent=4)

    device = torch.device(args.device)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    model.to(device)


    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)
        model_without_ddp = model.module
        # model._set_static_graph()
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    if args.freeze_lang_encoder:
        for name, p in model.named_parameters():
            
            if 'lang_encoder' in name and 'logit_scale_obj' not in name and 'logit_scale_verb' not in name and 'logit_scale_hoi' not in name:
                p.requires_grad = False

            if 'obj_visual_projection' in name or 'verb_visual_projection' in name:
                p.requires_grad = False


    if args.group_weight_decay:
        norm_layer_params = []
        embedding_layer_params = []
        for name, param in model.named_parameters():
            if 'norm' in name and 'backbone' not in name:  # add other normalization types if needed
                norm_layer_params.append(param)
            if 'query_feat' in name or 'query_embed' in name or 'level_embed' in name:
                embedding_layer_params.append(param)

        param_dicts = [
            {
                "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": args.lr_backbone,
            },
            {
                "params": embedding_layer_params,
                "weight_decay": 0.0,
            },
            {"params": norm_layer_params, "weight_decay": 0.0},
        ]
    else:
        param_dicts = [
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        ]

    if args.dataset_file == 'hico+vcoco+psg':
        dataset_train1, dataset_train2, dataset_train3 = build_dataset(image_set='train', args=args)
    elif args.dataset_file == 'hico+psg':
        dataset_train1, dataset_train2 = build_dataset(image_set='train', args=args)
    elif args.dataset_file == 'hico+vcoco':
        dataset_train1, dataset_train2 = build_dataset(image_set='train', args=args)
    else:
        dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='test', args=args)

    if args.distributed:
        if args.dataset_file == 'hico+vcoco+psg':
            sampler_train1 = DistributedSampler(dataset_train1)
            sampler_train2 = DistributedSampler(dataset_train2)
            sampler_train3 = DistributedSampler(dataset_train3)
        elif args.dataset_file == 'hico+vcoco' or args.dataset_file == 'hico+psg':
            sampler_train1 = DistributedSampler(dataset_train1)
            sampler_train2 = DistributedSampler(dataset_train2)
        else:
            sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        if args.dataset_file == 'hico+vcoco' or args.dataset_file == 'hico+psg':
            sampler_train1 = torch.utils.data.RandomSampler(dataset_train1)
            sampler_train2 = torch.utils.data.RandomSampler(dataset_train2)
        else:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if args.dataset_file == 'hico+vcoco' or args.dataset_file == 'hico+psg':
        batch_sampler_train1 = torch.utils.data.BatchSampler(
            sampler_train1, args.batch_size, drop_last=True)
        batch_sampler_train2 = torch.utils.data.BatchSampler(
            sampler_train2, args.batch_size, drop_last=True)
    elif args.dataset_file == 'hico+vcoco+psg':
        batch_sampler_train1 = torch.utils.data.BatchSampler(
            sampler_train1, args.batch_size, drop_last=True)
        batch_sampler_train2 = torch.utils.data.BatchSampler(
            sampler_train2, args.batch_size, drop_last=True)
        batch_sampler_train3 = torch.utils.data.BatchSampler(
            sampler_train3, args.batch_size, drop_last=True)
    else:
        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, args.batch_size, drop_last=True)

    # with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True) as prof:
    if args.dataset_file == 'hico+vcoco' or args.dataset_file == 'hico+psg': 
        data_loader_train1 = DataLoader(dataset_train1,
                        batch_sampler=batch_sampler_train1,
                        collate_fn=trivial_batch_collator,
                        num_workers=args.num_workers)
        data_loader_train2 = DataLoader(dataset_train2,
                        batch_sampler=batch_sampler_train2,
                        collate_fn=trivial_batch_collator,
                        num_workers=args.num_workers)
    elif args.dataset_file == 'hico+vcoco+psg':
        data_loader_train1 = DataLoader(dataset_train1,
                        batch_sampler=batch_sampler_train1,
                        collate_fn=trivial_batch_collator,
                        num_workers=args.num_workers)
        data_loader_train2 = DataLoader(dataset_train2,
                        batch_sampler=batch_sampler_train2,
                        collate_fn=trivial_batch_collator,
                        num_workers=args.num_workers)
        data_loader_train3 = DataLoader(dataset_train3,
                        batch_sampler=batch_sampler_train3,
                        collate_fn=trivial_batch_collator,
                        num_workers=args.num_workers)
    else:
        data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                    collate_fn=trivial_batch_collator, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=trivial_batch_collator, num_workers=args.num_workers)

    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)

    if args.schedule is None:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    if args.test_epoch != -1:
        args.pretrained = args.output_dir + '/checkpoint_ep' + str(args.test_epoch) + '.pth'
    
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
    elif args.pretrained:
        checkpoint = torch.load(args.pretrained, map_location='cpu')
        if args.eval:
            model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
            # model_without_ddp.load_state_dict(checkpoint, strict=False) # for xdecoder pretrained weights
        else:
            if args.skip_mask_cls_init and args.psg:
                keys_to_remove = ['hoi_decoder.class_embed_sub', 'hoi_decoder.class_embed_obj', 'hoi_decoder.class_embed_verb', 'hoi_decoder.mask_embed_sub.layers.0.weight', 'hoi_decoder.mask_embed_sub.layers.0.bias', 'hoi_decoder.mask_embed_sub.layers.1.weight', 'hoi_decoder.mask_embed_sub.layers.1.bias', 'hoi_decoder.mask_embed_sub.layers.2.weight', 'hoi_decoder.mask_embed_sub.layers.2.bias', 'hoi_decoder.mask_embed_obj.layers.0.weight', 'hoi_decoder.mask_embed_obj.layers.0.bias', 'hoi_decoder.mask_embed_obj.layers.1.weight', 'hoi_decoder.mask_embed_obj.layers.1.bias', 'hoi_decoder.mask_embed_obj.layers.2.weight', 'hoi_decoder.mask_embed_obj.layers.2.bias']
                for key in keys_to_remove:
                    checkpoint.pop(key)
            if args.random_vis_model:
                checkpoint = {k: v for k, v in checkpoint.items() if 'lang_encoder' in k}


            model_without_ddp.load_state_dict(checkpoint, strict=False) # for xdecoder pretrained weights
         
    if args.eval:
    
        if args.dataset_file == 'hico':
            if args.flexible_eval_task=="generic_eval":
                test_stats = evaluate_generic_hoi(args.dataset_file, model, postprocessors, data_loader_val, device, args)
            elif args.flexible_eval_task=="pred_verb" or args.flexible_eval_task=="pred_sub" or args.flexible_eval_task=="pred_obj":
                test_stats = evaluate_flexible_hoi(args.dataset_file, model, postprocessors, data_loader_val, device, args)
            else:
                test_stats = evaluate_generic_hoi(args.dataset_file, model, postprocessors, data_loader_val, device, args)
        elif args.dataset_file == 'vcoco':
            test_stats = evaluate_generic_hoi(args.dataset_file, model, postprocessors, data_loader_val, device, args)
        elif args.dataset_file == 'psg':
            if args.flexible_eval_task=="pred_verb" or args.flexible_eval_task=="pred_sub" or args.flexible_eval_task=="pred_obj":
                test_stats = evaluate_flexible_psg(args.dataset_file, model, postprocessors, data_loader_val, device, args)
            else:
                test_stats = evaluate_psg(args.dataset_file, model, postprocessors, data_loader_val, device, args)
        
        test_stats.update({"test_epoch": args.test_epoch})
        if args.flexible_eval_task is not None: 
            file_name = "log_test_" + args.flexible_eval_task + '.txt'
        else:
            file_name = "log_test.txt"
        if utils.is_main_process():
            with (output_dir / file_name).open("a") as f:
                f.write(json.dumps(test_stats) + "\n")
        return


    print("Start training")
    start_time = time.time()
    best_performance = 0
    if args.dataset_file == 'hico+vcoco' or args.dataset_file == 'hico+psg':
        iterator_train1 = iter(data_loader_train1)
        iterator_train2 = iter(data_loader_train2)
    elif args.dataset_file == 'hico+vcoco+psg':
        iterators = {
            'hico': iter(data_loader_train1),
            'vcoco': iter(data_loader_train2),
            'psg': iter(data_loader_train3)
        }
    
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            if args.dataset_file == 'hico+vcoco+psg':
                sampler_train1.set_epoch(epoch)
                sampler_train2.set_epoch(epoch)
                sampler_train3.set_epoch(epoch)
            elif args.dataset_file == 'hico+vcoco' or args.dataset_file == 'hico+psg':
                sampler_train1.set_epoch(epoch)
                sampler_train2.set_epoch(epoch)
            else:
                sampler_train.set_epoch(epoch)

        if args.dataset_file == 'hico+vcoco+psg':
            (criterion1, criterion2, criterion3) = criterion
            train_stats = train_one_epoch_three(
                model, criterion1, criterion2, criterion3, data_loader_train1, data_loader_train2, data_loader_train3, iterators, optimizer, device, epoch,
                args.clip_max_norm, None, args.flexible_grounding)
            for key in iterators:
                iterators[key] = iter(iterators[key])
            lr_scheduler.step()
        elif args.dataset_file == 'hico+vcoco' or args.dataset_file == 'hico+psg':
            (criterion1, criterion2) = criterion
            if args.dataset_file == 'hico+vcoco':
                train_stats = train_one_epoch_multi(
                    model, criterion1, criterion2, data_loader_train1, data_loader_train2, iterator_train1, iterator_train2, optimizer, device, epoch,
                    args.clip_max_norm, None, args.flexible_grounding)
            elif args.dataset_file == 'hico+psg':
                train_stats = train_one_epoch_multi(
                    model, criterion2, criterion1, data_loader_train2, data_loader_train1, iterator_train2, iterator_train1, optimizer, device, epoch,
                    args.clip_max_norm, None, args.flexible_grounding)
            iterator_train1 = iter(data_loader_train1)
            iterator_train2 = iter(data_loader_train2)
            lr_scheduler.step()
        else:
            if args.two_stage:
                train_stats = train_one_epoch_two_stage(
                    model, criterion, data_loader_train, optimizer, device, epoch,
                    args.clip_max_norm, None, args.flexible_grounding, args.psg)
            else:
                train_stats = train_one_epoch(
                    model, criterion, data_loader_train, optimizer, device, epoch,
                    args.clip_max_norm, args.flexible_grounding, args)
                if args.schedule is None:
                    lr_scheduler.step()

        if epoch%args.save_model_epoch == 0:
            checkpoint_path = os.path.join(output_dir, 'checkpoint_ep{}.pth'.format(epoch+1))

            if args.schedule is None:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)
            else:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)
        
        if epoch == args.epochs - 1:
            checkpoint_path = os.path.join(output_dir, 'checkpoint_ep{}.pth'.format(args.epochs))
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
            }, checkpoint_path)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                    #  **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Flexible HOI training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    main(args)
