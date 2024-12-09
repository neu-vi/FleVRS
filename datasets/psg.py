
import torch
import numpy as np
import json
import cv2
import random
import os
from detectron2.data.detection_utils import read_image

import datasets.transforms as T
import datasets.transforms_detr as T_detr
import torchvision.transforms.functional as F
from PIL import Image
from panopticapi.utils import rgb2id
from evaluation import sgg_evaluation
from util.misc import Result
from collections import defaultdict

def make_coco_transforms(image_set, args):
    
    if args.hilo_data_aug:

        normalize = T_detr.Compose([
            T_detr.ToTensor(),
            T_detr.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
        scales = [544, 576, 608, 640, 672, 704, 736, 768, 800]

        if image_set == 'train':
            return T_detr.Compose([
                T_detr.RandomHorizontalFlip(),
                T_detr.RandomSelect(
                    T_detr.RandomResize(scales, max_size=1333),
                    T_detr.Compose([
                        T_detr.RandomResize([400, 500, 600]),
                        T_detr.RandomSizeCrop(384, 600),
                        T_detr.RandomResize(scales, max_size=1333),
                    ])
                ),
                normalize,
            ])

        if image_set == 'test':
            return T_detr.Compose([
                T_detr.RandomResize([800], max_size=1333),
                normalize,
            ])
    else:
        normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        if args.input_img_size == 1024:
            scales = [408, 510, 612, 714, 816, 918, 1020, 1122, 1224, 1326, 1428] 
        elif args.input_img_size == 640:
            scales = [544, 576, 608, 640, 672, 704, 736, 768, 800]
   

        if image_set == 'train':
            return T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomResize(scales, max_size=1333),
                T.FixedSizeCrop([args.input_img_size, args.input_img_size]),
                normalize,
            ])
        if image_set == 'test':
            return T.Compose([
                T.RandomResize([800], max_size=1333),
                normalize,
            ])
    raise ValueError(f'unknown {image_set}')

class PanopticSceneGraphDataset:
    def __init__(self, img_folder, ann_file, num_queries, transforms=None, image_set='train', flexible_task=False, flexible_eval_task=None):
        assert image_set in ['train', 'test'], image_set
        self.image_set = image_set

        with open(ann_file, 'r') as f:
            dataset = json.load(f)

        for d in dataset['data']:
            # NOTE: 0-index for object class labels
            # for s in d['segments_info']:
            #     s['category_id'] += 1

            # for a in d['annotations']:
            #     a['category_id'] += 1

            # NOTE: 1-index for predicate class labels
            for r in d['relations']:
                r[2] += 1


        # Filter out images with zero relations. 
        # Comment out this part for competition files
        dataset['data'] = [
            d for d in dataset['data'] if len(d['relations']) != 0
        ]

        if self.image_set == 'train':
            self.data = [
                d for d in dataset['data']
                if d['image_id'] not in dataset['test_image_ids']
            ]
            # self.data = self.data[:100] # for quick debug
        elif self.image_set == 'test':
            self.data = [
                d for d in dataset['data']
                if d['image_id'] in dataset['test_image_ids']
            ]

        # Init image infos
        self.data_infos = []
        for d in self.data:
            self.data_infos.append({
                'filename': d['file_name'],
                'height': d['height'],
                'width': d['width'],
                'id': d['image_id'],
            })
        self.img_ids = [d['id'] for d in self.data_infos]

        # Define classes, 0-index
        # Class ids should range from 0 to (num_classes - 1)
        self.THING_CLASSES = dataset['thing_classes']
        self.STUFF_CLASSES = dataset['stuff_classes']
        self.CLASSES = self.THING_CLASSES + self.STUFF_CLASSES
        self.PREDICATES = dataset['predicate_classes']

        print('Number of images: {}'.format(len(self.data)))
        print('# Object Classes: {}'.format(len(self.CLASSES)))
        print('# Relation Classes: {}'.format(len(self.PREDICATES)))
    

        self.img_folder = img_folder

        self._valid_verb_ids = list(range(0, 56))

        self._transforms = transforms
        self.num_queries = num_queries

        self.flexible_task = flexible_task
        self.flexible_eval_task = flexible_eval_task

    def __getitem__(self, index):
        ann = self.data[index]
        img_name = ann['file_name']
        img_path = os.path.join(self.img_folder, img_name)
        
        if self.image_set == 'train' and len(ann['relations']) > self.num_queries:
            ann['relations'] = ann['relations'][:self.num_queries]

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = Image.fromarray(img[:, :, ::-1]).convert('RGB')
        w, h = img.size
        
        masks = []
        mask_cat_id_list = []
        seg_map = read_image(self.img_folder + '/' + ann["pan_seg_file_name"], format="RGB")
        seg_map = rgb2id(seg_map)
        for i, s in enumerate(ann["segments_info"]):
            mask_cat_id = s["category_id"]
            mask_cat_id_list.append(mask_cat_id)
            masks.append(seg_map == s["id"])
        # pdb.set_trace()
        masks = torch.stack([torch.from_numpy(m) for m in masks])
        masks = torch.as_tensor(masks, dtype=torch.bool)
        
        if self.image_set == 'train':
            classes = [(i, obj) for i, obj in enumerate(mask_cat_id_list)]
        else:
            classes = [obj for obj in mask_cat_id_list]
        classes = torch.tensor(classes, dtype=torch.int64)

        target = {}
        target['orig_size'] = torch.as_tensor([int(h), int(w)])
        target['size'] = torch.as_tensor([int(h), int(w)])
        
        if self.image_set == 'train':
            
            target['masks'] = masks
            target['labels'] = classes # list of [index, category_id]
            # pdb.set_trace()
            if self._transforms is not None:
                
                img, target = self._transforms(img, target) # may reduce some masks due to crop
            kept_mask_indices = [label[0] for label in target['labels']]
            target['labels'] = target['labels'][:, 1] # list of kept category_id
            
            sub_labels, obj_labels, verb_labels, sub_masks, obj_masks = [], [], [], [], []
            
            all_rel_sets = defaultdict(list)
            for sop in ann['relations']:
                if sop[0] not in kept_mask_indices or sop[1] not in kept_mask_indices:
                    continue

                sub_obj_pair = (sop[0], sop[1])
                all_rel_sets[sub_obj_pair].append(sop[2])
            if len(all_rel_sets) != 0:
                gt_rels = [(k[0], k[1], np.random.choice(v))
                        for k, v in all_rel_sets.items()]
            
                for (sid, oid, rid) in gt_rels:

                    sub_labels.append(target['labels'][kept_mask_indices.index(sid)])
                    obj_labels.append(target['labels'][kept_mask_indices.index(oid)])
                    verb_labels.append(torch.tensor(rid, dtype=torch.int64))
                    
                    sub_mask = target['masks'][kept_mask_indices.index(sid)]
                    obj_mask = target['masks'][kept_mask_indices.index(oid)]
                    
                    sub_masks.append(sub_mask)
                    obj_masks.append(obj_mask)
                assert len(sub_masks) == len(obj_masks)

            target['filename'] = ann['file_name']
            if len(all_rel_sets) == 0:
                # print("no rel!!")
                target['sub_labels'] = torch.zeros((0,), dtype=torch.int64)
                target['obj_labels'] = torch.zeros((0,), dtype=torch.int64)
                target['verb_labels'] = torch.zeros((0,), dtype=torch.int64)
                target['sub_masks'] = torch.zeros((0, target['size'][0], target['size'][1]), dtype=torch.bool)
                target['obj_masks'] = torch.zeros((0, target['size'][0], target['size'][1]), dtype=torch.bool)
                target['matching_labels'] = torch.zeros((0,), dtype=torch.int64)

                if self.flexible_task: 
                    grounding = {}
                    text_grd = ['{}'.format('none')]
                    hash_grd = np.array([hash(txt) for txt in text_grd])
                    grounding = {'gtask': None, 
                                 'sub_gmasks': torch.zeros((0, target['size'][0], target['size'][1]), dtype=torch.bool), 
                                 'obj_gmasks': torch.zeros((0, target['size'][0], target['size'][1]), dtype=torch.bool), 
                                 'gverb_classes': torch.zeros((0,), dtype=torch.int64), 
                                 'gobj_classes': torch.zeros((0,), dtype=torch.int64),
                                 'gsub_classes': torch.zeros((0,), dtype=torch.int64),
                                 'gtext': text_grd, 
                                 'ghash': hash_grd}
                    
                    target['grounding'] = grounding
            else:
                target['sub_labels'] = torch.stack(sub_labels)
                target['obj_labels'] = torch.stack(obj_labels)
                target['verb_labels'] = torch.stack(verb_labels)
                target['sub_masks'] = torch.stack(sub_masks)
                target['obj_masks'] = torch.stack(obj_masks)
                target['matching_labels'] = torch.ones_like(target['obj_labels'])
                if self.flexible_task: 
                    task_pool = ['pred_verb', 'pred_obj', 'pred_sub']
                    task_name = random.choice(task_pool)
                    # print(task_name)
                    triplet_num = len(obj_labels)
                    # random choose a triplet, to construct prompt
                    rand_id = np.random.choice(np.arange(triplet_num))
                    obj_class_id = obj_labels[rand_id]
                    obj_classname = self.CLASSES[obj_labels[rand_id]]

                    verb_class_id = verb_labels[rand_id]
                    verb_classname = self.PREDICATES[verb_class_id-1]

                    sub_class_id = sub_labels[rand_id]
                    sub_classname = self.CLASSES[sub_labels[rand_id]]

                    grounding = {}
                    
                    if task_name == 'pred_verb':
                        text_grd = ['<sub>{}<sub/><obj>{}<obj/>'.format(sub_classname, obj_classname)]
                        # find all sub-obj pairs
                        obj_gids = [i for i, value in enumerate(obj_labels) if value == obj_class_id and sub_labels[i]==sub_class_id]
                    elif task_name == 'pred_obj':
                        text_grd = ['<sub>{}<sub/><verb>{}<verb/>'.format(sub_classname, verb_classname)]
                        obj_gids = [i for i, value in enumerate(verb_labels) if value==verb_class_id and sub_labels[i]==sub_class_id]
                    elif task_name == 'pred_sub':
                        text_grd = ['<obj>{}<obj/><verb>{}<verb/>'.format(obj_classname, verb_classname)]
                        obj_gids = [i for i, value in enumerate(verb_labels) if value==verb_class_id and obj_labels[i]==obj_class_id]

                    hash_grd = np.array([hash(txt) for txt in text_grd])
                    obj_gmasks = [obj_masks[i] for i in obj_gids]
                    sub_gmasks = [sub_masks[i] for i in obj_gids]
                    gverbs = [verb_labels[i] for i in obj_gids]
                    gobjs = [obj_labels[i] for i in obj_gids]
                    gsubs = [sub_labels[i] for i in obj_gids]
                    grounding = {'gtask': task_name, 
                                'sub_gmasks': torch.stack(sub_gmasks), 
                                'obj_gmasks': torch.stack(obj_gmasks), 
                                'gverb_classes': torch.stack(gverbs), 
                                'gobj_classes': torch.stack(gobjs),
                                'gsub_classes': torch.stack(gsubs),
                                'gtext': text_grd, 
                                'ghash': hash_grd}
                    
                    target['grounding'] = grounding
        else:
            target['filename'] = ann['file_name']
            target['masks'] = masks
            target['labels'] = classes
            target['id'] = index

            if self._transforms is not None:
                img, _ = self._transforms(img, None)
            sops = []
            for sop in ann['relations']:
                sops.append(sop)
            target['sops'] = torch.as_tensor(sops, dtype=torch.int64)

            if self.flexible_eval_task=="pred_verb" or self.flexible_eval_task=="pred_sub" or self.flexible_eval_task=="pred_obj":
                if len(sops)==0:
                    target['grounding'] = None
                else:
                    rand_sop_id = np.random.choice(np.arange(len(sops)))

                    obj_class_id = classes[sops[rand_sop_id][1]]
                    obj_classname = self.CLASSES[obj_class_id]

                    verb_class_id = sops[rand_sop_id][2]
                    verb_classname = self.PREDICATES[verb_class_id-1]

                    sub_class_id = classes[sops[rand_sop_id][0]]
                    sub_classname = self.CLASSES[sub_class_id]

                    if self.flexible_eval_task == 'pred_verb':
                        text_grd = ['<sub>{}<sub/><obj>{}<obj/>'.format(sub_classname, obj_classname)]
                        # find all sub-obj pairs
                        g_sops = []
                        for g_sop in sops:
                            if classes[g_sop[1]] == obj_class_id and classes[g_sop[0]] == sub_class_id:
                                g_sops.append(g_sop)
                    elif self.flexible_eval_task == 'pred_obj':
                        text_grd = ['<sub>{}<sub/><verb>{}<verb/>'.format(sub_classname, verb_classname)]
                        g_sops = []
                        for g_sop in sops:
                            if g_sop[2] == verb_class_id and classes[g_sop[0]] == sub_class_id:
                                g_sops.append(g_sop)
                    elif self.flexible_eval_task == 'pred_sub':
                        text_grd = ['<obj>{}<obj/><verb>{}<verb/>'.format(obj_classname, verb_classname)]
                        g_sops = []
                        for g_sop in sops:
                            if g_sop[2] == verb_class_id and classes[g_sop[1]] == obj_class_id:
                                g_sops.append(g_sop)

                    hash_grd = np.array([hash(txt) for txt in text_grd])
                    grounding = {'gtask': self.flexible_eval_task, 
                                 'gsops': torch.as_tensor(g_sops, dtype=torch.int64),
                                 'p_sub': torch.as_tensor(sub_class_id),
                                 'p_verb': torch.as_tensor(verb_class_id),
                                 'p_obj': torch.as_tensor(obj_class_id),
                                 'gtext': text_grd, 
                                 'ghash': hash_grd}
                    target['grounding'] = grounding

                        


        target['image'] = img
        target['source'] = 'psg'
        

        return target

    def __len__(self):
        return len(self.data)

    def load_correct_mat(self, path):
        self.correct_mat = np.load(path)

    def get_ann_info(self, idx):
        d = self.data[idx]

        # Process bbox annotations
        gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        gt_bboxes = np.array([a['bbox'] for a in d['annotations']],
                                 dtype=np.float32)
        gt_labels = np.array([a['category_id'] for a in d['annotations']],
                                 dtype=np.int64)
        

        # Process segment annotations
        gt_mask_infos = []
        for s in d['segments_info']:
            gt_mask_infos.append({
                'id': s['id'],
                'category': s['category_id'],
                'is_thing': s['isthing']
            })

        # Process relationship annotations
        gt_rels = d['relations'].copy()

        # Filter out dupes!
        if self.image_set == 'train':
            all_rel_sets = defaultdict(list)
            for (o0, o1, r) in gt_rels:
                all_rel_sets[(o0, o1)].append(r)
            gt_rels = [(k[0], k[1], np.random.choice(v))
                       for k, v in all_rel_sets.items()]
            gt_rels = np.array(gt_rels, dtype=np.int32)
        else:
            # for test or val set, filter the duplicate triplets,
            # but allow multiple labels for each pair
            all_rel_sets = []
            for (o0, o1, r) in gt_rels:
                if (o0, o1, r) not in all_rel_sets:
                    all_rel_sets.append((o0, o1, r))
            gt_rels = np.array(all_rel_sets, dtype=np.int32)

        # add relation to target
        num_box = len(gt_mask_infos)
        relation_map = np.zeros((num_box, num_box), dtype=np.int64)
        for i in range(gt_rels.shape[0]):
            # If already exists a relation?
            if relation_map[int(gt_rels[i, 0]), int(gt_rels[i, 1])] > 0:
                if random.random() > 0.5:
                    relation_map[int(gt_rels[i, 0]),
                                 int(gt_rels[i, 1])] = int(gt_rels[i, 2])
            else:
                relation_map[int(gt_rels[i, 0]),
                             int(gt_rels[i, 1])] = int(gt_rels[i, 2])

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            rels=gt_rels,
            rel_maps=relation_map,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_mask_infos,
            seg_map=d['pan_seg_file_name'],
        )

        return ann

    def evaluate(
        self,
        results,
        metric='sgdet',
        logger=None,
        jsonfile_prefix=None,
        classwise=True,
        multiple_preds=False,
        iou_thrs=0.5,
        nogc_thres_num=None,
        detection_method='pan_seg',
        **kwargs,
    ):
        """Overwritten evaluate API:

        For each metric in metrics, it checks whether to invoke ps or sg
        evaluation. if the metric is not 'sg', the evaluate method of super
        class is invoked to perform Panoptic Segmentation evaluation. else,
        perform scene graph evaluation.
        """
        metrics = metric if isinstance(metric, list) else [metric]

        # Available metrics
        allowed_sg_metrics = ['predcls', 'sgcls', 'sgdet']

        sg_metrics, od_metrics = [], []
        for m in metrics:
            if m in allowed_sg_metrics:
                sg_metrics.append(m)
            else:
                raise ValueError('Unknown metric {}.'.format(m))

        if len(sg_metrics) > 0:
            """Invoke scene graph evaluation.

            prepare the groundtruth and predictions. Transform the predictions
            of key-wise to image-wise. Both the value in gt_results and
            det_results are numpy array.
            """
            if not hasattr(self, 'test_gt_results'):
                print('\nLoading testing groundtruth...\n')
                gt_results = []
                for i in range(len(self)):
                    ann = self.get_ann_info(i)

                    # NOTE: Change to object class labels 1-index here
                    ann['labels'] += 1

                    # load gt pan_seg masks
                    segment_info = ann['masks']
                    gt_img = read_image(self.img_folder + '/' + ann['seg_map'],
                                        format='RGB')
                    gt_img = gt_img.copy()  # (H, W, 3)

                    seg_map = rgb2id(gt_img)

                    # get separate masks
                    gt_masks = []
                    labels_coco = []
                    for _, s in enumerate(segment_info):
                        label = self.CLASSES[s['category']]
                        labels_coco.append(label)
                        gt_masks.append(seg_map == s['id'])
                    # load gt pan seg masks done

                    gt_results.append(
                        Result(
                            bboxes=ann['bboxes'],
                            labels=ann['labels'],
                            rels=ann['rels'],
                            relmaps=ann['rel_maps'],
                            rel_pair_idxes=ann['rels'][:, :2],
                            rel_labels=ann['rels'][:, -1],
                            masks=gt_masks,
                        ))

                print('\n')
                self.test_gt_results = gt_results

            return sgg_evaluation(
                sg_metrics,
                groundtruths=self.test_gt_results,
                predictions=results,
                iou_thrs=iou_thrs,
                # logger=logger,
                ind_to_predicates=['__background__'] + self.PREDICATES,
                multiple_preds=multiple_preds,
                # predicate_freq=self.predicate_freq,
                nogc_thres_num=nogc_thres_num,
                detection_method=detection_method,
            )



def build(image_set, args):
    assert image_set in ['train', 'test'], image_set
    
    annotation_file = os.path.join(args.psg_folder, 'psg.json')

    dataset = PanopticSceneGraphDataset(img_folder=args.coco_path, ann_file=annotation_file, num_queries=args.num_queries, transforms=make_coco_transforms(image_set, args), image_set=image_set)
    return dataset