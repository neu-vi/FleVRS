from torchvision.datasets.vision import VisionDataset
import torch
import numpy as np
import cv2
import random
import os

import datasets.transforms as T
from datasets.vrd_text_label import seen_object_ids, unseen_object_ids, seen_object_names, predicate_classes, full_object_names
import torchvision.transforms.functional as F
from PIL import Image
import pickle
from collections import defaultdict


def make_vrd_transforms(image_set, args):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if args.input_img_size == 1024:
        scales = [408, 510, 612, 714, 816, 918, 1020, 1122, 1224, 1326, 1428, 1530, 1632, 1734, 1836, 1938, 2040] 
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


class VRD(VisionDataset):
   
    def __init__(self, root, annFile, num_queries, transform=None, target_transform=None, transforms=None, image_set='train', flexible_task=False, flexible_eval_task=None, flexible_test_set=None, unseen_type=None):
        assert image_set in ['train', 'test'], image_set
        self.image_set = image_set
        super(VRD, self).__init__(transforms, transform, target_transform)

        self.root = root
        if self.image_set=='train':
            self._valid_obj_ids = seen_object_ids
            with open('/work/vig/Datasets/VRD/seen_obj_files.pkl', 'rb') as f:
                self.annotation_files = pickle.load(f)
            self.ann_dir = annFile
            self.ids = list(range(len(self.annotation_files)))
        else:
            self._valid_obj_ids = list(range(0, 100))
            # self._valid_obj_ids = unseen_object_ids
            with open(annFile, 'rb') as f:
                self.annotations = pickle.load(f)
            self.ids = list(range(len(self.annotations)))

        self._valid_verb_ids = list(range(0, 70))

        self.flexible_task = flexible_task # can be "pred_verb"/"pred_sub"/"pred_obj"
        self.flexible_eval_task = flexible_eval_task # can be "pred_verb"/"pred_sub"/"pred_obj"

        self._transforms = transforms
        self.num_queries = num_queries

    def __getitem__(self, index):
        # print(self.flexible_task)
        
        if self.image_set == 'train':
            file_path = os.path.join(self.ann_dir, self.annotation_files[self.ids[index]])
            with open(file_path, 'rb') as f:
                ann = pickle.load(f)
        else:
            ann = self.annotations[self.ids[index]]

        img_name = ann['file_name']
        target = ann['annotations'] 

        if self.image_set=='train':
            img_path = os.path.join('data/VRD/sg_dataset/sg_train_images', img_name)
        elif self.image_set=='test':
            img_path = os.path.join('data/VRD/sg_dataset/sg_test_images', img_name)
        else:  # For single image visualization.
            raise NotImplementedError()
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = Image.fromarray(img[:, :, ::-1]).convert('RGB')
        w, h = img.size
       
        if self.image_set == 'train':
            skip_pair = []
            new_ann = []
            for sop in ann['sop_annotations']:
                if ann['annotations'][sop['subject_id']]['category'] in unseen_object_ids or ann['annotations'][sop['object_id']]['category'] in unseen_object_ids:
                    skip_pair.append((sop['subject_id'], sop['object_id']))
            for sop in ann['sop_annotations']:
                if sop['subject_id'] >= len(ann['annotations']) or sop['object_id'] >= len(
                        ann['annotations']):
                    new_ann = []
                    break
                if (sop['subject_id'], sop['object_id']) not in skip_pair:
                    new_ann.append(sop)
            assert len(new_ann) > 0
            ann['sop_annotations'] = new_ann

        if self.image_set == 'train' and len(ann['sop_annotations']) > self.num_queries:
            ann['sop_annotations'] = ann['sop_annotations'][:self.num_queries]

        masks = torch.stack([torch.from_numpy(obj['mask'][0]) for obj in ann['annotations']])
        masks = torch.as_tensor(masks, dtype=torch.bool)
        
        if self.image_set == 'train':
            classes = []
            for i, obj in enumerate(ann['annotations']):
                if obj['category'] in unseen_object_ids:
                    classes.append((i, -1))
                else:
                    classes.append((i, self._valid_obj_ids.index(obj['category'])))
            # classes = [(i, self._valid_obj_ids.index(obj['category_id'])) for i, obj in enumerate(ann['annotations'])]
        else:
            classes = [self._valid_obj_ids.index(obj['category']) for obj in ann['annotations']]
        classes = torch.tensor(classes, dtype=torch.int64)

        target = {}
        target['orig_size'] = torch.as_tensor([int(h), int(w)])
        target['size'] = torch.as_tensor([int(h), int(w)])
        
        if self.image_set == 'train':
            
            target['masks'] = masks
            target['labels'] = classes # list of [index, category_id]

            if self._transforms is not None:
                img, target = self._transforms(img, target) # may reduce some masks due to crop
            kept_mask_indices = [label[0] for label in target['labels']]
            target['labels'] = target['labels'][:, 1] # list of kept category_id
            
            sub_labels, obj_labels, verb_labels, sub_masks, obj_masks, hoi_labels = [], [], [], [], [], []
            # sub_obj_pairs = []
            all_rel_sets = defaultdict(list)
            for sop in ann['sop_annotations']:
                if sop['subject_id'] not in kept_mask_indices or sop['object_id'] not in kept_mask_indices:
                    continue
              
                sub_obj_pair = (sop['subject_id'], sop['object_id'])
                all_rel_sets[sub_obj_pair].append(sop['category_id'])
            if len(all_rel_sets) != 0:
                gt_rels = [(k[0], k[1], np.random.choice(v))
                        for k, v in all_rel_sets.items()]
                for (sid, oid, rid) in gt_rels:
                    obj_labels.append(target['labels'][kept_mask_indices.index(oid)])
                    sub_labels.append(target['labels'][kept_mask_indices.index(sid)])
                    verb_labels.append(torch.tensor(rid, dtype=torch.int64))

                    sub_mask = target['masks'][kept_mask_indices.index(sid)]
                    obj_mask = target['masks'][kept_mask_indices.index(oid)]
                    sub_masks.append(sub_mask)
                    obj_masks.append(obj_mask)
                assert len(sub_masks) == len(obj_masks)
                    


            target['filename'] = ann['file_name']
            if len(all_rel_sets) == 0:
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
                target['obj_labels'] = torch.stack(obj_labels)
                target['sub_labels'] = torch.stack(sub_labels)
                
                target['verb_labels'] = torch.stack(verb_labels)
                target['sub_masks'] = torch.stack(sub_masks)
                target['obj_masks'] = torch.stack(obj_masks)
                target['matching_labels'] = torch.ones_like(target['obj_labels'])
                if self.flexible_task: 
                    task_pool = ['pred_obj', 'pred_sub', 'pred_sub_obj']
                    task_name = random.choice(task_pool)
                   
                    triplet_num = len(obj_labels)
                    # random choose a triplet, to construct prompt
                    rand_id = np.random.choice(np.arange(triplet_num))
                    obj_class_id = obj_labels[rand_id]
                    obj_classname = seen_object_names[obj_labels[rand_id]]

                    sub_class_id = sub_labels[rand_id]
                    sub_classname = seen_object_names[sub_labels[rand_id]]

                    verb_class_id = verb_labels[rand_id]
                    verb_classname = predicate_classes[verb_class_id]
                    grounding = {}
                    
                    if task_name == 'pred_obj':
                        text_grd = ['<sub>{}<sub/><verb>{}<verb/>'.format(sub_classname, verb_classname)]
                        obj_gids = [i for i, value in enumerate(sub_labels) if verb_labels[i]==verb_class_id and value==sub_class_id]
                    elif task_name == 'pred_sub':
                        text_grd = ['<obj>{}<obj/><verb>{}<verb/>'.format(obj_classname, verb_classname)]
                        obj_gids = [i for i, value in enumerate(obj_labels) if verb_labels[i]==verb_class_id and value==obj_class_id]
                        
                    elif task_name == 'pred_sub_obj':
                        text_grd = ['<verb>{}<verb/>'.format(verb_classname)]
                        obj_gids = [i for i, value in enumerate(obj_labels) if value == obj_class_id and sub_labels[i]==sub_class_id]

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
            for sop in ann['sop_annotations']:
                sops.append((sop['subject_id'], sop['object_id'], sop['category_id']))
            target['sops'] = torch.as_tensor(sops, dtype=torch.int64)
            if self.flexible_eval_task=="pred_verb" or self.flexible_eval_task=="pred_sub" or self.flexible_eval_task=="pred_obj" or self.flexible_eval_task=="pred_sub_obj":
                if len(sops)==0:
                    target['grounding'] = None
                else:
                    rand_sop_id = np.random.choice(np.arange(len(sops)))

                    sub_class_id = classes[sops[rand_sop_id][0]]
                    obj_class_id = classes[sops[rand_sop_id][1]]
                    verb_class_id = sops[rand_sop_id][2]

                    sub_classname = full_object_names[classes[sops[rand_sop_id][0]]]
                    obj_classname = full_object_names[classes[sops[rand_sop_id][1]]]
                    verb_classname = predicate_classes[verb_class_id]
                    if self.flexible_eval_task == 'pred_obj':
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
                    elif self.flexible_eval_task == 'pred_sub_obj':
                        text_grd = ['<verb>{}<verb/>'.format(verb_classname)]
                        g_sops = []
                        for g_sop in sops:
                            if g_sop[2] == verb_class_id:
                                g_sops.append(g_sop)
        
                    hash_grd = np.array([hash(txt) for txt in text_grd])
                   
                    grounding = {'gtask': self.flexible_task, 
                                 'gsops': torch.as_tensor(g_sops, dtype=torch.int64),
                                 'p_verb': torch.as_tensor(verb_class_id),
                                 'p_obj': torch.as_tensor(obj_class_id),
                                 'p_sub': torch.as_tensor(sub_class_id),
                                 'gtext': text_grd, 
                                 'ghash': hash_grd}

                    target['grounding'] = grounding

        target['image'] = img
        target['source'] = 'vrd'
        return target

    def __len__(self):
        return len(self.ids)

def build(image_set, args):
    assert image_set in ['train', 'test'], image_set

    if image_set == 'train':
        annotation_dir = 'data/VRD/train_masks_merged_filt'
        
    else:
        annotation_dir = 'data/VRD/test_masks_merged_filt.pkl'

    dataset = VRD(root=args.vrd_path, annFile=annotation_dir, num_queries=args.num_queries,
                           transforms=make_vrd_transforms(image_set, args), image_set=image_set, flexible_task=args.flexible_grounding, flexible_eval_task=args.flexible_eval_task, flexible_test_set=args.flexible_test_set)

    return dataset

        