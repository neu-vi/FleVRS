# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# ------------------------------------------------------------------------

from torchvision.datasets.vision import VisionDataset
import torch
import numpy as np
import json
import cv2
import random
import os

import datasets.transforms as T
from datasets.hico_text_label import hico_verb_names, hico_unseen_index
import torchvision.transforms.functional as F
from PIL import Image
import pickle
from collections import defaultdict


coco_classes_originID = {
    "person": 1,
    "bicycle": 2,
    "car": 3,
    "motorcycle": 4,
    "airplane": 5,
    "bus": 6,
    "train": 7,
    "truck": 8,
    "boat": 9,
    "traffic light": 10,
    "fire hydrant": 11,
    "stop sign": 13,
    "parking meter": 14,
    "bench": 15,
    "bird": 16,
    "cat": 17,
    "dog": 18,
    "horse": 19,
    "sheep": 20,
    "cow": 21,
    "elephant": 22,
    "bear": 23,
    "zebra": 24,
    "giraffe": 25,
    "backpack": 27,
    "umbrella": 28,
    "handbag": 31,
    "tie": 32,
    "suitcase": 33,
    "frisbee": 34,
    "skis": 35,
    "snowboard": 36,
    "sports ball": 37,
    "kite": 38,
    "baseball bat": 39,
    "baseball glove": 40,
    "skateboard": 41,
    "surfboard": 42,
    "tennis racket": 43,
    "bottle": 44,
    "wine glass": 46,
    "cup": 47,
    "fork": 48,
    "knife": 49,
    "spoon": 50,
    "bowl": 51,
    "banana": 52,
    "apple": 53,
    "sandwich": 54,
    "orange": 55,
    "broccoli": 56,
    "carrot": 57,
    "hot dog": 58,
    "pizza": 59,
    "donut": 60,
    "cake": 61,
    "chair": 62,
    "couch": 63,
    "potted plant": 64,
    "bed": 65,
    "dining table": 67,
    "toilet": 70,
    "tv": 72,
    "laptop": 73,
    "mouse": 74,
    "remote": 75,
    "keyboard": 76,
    "cell phone": 77,
    "microwave": 78,
    "oven": 79,
    "toaster": 80,
    "sink": 81,
    "refrigerator": 82,
    "book": 84,
    "clock": 85,
    "vase": 86,
    "scissors": 87,
    "teddy bear": 88,
    "hair drier": 89,
    "toothbrush": 90,
}


coco_instance_ID_to_name = {
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "airplane",
    6: "bus",
    7: "train",
    8: "truck",
    9: "boat",
    10: "traffic light",
    11: "fire hydrant",
    13: "stop sign",
    14: "parking meter",
    15: "bench",
    16: "bird",
    17: "cat",
    18: "dog",
    19: "horse",
    20: "sheep",
    21: "cow",
    22: "elephant",
    23: "bear",
    24: "zebra",
    25: "giraffe",
    27: "backpack",
    28: "umbrella",
    31: "handbag",
    32: "tie",
    33: "suitcase",
    34: "frisbee",
    35: "skis",
    36: "snowboard",
    37: "sports ball",
    38: "kite",
    39: "baseball bat",
    40: "baseball glove",
    41: "skateboard",
    42: "surfboard",
    43: "tennis racket",
    44: "bottle",
    46: "wine glass",
    47: "cup",
    48: "fork",
    49: "knife",
    50: "spoon",
    51: "bowl",
    52: "banana",
    53: "apple",
    54: "sandwich",
    55: "orange",
    56: "broccoli",
    57: "carrot",
    58: "hot dog",
    59: "pizza",
    60: "donut",
    61: "cake",
    62: "chair",
    63: "couch",
    64: "potted plant",
    65: "bed",
    67: "dining table",
    70: "toilet",
    72: "tv",
    73: "laptop",
    74: "mouse",
    75: "remote",
    76: "keyboard",
    77: "cell phone",
    78: "microwave",
    79: "oven",
    80: "toaster",
    81: "sink",
    82: "refrigerator",
    84: "book",
    85: "clock",
    86: "vase",
    87: "scissors",
    88: "teddy bear",
    89: "hair drier",
    90: "toothbrush",
}


def make_hico_transforms(image_set, args):

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

class HoiDetection(VisionDataset):

    def __init__(self, root, annFile, box_ann_file, open_voc_file_ids, num_queries, transform=None, target_transform=None, transforms=None, image_set='train', flexible_task=False, flexible_eval_task=None, flexible_test_set=None, unseen_type=None):
        assert image_set in ['train', 'test'], image_set
        self.image_set = image_set
        super(HoiDetection, self).__init__(transforms, transform, target_transform)

        self.root = root
        self._valid_obj_ids = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
                               14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                               24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
                               37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
                               48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
                               58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
                               72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
                               82, 84, 85, 86, 87, 88, 89, 90)
        verb_list = list(range(1, 118))
        verb_list.remove(58) # no interaction
        self._valid_verb_ids = verb_list

        self.flexible_task = flexible_task # can be "pred_verb"/"pred_sub"/"pred_obj"
        self.flexible_eval_task = flexible_eval_task # can be "pred_verb"/"pred_sub"/"pred_obj"

        if flexible_test_set is not None:
            with open(flexible_test_set, 'rb') as f:
                self.triplet_ids_test = pickle.load(f)

        self.unseen_type = unseen_type

        if unseen_type==None:
            self.unseen_index = None
        elif self.unseen_type == 'uc_rf':
            self.unseen_index = hico_unseen_index["rare_first"]
        elif self.unseen_type == 'uc_nf':
            self.unseen_index = hico_unseen_index["non_rare_first"]
        elif self.unseen_type == 'uo':
            self.unseen_index = hico_unseen_index["unseen_object"]
            if self.image_set=='train':
                self.unseen_obj = hico_unseen_index["unseen_object_class"]
                tmp = []
                for idx, obj_id in enumerate(self._valid_obj_ids):
                    if idx in self.unseen_obj:
                        continue
                    else:
                        tmp.append(obj_id)
                self._valid_obj_ids = tmp
                
        elif self.unseen_type == 'uv':
            self.unseen_index = hico_unseen_index["unseen_verb"]
            if self.image_set=='train':
                self.unseen_verb = hico_unseen_index["unseen_verb_class"]
                tmp = []
                for idx, verb_id in enumerate(verb_list):
                    if verb_id == 58: # no_interaction
                        continue
                    if idx in self.unseen_verb:
                        continue
                    else:
                        tmp.append(verb_id)
                self._valid_verb_ids = tmp

        if image_set == 'train' and unseen_type==None:
            self.annotation_files = os.listdir(annFile)
            self.ann_dir = annFile
            self.ids = torch.arange(len(self.annotation_files))

        elif image_set == 'train' and self.unseen_type == 'uc_rf':
            self.ann_dir = annFile
            with open(os.path.join(root, open_voc_file_ids['uc_rf']), 'rb') as f:
                self.annotation_files = pickle.load(f)
            self.ids = torch.arange(len(self.annotation_files))
            
        elif image_set == 'train' and self.unseen_type == 'uc_nf':
            self.ann_dir = annFile
            with open(os.path.join(root, open_voc_file_ids['uc_nf']), 'rb') as f:
                self.annotation_files = pickle.load(f)
            self.ids = torch.arange(len(self.annotation_files))
        elif image_set == 'train' and self.unseen_type == 'uo':
            self.ann_dir = annFile
            with open(os.path.join(root, open_voc_file_ids['uo']), 'rb') as f:
                self.annotation_files = pickle.load(f)
            self.ids = torch.arange(len(self.annotation_files))
        elif image_set == 'train' and self.unseen_type == 'uv':
            self.ann_dir = annFile
            with open(os.path.join(root, open_voc_file_ids['uv']), 'rb') as f:
                self.annotation_files = pickle.load(f)
            self.ids = torch.arange(len(self.annotation_files))   
        else:
            with open(annFile, 'rb') as f:
                self.annotations = pickle.load(f)
            self.ids = list(range(len(self.annotations)))

        self._transforms = transforms
        self.num_queries = num_queries

    def __getitem__(self, index):
        
        if self.image_set == 'train':
            file_path = os.path.join(self.ann_dir, self.annotation_files[self.ids[index]])
            with open(file_path, 'rb') as f:
                ann = pickle.load(f)
        else:
            ann = self.annotations[self.ids[index]]
            

        img_name = ann['file_name']
        target = ann['annotations'] 

        if 'train2015' in img_name:
            img_path = self.root + '/images/train2015/%s' % img_name
        elif 'test2015' in img_name:
            img_path = self.root + '/images/test2015/%s' % img_name
        else:  # For single image visualization.
            raise NotImplementedError()
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = Image.fromarray(img[:, :, ::-1]).convert('RGB')
        w, h = img.size
       
        if self.unseen_index is not None and self.image_set == 'train':
            skip_pair = []
            new_ann = []
            for hoi in ann['hoi_annotation']:
                if hoi['hoi_category_id'] - 1 in self.unseen_index:
                    skip_pair.append((hoi['subject_id'], hoi['object_id']))
            for hoi in ann['hoi_annotation']:
                if hoi['subject_id'] >= len(ann['annotations']) or hoi['object_id'] >= len(
                        ann['annotations']):
                    new_ann = []
                    break
                if (hoi['subject_id'], hoi['object_id']) not in skip_pair:
                    new_ann.append(hoi)
            assert len(new_ann) > 0
            ann['hoi_annotation'] = new_ann

            if self.unseen_type=='uo':
                new_mask_ann = []
                for mask_ann in ann['annotations']:
                    if mask_ann['category_id'] in self._valid_obj_ids:
                        new_mask_ann.append(mask_ann)
                ann['annotations'] = new_mask_ann

        
        if self.image_set == 'train' and len(ann['hoi_annotation']) > self.num_queries:
            ann['hoi_annotation'] = ann['hoi_annotation'][:self.num_queries]

        masks = torch.stack([torch.from_numpy(obj['mask'][0]) for obj in ann['annotations']])
        masks = torch.as_tensor(masks, dtype=torch.bool)
        
        if self.image_set == 'train':
            classes = [(i, self._valid_obj_ids.index(obj['category_id'])) for i, obj in enumerate(ann['annotations'])]
        else:
            classes = [self._valid_obj_ids.index(obj['category_id']) for obj in ann['annotations']]
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
            
            obj_labels, verb_labels, sub_masks, obj_masks = [], [], [], []
            sub_obj_pairs = []
            for hoi in ann['hoi_annotation']:
                if hoi['subject_id'] not in kept_mask_indices or hoi['object_id'] not in kept_mask_indices:
                    continue
                if self.unseen_index is not None:
                    assert hoi['hoi_category_id'] - 1 not in self.unseen_index
                sub_obj_pair = (hoi['subject_id'], hoi['object_id'])
                if sub_obj_pair in sub_obj_pairs:
                    if hoi['category_id'] == 58: # remove no interaction
                        continue
                    verb_labels[sub_obj_pairs.index(sub_obj_pair)][self._valid_verb_ids.index(hoi['category_id'])] = 1
                else:
                    if hoi['category_id'] == 58: # remove no interaction
                        continue
                    sub_obj_pairs.append(sub_obj_pair)
                    obj_labels.append(target['labels'][kept_mask_indices.index(hoi['object_id'])])
                    verb_label = [0 for _ in range(len(self._valid_verb_ids))]
                    verb_label[self._valid_verb_ids.index(hoi['category_id'])] = 1

                    sub_mask = target['masks'][kept_mask_indices.index(hoi['subject_id'])]
                    obj_mask = target['masks'][kept_mask_indices.index(hoi['object_id'])]

                    verb_labels.append(verb_label)
                    
                    sub_masks.append(sub_mask)
                    obj_masks.append(obj_mask)
                
            target['filename'] = ann['file_name']
            if len(sub_obj_pairs) == 0:
                target['obj_labels'] = torch.zeros((0,), dtype=torch.int64)
                target['verb_labels'] = torch.zeros((0, len(self._valid_verb_ids)), dtype=torch.float32)
                
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
                                 'gverb_classes': torch.zeros((0, len(self._valid_verb_ids)), dtype=torch.float32), 
                                 'gobj_classes': torch.zeros((0,), dtype=torch.int64),
                                 'gtext': text_grd, 
                                 'ghash': hash_grd}
                    
                    target['grounding'] = grounding
            else:
                target['obj_labels'] = torch.stack(obj_labels)
               
                target['verb_labels'] = torch.as_tensor(verb_labels, dtype=torch.float32)
                
                target['sub_masks'] = torch.stack(sub_masks)
                target['obj_masks'] = torch.stack(obj_masks)

                target['matching_labels'] = torch.ones_like(target['obj_labels'])
                if self.flexible_task: 
                    task_pool = ['pred_verb', 'pred_obj', 'pred_sub']
                    task_name = random.choice(task_pool)
    
                    triplet_num = len(obj_labels)
                    # random choose a triplet, to construct prompt
                    rand_id = np.random.choice(np.arange(triplet_num))
                    obj_class_id = obj_labels[rand_id]
                    obj_classname = list(coco_classes_originID.keys())[obj_labels[rand_id]]

                    verb_class_id = np.random.choice(np.where(np.array(verb_labels[rand_id]) == 1)[0])
                    verb_classname = hico_verb_names[verb_class_id]
                    grounding = {}
                    
                    if task_name == 'pred_verb':
                        text_grd = ['<sub>person<sub/><obj>{}<obj/>'.format(obj_classname)]
                        # find all sub-obj pairs
                        obj_gids = [i for i, value in enumerate(obj_labels) if value == obj_class_id]
                    elif task_name == 'pred_obj':
                        text_grd = ['<sub>person<sub/><verb>{}<verb/>'.format(verb_classname)]
                        obj_gids = [i for i, value in enumerate(verb_labels) if value[verb_class_id]==1]
                    elif task_name == 'pred_sub':
                        text_grd = ['<obj>{}<obj/><verb>{}<verb/>'.format(obj_classname, verb_classname)]
                        obj_gids = []
                        for i, value in enumerate(obj_labels):
                            if value==obj_class_id and verb_labels[i][verb_class_id]==1:
                                obj_gids.append(i)

                    hash_grd = np.array([hash(txt) for txt in text_grd])
                    obj_gmasks = [obj_masks[i] for i in obj_gids]
                    sub_gmasks = [sub_masks[i] for i in obj_gids]
                    gverbs = [verb_labels[i] for i in obj_gids]
                    gobjs = [obj_labels[i] for i in obj_gids]
                    grounding = {'gtask': task_name, 
                                'sub_gmasks': torch.stack(sub_gmasks), 
                                'obj_gmasks': torch.stack(obj_gmasks), 
                                'gverb_classes': torch.as_tensor(gverbs, dtype=torch.float32), 
                                'gobj_classes': torch.stack(gobjs),
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
            hois = []
            for hoi in ann['hoi_annotation']:
                if hoi['category_id'] == 58: # remove no interaction
                    continue
                hois.append((hoi['subject_id'], hoi['object_id'], self._valid_verb_ids.index(hoi['category_id'])))
            target['hois'] = torch.as_tensor(hois, dtype=torch.int64)
            if self.flexible_eval_task=="pred_verb" or self.flexible_eval_task=="pred_sub" or self.flexible_eval_task=="pred_obj":
                if len(hois)==0:
                    target['grounding'] = None
                else:
                    rand_hoi_id = self.triplet_ids_test[ann['file_name']]

                    obj_class_id = classes[hois[rand_hoi_id][1]]
                    verb_class_id = hois[rand_hoi_id][2]

                    obj_classname = list(coco_classes_originID.keys())[classes[hois[rand_hoi_id][1]]]
                    verb_classname = hico_verb_names[hois[rand_hoi_id][2]]
                    if self.flexible_eval_task == 'pred_verb':
                        text_grd = ['<sub>person<sub/><obj>{}<obj/>'.format(obj_classname)]
                        g_hois = []
                        for g_hoi in ann['hoi_annotation']:
                            if g_hoi['category_id'] == 58: # remove no interaction
                                continue
                            if classes[g_hoi['object_id']] == obj_class_id:
                                g_hois.append((g_hoi['subject_id'], g_hoi['object_id'], self._valid_verb_ids.index(g_hoi['category_id'])))
                    elif self.flexible_eval_task == 'pred_obj':
                        text_grd = ['<sub>person<sub/><verb>{}<verb/>'.format(verb_classname)]
                        g_hois = []
                        for g_hoi in ann['hoi_annotation']:
                            if g_hoi['category_id'] == 58: # remove no interaction
                                continue
                            if self._valid_verb_ids.index(g_hoi['category_id']) == verb_class_id:
                                g_hois.append((g_hoi['subject_id'], g_hoi['object_id'], self._valid_verb_ids.index(g_hoi['category_id'])))
                    elif self.flexible_eval_task == 'pred_sub':
                        text_grd = ['<obj>{}<obj/><verb>{}<verb/>'.format(obj_classname, verb_classname)]
                        g_hois = []
                        for g_hoi in ann['hoi_annotation']:
                            if g_hoi['category_id'] == 58: # remove no interaction
                                continue
                            if self._valid_verb_ids.index(g_hoi['category_id']) == verb_class_id and classes[g_hoi['object_id']] == obj_class_id:
                                g_hois.append((g_hoi['subject_id'], g_hoi['object_id'], self._valid_verb_ids.index(g_hoi['category_id'])))

                    hash_grd = np.array([hash(txt) for txt in text_grd])
                
                    grounding = {'gtask': self.flexible_task, 
                                 'ghois': torch.as_tensor(g_hois, dtype=torch.int64),
                                 'p_verb': torch.as_tensor(verb_class_id),
                                 'p_obj': torch.as_tensor(obj_class_id),
                                 'gtext': text_grd, 
                                 'ghash': hash_grd}

                    target['grounding'] = grounding

        target['image'] = img
        target['source'] = 'hico'
        return target

    def __len__(self):
        return len(self.ids)

    def load_correct_mat(self, path):
        self.correct_mat = np.load(path)

    def set_rare_hois(self, anno_file):
        with open(anno_file, 'r') as f:
            annotations = json.load(f)

        if self.unseen_index is None:
            # no unseen category, use rare to evaluate
            counts = defaultdict(lambda: 0)
            for img_anno in annotations:
                hois = img_anno['hoi_annotation']
                bboxes = img_anno['annotations']
                for hoi in hois:
                    if hoi['category_id']==58:
                        continue
                    triplet = (self._valid_obj_ids.index(bboxes[hoi['subject_id']]['category_id']),
                            self._valid_obj_ids.index(bboxes[hoi['object_id']]['category_id']),
                            self._valid_verb_ids.index(hoi['category_id']))
                    counts[triplet] += 1
            self.rare_triplets = []
            self.non_rare_triplets = []
            for triplet, count in counts.items():
                if count < 10:
                    self.rare_triplets.append(triplet)
                else:
                    self.non_rare_triplets.append(triplet)
            print("rare:{}, non-rare:{}".format(len(self.rare_triplets), len(self.non_rare_triplets)))
        else:
            self.rare_triplets = []
            self.non_rare_triplets = []
            for img_anno in annotations:
                hois = img_anno['hoi_annotation']
                bboxes = img_anno['annotations']
                for hoi in hois:
                    if hoi['category_id'] == 58:
                        continue
                    triplet = (self._valid_obj_ids.index(bboxes[hoi['subject_id']]['category_id']),
                            self._valid_obj_ids.index(bboxes[hoi['object_id']]['category_id']),
                            self._valid_verb_ids.index(hoi['category_id']))
                    if hoi['hoi_category_id'] - 1 in self.unseen_index:
                        self.rare_triplets.append(triplet)
                    else:
                        self.non_rare_triplets.append(triplet)
            print("unseen:{}, seen:{}".format(len(self.rare_triplets), len(self.non_rare_triplets)))


def build(image_set, args):
    assert image_set in ['train', 'test'], image_set

    open_voc_file_ids = {
        "uc_rf": 'annotations/open_voc/uc_rf_ids_train.pkl',
        "uc_nf": 'annotations/open_voc/uc_nf_ids_train.pkl',
        "uo": 'annotations/open_voc/uo_ids_train.pkl',
        "uv": 'annotations/open_voc/uv_ids_train.pkl',
    }

    if image_set == 'train':
        annotation_dir = os.path.join(args.hoi_path, 'annotations/trainval_hico_samL_mask_filt')
        
    else:
        annotation_dir = os.path.join(args.hoi_path, 'annotations/test_hico_w_sam_mask_merged.pkl')
    CORRECT_MAT_PATH = os.path.join(args.hoi_path, 'annotations/corre_hico_filtered_nointer.npy')
    
    box_annotation_files = {'train': os.path.join(args.hoi_path, 'annotations/trainval_hico.json'),
                            'test': os.path.join(args.hoi_path, 'annotations/test_hico.json')}
    
    dataset = HoiDetection(root=args.hoi_path, annFile=annotation_dir, open_voc_file_ids=open_voc_file_ids, num_queries=args.num_queries,
                           transforms=make_hico_transforms(image_set, args), image_set=image_set, flexible_task=args.flexible_grounding, flexible_eval_task=args.flexible_eval_task, flexible_test_set=args.flexible_test_set, unseen_type=args.unseen_type)

    if image_set == 'test':
        dataset.set_rare_hois(box_annotation_files['train'])
        dataset.load_correct_mat(CORRECT_MAT_PATH)
    return dataset


        