from PIL import Image
import os, pickle
import numpy as np

import torch
import torch.utils.data
import cv2
from datasets.vcoco_text_label import vcoco_hoi_text_label

import datasets.transforms as T

class VCOCO(torch.utils.data.Dataset):

    def __init__(self, img_set, img_folder, anno_file, transforms, num_queries):
        self.img_set = img_set
        self.img_folder = img_folder

        self._transforms = transforms

        self.num_queries = num_queries

        self._valid_obj_ids = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
                               14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                               24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
                               37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
                               48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
                               58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
                               72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
                               82, 84, 85, 86, 87, 88, 89, 90)
        self._valid_verb_ids = range(29)
        self.hoi_text_label_ids = list(vcoco_hoi_text_label.keys())
        if img_set == 'train':
            self.annotation_files = os.listdir(anno_file)
            self.ann_dir = anno_file
            self.ids = torch.arange(len(self.annotation_files))
        else:
            with open(anno_file, 'rb') as f:
                self.annotations = pickle.load(f)
            self.ids = list(range(len(self.annotations)))
        

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):

        if self.img_set == 'train':
            file_path = os.path.join(self.ann_dir, self.annotation_files[self.ids[idx]])
            with open(file_path, 'rb') as f:
                img_anno = pickle.load(f)
        else:
            img_anno = self.annotations[self.ids[idx]]

        img_name = img_anno['file_name']
        target = img_anno['annotations'] 

        if self.img_set == 'train' and len(img_anno['hoi_annotation']) > self.num_queries:
            img_anno['hoi_annotation'] = img_anno['hoi_annotation'][:self.num_queries]
        if 'train2014' in img_name:
            img_path = self.img_folder + '/train2014/%s' % img_name
        elif 'val2014' in img_name:
            img_path = self.img_folder + '/val2014/%s' % img_name
        else:  # For single image visualization.
            raise NotImplementedError()

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = Image.fromarray(img[:, :, ::-1]).convert('RGB')
        w, h = img.size

    
        masks = torch.stack([torch.from_numpy(obj['mask'][0]) for obj in img_anno['annotations']])
        masks = torch.as_tensor(masks, dtype=torch.bool)

        if self.img_set == 'train':
            # Add index for confirming which boxes are kept after image transformation
            classes = [(i, self._valid_obj_ids.index(obj['category_id'])) for i, obj in enumerate(img_anno['annotations'])]
        else:
            classes = [self._valid_obj_ids.index(obj['category_id']) for obj in img_anno['annotations']]
        classes = torch.tensor(classes, dtype=torch.int64)

        target = {}
        target['orig_size'] = torch.as_tensor([int(h), int(w)])
        target['size'] = torch.as_tensor([int(h), int(w)])
        if self.img_set == 'train':
            target['masks'] = masks
            target['labels'] = classes # list of [index, category_id]

            if self._transforms is not None:
                img, target = self._transforms(img, target)

            kept_mask_indices = [label[0] for label in target['labels']]
            target['labels'] = target['labels'][:, 1] # list of kept category_id
            
            obj_labels, verb_labels, sub_masks, obj_masks = [], [], [], []
            sub_obj_pairs = []
            hoi_labels = [] # triplet
            for hoi in img_anno['hoi_annotation']:
                if hoi['subject_id'] not in kept_mask_indices or \
                   (hoi['object_id'] != -1 and hoi['object_id'] not in kept_mask_indices):
                    continue

                if hoi['object_id'] == -1:
                    verb_obj_pair = (self._valid_verb_ids.index(hoi['category_id']), 80)
                else:
                    verb_obj_pair = (self._valid_verb_ids.index(hoi['category_id']),
                                     target['labels'][kept_mask_indices.index(hoi['object_id'])])
                
                if verb_obj_pair not in self.hoi_text_label_ids:
                    continue

                sub_obj_pair = (hoi['subject_id'], hoi['object_id'])
                if sub_obj_pair in sub_obj_pairs:
                    verb_labels[sub_obj_pairs.index(sub_obj_pair)][self._valid_verb_ids.index(hoi['category_id'])] = 1
                    hoi_labels[sub_obj_pairs.index(sub_obj_pair)][self.hoi_text_label_ids.index(verb_obj_pair)] = 1
                else:
                    sub_obj_pairs.append(sub_obj_pair)
                    if hoi['object_id'] == -1: # no object, only <person, verb>
                        obj_labels.append(torch.tensor(len(self._valid_obj_ids)))
                    else:
                        obj_labels.append(target['labels'][kept_mask_indices.index(hoi['object_id'])])
                    verb_label = [0 for _ in range(len(self._valid_verb_ids))]
                    verb_label[self._valid_verb_ids.index(hoi['category_id'])] = 1

                    hoi_label = [0] * len(self.hoi_text_label_ids)
                    hoi_label[self.hoi_text_label_ids.index(verb_obj_pair)] = 1

                    sub_mask = target['masks'][kept_mask_indices.index(hoi['subject_id'])]
                    if hoi['object_id'] == -1:
                        obj_mask = torch.zeros((target['size'][0], target['size'][1]), dtype=torch.bool)
                    else:
                        obj_mask = target['masks'][kept_mask_indices.index(hoi['object_id'])]
                    verb_labels.append(verb_label)
                    hoi_labels.append(hoi_label)
                    sub_masks.append(sub_mask)
                    obj_masks.append(obj_mask)

            target['filename'] = img_anno['file_name']
            if len(sub_obj_pairs) == 0:
                target['obj_labels'] = torch.zeros((0,), dtype=torch.int64)
                target['verb_labels'] = torch.zeros((0, len(self._valid_verb_ids)), dtype=torch.float32)
                target['hoi_labels'] = torch.zeros((0, len(self.hoi_text_label_ids)), dtype=torch.float32)
                target['sub_masks'] = torch.zeros((0, target['size'][0], target['size'][1]), dtype=torch.bool)
                target['obj_masks'] = torch.zeros((0, target['size'][0], target['size'][1]), dtype=torch.bool)
                target['matching_labels'] = torch.zeros((0,), dtype=torch.int64)
            else:
                target['obj_labels'] = torch.stack(obj_labels)
                target['verb_labels'] = torch.as_tensor(verb_labels, dtype=torch.float32)
                target['hoi_labels'] = torch.as_tensor(hoi_labels, dtype=torch.float32)
                target['sub_masks'] = torch.stack(sub_masks)
                target['obj_masks'] = torch.stack(obj_masks)
                target['matching_labels'] = torch.ones_like(target['obj_labels'])
        else:
            target['filename'] = img_anno['file_name']
            target['masks'] = masks
            target['labels'] = classes
            target['id'] = idx
            target['img_id'] = int(img_anno['file_name'].rstrip('.jpg').split('_')[2])

            if self._transforms is not None:
                img, _ = self._transforms(img, None)

            hois = []
            for hoi in img_anno['hoi_annotation']:
                hois.append((hoi['subject_id'], hoi['object_id'], self._valid_verb_ids.index(hoi['category_id'])))
            target['hois'] = torch.as_tensor(hois, dtype=torch.int64)

        target['image'] = img
        target['source'] = 'vcoco'
        return target

    def load_correct_mat(self, path):
        self.correct_mat = np.load(path)


# Add color jitter to coco transforms
def make_vcoco_transforms(image_set, args):

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
            T.ColorJitter(.4, .4, .4),
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


def build(image_set, args):
    if image_set == 'train':
        annotation_dir = os.path.join(args.vcoco_path, 'annotations/trainval_vcoco_w_saml_mask')
    else:
        annotation_dir = os.path.join(args.vcoco_path, 'annotations/test_vcoco_w_saml_mask_merged.pkl')
    
    CORRECT_MAT_PATH = os.path.join(args.vcoco_path, 'annotations/corre_vcoco.npy')

    img_folder = os.path.join(args.vcoco_path, 'images')
    dataset = VCOCO(image_set, img_folder, annotation_dir, transforms=make_vcoco_transforms(image_set, args),
                    num_queries=args.num_queries)
    if image_set == 'test':
        dataset.load_correct_mat(CORRECT_MAT_PATH)
    return dataset