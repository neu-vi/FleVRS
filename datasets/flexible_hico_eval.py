import numpy as np
from collections import defaultdict
from bitarray import bitarray
import pickle
import pycocotools.mask as mask_util
import numpy as np


def compute_iou(mask1, mask2):
    iou = mask_util.iou([mask1], [mask2], [0])
    
    return iou[0][0]

class Flexible_HICOEvaluator():
    def __init__(self, preds, gts, correct_mat, args):
        self.overlap_iou = 0.5
        self.fp = defaultdict(list)
        self.tp = defaultdict(list)
        self.score = defaultdict(list)
        self.sum_gts = defaultdict(lambda: 0)
        self.gt_triplets = []
        self.flexible_eval_task = args.flexible_eval_task
        self.max_hois = 100

        

        # when evaluating mask map, exclude those no interaction images 
        with open(args.exclude_filenames_path, 'rb') as f:
            self.exclude_filenames = pickle.load(f)

        self.preds = []
        for index, img_preds in enumerate(preds):

            img_preds = {k: v.to('cpu').numpy() if k != 'masks' and v is not None else v for k, v in img_preds.items()}

            masks = []
            for i in range(2*len(img_preds['sub_ids'])):
                if i < len(img_preds['sub_ids']):
                    masks.append({
                        'mask': img_preds['masks'][i],
                        'category_id': 0
                    })
                else:
                    if img_preds['gobjs'] is not None:
                        l = img_preds['gobjs'][i-len(img_preds['sub_ids'])]
                    else:
                        l = gts[index]['p_obj'].item()
                    masks.append({
                        'mask': img_preds['masks'][i],
                        'category_id': l
                    })

            hoi_scores = img_preds['matching_s']
            
            if self.flexible_eval_task == 'pred_verb':
                subject_ids = np.tile(img_preds['sub_ids'], (hoi_scores.shape[1], 1)).T
                object_ids = np.tile(img_preds['obj_ids'], (hoi_scores.shape[1], 1)).T
            else:
                subject_ids = img_preds['sub_ids']
                object_ids = img_preds['obj_ids']

            if self.flexible_eval_task == 'pred_verb':
                verb_labels = np.tile(np.arange(hoi_scores.shape[1]), (hoi_scores.shape[0], 1))
                # verb_labels = verb_labels.ravel()
            else:
                verb_labels = np.array([gts[index]['p_verb'] for _ in object_ids])
            verb_labels = verb_labels.ravel()

            hoi_scores = hoi_scores.ravel()
            subject_ids = subject_ids.ravel()
            object_ids = object_ids.ravel()

            if len(subject_ids) > 0:
                if img_preds['gobjs'] is not None:
                    object_labels = np.array([ol for ol in img_preds['gobjs']])
                else:
                    object_labels = np.array([gts[index]['p_obj'] for _ in object_ids])
                corr_masks = correct_mat[verb_labels, object_labels]
                hoi_scores *= corr_masks

                hois = [{'subject_id': subject_id, 'object_id': object_id, 'category_id': category_id, 'score': score} for
                        subject_id, object_id, category_id, score in zip(subject_ids, object_ids, verb_labels, hoi_scores)]
                hois.sort(key=lambda k: (k.get('score', 0)), reverse=True)
                if self.flexible_eval_task == 'pred_verb':
                    hois = hois[:self.max_hois]
            else:
                hois = []

            filename = gts[index]['filename']
            if filename in self.exclude_filenames:
                continue
            else: # already sorted
                self.preds.append({
                    'filename':filename,
                    'predictions': masks,
                    'hoi_prediction': hois
                })

        self.gts = []
        for i, img_gts in enumerate(gts):
            filename = img_gts['filename']
            if filename in self.exclude_filenames:
                continue
            img_gts = {k: v.to('cpu').numpy() if k != 'filename' and k != 'prompt' and k != 'masks' else v for k, v in img_gts.items()}
            
            self.gts.append({
                'filename':filename,
                'annotations': [{'mask': mask, 'category_id': label} for mask, label in zip(img_gts['masks'], img_gts['labels'])],
                'hoi_annotation': [{'subject_id': ghoi[0], 'object_id': ghoi[1], 'category_id': ghoi[2]} for ghoi in img_gts['ghois']]
            })
            for hoi in self.gts[-1]['hoi_annotation']:
                triplet = (self.gts[-1]['annotations'][hoi['subject_id']]['category_id'],
                           self.gts[-1]['annotations'][hoi['object_id']]['category_id'],
                           hoi['category_id'])

                if triplet not in self.gt_triplets:
                    self.gt_triplets.append(triplet)

                self.sum_gts[triplet] += 1

        print('init finish')

    def evaluate(self):
        for img_preds, img_gts in zip(self.preds, self.gts):
            if img_gts['filename'] in self.exclude_filenames:
                continue
            pred_masks = img_preds['predictions']
            pred_hois = img_preds['hoi_prediction']
            
            gt_masks = img_gts['annotations']
            gt_hois = img_gts['hoi_annotation']

            if len(gt_masks) != 0:
                mask_pairs, mask_overlaps = self.compute_iou_mat(gt_masks, pred_masks)
                self.compute_fptp(pred_hois, gt_hois, mask_pairs, pred_masks, mask_overlaps)
            else:
                for pred_hoi in pred_hois:
                    triplet = [pred_masks[pred_hoi['subject_id']]['category_id'],
                               pred_masks[pred_hoi['object_id']]['category_id'], pred_hoi['category_id']]
                    if triplet not in self.gt_triplets:
                        continue
                    self.tp[triplet].append(0)
                    self.fp[triplet].append(1)
                    self.score[triplet].append(pred_hoi['score'])
        map = self.compute_map()
        return map

    def compute_iou_mat(self, mask_list1, mask_list2):
        iou_mat = np.zeros((len(mask_list1), len(mask_list2)))
        if len(mask_list1) == 0 or len(mask_list2) == 0:
            return {}
        for i, mask1 in enumerate(mask_list1):
            for j, mask2 in enumerate(mask_list2):
                iou_i = self.compute_IOU(mask1, mask2)
                iou_mat[i, j] = iou_i

        iou_mat_ov=iou_mat.copy()
        iou_mat[iou_mat>=self.overlap_iou] = 1
        iou_mat[iou_mat<self.overlap_iou] = 0

        match_pairs = np.nonzero(iou_mat)
        match_pairs_dict = {}
        match_pair_overlaps = {}
        if iou_mat.max() > 0:
            for i, pred_id in enumerate(match_pairs[1]):
                if pred_id not in match_pairs_dict.keys():
                    match_pairs_dict[pred_id] = []
                    match_pair_overlaps[pred_id]=[]
                match_pairs_dict[pred_id].append(match_pairs[0][i])
                match_pair_overlaps[pred_id].append(iou_mat_ov[match_pairs[0][i],pred_id])
        return match_pairs_dict, match_pair_overlaps
    
    def compute_IOU(self, mask1, mask2):
        if isinstance(mask1['category_id'], str):
            mask1['category_id'] = int(mask1['category_id'].replace('\n', ''))
        if isinstance(mask2['category_id'], str):
            mask2['category_id'] = int(mask2['category_id'].replace('\n', ''))
        if mask1['category_id'] == mask2['category_id']:
            m1 = mask1['mask']
            m2 = mask2['mask']
            return compute_iou(m1, m2)
        else:
            return 0
        
        
    def voc_ap(self, rec, prec):
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
        return ap
    
    def compute_fptp(self, pred_hois, gt_hois, match_pairs, pred_masks, mask_overlaps):
        pos_pred_ids = match_pairs.keys()
        vis_tag = np.zeros(len(gt_hois))
        pred_hois.sort(key=lambda k: (k.get('score', 0)), reverse=True)
        if len(pred_hois) != 0:
            for pred_hoi in pred_hois:
                is_match = 0
                if len(match_pairs) != 0 and pred_hoi['subject_id'] in pos_pred_ids and pred_hoi['object_id'] in pos_pred_ids:
                    pred_sub_ids = match_pairs[pred_hoi['subject_id']]
                    pred_obj_ids = match_pairs[pred_hoi['object_id']]
                    pred_sub_overlaps = mask_overlaps[pred_hoi['subject_id']]
                    pred_obj_overlaps = mask_overlaps[pred_hoi['object_id']]
                    pred_category_id = pred_hoi['category_id']
                    max_overlap = 0
                    max_gt_hoi = 0
                    for gt_hoi in gt_hois:
                        if gt_hoi['subject_id'] in pred_sub_ids and gt_hoi['object_id'] in pred_obj_ids \
                           and pred_category_id == gt_hoi['category_id']:
                            is_match = 1
                            min_overlap_gt = min(pred_sub_overlaps[pred_sub_ids.index(gt_hoi['subject_id'])],
                                                 pred_obj_overlaps[pred_obj_ids.index(gt_hoi['object_id'])])
                            if min_overlap_gt > max_overlap:
                                max_overlap = min_overlap_gt
                                max_gt_hoi = gt_hoi
                triplet = (pred_masks[pred_hoi['subject_id']]['category_id'], pred_masks[pred_hoi['object_id']]['category_id'],
                           pred_hoi['category_id'])
                if triplet not in self.gt_triplets:
                    continue
                if is_match == 1 and vis_tag[gt_hois.index(max_gt_hoi)] == 0:
                    self.fp[triplet].append(0)
                    self.tp[triplet].append(1)
                    vis_tag[gt_hois.index(max_gt_hoi)] =1
                else:
                    self.fp[triplet].append(1)
                    self.tp[triplet].append(0)
                self.score[triplet].append(pred_hoi['score'])

    def compute_map(self):
        ap = defaultdict(lambda: 0)
        # rare_ap = defaultdict(lambda: 0)
        # non_rare_ap = defaultdict(lambda: 0)
        max_recall = defaultdict(lambda: 0)
        for triplet in self.gt_triplets:
            sum_gts = self.sum_gts[triplet]
            if sum_gts == 0:
                continue

            tp = np.array((self.tp[triplet]))
            fp = np.array((self.fp[triplet]))
            if len(tp) == 0:
                ap[triplet] = 0
                max_recall[triplet] = 0
                
                continue

            score = np.array(self.score[triplet])
            sort_inds = np.argsort(-score)
            fp = fp[sort_inds]
            tp = tp[sort_inds]
            fp = np.cumsum(fp)
            tp = np.cumsum(tp)
            rec = tp / sum_gts
            prec = tp / (fp + tp)
            ap[triplet] = self.voc_ap(rec, prec)
            max_recall[triplet] = np.amax(rec)
         
        m_ap = np.mean(list(ap.values()))

        m_max_recall = np.mean(list(max_recall.values()))

        print('--------------------')
        print('mAP: {}  mean max recall: {}'.format(m_ap, m_max_recall))
        print('--------------------')

        return {'mAP': m_ap, 'mean max recall': m_max_recall}

class HICOEvaluator():
    def __init__(self, preds, gts, rare_triplets, non_rare_triplets, correct_mat, args):
        self.overlap_iou = 0.5
        self.max_hois = 100

        self.use_nms_filter = args.use_nms_filter
        self.thres_nms = args.thres_nms
        self.nms_alpha = args.nms_alpha
        self.nms_beta = args.nms_beta

        self.rare_triplets = rare_triplets
        self.non_rare_triplets = non_rare_triplets

        self.fp = defaultdict(list)
        self.tp = defaultdict(list)
        self.score = defaultdict(list)
        self.sum_gts = defaultdict(lambda: 0)
        self.gt_triplets = []

        # when evaluating mask map, exclude those no interaction images 
        with open(args.exclude_filenames_path, 'rb') as f:
            self.exclude_filenames = pickle.load(f)

        self.preds = []
        for index, img_preds in enumerate(preds):
            img_preds = {k: v.to('cpu').numpy() if k != 'masks' else v for k, v in img_preds.items()}
            
            masks = [{'mask': mask, 'category_id': label} for mask, label in zip(img_preds['masks'], img_preds['labels'])]
            hoi_scores = img_preds['verb_scores']
            verb_labels = np.tile(np.arange(hoi_scores.shape[1]), (hoi_scores.shape[0], 1))
            subject_ids = np.tile(img_preds['sub_ids'], (hoi_scores.shape[1], 1)).T
            object_ids = np.tile(img_preds['obj_ids'], (hoi_scores.shape[1], 1)).T

            hoi_scores = hoi_scores.ravel()
            verb_labels = verb_labels.ravel()
            subject_ids = subject_ids.ravel()
            object_ids = object_ids.ravel()

            if len(subject_ids) > 0:
                object_labels = np.array([masks[object_id]['category_id'] for object_id in object_ids])
                corr_masks = correct_mat[verb_labels, object_labels]
                hoi_scores *= corr_masks

                hois = [{'subject_id': subject_id, 'object_id': object_id, 'category_id': category_id, 'score': score} for
                        subject_id, object_id, category_id, score in zip(subject_ids, object_ids, verb_labels, hoi_scores)]
                hois.sort(key=lambda k: (k.get('score', 0)), reverse=True)
                hois = hois[:self.max_hois]
            else:
                hois = []

            filename = gts[index]['filename']
            if filename in self.exclude_filenames:
                continue
            else:
                self.preds.append({
                    'filename':filename,
                    'predictions': masks,
                    'hoi_prediction': hois
                })


        if self.use_nms_filter:
            self.preds = self.triplet_nms_filter(self.preds)


        self.gts = []
        for i, img_gts in enumerate(gts):
            filename = img_gts['filename']
            if filename in self.exclude_filenames:
                continue
            img_gts = {k: v.to('cpu').numpy() if k != 'id' and k != 'filename' and k != 'masks' else v for k, v in img_gts.items()}
            self.gts.append({
                'filename':filename,
                'annotations': [{'mask': mask, 'category_id': label} for mask, label in zip(img_gts['masks'], img_gts['labels'])],
                'hoi_annotation': [{'subject_id': hoi[0], 'object_id': hoi[1], 'category_id': hoi[2]} for hoi in img_gts['hois']]
            })
            for hoi in self.gts[-1]['hoi_annotation']:
                triplet = (self.gts[-1]['annotations'][hoi['subject_id']]['category_id'],
                           self.gts[-1]['annotations'][hoi['object_id']]['category_id'],
                           hoi['category_id'])

                if triplet not in self.gt_triplets:
                    self.gt_triplets.append(triplet)

                self.sum_gts[triplet] += 1

        print('init finish')

    def evaluate(self):
        for img_preds, img_gts in zip(self.preds, self.gts):
            if img_gts['filename'] in self.exclude_filenames:
                continue
            pred_masks = img_preds['predictions']
            gt_masks = img_gts['annotations']
            pred_hois = img_preds['hoi_prediction']
            gt_hois = img_gts['hoi_annotation']
            if len(gt_masks) != 0:
                mask_pairs, mask_overlaps = self.compute_iou_mat(gt_masks, pred_masks)
                self.compute_fptp(pred_hois, gt_hois, mask_pairs, pred_masks, mask_overlaps)
            else:
                for pred_hoi in pred_hois:
                    triplet = [pred_masks[pred_hoi['subject_id']]['category_id'],
                               pred_masks[pred_hoi['object_id']]['category_id'], pred_hoi['category_id']]
                    if triplet not in self.gt_triplets:
                        continue
                    self.tp[triplet].append(0)
                    self.fp[triplet].append(1)
                    self.score[triplet].append(pred_hoi['score'])
        map = self.compute_map()
        return map

    def compute_map(self):
        ap = defaultdict(lambda: 0)
        rare_ap = defaultdict(lambda: 0)
        non_rare_ap = defaultdict(lambda: 0)
        max_recall = defaultdict(lambda: 0)
        for triplet in self.gt_triplets:
            sum_gts = self.sum_gts[triplet]
            if sum_gts == 0:
                continue

            tp = np.array((self.tp[triplet]))
            fp = np.array((self.fp[triplet]))
            if len(tp) == 0:
                ap[triplet] = 0
                max_recall[triplet] = 0
                if triplet in self.rare_triplets:
                    rare_ap[triplet] = 0
                elif triplet in self.non_rare_triplets:
                    non_rare_ap[triplet] = 0
                else:
                    print('Warning: triplet {} is neither in rare triplets nor in non-rare triplets'.format(triplet))
                continue

            score = np.array(self.score[triplet])
            sort_inds = np.argsort(-score)
            fp = fp[sort_inds]
            tp = tp[sort_inds]
            fp = np.cumsum(fp)
            tp = np.cumsum(tp)
            rec = tp / sum_gts
            prec = tp / (fp + tp)
            ap[triplet] = self.voc_ap(rec, prec)
            max_recall[triplet] = np.amax(rec)
            if triplet in self.rare_triplets:
                rare_ap[triplet] = ap[triplet]
            elif triplet in self.non_rare_triplets:
                non_rare_ap[triplet] = ap[triplet]
            else:
                print('Warning: triplet {} is neither in rare triplets nor in non-rare triplets'.format(triplet))
        m_ap = np.mean(list(ap.values()))
        m_ap_rare = np.mean(list(rare_ap.values()))
        m_ap_non_rare = np.mean(list(non_rare_ap.values()))
        m_max_recall = np.mean(list(max_recall.values()))

        print('--------------------')
        print('mAP: {} mAP rare: {}  mAP non-rare: {}  mean max recall: {}'.format(m_ap, m_ap_rare, m_ap_non_rare, m_max_recall))
        print('--------------------')

        return {'mAP': m_ap, 'mAP rare': m_ap_rare, 'mAP non-rare': m_ap_non_rare, 'mean max recall': m_max_recall}
        # return {'mAP': m_ap}

    def voc_ap(self, rec, prec):
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
        return ap

    def compute_fptp(self, pred_hois, gt_hois, match_pairs, pred_masks, mask_overlaps):
        pos_pred_ids = match_pairs.keys()
        vis_tag = np.zeros(len(gt_hois))
        pred_hois.sort(key=lambda k: (k.get('score', 0)), reverse=True)
        if len(pred_hois) != 0:
            for pred_hoi in pred_hois:
                is_match = 0
                if len(match_pairs) != 0 and pred_hoi['subject_id'] in pos_pred_ids and pred_hoi['object_id'] in pos_pred_ids:
                    pred_sub_ids = match_pairs[pred_hoi['subject_id']]
                    pred_obj_ids = match_pairs[pred_hoi['object_id']]
                    pred_sub_overlaps = mask_overlaps[pred_hoi['subject_id']]
                    pred_obj_overlaps = mask_overlaps[pred_hoi['object_id']]
                    pred_category_id = pred_hoi['category_id']
                    max_overlap = 0
                    max_gt_hoi = 0
                    for gt_hoi in gt_hois:
                        if gt_hoi['subject_id'] in pred_sub_ids and gt_hoi['object_id'] in pred_obj_ids \
                           and pred_category_id == gt_hoi['category_id']:
                            is_match = 1
                            min_overlap_gt = min(pred_sub_overlaps[pred_sub_ids.index(gt_hoi['subject_id'])],
                                                 pred_obj_overlaps[pred_obj_ids.index(gt_hoi['object_id'])])
                            if min_overlap_gt > max_overlap:
                                max_overlap = min_overlap_gt
                                max_gt_hoi = gt_hoi
                triplet = (pred_masks[pred_hoi['subject_id']]['category_id'], pred_masks[pred_hoi['object_id']]['category_id'],
                           pred_hoi['category_id'])
                if triplet not in self.gt_triplets:
                    continue
                if is_match == 1 and vis_tag[gt_hois.index(max_gt_hoi)] == 0:
                    self.fp[triplet].append(0)
                    self.tp[triplet].append(1)
                    vis_tag[gt_hois.index(max_gt_hoi)] =1
                else:
                    self.fp[triplet].append(1)
                    self.tp[triplet].append(0)
                self.score[triplet].append(pred_hoi['score'])

    def compute_iou_mat(self, mask_list1, mask_list2):
        iou_mat = np.zeros((len(mask_list1), len(mask_list2)))
        if len(mask_list1) == 0 or len(mask_list2) == 0:
            return {}
        for i, mask1 in enumerate(mask_list1):
            for j, mask2 in enumerate(mask_list2):
                iou_i = self.compute_IOU(mask1, mask2)
                iou_mat[i, j] = iou_i

        iou_mat_ov=iou_mat.copy()
        iou_mat[iou_mat>=self.overlap_iou] = 1
        iou_mat[iou_mat<self.overlap_iou] = 0

        match_pairs = np.nonzero(iou_mat)
        match_pairs_dict = {}
        match_pair_overlaps = {}
        if iou_mat.max() > 0:
            for i, pred_id in enumerate(match_pairs[1]):
                if pred_id not in match_pairs_dict.keys():
                    match_pairs_dict[pred_id] = []
                    match_pair_overlaps[pred_id]=[]
                match_pairs_dict[pred_id].append(match_pairs[0][i])
                match_pair_overlaps[pred_id].append(iou_mat_ov[match_pairs[0][i],pred_id])
        return match_pairs_dict, match_pair_overlaps

    def compute_IOU(self, mask1, mask2):
        if isinstance(mask1['category_id'], str):
            mask1['category_id'] = int(mask1['category_id'].replace('\n', ''))
        if isinstance(mask2['category_id'], str):
            mask2['category_id'] = int(mask2['category_id'].replace('\n', ''))
        if mask1['category_id'] == mask2['category_id']:
            m1 = mask1['mask']
            m2 = mask2['mask']
            return compute_iou(m1, m2)
        else:
            return 0

    def triplet_nms_filter(self, preds):
        preds_filtered = []
        for img_preds in preds:
            pred_masks = img_preds['predictions']
            pred_hois = img_preds['hoi_prediction']
            all_triplets = {}
            for index, pred_hoi in enumerate(pred_hois):
                triplet = str(pred_masks[pred_hoi['subject_id']]['category_id']) + '_' + \
                          str(pred_masks[pred_hoi['object_id']]['category_id']) + '_' + str(pred_hoi['category_id'])

                if triplet not in all_triplets:
                    all_triplets[triplet] = {'subs':[], 'objs':[], 'scores':[], 'indexes':[]}
                all_triplets[triplet]['subs'].append(pred_masks[pred_hoi['subject_id']]['mask'])
                all_triplets[triplet]['objs'].append(pred_masks[pred_hoi['object_id']]['mask'])
                all_triplets[triplet]['scores'].append(pred_hoi['score'])
                all_triplets[triplet]['indexes'].append(index)

            all_keep_inds = []
            for triplet, values in all_triplets.items():
                subs, objs, scores = values['subs'], values['objs'], values['scores']
                keep_inds = self.pairwise_nms(np.array(subs), np.array(objs), np.array(scores))

                keep_inds = list(np.array(values['indexes'])[keep_inds])
                all_keep_inds.extend(keep_inds)

            preds_filtered.append({
                'filename': img_preds['filename'],
                'predictions': pred_masks,
                'hoi_prediction': list(np.array(img_preds['hoi_prediction'])[all_keep_inds])
                })

        return preds_filtered

    def pairwise_nms(self, subs, objs, scores):
        
        order = scores.argsort()[::-1]
        selected_indices = []

        for i in order:
            keep_pair = True
            for selected_pair in selected_indices:
                sub_iou = compute_iou(subs[i], subs[selected_pair])
                obj_iou = compute_iou(objs[i], objs[selected_pair])
                if sub_iou > 0.5 and obj_iou > 0.5:
                    keep_pair = False
                    break
            if keep_pair:
                selected_indices.append(i)
        return selected_indices




