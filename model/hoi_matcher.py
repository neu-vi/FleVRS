import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou

def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
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
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


batch_dice_loss_jit = torch.jit.script(
    batch_dice_loss
)  # type: torch.jit.ScriptModule


def batch_sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor):
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
    hw = inputs.shape[1]

    pos = F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )

    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum(
        "nc,mc->nm", neg, (1 - targets)
    )

    return loss / hw


batch_sigmoid_ce_loss_jit = torch.jit.script(
    batch_sigmoid_ce_loss
)  # type: torch.jit.ScriptModule



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



class HungarianMatcherHOI(nn.Module):
    def __init__(self, cost_obj_class: float = 1, cost_verb_class: float = 1, cost_hoi_class: float =1 , cost_mask: float = 1,
                 cost_dice: float = 1, cost_reg: float=1, cost_giou: float=1, num_points: int = 0, task_switch={'mask': False}):
        super().__init__()
        self.cost_obj_class = cost_obj_class
        self.cost_verb_class = cost_verb_class
        self.cost_hoi_class = cost_hoi_class

        if task_switch['mask']:
            self.cost_mask = cost_mask
            self.cost_dice = cost_dice

        if task_switch['box']:
            self.cost_reg = cost_reg
            self.cost_giou = cost_giou

        self.num_points = num_points
        self.task_switch = task_switch
        if task_switch['psg'] or task_switch['vrd']:
            self.cost_sub_class = cost_obj_class

        assert cost_obj_class != 0 or cost_verb_class != 0 or cost_mask != 0 or cost_dice != 0, 'all costs cant be 0'

    @torch.no_grad()
    def forward(self, outputs, targets, mode='default', extra={}):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_masks": Tensor of dim [batch_size, num_queries, H_pred, W_pred] with the predicted masks

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "masks": Tensor of dim [num_target_boxes, H_gt, W_gt] containing the target masks

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        if mode == 'default':
            return self.memory_efficient_forward(outputs, targets)
        elif mode == 'grounding':
            return self.grounding_forward(outputs, targets, extra)
        elif mode=='psg':
            return self.psg_forward(outputs, targets)
        elif mode=='vrd':
            return self.vrd_forward(outputs, targets)
    
    @torch.no_grad()
    def grounding_forward(self, outputs, targets, extra):
        """More memory-friendly matching"""
        bs, num_queries = outputs["pred_gsub_masks"].shape[:2]
        
        if bs == 0 or len(targets) == 0:
            return None

        indices = []
        # Iterate through batch size
        for b in range(bs):
            out_prob = outputs["pred_logits"][b]
            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            cost_class = -out_prob.softmax(dim=0)

            out_sub_mask = outputs["pred_gsub_masks"][b]  # [num_queries, H_pred, W_pred]
            # gt masks are already padded when preparing target
            tgt_sub_mask = targets[b]["grounding_sub_mask"].to(out_sub_mask)

            out_sub_mask = out_sub_mask[:, None]
            tgt_sub_mask = tgt_sub_mask[:, None]

            out_obj_mask = outputs["pred_gobj_masks"][b]  # [num_queries, H_pred, W_pred]
            # gt masks are already padded when preparing target
            tgt_obj_mask = targets[b]["grounding_obj_mask"].to(out_obj_mask)

            out_obj_mask = out_obj_mask[:, None]
            tgt_obj_mask = tgt_obj_mask[:, None]
            
            # all masks share the same set of points for efficient matching!
            point_coords = torch.rand(1, self.num_points, 2, device=out_sub_mask.device, dtype=tgt_sub_mask.dtype)
            
            tgt_sub_mask = point_sample(tgt_sub_mask, point_coords.repeat(tgt_sub_mask.shape[0], 1, 1), align_corners=False).squeeze(1)
            tgt_obj_mask = point_sample(tgt_obj_mask, point_coords.repeat(tgt_obj_mask.shape[0], 1, 1), align_corners=False).squeeze(1)
            out_sub_mask = point_sample(out_sub_mask, point_coords.repeat(out_sub_mask.shape[0], 1, 1), align_corners=False).squeeze(1)
            out_obj_mask = point_sample(out_obj_mask, point_coords.repeat(out_obj_mask.shape[0], 1, 1), align_corners=False).squeeze(1)
            

            with autocast(enabled=False):
                out_sub_mask = out_sub_mask.float()
                tgt_sub_mask = tgt_sub_mask.float()

                out_obj_mask = out_obj_mask.float()
                tgt_obj_mask = tgt_obj_mask.float()

                # Compute the focal loss between masks
                cost_sub_mask = batch_sigmoid_ce_loss_jit(out_sub_mask, tgt_sub_mask)
                cost_obj_mask = batch_sigmoid_ce_loss_jit(out_obj_mask, tgt_obj_mask)
                # Compute the dice loss betwen masks
                cost_sub_dice = batch_dice_loss_jit(out_sub_mask, tgt_sub_mask)
                cost_obj_dice = batch_dice_loss_jit(out_obj_mask, tgt_obj_mask) 

                if cost_sub_mask.shape[1] == 0:
                    cost_mask = cost_sub_mask
                else:
                    cost_mask = torch.stack((cost_sub_mask, cost_obj_mask)).max(dim=0)[0]

                if cost_sub_dice.shape[1] == 0:
                    cost_dice = cost_sub_dice
                else:
                    cost_dice = torch.stack((cost_sub_dice, cost_obj_dice)).max(dim=0)[0]
                
            # Final cost matrix
            C = (
                self.cost_mask * cost_mask
                + self.cost_obj_class * cost_class
                + self.cost_dice * cost_dice
            )
            C = C.reshape(num_queries, -1).cpu()
            indices.append(linear_sum_assignment(C))

        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]
    @torch.no_grad()
    def vrd_forward(self, outputs, targets):
        bs, num_queries = outputs['pred_obj_logits'].shape[:2] 

        indices = []

        for b in range(bs):
            
            out_sub_prob = outputs['pred_sub_logits'][b].softmax(-1) # 100 x 81
            out_obj_prob = outputs['pred_obj_logits'][b].softmax(-1) # 100 x 81
            
            out_sub_mask = outputs['pred_sub_masks'][b] # 100 x 160 x 160
            out_obj_mask = outputs['pred_obj_masks'][b]
            
            tgt_sub_labels = targets[b]['sub_labels']
            tgt_obj_labels = targets[b]['obj_labels']
            
            tgt_sub_mask = targets[b]['sub_masks'].to(out_sub_mask)
            tgt_obj_mask = targets[b]['obj_masks'].to(out_obj_mask)

        
            cost_obj_class = -out_obj_prob[:, tgt_obj_labels]
            if self.task_switch['vrd']:
                cost_sub_class = -out_sub_prob[:, tgt_sub_labels]

            
            # out_verb_prob = outputs['pred_verb_logits'][b].softmax(-1) # 100 x 117
            # tgt_verb_labels = targets[b]['verb_labels']

            # cost_verb_class = -out_verb_prob[:, tgt_verb_labels]

            out_verb_prob = outputs['pred_verb_logits'][b].sigmoid() # 100 x 117
            tgt_verb_labels = targets[b]['verb_labels']
            tgt_verb_labels_permute = tgt_verb_labels.permute(1, 0)
            cost_verb_class = -(out_verb_prob.matmul(tgt_verb_labels_permute) / \
                        (tgt_verb_labels_permute.sum(dim=0, keepdim=True) + 1e-4) + \
                        (1 - out_verb_prob).matmul(1 - tgt_verb_labels_permute) / \
                        ((1 - tgt_verb_labels_permute).sum(dim=0, keepdim=True) + 1e-4)) / 2
            # cost_hoi_class = self.cost_verb_class * cost_verb_class
            
            out_sub_mask = out_sub_mask[:, None]
            out_obj_mask = out_obj_mask[:, None]
            tgt_sub_mask = tgt_sub_mask[:, None]
            tgt_obj_mask = tgt_obj_mask[:, None]
            # all masks share the same set of points for efficient matching!
            point_coords = torch.rand(1, self.num_points, 2, device=out_sub_mask.device, dtype=tgt_sub_mask.dtype)
            
            tgt_sub_mask = point_sample(tgt_sub_mask, point_coords.repeat(tgt_sub_mask.shape[0], 1, 1), align_corners=False).squeeze(1)
            tgt_obj_mask = point_sample(tgt_obj_mask, point_coords.repeat(tgt_obj_mask.shape[0], 1, 1), align_corners=False).squeeze(1)
            out_sub_mask = point_sample(out_sub_mask, point_coords.repeat(out_sub_mask.shape[0], 1, 1), align_corners=False).squeeze(1)
            out_obj_mask = point_sample(out_obj_mask, point_coords.repeat(out_obj_mask.shape[0], 1, 1), align_corners=False).squeeze(1)

            with autocast(enabled=False):
                out_sub_mask = out_sub_mask.float()
                out_obj_mask = out_obj_mask.float()
                tgt_sub_mask = tgt_sub_mask.float()
                tgt_obj_mask = tgt_obj_mask.float()
                # Compute the focal loss between masks
                cost_sub_mask = batch_sigmoid_ce_loss_jit(out_sub_mask, tgt_sub_mask)
                cost_obj_mask = batch_sigmoid_ce_loss_jit(out_obj_mask, tgt_obj_mask)

                if cost_sub_mask.shape[1] == 0:
                    cost_mask = cost_sub_mask
                else:
                    cost_mask = torch.stack((cost_sub_mask, cost_obj_mask)).max(dim=0)[0]

                # Compute the dice loss betwen masks
                cost_sub_dice = batch_dice_loss_jit(out_sub_mask, tgt_sub_mask)
                cost_obj_dice = batch_dice_loss_jit(out_obj_mask, tgt_obj_mask) 
                
                if cost_sub_dice.shape[1] == 0:
                    cost_dice = cost_sub_dice
                else:
                    cost_dice = torch.stack((cost_sub_dice, cost_obj_dice)).max(dim=0)[0]

            # Final cost matrix
        
            C = (
                    self.cost_sub_class * cost_sub_class
                    + self.cost_obj_class * cost_obj_class
                    + self.cost_verb_class * cost_verb_class
                    + self.cost_mask * cost_mask
                    + self.cost_dice * cost_dice
                )
            C = C.reshape(num_queries, -1).cpu()
            indices.append(linear_sum_assignment(C))
        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]
    @torch.no_grad()
    def psg_forward(self, outputs, targets):
        bs, num_queries = outputs['pred_obj_logits'].shape[:2] 

        indices = []

        for b in range(bs):
            
            out_sub_prob = outputs['pred_sub_logits'][b].softmax(-1) # 100 x 81
            out_obj_prob = outputs['pred_obj_logits'][b].softmax(-1) # 100 x 81
            
            out_sub_mask = outputs['pred_sub_masks'][b] # 100 x 160 x 160
            out_obj_mask = outputs['pred_obj_masks'][b]
            
            tgt_sub_labels = targets[b]['sub_labels']
            tgt_obj_labels = targets[b]['obj_labels']
            
            tgt_sub_mask = targets[b]['sub_masks'].to(out_sub_mask)
            tgt_obj_mask = targets[b]['obj_masks'].to(out_obj_mask)

        
            cost_obj_class = -out_obj_prob[:, tgt_obj_labels]
            if self.task_switch['psg'] or self.task_switch['vrd']:
                cost_sub_class = -out_sub_prob[:, tgt_sub_labels]

            
            out_verb_prob = outputs['pred_verb_logits'][b].softmax(-1) # 100 x 117
            tgt_verb_labels = targets[b]['verb_labels']

            cost_verb_class = -out_verb_prob[:, tgt_verb_labels]
            
            out_sub_mask = out_sub_mask[:, None]
            out_obj_mask = out_obj_mask[:, None]
            tgt_sub_mask = tgt_sub_mask[:, None]
            tgt_obj_mask = tgt_obj_mask[:, None]
            # all masks share the same set of points for efficient matching!
            point_coords = torch.rand(1, self.num_points, 2, device=out_sub_mask.device, dtype=tgt_sub_mask.dtype)
            
            tgt_sub_mask = point_sample(tgt_sub_mask, point_coords.repeat(tgt_sub_mask.shape[0], 1, 1), align_corners=False).squeeze(1)
            tgt_obj_mask = point_sample(tgt_obj_mask, point_coords.repeat(tgt_obj_mask.shape[0], 1, 1), align_corners=False).squeeze(1)
            out_sub_mask = point_sample(out_sub_mask, point_coords.repeat(out_sub_mask.shape[0], 1, 1), align_corners=False).squeeze(1)
            out_obj_mask = point_sample(out_obj_mask, point_coords.repeat(out_obj_mask.shape[0], 1, 1), align_corners=False).squeeze(1)

            with autocast(enabled=False):
                out_sub_mask = out_sub_mask.float()
                out_obj_mask = out_obj_mask.float()
                tgt_sub_mask = tgt_sub_mask.float()
                tgt_obj_mask = tgt_obj_mask.float()
                # Compute the focal loss between masks
                cost_sub_mask = batch_sigmoid_ce_loss_jit(out_sub_mask, tgt_sub_mask)
                cost_obj_mask = batch_sigmoid_ce_loss_jit(out_obj_mask, tgt_obj_mask)

                if cost_sub_mask.shape[1] == 0:
                    cost_mask = cost_sub_mask
                else:
                    cost_mask = torch.stack((cost_sub_mask, cost_obj_mask)).max(dim=0)[0]

                # Compute the dice loss betwen masks
                cost_sub_dice = batch_dice_loss(out_sub_mask, tgt_sub_mask)
                cost_obj_dice = batch_dice_loss(out_obj_mask, tgt_obj_mask) 

                # cost_sub_dice = batch_dice_loss_jit(out_sub_mask, tgt_sub_mask)
                # cost_obj_dice = batch_dice_loss_jit(out_obj_mask, tgt_obj_mask) 
                
                if cost_sub_dice.shape[1] == 0:
                    cost_dice = cost_sub_dice
                else:
                    cost_dice = torch.stack((cost_sub_dice, cost_obj_dice)).max(dim=0)[0]

            # Final cost matrix
        
            C = (
                    self.cost_sub_class * cost_sub_class
                    + self.cost_obj_class * cost_obj_class
                    + self.cost_verb_class * cost_verb_class
                    + self.cost_mask * cost_mask
                    + self.cost_dice * cost_dice
                )
            C = C.reshape(num_queries, -1).cpu()
            indices.append(linear_sum_assignment(C))
        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]
    @torch.no_grad()
    def memory_efficient_forward(self, outputs, targets):
        bs, num_queries = outputs['pred_obj_logits'].shape[:2] 

        indices = []

        for b in range(bs):
            if self.task_switch['psg'] and 'pred_sub_logits' in outputs:
                out_sub_prob = outputs['pred_sub_logits'][b].softmax(-1) # 100 x 81
            out_obj_prob = outputs['pred_obj_logits'][b].softmax(-1) # 100 x 81
            
            if self.task_switch['mask']:
                out_sub_mask = outputs['pred_sub_masks'][b] # 100 x 160 x 160
                out_obj_mask = outputs['pred_obj_masks'][b]
                tgt_sub_mask = targets[b]['sub_masks'].to(out_sub_mask)
                tgt_obj_mask = targets[b]['obj_masks'].to(out_obj_mask)

                out_sub_mask = out_sub_mask[:, None]
                out_obj_mask = out_obj_mask[:, None]
                tgt_sub_mask = tgt_sub_mask[:, None]
                tgt_obj_mask = tgt_obj_mask[:, None]
                # all masks share the same set of points for efficient matching!
                point_coords = torch.rand(1, self.num_points, 2, device=out_sub_mask.device, dtype=tgt_sub_mask.dtype)
                
                tgt_sub_mask = point_sample(tgt_sub_mask, point_coords.repeat(tgt_sub_mask.shape[0], 1, 1), align_corners=False).squeeze(1)
                tgt_obj_mask = point_sample(tgt_obj_mask, point_coords.repeat(tgt_obj_mask.shape[0], 1, 1), align_corners=False).squeeze(1)
                out_sub_mask = point_sample(out_sub_mask, point_coords.repeat(out_sub_mask.shape[0], 1, 1), align_corners=False).squeeze(1)
                out_obj_mask = point_sample(out_obj_mask, point_coords.repeat(out_obj_mask.shape[0], 1, 1), align_corners=False).squeeze(1)

                with autocast(enabled=False):
                    out_sub_mask = out_sub_mask.float()
                    out_obj_mask = out_obj_mask.float()
                    tgt_sub_mask = tgt_sub_mask.float()
                    tgt_obj_mask = tgt_obj_mask.float()
                    # Compute the focal loss between masks
                    cost_sub_mask = batch_sigmoid_ce_loss_jit(out_sub_mask, tgt_sub_mask)
                    cost_obj_mask = batch_sigmoid_ce_loss_jit(out_obj_mask, tgt_obj_mask)

                    if cost_sub_mask.shape[1] == 0:
                        cost_mask = cost_sub_mask
                    else:
                        cost_mask = torch.stack((cost_sub_mask, cost_obj_mask)).max(dim=0)[0]

                    # Compute the dice loss betwen masks
                    # cost_sub_dice = batch_dice_loss_jit(out_sub_mask, tgt_sub_mask)
                    # cost_obj_dice = batch_dice_loss_jit(out_obj_mask, tgt_obj_mask) 
                    cost_sub_dice = batch_dice_loss(out_sub_mask, tgt_sub_mask)
                    cost_obj_dice = batch_dice_loss(out_obj_mask, tgt_obj_mask)
                    
                    if cost_sub_dice.shape[1] == 0:
                        cost_dice = cost_sub_dice
                    else:
                        cost_dice = torch.stack((cost_sub_dice, cost_obj_dice)).max(dim=0)[0]
            
            if self.task_switch['box']:
                out_sub_box = outputs['pred_sub_boxes'][b]
                out_obj_box = outputs['pred_obj_boxes'][b]
                tgt_sub_box = targets[b]['sub_boxes'].to(out_sub_box)
                tgt_obj_box = targets[b]['obj_boxes'].to(out_obj_box)

                cost_sub_bbox = torch.cdist(out_sub_box, tgt_sub_box, p=1)
                cost_obj_bbox = torch.cdist(out_obj_box, tgt_obj_box, p=1) * (tgt_obj_box != 0).any(dim=1).unsqueeze(0)

                if cost_sub_bbox.shape[1] == 0:
                    cost_bbox = cost_sub_bbox
                else:
                    cost_bbox = torch.stack((cost_sub_bbox, cost_obj_bbox)).max(dim=0)[0]

                cost_sub_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_sub_box), box_cxcywh_to_xyxy(tgt_sub_box))
                cost_obj_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_obj_box), box_cxcywh_to_xyxy(tgt_obj_box)) + \
                        cost_sub_giou * (tgt_obj_box == 0).all(dim=1).unsqueeze(0)
                if cost_sub_giou.shape[1] == 0:
                    cost_giou = cost_sub_giou
                else:
                    cost_giou = torch.stack((cost_sub_giou, cost_obj_giou)).max(dim=0)[0]

            if self.task_switch['psg'] and 'pred_sub_logits' in outputs:
                tgt_sub_labels = targets[b]['sub_labels']
            tgt_obj_labels = targets[b]['obj_labels']           

            cost_obj_class = -out_obj_prob[:, tgt_obj_labels]
            if self.task_switch['psg'] and 'pred_sub_logits' in outputs:
                cost_sub_class = -out_sub_prob[:, tgt_sub_labels]

            if 'pred_hoi_logits' in outputs:
                out_hoi_prob = outputs['pred_hoi_logits'][b].sigmoid()
                tgt_hoi_labels = targets[b]['hoi_labels']
                tgt_hoi_labels_permute = tgt_hoi_labels.permute(1, 0)
                cost_hoi_class = -(out_hoi_prob.matmul(tgt_hoi_labels_permute) / \
                            (tgt_hoi_labels_permute.sum(dim=0, keepdim=True) + 1e-4) + \
                            (1 - out_hoi_prob).matmul(1 - tgt_hoi_labels_permute) / \
                            ((1 - tgt_hoi_labels_permute).sum(dim=0, keepdim=True) + 1e-4)) / 2
                cost_hoi_class = self.cost_hoi_class * cost_hoi_class
            else:
                out_verb_prob = outputs['pred_verb_logits'][b].sigmoid() # 100 x 117
                tgt_verb_labels = targets[b]['verb_labels']
                # import ipdb; ipdb.set_trace()
                tgt_verb_labels_permute = tgt_verb_labels.permute(1, 0)
                cost_verb_class = -(out_verb_prob.matmul(tgt_verb_labels_permute) / \
                            (tgt_verb_labels_permute.sum(dim=0, keepdim=True) + 1e-4) + \
                            (1 - out_verb_prob).matmul(1 - tgt_verb_labels_permute) / \
                            ((1 - tgt_verb_labels_permute).sum(dim=0, keepdim=True) + 1e-4)) / 2
                cost_hoi_class = self.cost_verb_class * cost_verb_class

            # Final cost matrix
            if self.task_switch['psg'] and 'pred_sub_logits' in outputs:
                C = (
                    self.cost_sub_class * cost_sub_class
                    + self.cost_obj_class * cost_obj_class
                    + self.cost_verb_class * cost_verb_class
                    + self.cost_mask * cost_mask
                    + self.cost_dice * cost_dice
                )
            elif self.task_switch['hoi']:
                # if 'pred_hoi_logits' in outputs:
                #     C = (
                #     self.cost_obj_class * cost_obj_class
                #     + self.cost_verb_class * cost_verb_class
                #     + self.cost_mask * cost_mask
                #     + self.cost_dice * cost_dice
                #     + self.cost_hoi_class * cost_hoi_class
                # )
                # else: 
                if self.task_switch['mask'] and self.task_switch['box']:
                    C = (
                        self.cost_obj_class * cost_obj_class
                        + self.cost_hoi_class * cost_hoi_class
                        + self.cost_mask * cost_mask
                        + self.cost_dice * cost_dice
                        + self.cost_reg * cost_bbox
                        + self.cost_giou * cost_giou
                    )
                elif self.task_switch['mask']:
                    C = (
                        self.cost_obj_class * cost_obj_class
                        + self.cost_hoi_class * cost_hoi_class
                        + self.cost_mask * cost_mask
                        + self.cost_dice * cost_dice
                    )
                elif self.task_switch['box']:
                    C = (
                        self.cost_obj_class * cost_obj_class
                        + self.cost_hoi_class * cost_hoi_class
                        + self.cost_reg * cost_bbox
                        + self.cost_giou * cost_giou
                    )
                else:
                    raise NotImplementedError
            C = C.reshape(num_queries, -1).cpu()
            indices.append(linear_sum_assignment(C))
        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]


class HungarianMatcherM2F(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_mask: float = 1, cost_dice: float = 1, num_points: int = 0):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_mask: This is the relative weight of the focal loss of the binary mask in the matching cost
            cost_dice: This is the relative weight of the dice loss of the binary mask in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice

        assert cost_class != 0 or cost_mask != 0 or cost_dice != 0, "all costs cant be 0"

        self.num_points = num_points

    @torch.no_grad()
    def memory_efficient_forward(self, outputs, targets):
        """More memory-friendly matching"""
        bs, num_queries = outputs["pred_logits"].shape[:2]

        indices = []

        # Iterate through batch size
        for b in range(bs):

            out_prob = outputs["pred_logits"][b].softmax(-1)  # [num_queries, num_classes]
            tgt_ids = targets[b]["labels"]

            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            cost_class = -out_prob[:, tgt_ids]

            out_mask = outputs["pred_masks"][b]  # [num_queries, H_pred, W_pred]
            # gt masks are already padded when preparing target
            tgt_mask = targets[b]["masks"].to(out_mask)

            out_mask = out_mask[:, None]
            tgt_mask = tgt_mask[:, None]
            # all masks share the same set of points for efficient matching!
            point_coords = torch.rand(1, self.num_points, 2, device=out_mask.device)
            # get gt labels
            tgt_mask = point_sample(
                tgt_mask,
                point_coords.repeat(tgt_mask.shape[0], 1, 1),
                align_corners=False,
            ).squeeze(1)

            out_mask = point_sample(
                out_mask,
                point_coords.repeat(out_mask.shape[0], 1, 1),
                align_corners=False,
            ).squeeze(1)

            with autocast(enabled=False):
                out_mask = out_mask.float()
                tgt_mask = tgt_mask.float()
                # Compute the focal loss between masks
                cost_mask = batch_sigmoid_ce_loss_jit(out_mask, tgt_mask)

                # Compute the dice loss betwen masks
                cost_dice = batch_dice_loss_jit(out_mask, tgt_mask)
            
            # Final cost matrix
            C = (
                self.cost_mask * cost_mask
                + self.cost_class * cost_class
                + self.cost_dice * cost_dice
            )
            C = C.reshape(num_queries, -1).cpu()

            indices.append(linear_sum_assignment(C))

        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]

    @torch.no_grad()
    def forward(self, outputs, targets):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_masks": Tensor of dim [batch_size, num_queries, H_pred, W_pred] with the predicted masks

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "masks": Tensor of dim [num_target_boxes, H_gt, W_gt] containing the target masks

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        return self.memory_efficient_forward(outputs, targets)

    def __repr__(self, _repr_indent=4):
        head = "Matcher " + self.__class__.__name__
        body = [
            "cost_class: {}".format(self.cost_class),
            "cost_mask: {}".format(self.cost_mask),
            "cost_dice: {}".format(self.cost_dice),
        ]
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)

def build_matcher(args, name=None):
    task_switch = {'mask': False, 'box': False, 'hoi': False, 'psg': False, 'grounding': False, 'vrd': False}
    if args.psg and name!="hico":
        task_switch['psg'] = True
        task_switch['mask'] = True
    if args.vrd:
        task_switch['vrd'] = True
    if args.hoi and name!="psg":
        task_switch['hoi'] = True
        if args.use_mask:
            task_switch['mask'] = True
        if args.use_box:
            task_switch['box'] = True
    if args.flexible_grounding:
        task_switch['grounding'] = True
    
    if args.two_stage:
        return HungarianMatcherM2F(cost_class=args.set_cost_obj_class, cost_mask=args.set_cost_mask, cost_dice=args.set_cost_dice, num_points=args.num_points)
    else:
        return HungarianMatcherHOI(cost_obj_class=args.set_cost_obj_class, cost_verb_class=args.set_cost_verb_class, cost_hoi_class=args.set_cost_hoi_class,
                               cost_mask=args.set_cost_mask, cost_dice=args.set_cost_dice, cost_reg=args.set_cost_reg, cost_giou=args.set_cost_giou,
                               num_points=args.num_points, task_switch=task_switch)


