import torch.utils.data
import torchvision

from .flexible_hico import build as build_hico
from .psg import build as build_psg
from .vcoco import build as build_vcoco
from .vrd import build as build_vrd

def build_dataset(image_set, args):
    if args.dataset_file == 'hico':
        return build_hico(image_set, args)
   
    if args.dataset_file == 'psg':
        return build_psg(image_set, args)
    
    if args.dataset_file == 'vcoco':
        return build_vcoco(image_set, args)
    if args.dataset_file == 'vrd':
        return build_vrd(image_set, args)

    raise ValueError(f'dataset {args.dataset_file} not supported')
