import os
import random
import numpy as np
import torch
import multiprocessing as mp

n_cores = max(mp.cpu_count()-2, 1)
seed = 42
generator = torch.Generator().manual_seed(seed)

def set_seed(seed=seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


cell_types_lung = {
    'Tumor_cell_LUAD': [
        'Tumor cells LUAD',
        'Tumor cells LUAD EMT',
        'Tumor cells LUAD MSLN',
        'Tumor cells LUAD NE',
        'Tumor cells LUAD mitotic',
    ],
    'Stromal_cell': [
        'stromal dividing',
        'Fibroblast adventitial',
        'Fibroblast alveolar',
        'Fibroblast peribronchial',
        'Smooth muscle cell',
        'Mesothelial',
    ],
    'Pericyte': [
        'Pericyte',
    ],
    'Endothelial_cell': [
        'Endothelial cell arterial',
        'Endothelial cell capillary',
        'Endothelial cell lymphatic',
        'Endothelial cell venous',
    ],
    'Lymphocyte': [
         'B cell',
         'B cell dividing',
         'Plasma cell',
         'Plasma cell dividing',
         'T cell regulatory',
         'T cell CD4',
         'T cell CD4 dividing',
         'T cell CD8 activated',
         'T cell CD8 dividing',
         'T cell CD8 effector memory',
         'T cell CD8 naive',
         'T cell CD8 terminally exhausted',
         'T cell NK-like',
         'NK cell',
         'NK cell dividing',
    ],
#    'Other': [
#        'Tumor cells LUSC',
#        'Tumor cells LUSC mitotic',
#        'Tumor cells NSCLC mixed',
#        'Alveolar cell type 1',
#        'Alveolar cell type 2',
#        'transitional club/AT2',
#        'Ciliated',
#        'Club',
#        'Macrophage',
#        'Macrophage alveolar',
#        'Mast cell',
#        'Monocyte classical',
#        'Monocyte non-classical',
#        'Neutrophils',
#        'DC mature',
#        'cDC1',
#        'cDC2',
#        'pDC',
#        'ROS1+ healthy epithelial',
#        'myeloid dividing',
#    ],
}
