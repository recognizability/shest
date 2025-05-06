import os
import random
import numpy as np
import torch
import multiprocessing as mp
import seaborn as sns

n_cores = max(mp.cpu_count()-2, 1)
seed = 42
generator = torch.Generator().manual_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


cell_types = {
    'lung':{
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
    },
}

class Config:
    def __init__(self, args):
        self.stem_directory = f"{args.platform}/{args.source}/{args.sample}/"
        self.stem_file = f"{args.platform}_{args.source}_{args.sample}"

        self.cell_types = next((cell_type_values for organ, cell_type_values in cell_types.items() if organ in args.sample.lower()), {}) #for the organ
        self.cell_subtypes = sum(self.cell_types.values(), [])

        palette = 'blend:red,orange,green,blue'
        self.palette_type = dict(zip(
            self.cell_types.keys(),
            sns.color_palette(palette, n_colors=len(self.cell_types)).as_hex()
        ))
        self.palette_subtype = dict(zip(
            self.cell_subtypes,
            sns.color_palette(palette, n_colors=len(self.cell_subtypes)).as_hex()
        ))

        self.angles = [0, 90, 180, 270] if args.rotate else [0]
