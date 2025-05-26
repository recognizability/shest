import os
import random
import numpy as np
from sklearn.preprocessing import LabelEncoder
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


cell_types = { #cell subtypes from CZ CELLxGENE Discover
    'lung':{
        'Tumor_cell_LUAD': [
            'Tumor cells LUAD',
            'Tumor cells LUAD EMT',
            'Tumor cells LUAD MSLN',
            'Tumor cells LUAD NE',
            'Tumor cells LUAD mitotic',
            'Alveolar cell type 2',
            'Macrophage alveolar',
        ],
        "Macrophage": [
            "Macrophage",
        ],
        'Endothelial_cell': [
            'Endothelial cell arterial',
            'Endothelial cell capillary',
            'Endothelial cell lymphatic',
            'Endothelial cell venous',
        ],
        'Stromal_cell': [
            'Fibroblast adventitial',
            'Fibroblast alveolar',
            'Fibroblast peribronchial',
            'Smooth muscle cell',
            'Mesothelial',
            'Pericyte',
        ],
        'Plasma_cell': [
             'Plasma cell',
        ],
        'B_cell': [
             'B cell',
        ],
        'T_cell': [
             'T cell regulatory',
             'T cell CD4',
             'T cell CD8 activated',
             'T cell CD8 effector memory',
             'T cell CD8 naive',
             'T cell CD8 terminally exhausted',
             'T cell NK-like',
        ],
    },
    'breast': {
        "Tumor_cell": [
            "luminal epithelial cell of mammary gland", 
            "mammary gland epithelial cell",
        ], 
        "Macrophage": [
            "macrophage",
        ], 
        "Endothelial_cell": [
            "endothelial cell", 
            "endothelial cell of lymphatic vessel"
        ], 
        'Stromal_cell': [
            "fibroblast of breast",
            "pericyte",
        ], 
        'Plasma_cell': [
            "plasmablast",
        ], 
        'B_cell': [
            "memory B cell", 
            "naive B cell", 
        ], 
        'T_cell': [
            "T cell",
            "CD4-positive, alpha-beta T cell",
            "CD8-positive, alpha-beta T cell",
            "mature NK T cell",
            "natural killer cell",
        ],
    },
}

class Config:
    def __init__(self, args):
        self.stem_directory = f"{args.platform}/{args.source}/{args.sample}/"
        self.stem_file = f"{args.platform}_{args.source}_{args.sample}"

        self.cell_types = next((cell_type_values for organ, cell_type_values in cell_types.items() if organ in args.sample.lower()), {}) #for the organ
        self.cell_subtypes = sum(self.cell_types.values(), [])

#        palette = 'blend:red,orange,yellow,green,blue,navy'
        palette = 'gist_ncar_r'
        self.palette_type = dict(zip(
            self.cell_types.keys(),
            sns.color_palette(palette, n_colors=len(self.cell_types)).as_hex()
        ))
        self.palette_subtype = dict(zip(
            self.cell_subtypes,
            sns.color_palette(palette, n_colors=len(self.cell_subtypes)).as_hex()
        ))

        self.label_encoder = LabelEncoder()
        if 'subtype' in args.cell_type:
            parameters =  self.cell_subtypes
        else:
            parameters = list(self.cell_types.keys())
        self.label_encoder.fit(parameters)
        self.label_encoder.classes_ = np.array(parameters)
        self.classes = self.label_encoder.classes_ #for classification
        print('The classes are:', self.classes)
