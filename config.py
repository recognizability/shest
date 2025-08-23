import os
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch
import multiprocessing as mp
import seaborn as sns
from pprint import pprint

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
            'Pericyte',
        ],
        'Plasma_cell': [
            'Plasma cell',
        ],
        'Lymphocyte': [
            'B cell',
            'T cell regulatory',
            'T cell CD4',
            'T cell CD8 activated',
            'T cell CD8 effector memory',
            'T cell CD8 naive',
            'T cell CD8 terminally exhausted',
            'T cell NK-like',
            'NK cell', # not B or T
        ],
    },
    'breast': {
        "Tumor_cell_BRCA": [
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
        ], 
        'Lymphocyte': [
            "memory B cell", 
            "naive B cell", 
            "T cell",
            "CD4-positive, alpha-beta T cell",
            "CD8-positive, alpha-beta T cell",
            "mature NK T cell",
            "natural killer cell",
        ],
    },
    'skin':{
        'Tumor_cell_SKCM':[
            'malignant cell',
        ],
        "Macrophage": [
        ], 
        'Endothelial_cell':[
            'endothelial cell',
        ],
        'Stromal_cell':[
            'fibroblast',
        ],
        'Plasma_cell': [
        ],
        'Lymphocyte':[
            'B cell',
            'T cell',
        ],
    }
}

cell_subtype = {
    'lung':'cell_type_tumor',
    'breast':'cell_type',
    'skin':'cell_type',
}

class Config:
    def __init__(self, args):
        self.organ = args.organ.lower()

        self.cell_types = next((cell_type_values for organ, cell_type_values in cell_types.items() if organ == self.organ), {}) #for the organ
        self.cell_subtype = next((subtype for organ, subtype in cell_subtype.items() if organ == self.organ), {}) #for the organ
        self.cell_subtypes = sum(self.cell_types.values(), [])

        palette = 'nipy_spectral_r'
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

        self.gene_panel = pd.read_csv(args.raw_directory + args.platform + '/XeniumPrimeHuman5Kpan_tissue_pathways_metadata.csv')['gene_name'].values

        pprint(self.__dict__)
