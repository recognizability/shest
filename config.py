seed = 42
n_cores = max(mp.cpu_count()-2, 1)

cell_types_lung = {
    'Tumor_cell_LUAD': [
        'Tumor cells LUAD',
        'Tumor cells LUAD EMT',
        'Tumor cells LUAD MSLN',
        'Tumor cells LUAD NE',
        'Tumor cells LUAD mitotic'
    ],
    'Stromal_cell': [
        'stromal dividing',
        'Fibroblast adventitial',
        'Fibroblast alveolar',
        'Fibroblast peribronchial'
    ],
    'Pericyte':[
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
         'NK cell dividing'
    ],
    'Other': [],
}
