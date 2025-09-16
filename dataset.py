import os
import json
import pickle
import joblib
from joblib import Parallel, delayed
from tqdm import tqdm
from collections import Counter
import numpy as np
import pandas as pd
from scipy import sparse

import matplotlib.pyplot as plt
import shapely
from skimage.draw import polygon2mask
from skimage.color import rgb2hed, hed2rgb

import scanpy as sc
import spatialdata as sd
from spatialdata.transformations import get_transformation
import spatialdata_plot
from spatialdata_io import xenium
import tacco as tc

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

from config import n_cores, generator, seed, set_seed, side, tile, lower_micrometer, upper_micrometer

sc.settings.n_jobs = n_cores

class Preprocessing():
    def __init__(self, args, config, source, sample):
        self.raw_directory = args.raw_directory
        self.source = source
        self.sample = sample
        self.stem_file = f"{args.platform}_{source}_{sample}"
        self.directory = args.directory
        self.sc_annotate = args.sc_annotate
        self.organ = config.organ
        self.cell_types = config.cell_types
        self.cell_subtype = config.cell_subtype
        self.cell_subtypes = config.cell_subtypes
        self.subtype_to_type = {subtype: category for category, subtypes in self.cell_types.items() for subtype in subtypes}

        self.stem_directory = f"{args.platform}/{source}/{sample}/"
        self.pixel_size = json.load(open(self.raw_directory + self.stem_directory + "experiment.xenium"))['pixel_size'] # micrometers per pixel

        self.sdata = self._prepare_sdata()
        self.affine = get_transformation(self.sdata.images['he_image']).to_affine_matrix(input_axes=('x', 'y'), output_axes=('x', 'y'))
        self.nucleus_boundaries = self.sdata.shapes["nucleus_boundaries"]
        adata_raw = self.sdata.tables['table']
        self.adata = adata_raw[:, adata_raw.var_names.isin(config.gene_panel)].copy() 
        self.adata.obs.index = self.adata.obs['cell_id']

        self.processing_directory = self.directory + 'dataset/' + self.stem_directory
        os.makedirs(self.processing_directory, exist_ok=True)

        self.image = self.sdata.images["he_image"]["scale0"]["image"].values
        self.height, self.width = self.image.shape[1:]
        self.save_image = args.save_image

        self.annotated_cell_ids = None
        self.image_ids = None
        self.images = None
        self.cell_ids = None
        self.exteriors = None
        
        self.sc_annotation()
        self.cell_images()
        self.annotation()

    def _prepare_sdata(self):
        path_zarr = self.directory + "dataset/" + self.stem_directory + "data.zarr"
        if os.path.exists(path_zarr):
            print("Loading the sdata ...", end=' ')
            sdata = sd.SpatialData.read(path_zarr)
            print('done.')
        else:
            path = self.raw_directory + self.stem_directory
            sdata = xenium(path=path, n_jobs=n_cores)
            print(f"Saving {path_zarr} ...")
            sdata.write(path_zarr) 
            print('done.')
        return sdata

    def _single_cell_reference(self):
        if self.organ == 'lung':
            print("Loading LuCA single cell reference ... ", end='')
            ref = sc.read_h5ad(self.raw_directory + 'cz_sc_reference/dd538ee7-f5e4-49e9-9f1e-2a1ea5246cf4.h5ad')
            ref = ref[ref.obs['platform']!='Smart-seq2'].copy()
        if self.organ == 'breast':
            print("Loading breast cancer single cell reference ... ", end='')
            ref = sc.read_h5ad(self.raw_directory + 'cz_sc_reference/966b60ee-b416-44bd-981c-817bfc476646.h5ad')
        if self.organ == 'skin':
            print("Loading skin cancer single cell reference ... ", end='')
            ref = sc.read_h5ad(self.raw_directory + 'cz_sc_reference/f6e35982-3bef-47fe-b14a-60d2e8965f20.h5ad')
            ref = ref[ref.obs['disease'].str.contains('melanoma', case=False, na=False) & ref.obs['tissue'].str.contains('skin', case=False, na=False)].copy()
        ref.index = ref.var.feature_name
        ref.var.index = ref.var.feature_name
        print(ref.shape)
        return ref

    def sc_annotation(self):
        os.makedirs(self.processing_directory + f'annotation/', exist_ok=True)
        sc_annotation_csv = self.processing_directory + f'annotation/sc_annotation_{self.cell_subtype}.csv'
        sc_annotation_h5ad = self.processing_directory + f'annotation/sc_annotation_{self.cell_subtype}.h5ad'
        if not (os.path.exists(sc_annotation_csv) and os.path.exists(sc_annotation_h5ad)) or self.sc_annotate:
            ref = self._single_cell_reference()
            print('Annotating types of the cells ... ')
            self.adata.obs[self.cell_subtype] = tc.tl.annotate(
                self.adata,
                ref,
                annotation_key=self.cell_subtype,
                assume_valid_counts=True,
                remove_constant_genes=False,
            ).T.idxmax()
            self.adata.obs[self.cell_subtype].to_csv(sc_annotation_csv)
            print(len(self.adata.obs[self.cell_subtype].unique()), 'subtypes are annotated.')

            print('Setting neighbors for each cell ...')
            sc.pp.neighbors(self.adata, random_state=seed)
            print('Making UMAPs for each cell ...')
            sc.tl.umap(self.adata, random_state=seed)

            print("Saving the annotations ... ", end='')
            self.adata.write_h5ad(sc_annotation_h5ad)
            print("done.")
        else: 
            print("Loading the annotations ... ", end='')
            self.adata = sc.read_h5ad(sc_annotation_h5ad)
            print("done.")

        self.adata.obs.index = self.adata.obs['cell_id']
        self.adata.obs = self.adata.obs.drop(columns='cell_id')
        inclusion = self.adata.obs[self.cell_subtype].isin(self.cell_subtypes)
        self.adata.obs.loc[inclusion, 'cell_subtype_expression'] = self.adata.obs.loc[inclusion, self.cell_subtype]
        self.adata.obs['cell_subtype_expression'] = pd.Categorical(self.adata.obs['cell_subtype_expression'], categories=self.cell_subtypes, ordered=True)
        self.adata.obs['cell_type_expression'] = self.adata.obs[self.cell_subtype].map(self.subtype_to_type)
        self.adata.obs['cell_type_expression'] = pd.Categorical(self.adata.obs['cell_type_expression'], categories=self.cell_types.keys(), ordered=True)

    def _transform(self, polygon):
        x, y = polygon.exterior.xy
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        x = x / self.pixel_size
        y = y / self.pixel_size

        coords = np.stack([x, y], axis=1)
        ones = np.ones((coords.shape[0], 1))
        three_dimension = np.hstack([coords, ones]).T
        inverse_affine = np.linalg.inv(self.affine)
        transformed = inverse_affine @ three_dimension
        x, y = transformed[:2]

        coords = np.stack([x, y], axis=1)
        return shapely.geometry.Polygon(coords).exterior

    def cell_images(self):
        self.exteriors = self.nucleus_boundaries.geometry.apply(lambda polygon: self._transform(polygon))
        images_directory = os.path.join(self.processing_directory, 'images/')
        os.makedirs(images_directory, exist_ok=True)
        images_file = images_directory + f"images.pkl"
        if not os.path.exists(images_file) or self.save_image:
            lower = int(round(lower_micrometer/self.pixel_size)) #pixels
            upper = int(round(upper_micrometer/self.pixel_size)) #pixels
            half = upper // 2
            self.images = {}
            print("Filtering and preparing cell images ...")
            for cell_id, polygon in tqdm(exteriors.items(), total=len(exteriors)):
                x, y = polygon.xy
                if np.min(x) < 0 or np.max(x) >= self.width or np.min(y) < 0 or np.max(y) >= self.height:
                    continue
                nucleus = max(np.max(y) - np.min(y), np.max(x) - np.min(x))
                nucleus = int(round(tile*nucleus/upper))
                if nucleus < lower or nucleus >= upper:
                    continue
                x_centroid = int(round(polygon.centroid.x))
                y_centroid = int(round(polygon.centroid.y))
                y_lower = y_centroid - half
                y_upper = y_centroid + half
                x_left = x_centroid - half
                x_right = x_centroid + half
                if x_left < 0 or x_right >= self.width or y_lower < 0 or y_upper >= self.height:
                    continue

                window_image = self.image[:, y_lower:y_upper, x_left:x_right].copy()
                window_image = torch.from_numpy(window_image).float()
                window_image = F.interpolate(window_image.unsqueeze(0), size=tile, mode="bilinear", align_corners=False).squeeze(0)

                polygon_shifted = [(
                    int(round(tile*(y - y_lower)/upper)),
                    int(round(tile*(x - x_left)/upper)),
                ) for x, y in polygon.coords]
                mask = polygon2mask(window_image.shape[1:], polygon_shifted)
                mask = torch.from_numpy(mask).unsqueeze(0).float()
                window_image_masked = window_image*mask

                center = window_image.shape[1] // 2
                nucleus_image = window_image[:, center-nucleus//2:center+nucleus//2+1, center-nucleus//2:center+nucleus//2+1]
                nucleus_image = F.interpolate(nucleus_image.unsqueeze(0), size=tile, mode="bilinear", align_corners=False).squeeze(0)
                nucleus_image_masked = window_image_masked[:, center-nucleus//2:center+nucleus//2+1, center-nucleus//2:center+nucleus//2+1]
                nucleus_image_masked = F.interpolate(nucleus_image_masked.unsqueeze(0), size=tile, mode="bilinear", align_corners=False).squeeze(0)

                self.images[cell_id] = torch.cat([
                    torch.cat([window_image, window_image_masked], dim=2),
                    torch.cat([nucleus_image, nucleus_image_masked], dim=2)
                ], dim=1)

            print("Saving the images ...", end=' ')
            joblib.dump(self.images, os.path.join(images_file), compress=0)
            print("done.")
        else:
            print("Loading the images ...", end=' ')
            with open(file=os.path.join(images_file), mode='rb') as f:
                self.images = joblib.load(f)
            print("done.")

        self.image_ids = list(self.images.keys())
        print(len(self.image_ids), "images are prepared.")

    def annotation(self):
        he_annotation_file = self.processing_directory + f"annotation/he_annotation.csv"
        if os.path.exists(he_annotation_file):
            he_annotation = pd.read_csv(he_annotation_file, index_col='cell_id')
            he_annotation['cell_type_morphology'] = he_annotation['group'].astype(str)
            self.adata.obs = self.adata.obs.merge(he_annotation['cell_type_morphology'], how='left', left_index=True, right_index=True)
            print(f"{len(self.adata.obs['cell_type_morphology'].dropna())} cells are annotated by thier morphologies.")

            inclusion_annotation = self.adata.obs['cell_type_expression'].astype(str) == self.adata.obs['cell_type_morphology'].astype(str)
            self.adata.obs.loc[inclusion_annotation, 'cell_type_annotation'] = self.adata.obs.loc[inclusion_annotation, 'cell_type_expression']
            self.annotated_cell_ids = self.adata.obs[self.adata.obs['cell_type_annotation'].notna()].index
            print(f'Only the {len(self.annotated_cell_ids)} cells are annotated by the morphology and the single cell reference')
            self.adata.obs.loc[inclusion_annotation, 'cell_subtype_annotation'] = self.adata.obs.loc[inclusion_annotation, 'cell_subtype_expression']

            self.cell_ids = sorted(list(set(self.annotated_cell_ids) & set(self.image_ids)))
            print(f"Only {len(self.cell_ids)} cells common to both annotation and area filtering are prepared.")
            self.adata.obs['cell_type'] = self.adata.obs['cell_type_annotation'].where(self.adata.obs.index.isin(self.cell_ids), other=np.nan)
            self.adata.obs['cell_subtype'] = self.adata.obs['cell_subtype_annotation'].where(self.adata.obs.index.isin(self.cell_ids), other=np.nan)
            
            self.adata.obs = self.adata.obs[['cell_subtype_expression', 'cell_type_expression', 'cell_type_morphology', 'cell_subtype_annotation', 'cell_type_annotation', 'cell_subtype', 'cell_type']]

        self.adata.raw = self.adata
        self.adata.write(self.processing_directory + f'annotation/adata.h5ad')

class Images():
    def __init__(self, args, images, cell_ids=None):
        self.batch_size = args.batch_size
        self.images_raw = images
        self.cell_ids = list(self.images_raw.keys()) if cell_ids is None else cell_ids
        self.images = self._image_tensor()

    def _image_tensor(self):
        images = []
        print("Loading images ...")
        for cell_id in tqdm(self.cell_ids):
            image = self.images_raw[cell_id]
            images.append(image.unsqueeze(0))
        images = torch.cat(images, dim=0).contiguous() / 255.0
        mean = torch.tensor([0.707223, 0.578729, 0.703617]).view(1, 3, 1, 1) #from H-optimus-0
        std = torch.tensor([0.211883, 0.230117, 0.177517]).view(1, 3, 1, 1) #from H-optimus-0
        images = (images - mean) / std
        print(len(images), "images are converted into a tensor.")
        return images

    def __len__(self):
        return len(self.cell_ids)

    def __getitem__(self, i):
        cell_id = self.cell_ids[i]
        image = self.images[i]
        return cell_id, image

    def loader(self):
        total = len(self)
        return DataLoader(self, batch_size=self.batch_size, shuffle=False, pin_memory=True)

class Load(Images):
    def __init__(self, args, config, source, sample):
        self.directory = args.directory
        self.cell_type = args.cell_type
        self.cell_types = config.cell_types
        self.cell_subtypes = config.cell_subtypes
        self.palette_type = config.palette_type
        self.palette_subtype = config.palette_subtype
        self.batch_size = args.batch_size
        self.stem_file = f"{args.platform}_{source}_{sample}"

        preprocessed = Preprocessing(args, config, source, sample)

        stem_directory = f"{args.platform}/{source}/{sample}/"
        processing_directory = self.directory + 'dataset/' + stem_directory
        self.images_raw = preprocessed.images
        self.image_ids = preprocessed.image_ids
        print(len(self.image_ids), "images of the cells are prepared.")

        print('Expression profile and their cell types loading ... ', end='')
        self.adata_raw = preprocessed.adata
        print(self.adata_raw.shape)
        self.columns = self.adata_raw.obs.columns
        type_ids = self.adata_raw.obs[self.cell_type].dropna().index.tolist()
        print(len(type_ids), "of annotated cells are loaded.")

        self.cell_ids = sorted(list(set(self.image_ids) & set(type_ids)))
        print('Common', len(self.cell_ids), "cells are selected.")

        super().__init__(args, self.images_raw, self.cell_ids)
        self.image = preprocessed.image
        self.width = preprocessed.width
        self.height = preprocessed.height
        self.exteriors = preprocessed.exteriors

        self.adata = self.adata_raw[self.cell_ids, :].copy()
        self.cell_type_encoded = self.cell_type + '_encoded'
        self.label_encoder = config.label_encoder
        self.adata.obs[self.cell_type_encoded] = self.label_encoder.transform(self.adata.obs[self.cell_type])
        self.genes = self.adata.var_names.tolist()

        cell_indices = self.adata.obs_names.get_indexer(self.cell_ids)
        self.expressions = torch.from_numpy(self.adata.X[cell_indices].toarray()).float()
        self.labels_raw = self.adata.obs[self.cell_type].tolist()
        self.labels = torch.from_numpy(self.adata.obs.iloc[cell_indices][self.cell_type_encoded].to_numpy()).long()
        self.classes = config.classes

    def __getitem__(self, i):
        cell_id = self.cell_ids[i]
        image = self.images[i]
        expression = self.expressions[i]
        label = self.labels[i]

        return cell_id, image, expression, label

class Dataset():
    def __init__(self, args, config):
        self.batch_size = args.batch_size
        self.classes = config.classes
        cell_ids = []
        images = []
        expressions = []
        labels_raw = []
        labels = []
        genes = []
        stem_file = args.platform
        columns = []
        for source, sample in zip(args.sources, args.samples, strict=True):
            loaded = Load(args, config, source, sample)
            cell_ids.extend(loaded.cell_ids)
            images.append(loaded.images)
            expressions.append(loaded.expressions)
            labels_raw.extend(loaded.labels_raw)
            labels.append(loaded.labels)
            for gene in loaded.genes:
                if gene not in genes:
                    genes.append(gene)
            stem_file += f"_{source}_{sample}"
            columns.extend(loaded.columns)
            del loaded

        self.genes = genes
        self.stem_file = stem_file
        self.columns = columns
        print('stem_file:', self.stem_file)

        class_indices = {c: [i for i, label in enumerate(labels_raw) if label == c] for c in self.classes}
        print("classes:", {c:len(indices) for c, indices in class_indices.items()})

        images = torch.cat(images, dim=0).contiguous()
        expressions = torch.cat(expressions, dim=0).contiguous()
        labels = torch.cat(labels, dim=0).contiguous()

        if args.not_augment:
            self.cell_ids = cell_ids
            self.images = images
            self.expressions = expressions
            self.labels = labels
        else:
            max_count = max(len(indices) for indices in class_indices.values())
            angles = [0, 90, 180, 270]
            flips = [None, "h", "v", "hv"]
            rng = np.random.default_rng(seed)
            augmented_images = []
            augmented_expressions = []
            augmented_labels = []
            augmented_cell_ids = []
            print("Augmentating the dataset ...")
            for c, indices in tqdm(class_indices.items()):
                repeat = max_count // len(indices)
                remain = max_count % len(indices)
                expanded_indices = indices * repeat + indices[:remain]
                for i in expanded_indices:
                    image = images[i]
                    image_tiles = [
                        image[..., :tile, :tile],
                        image[..., :tile, tile:],
                        image[..., tile:, :tile],
                        image[..., tile:, tile:],
                    ]
                    expression = expressions[i]
                    label = labels[i]
                    cell_id = cell_ids[i]
                    angle = int(rng.choice(angles))
                    flip = str(rng.choice(flips))
                    if angle != 0:
                        image_tiles = [transforms.functional.rotate(t, angle) for t in image_tiles]
                    if flip == "h":
                        image_tiles = [transforms.functional.hflip(t) for t in image_tiles]
                    elif flip == "v":
                        image_tiles = [transforms.functional.vflip(t) for t in image_tiles]
                    elif flip == "hv":
                        image_tiles = [transforms.functional.hflip(transforms.functional.vflip(t)) for t in image_tiles]
                    image = torch.cat([
                        torch.cat(image_tiles[:2], dim=-1),
                        torch.cat(image_tiles[2:], dim=-1),
                    ], dim=-2)
                    augmented_images.append(image.unsqueeze(0)) 
                    augmented_expressions.append(expression.unsqueeze(0)) 
                    augmented_labels.append(label.unsqueeze(0)) 
                    augmented_cell_ids.append(cell_id)

            print("augmented_labels:", dict(Counter([int(label) for label in augmented_labels])))
            self.cell_ids = augmented_cell_ids
            self.images = torch.cat(augmented_images, dim=0)
            self.expressions = torch.cat(augmented_expressions, dim=0)
            self.labels = torch.cat(augmented_labels, dim=0)

        print('cell_ids:', len(self.cell_ids))

    def __len__(self):
        return len(self.cell_ids)

    def __getitem__(self, i):
        cell_id = self.cell_ids[i]
        image = self.images[i]
        expression = self.expressions[i]
        label = self.labels[i]

        return cell_id, image, expression, label

    def loader(self, split=0.8):
        total = len(self)
        if split==1 or split==0:
            data_loader = DataLoader(self, batch_size=self.batch_size, shuffle=False, pin_memory=True)
            return data_loader
        else:
            train_size = int(split * total)
            test_size = total - train_size
            train_dataset, test_dataset = random_split(self, [train_size, test_size], generator=generator)
            print(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True)
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True)
            return train_loader, test_loader
