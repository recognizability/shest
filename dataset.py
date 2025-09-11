import numpy as np
from collections import Counter
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

from config import generator, seed, set_seed
from preprocess import Preprocessing

class Quadruple():
    def __init__(self, images, cell_ids=None):
        self.images_raw = images
        self.cell_ids = list(images.keys()) if cell_ids is None else cell_ids
        print(len(self.cell_ids), "images of the cells are prepared.")
        self.images = self._image_load()

    def __len__(self):
        return len(self.cell_ids)

    def _resize(self, image, size):
        return F.interpolate(image.unsqueeze(0), size=size, mode="bilinear", align_corners=False).squeeze()

    def _weight(self, steps):
        coords = torch.linspace(-1, 1, steps=steps)
        weight = ((coords[:, None]**2 + coords[None, :]**2 - 2) / 2)**2
        return weight

    def _image_load(self):
        length = 224
        half = length // 2
        weight = self._weight(half).unsqueeze(0)
        images = []
        print("Image processing ... ", end='')
        for cell_id in self.cell_ids:
            image_dict = self.images_raw[cell_id]
            window_image = torch.from_numpy(image_dict['window_image'])
            center = window_image.shape[1] // 2
            nucleus = image_dict['nucleus']
            nucleus_image = window_image[:, center-nucleus//2:center+nucleus//2+1, center-nucleus//2:center+nucleus//2+1]

            image_top_left = self._resize(window_image, half)
            image_bottom_left = self._resize(nucleus_image, half)
            image_top_right = image_top_left * weight
            image_bottom_right = image_bottom_left * weight

            image = torch.zeros(3, length, length, dtype=torch.uint8, device=window_image.device)
            image[:, :half, :half] = image_top_left
            image[:, :half, half:] = image_top_right
            image[:, half:, :half] = image_bottom_left
            image[:, half:, half:] = image_bottom_right

            images.append(image.unsqueeze(0))

        images = torch.cat(images, dim=0).contiguous() / 255.0
        mean = torch.tensor([0.707223, 0.578729, 0.703617]).view(1, 3, 1, 1)
        std = torch.tensor([0.211883, 0.230117, 0.177517]).view(1, 3, 1, 1)
        images = (images - mean) / std
        print(len(images), "images are loaded.")
        return images

    def __getitem__(self, i):
        cell_id = self.cell_ids[i]
        image = self.images[i]
        return cell_id, image

class Infer(Quadruple):
    def __init__(self, images):
        super().__init__(images)

    def loader(self, batch_size):
        total = len(self)
        return DataLoader(self, batch_size=batch_size, shuffle=False, pin_memory=True)

class Load(Quadruple):
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

        super().__init__(self.images_raw, self.cell_ids)

        self.adata = self.adata_raw[self.cell_ids, :].copy()
        self.cell_type_encoded = self.cell_type + '_encoded'
        self.label_encoder = config.label_encoder
        self.adata.obs[self.cell_type_encoded] = self.label_encoder.transform(self.adata.obs[self.cell_type])
        self.genes = self.adata.var_names.tolist()

        cell_indices = self.adata.obs_names.get_indexer(self.cell_ids)
        self.expressions = torch.from_numpy(self.adata.X[cell_indices].toarray()).float()
        self.labels_raw = self.adata.obs[self.cell_type].tolist()
        self.labels =  torch.from_numpy(self.adata.obs.iloc[cell_indices][self.cell_type_encoded].to_numpy(dtype=np.int64))
        self.classes = config.classes

    def __getitem__(self, i):
        cell_id = self.cell_ids[i]
        image = self.images[i]
        expression = self.expressions[i]
        label = self.labels[i]

        return cell_id, image, expression, label

class Dataset():
    def __init__(self, args, config):
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

        print('cell_ids:', len(cell_ids))
        print('stem_file:', stem_file)

        images = torch.cat(images, dim=0).contiguous()
        expressions = torch.cat(expressions, dim=0).contiguous()
        labels = torch.cat(labels, dim=0).contiguous()
        print("labels:", dict(Counter([int(label) for label in labels])))

        self.cell_ids = cell_ids
        self.genes = genes
        self.stem_file = stem_file
        self.columns = columns

        self.batch_size = args.batch_size

        class_indices = {c: [i for i, label in enumerate(labels_raw) if label == c] for c in self.classes}
        print("classes:", {c:len(indices) for c, indices in class_indices.items()})
        max_count = max(len(indices) for indices in class_indices.values())
        angles = [0, 90, 180, 270]
        flips = [None, "h", "v", "hv"]
        rng = np.random.default_rng(seed)
        class_counts = Counter()
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
                angle = int(rng.choice(angles))
                flip = str(rng.choice(flips))
                image = images[i]
                expression = expressions[i]
                label = labels[i]
                cell_id = cell_ids[i]
                if angle != 0:
                    image = transforms.functional.rotate(image, angle)
                if flip == "h":
                    image = transforms.functional.hflip(image)
                elif flip == "v":
                    image = transforms.functional.vflip(image)
                elif flip == "hv":
                    image = transforms.functional.hflip(transforms.functional.vflip(image))
                augmented_images.append(image.unsqueeze(0))
                augmented_expressions.append(expression.unsqueeze(0))
                augmented_labels.append(label.unsqueeze(0))
                augmented_cell_ids.append(cell_id)
                class_counts[c] += 1

        print("Augmented classes:", dict(class_counts))
        print("Augmented labels:", dict(Counter([int(label) for label in augmented_labels])))

        self.images = torch.cat(augmented_images, dim=0).contiguous()
        self.expressions = torch.cat(augmented_expressions, dim=0).contiguous()
        self.labels = torch.cat(augmented_labels, dim=0).contiguous()
        self.cell_ids = augmented_cell_ids

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
