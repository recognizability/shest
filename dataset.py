import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

import matplotlib.pyplot as plt

from config import generator
from preprocess import Preprocessing

#class Dataset():
#    def __init__(self, images, batch_size):
#        self.images = images
#        self.cell_ids = list(images.keys())
#        print(len(self.cell_ids), "images of the cells are prepared.")
#        self.batch_size = batch_size
#
#    def __len__(self):
#        return len(self.cell_ids)
#
#    def _resize(self, image, size):
#        return F.interpolate(image.unsqueeze(0), size=size, mode="bilinear", align_corners=False).squeeze()
#
#    def _weight(self, steps):
#        coords = torch.linspace(-1, 1, steps=steps)
#        weight = ((coords[:, None]**2 + coords[None, :]**2 - 2) / 2)**2
#        return weight
#
#    def __getitem__(self, i):
#        cell_id = self.cell_ids[i]
#        image_dict = self.images[cell_id]
#        window_image = torch.from_numpy(image_dict['window_image'].copy().transpose(2, 1, 0))
#        center = window_image.shape[1] // 2
#        nucleus = image_dict['nucleus']
#        nucleus_image = window_image[:, center-nucleus//2:center+nucleus//2+1, center-nucleus//2:center+nucleus//2+1]
#
#        length = 224
#        half = length // 2
#
#        weight = self._weight(half).to(window_image.device).unsqueeze(0)
#
#        image_top_left = self._resize(window_image, half)
#        image_top_right = image_top_left * weight
#        image_bottom_left = self._resize(nucleus_image, half)
#        image_bottom_right = image_bottom_left * weight
#
#        image = torch.zeros(3, length, length, dtype=torch.uint8, device=window_image.device)
#        image[:, :half, :half] = image_top_left
#        image[:, :half, half:] = image_top_right
#        image[:, half:, :half] = image_bottom_left
#        image[:, half:, half:] = image_bottom_right
#
#        if isinstance(image, torch.Tensor):
#            image = image.float().div(255)
#        else:
#            image = transforms.ToTensor()(image)
#        image = transforms.Normalize(
#            mean=(0.707223, 0.578729, 0.703617),
#            std=(0.211883, 0.230117, 0.177517)
#        )(image)
#
#        return cell_id, image
#
#    def loader(self):
#        total = len(self)
#        return DataLoader(self, batch_size=self.batch_size, shuffle=False, pin_memory=True)

class Dataset():
    def __init__(self, args, config, source, sample):
        self.directory = args.directory
        self.original_image = args.original_image
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
        type_ids = self.adata_raw.obs[self.cell_type].dropna().index.tolist()
        print(len(type_ids), "of annotated cells are loaded.")

        self.cell_ids = sorted(list(set(self.image_ids) & set(type_ids)))
        print('Common', len(self.cell_ids), "cells are selected.")

        self.images = self._image_load()

        self.adata = self.adata_raw[self.cell_ids, :].copy()
        self.cell_type_encoded = self.cell_type + '_encoded'
        self.label_encoder = config.label_encoder
        self.adata.obs[self.cell_type_encoded] = self.label_encoder.transform(self.adata.obs[self.cell_type])
        self.genes = self.adata.var_names.tolist()

        cell_indices = self.adata.obs_names.get_indexer(self.cell_ids)
        self.expressions = torch.from_numpy(self.adata.X[cell_indices].toarray()).float()
        self.labels =  torch.from_numpy(self.adata.obs.iloc[cell_indices][self.cell_type_encoded].to_numpy(dtype=np.int64))
        self.classes = config.classes

    def _resize(self, image, size):
        return F.interpolate(image.unsqueeze(0), size=size, mode="bilinear", align_corners=False).squeeze()

    def _weight(self, steps):
        coords = torch.linspace(-1, 1, steps=steps)
        weight = ((coords[:, None]**2 + coords[None, :]**2 - 2) / 2)**2
        return weight

    def _image_load(self):
        images = []
        print("Image processing ... ", end='')
        for cell_id in self.cell_ids:
            image_dict = self.images_raw[cell_id]
            window_image = torch.from_numpy(image_dict['window_image'])
            center = window_image.shape[1] // 2
            nucleus = image_dict['nucleus']
            nucleus_image = window_image[:, center-nucleus//2:center+nucleus//2+1, center-nucleus//2:center+nucleus//2+1]

            length = 224
            if self.original_image:
                image = self._resize(window_image, length)
            else:
                half = length // 2

                image_top_left = self._resize(window_image, half)
                image_bottom_left = self._resize(nucleus_image, half)

                weight = self._weight(half).to(window_image.device).unsqueeze(0)
                image_top_right = image_top_left * weight
                image_bottom_right = image_bottom_left * weight

                image = torch.zeros(3, length, length, dtype=torch.uint8, device=window_image.device)
                image[:, :half, :half] = image_top_left
                image[:, :half, half:] = image_top_right
                image[:, half:, :half] = image_bottom_left
                image[:, half:, half:] = image_bottom_right

            if isinstance(image, torch.Tensor):
                image = image.float().div(255)
            else:
                image = transforms.ToTensor()(image)

            image = transforms.Normalize(mean=(0.707223, 0.578729, 0.703617), std=(0.211883, 0.230117, 0.177517))(image) #from H-optimus-0

            images.append(image)

        images = torch.stack(images).contiguous()

        print(len(images), "images are loaded.")
        return images

    def __len__(self):
        return len(self.cell_ids)

    def __getitem__(self, i):
        cell_id = self.cell_ids[i]
        image = self.images[i]
        expression = self.expressions[i]
        label = self.labels[i]

        return cell_id, image, expression, label

    def draw_umaps_expression(self):
        cell_types = self.adata_raw.obs.columns
        fig, ax = plt.subplots(
            nrows=4,
            ncols=4,
            figsize=(6+6+6+6, 6+4+2+4),
            gridspec_kw={
                'width_ratios': [1, 1, 1, 1],
                'height_ratios': [1.5, 1, 0.5, 1]
            }
        )
        ax[0][1].axis('off')
        ax[1][1].axis('off')
        indices = {'expression':0, 'morphology':1, 'annotation':2}
        for cell_type in cell_types:
            if 'subtype' in cell_type:
                reindex = self.cell_subtypes
                palette = self.palette_subtype
                i = 0
            else:
                reindex = self.cell_types.keys()
                palette = self.palette_type
                i = 2
            j = next((index for string, index in indices.items() if string in cell_type), 3)
            counts = pd.DataFrame(self.adata_raw.obs.groupby(cell_type, observed=False).apply(len, include_groups=False), columns=['']).reindex(reindex).T
            ax[i][j] = sns.barplot(
               counts,
               orient = 'h',
               palette = palette,
               ax=ax[i][j]
            )
            ax[i][j].set_xlim(0, counts.max(axis=None, skipna=True) * 1.2 if pd.notna(counts.max(axis=None)) else 1)
            ax[i][j].set_title(f"n={counts.sum(axis=1).values[0]}")
            for container in ax[i][j].containers:
                ax[i][j].bar_label(container)
            sc.pl.umap(self.adata_raw, color=cell_type, palette=palette, ax=ax[i+1][j], show=False, legend_loc=None, size=1)
        fig.tight_layout()
        fig.savefig(self.directory + f"results/umaps_expression_{self.stem_file}_{self.cell_type}.png", bbox_inches="tight")
        plt.close()

    def loader(self, split=0.8):
        total = len(self)
        if split==1 or split==0:
            data_loader = DataLoader(self, batch_size=self.batch_size, shuffle=False, pin_memory=True)
            return data_loader
        else:
            train_size = int(split * total)
            test_size = total - train_size
            train_dataset, test_dateset = random_split(self, [train_size, test_size], generator=generator)
            print(f"Train size: {len(train_dataset)}, Test size: {len(test_dateset)}")
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True)
            test_loader = DataLoader(test_dateset, batch_size=self.batch_size, shuffle=False, pin_memory=True)
            return train_loader, test_loader

class MergedDataset(Dataset):
    def __init__(self, args, config):
        cell_ids = []
        images = []
        expressions = []
        labels = []
        genes = []
        stem_file = args.platform
        for source, sample in zip(args.sources, args.samples):
            dataset = Dataset(args, config, source, sample)
            cell_ids.extend(dataset.cell_ids)
            images.extend(dataset.images)
            expressions.extend(dataset.expressions)
            labels.extend(dataset.labels)
            for gene in dataset.genes:
                if gene not in genes:
                    genes.append(gene)
            stem_file += f"_{source}_{sample}"

        images = torch.stack(images)
        expressions = torch.stack(expressions)
        labels = torch.stack(labels)

        print('cell_ids:', len(cell_ids))
        print('images:', images.shape, type(images))
        print('expressions:', expressions.shape, type(expressions))
        print('labels:', labels.shape, type(labels))
        print('genes:', len(genes))
        print('stem_file:', stem_file)

        self.cell_ids = cell_ids
        self.images = images
        self.expressions = expressions
        self.labels = labels
        self.genes = genes
        self.stem_file = stem_file

        self.batch_size = args.batch_size

    def __len__(self):
        return len(self.cell_ids)

    def __getitem__(self, i):
        cell_id = self.cell_ids[i]
        image = self.images[i]
        expression = self.expressions[i]
        label = self.labels[i]

        return cell_id, image, expression, label
