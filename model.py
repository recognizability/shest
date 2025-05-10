import os
os.environ["PARAMETRICUMAP"] = "0"
os.environ["UMAP_DISABLE_PARAMETRIC"] = "True"

import sys
import gc
from glob import glob
from collections import Counter
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map

import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, f1_score
from umap import UMAP

import anndata
import scanpy as sc
import tacco as tc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import NegativeBinomial
from torch.utils.data import DataLoader, random_split

import torchvision.transforms as transforms
import torchvision.models as models
from conch.open_clip_custom import create_model_from_pretrained
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from config import n_cores, seed, set_seed, generator, device

sc.settings.n_jobs = n_cores
in_features = 1000 #output features of ViT or SwinTransformer

def preprocessing(adata):
    adata.var_names_make_unique()
    adata.obs_names_make_unique()
    
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    adata = adata[adata.obs.pct_counts_mt < 5, :].copy()

    return adata

class Dataset():
    def __init__(self, args, config):
        self.directory = args.directory
        self.upper = args.upper
        self.upper_string = f"upper{args.upper}"
        self.cell_type = args.cell_type
        self.cell_types = config.cell_types
        self.cell_subtypes = config.cell_subtypes
        self.palette_type = config.palette_type
        self.palette_subtype = config.palette_subtype
        self.batch_size = args.batch_size
        self.angles = config.angles
        self.stem_file = config.stem_file

        processing_directory = self.directory + 'dataset/' + config.stem_directory
        self.images = torch.load(processing_directory + f'images/images_{self.upper_string}.pt')
        self.image_ids = json.load(open(processing_directory + f"images/image_ids_{self.upper_string}.json"))
        print(len(self.image_ids), "images of the cells are prepared.")

        print('Expression profile and their cell types loading ... ', end='')
        adata_file = os.path.join(processing_directory, f'annotation/adata.h5ad')
        self.adata_raw = sc.read_h5ad(adata_file)
        print(self.adata_raw.shape)
        type_ids = self.adata_raw.obs[self.cell_type].dropna().index.tolist()
        print(len(type_ids), "of annotated cells are loaded.")

        self.cell_ids = sorted(list(set(self.image_ids) & set(type_ids)))
        print('Common', len(self.cell_ids), "cells are selected.")

        self.adata = self.adata_raw[self.cell_ids, :].copy()
        self.cell_type_encoded = self.cell_type + '_encoded'
        self.label_encoder = config.label_encoder
        self.adata.obs[self.cell_type_encoded] = self.label_encoder.transform(self.adata.obs[self.cell_type])
        self.genes = self.adata.var_names.tolist()
        self.classes = config.classes

        cell_indices = self.adata.obs_names.get_indexer(self.cell_ids)
        self.expressions = torch.from_numpy(self.adata.X[cell_indices].toarray()).float()
        self.labels =  torch.from_numpy(self.adata.obs.iloc[cell_indices][self.cell_type_encoded].to_numpy(dtype=np.int64))

    def __len__(self):
        return len(self.cell_ids) * len(self.angles)

    def __getitem__(self, i):
        base_i = i // len(self.angles)
        angle_i = i % len(self.angles)

        cell_id = self.cell_ids[base_i]
        angle = self.angles[angle_i]

        image = self.images[self.image_ids[cell_id]]
        image = transforms.functional.rotate(image, angle)
        image = transforms.Resize((224, 224))(image)
        if isinstance(image, torch.Tensor):
            image = image.float().div(255)
        else:
            image = transforms.ToTensor()(image)
        image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)

        expression = self.expressions[base_i]
        label = self.labels[base_i]

        return cell_id, image, expression, label

    def draw_umaps_expression(self):
        cell_types = self.adata_raw.obs.columns
#        fig, ax = plt.subplots(4, len(cell_types)//2, figsize=(2*len(cell_types), 20))
        fig, ax = plt.subplots(
            nrows=4,
            ncols=4,
            figsize=(6+6+6+6, 6+4+2+4),
            gridspec_kw={
                'width_ratios': [1, 1, 1, 1],
                'height_ratios': [1.5, 1, 0.5, 1]
            }
        )
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
            ax[i][j].set_xlim(0, counts.max(axis=None)*1.2)
            ax[i][j].set_title(f"n={counts.sum(axis=1).values[0]}")
            for container in ax[i][j].containers:
                ax[i][j].bar_label(container)
            sc.pl.umap(self.adata_raw, color=cell_type, palette=palette, ax=ax[i+1][j], show=False, legend_loc=None)
        fig.tight_layout()
        fig.savefig(self.directory + f"results/umaps_expression_{self.stem_file}_{self.upper_string}.png", bbox_inches="tight")
        plt.close()

    def loader(self, split=0.8):
        total = len(self)
        prefetch_factor = 32
        if split==1 or split==0:
            data_loader = DataLoader(self, batch_size=self.batch_size, shuffle=False, num_workers=n_cores, pin_memory=True, prefetch_factor=prefetch_factor, persistent_workers=True)
            return data_loader
        else:
            train_size = int(split * total)
            test_size = total - train_size
            train_dataset, test_dateset = random_split(self, [train_size, test_size], generator=generator)
            print(f"Train size: {len(train_dataset)}, Test size: {len(test_dateset)}")
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=n_cores, pin_memory=True, prefetch_factor=prefetch_factor, persistent_workers=True)
            test_loader = DataLoader(test_dateset, batch_size=self.batch_size, shuffle=False, num_workers=n_cores, pin_memory=True, prefetch_factor=prefetch_factor, persistent_workers=True)
            return train_loader, test_loader

class Encoder(nn.Module):
    def __init__(self, backbone="vit_b_16"):
        super().__init__()
        self.encoder = getattr(models, backbone)(weights="IMAGENET1K_V1")
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.encoder.heads.parameters():
            param.requires_grad = True #for only the last heads

    def forward(self, x):
        return self.encoder(x)

def reset_parameters(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu', generator=generator)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm1d):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

class Reconstructor(nn.Module):
    def __init__(self, out_features, in_features=in_features):
        super().__init__()
        hidden = 2048
        dropout = 0.3
        self.bn0 = nn.BatchNorm1d(in_features)
        self.fc1 = nn.Linear(in_features, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.do1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.do2 = nn.Dropout(dropout)
        self.fc_mean = nn.Linear(hidden, out_features)
        self.fc_overdispersion = nn.Linear(hidden, out_features)
        self.fc_probability = nn.Linear(hidden, out_features)

        reset_parameters(self)

    def forward(self, x):
        x = self.bn0(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.do1(x)
        x = F.relu(self.bn2(self.fc2(x)))        
        x = self.do2(x)
        mean = F.softplus(self.fc_mean(x))
        overdispersion = F.softplus(self.fc_overdispersion(x))
        probability = F.relu(self.fc_overdispersion(x))
        return mean, overdispersion, probability

class Classifier(nn.Module):
    def __init__(self, out_features, in_features=in_features):
        super().__init__()
        hidden = 2048
        dropout = 0.3
        self.bn0 = nn.BatchNorm1d(in_features)
        self.fc1 = nn.Linear(in_features, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.do1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.do2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden, out_features)

        reset_parameters(self)

    def forward(self, x):
        x = self.bn0(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.do1(x)
        x = F.relu(self.bn2(self.fc2(x)))        
        x = self.do2(x)
        x = self.fc3(x)
        return x

class Model(nn.Module):
    def __init__(self, n_genes, n_classes):
        super().__init__()
        self.encoder = Encoder()
        self.reconstructor = Reconstructor(out_features=n_genes)
        self.classifier = Classifier(out_features=n_classes)

    def forward(self, x):
        embedding = self.encoder(x)
        mean, overdispersion, probability = self.reconstructor(embedding)
        logits = self.classifier(embedding)
        return embedding, mean, overdispersion, probability, logits

class ZeroInflatedNegativeBinomialLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, mean, overdispersion, probability, target):
        mean = mean + self.eps
        overdispersion = overdispersion + self.eps
        probability = torch.clamp(probability, self.eps, 1-self.eps)

        total_count = 1.0 / overdispersion
        logits = mean.log() - (total_count + mean).log()
        nb = NegativeBinomial(total_count=total_count, logits=logits)
        
        log_zero_prob = torch.log(probability + (1 - probability) * torch.exp(nb.log_prob(torch.zeros_like(target)))) #if target == 0
        log_nonzero_prob = torch.log(1 - probability + self.eps) + nb.log_prob(target.float()) #if target > 0
        zero_mask = (target < self.eps).float()
        loss = - (zero_mask * log_zero_prob + (1 - zero_mask) * log_nonzero_prob)
        return loss.mean()

class Modeling():
    def __init__(self, args, config):
        self.directory = args.directory
        self.stem_file = config.stem_file
        self.cell_type = args.cell_type
        self.cell_types = config.cell_types
        self.palette_type = config.palette_type
        self.palette_subtype = config.palette_subtype
        self.palette = self.palette_subtype if self.cell_type == 'Cell_subtype_ST' else self.palette_type
        self.angles = config.angles
        self.angles_string = '_'.join(map(str, self.angles))
        self.upper = args.upper
        self.upper_string = f"upper{args.upper}"
        self.suffix = f"{self.upper_string}_{self.cell_type}_{self.angles_string}"

        dataset = Dataset(args, config)
        dataset.draw_umaps_expression()
        self.train_loader, self.test_loader = dataset.loader()

        self.genes = dataset.genes
        self.n_genes = len(self.genes)
        self.label_encoder = config.label_encoder
        self.classes = config.classes
        self.n_classes = len(self.classes)

        self.epochs = args.epochs
        self.lr = args.lr
        self.train = args.train
        self.model = Model(n_genes = self.n_genes, n_classes=self.n_classes)
        self.model.to(device)

        self.criterion_reconstruction = ZeroInflatedNegativeBinomialLoss()
        self.criterion_classification = nn.CrossEntropyLoss()

        self.images = None
        self.expressions = None
        self.embeddings = None
        self.reconstructions = None
        self.labels = None
        self.predictions = None

        self.load()

    def load(self):
        model_file = self.directory + f"models/model_{self.stem_file}_{self.suffix}.pth"
        if not os.path.isfile(model_file) or self.train:
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)

            gc.collect()
            torch.cuda.empty_cache()

            print(f"Training the model ...")
            escape = False
            best_loss_reconstruction = float('inf')
            best_loss_classification = float('inf')
            patience = 3
            counter = 0
            for epoch in range(self.epochs):
                self.model.train()
                train_loss_reconstruction = 0
                train_loss_classification = 0

                for cell_id, image, expression, label in tqdm(self.train_loader):
                    image = image.to(device, non_blocking=True)
                    expression = expression.to(device, non_blocking=True)
                    label = label.to(device, non_blocking=True)

                    embedding, mean, overdispersion, probability, logit = self.model(image)
                    if torch.isnan(embedding).any():
                        print("nan found in embedding")
                        escape = True
                        break

                    optimizer.zero_grad()
                    loss_reconstruction = self.criterion_reconstruction(mean, overdispersion, probability, expression)
                    loss_classification = self.criterion_classification(logit, label)
                    loss = loss_reconstruction + loss_classification
                    loss.backward()
                    optimizer.step()

                    train_loss_reconstruction += loss_reconstruction.detach()
                    train_loss_classification += loss_classification.detach()

                    del image, expression, label, embedding, mean, overdispersion, probability, logit

                if escape:
                    break

                average_loss_reconstruction = train_loss_reconstruction / len(self.train_loader)
                average_loss_classification = train_loss_classification / len(self.train_loader)
                print(f"Epoch: {epoch+1}/{self.epochs}, reconstruction loss: {average_loss_reconstruction:.5f}, classification loss: {average_loss_classification:.5f}")

                if epoch >= 20:
                    torch.save(self.model.state_dict(), model_file)

                    if best_loss_reconstruction == float('inf') or (best_loss_reconstruction - average_loss_reconstruction) / best_loss_reconstruction >= 0.001:
                        best_loss_reconstruction = average_loss_reconstruction
                    else:
                        counter += 1
                        print(f"Early stopping counter is updated as {counter}/{patience} by the reconstruction loss")

                    if best_loss_classification == float('inf') or (best_loss_classification - average_loss_classification) / best_loss_classification >= 0.001:
                        best_loss_classification = average_loss_classification
                    else:
                        counter += 1
                        print(f"Early stopping counter is updated as {counter}/{patience} by the classification loss")

                    if counter >= patience:
                        print("The training is stopped eraly.")
                        break

            gc.collect()
            torch.cuda.empty_cache()

        else: 
            print(f"Loading weights from {model_file} ...")
            self.model.load_state_dict(torch.load(model_file, map_location=device))

    def evaluate(self):
        self.model.eval()

        images = [] 
        expressions = []
        embeddings = []
        reconstructions = []
        labels = []
        predictions = []
        
        gc.collect()
        torch.cuda.empty_cache()

        test_loss_reconstruction = 0
        test_loss_classification = 0

        print(f"Evaluating the model ...")
        with torch.no_grad():
            for cell_id, image, expression, label in tqdm(self.test_loader):
                image = image.to(device, non_blocking=True)
                expression = expression.to(device, non_blocking=True)
                label = label.to(device, non_blocking=True)

                embedding, mean, overdispersion, probability, logit = self.model(image)
                prediction = torch.argmax(logit, dim=1)

                loss_reconstruction = self.criterion_reconstruction(mean, overdispersion, probability, expression)
                loss_classification = self.criterion_classification(logit, label)

                test_loss_reconstruction += loss_reconstruction.detach()
                test_loss_classification += loss_classification.detach()

                images.append(image.view(image.shape[0], -1).detach().cpu())
                expressions.append(expression.detach().cpu())
                labels.append(label.detach().cpu())
                embeddings.append(embedding.detach().cpu())
                reconstruction = mean
                reconstructions.append(reconstruction.detach().cpu())
                predictions.append(prediction.detach().cpu())

                del image, expression, label, embedding, mean, overdispersion, probability, logit, reconstruction, prediction
                
        print(f"Test Loss of reconstruction: {test_loss_reconstruction / len(self.test_loader):.5f}")
        print(f"Test Loss of classification: {test_loss_classification / len(self.test_loader):.5f}")

        self.images = torch.cat(images).numpy()
        self.expressions = torch.cat(expressions).numpy()
        self.embeddings = torch.cat(embeddings).numpy()
        self.reconstructions = torch.cat(reconstructions).numpy()
        self.labels = self.label_encoder.inverse_transform(
            torch.cat(labels).numpy()
        )
        self.predictions = self.label_encoder.inverse_transform(
            torch.cat(predictions).numpy()
        )

        del images, expressions, embeddings, reconstructions, labels, predictions
        
        gc.collect()
        torch.cuda.empty_cache()

    def draw_umaps_embedding(self):
        print("Standardizing ... ", end='')
        images_scaled, embeddings_scaled, expressions_scaled, reconstructions_scaled = thread_map(
            lambda x: StandardScaler().fit_transform(x), 
            [self.images, self.embeddings, self.expressions, self.reconstructions],
            max_workers=n_cores,
        )
        print(images_scaled.shape, embeddings_scaled.shape, expressions_scaled.shape, reconstructions_scaled.shape)

        print("2D umapping ... ", end='')
        image_umap, embedding_umap, expression_umap, reconstruction_umap = map(
            lambda x: UMAP(n_components=2, n_jobs=n_cores, low_memory=True, random_state=seed).fit_transform(x),
            [images_scaled, embeddings_scaled, expressions_scaled, reconstructions_scaled]
        )
        print(image_umap.shape, embedding_umap.shape, expression_umap.shape, reconstruction_umap.shape)

        umaps = {
            'H&E images': image_umap, 
            'embeddings': embedding_umap,
            'gene expressions': expression_umap, 
            'reconstructions': reconstruction_umap, 
        }

        fig, ax = plt.subplots(1, len(umaps), figsize=(13, 3))
        colors = [self.palette[cell_type] for cell_type in self.labels]
        
        for i, (title, coordinate) in enumerate(umaps.items()):
            ax[i].scatter(coordinate[:, 0], coordinate[:, 1], c=colors, s=1)
            ax[i].set_title(f"UMAP of {title}")
            ax[i].set_xticks([])
            ax[i].set_yticks([])
        
        markers = [plt.Line2D([0,0],[0,0], color=color, marker='o', linestyle='') for color in self.palette.values()]
        plt.legend(markers, self.palette.keys(), numpoints=1, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        fig.tight_layout()
        fig.savefig(self.directory + f"results/umaps_embedding_{self.stem_file}_{self.suffix}.png", bbox_inches="tight")
        plt.close()

    def draw_heatmap(self):
        adata_actual = anndata.AnnData(
            X = self.expressions,
            var = pd.DataFrame(index=self.genes),
            obs = pd.DataFrame({self.cell_type: self.labels})
        )
        adata_reconstruction = anndata.AnnData(
            X = self.reconstructions,
            var = pd.DataFrame(index=self.genes),
            obs = pd.DataFrame({self.cell_type: self.labels})
        )

        sc.pp.normalize_total(adata_actual, target_sum=1e4)
        sc.pp.log1p(adata_actual)
        adata_actual.obs[self.cell_type] = adata_actual.obs[self.cell_type].astype('category')

        sc.pp.normalize_total(adata_reconstruction, target_sum=1e4)
        sc.pp.log1p(adata_reconstruction)
        adata_reconstruction.obs[self.cell_type] = adata_reconstruction.obs[self.cell_type].astype('category')
        
        sc.tl.rank_genes_groups(adata_actual, groupby=self.cell_type)
        top_markers = []
        for cell_type in self.cell_types:
            top_markers.append(sc.get.rank_genes_groups_df(adata_actual, group=cell_type)['names'].values[:10].tolist())
        top_markers = sum(top_markers, [])
        
        sc.tl.dendrogram(adata_actual, groupby=self.cell_type)
        sc.tl.dendrogram(adata_reconstruction, groupby=self.cell_type)
        sc.tl.rank_genes_groups(adata_reconstruction, groupby=self.cell_type)

        cell_type_order = adata_actual.obs[self.cell_type].cat.categories.tolist()
        adata_actual.uns[self.cell_type+'_colors'] = [self.palette[cell_type] for cell_type in cell_type_order]
        adata_reconstruction.uns[self.cell_type+'_colors'] = [self.palette[cell_type] for cell_type in cell_type_order]

        sc.pl.rank_genes_groups_heatmap(adata_actual, var_names=top_markers, show=False, use_raw=False, dendrogram=True, show_gene_labels=True)
        plt.tight_layout()
        plt.savefig("/tmp/heatmap_actual.png")
        plt.close()
        sc.pl.rank_genes_groups_heatmap(adata_reconstruction, var_names=top_markers, show=False, use_raw=False, dendrogram=True, show_gene_labels=True)
        plt.tight_layout()
        plt.savefig("/tmp/heatmap_reconstruction.png")
        plt.close()

        fig, ax = plt.subplots(2, figsize=(12, 10))
        ax[0].imshow(mpimg.imread("/tmp/heatmap_actual.png"))
        ax[0].axis('off')
        ax[0].set_title("Actual")

        ax[1].imshow(mpimg.imread("/tmp/heatmap_reconstruction.png"))
        ax[1].axis('off')
        ax[1].set_title("Reconstructed")

        plt.tight_layout()
        plt.savefig(self.directory + f"results/expression_{self.stem_file}_{self.suffix}.png")
        plt.close()

    def draw_confusion_matrix(self):
        fig, ax = plt.subplots(1, 2, figsize=(len(self.classes)*2, len(self.classes)))

        cm = confusion_matrix(self.labels, self.predictions, labels=self.classes)
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=self.classes, yticklabels=self.classes, ax=ax[0])
        f1_weighted = f1_score(self.labels, self.predictions, average='weighted')
        print('Weighted F1 score', f1_weighted)
        ax[0].set_title(f"{self.cell_type} (weighted f1: {f1_weighted:.4f})")
        ax[0].set_xlabel('Prediction')
        ax[0].set_ylabel('Label')

        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_normalized, annot=True, fmt='.3f', xticklabels=self.classes, yticklabels=self.classes, ax=ax[1])
        f1_macro = f1_score(self.labels, self.predictions, average='macro')
        print('Macro F1 score', f1_macro)
        ax[1].set_title(f"{self.cell_type} (macro f1: {f1_macro:.4f})")
        ax[1].set_xlabel('Prediction')
        ax[1].set_ylabel('Label')

        fig.tight_layout()
        plt.savefig(self.directory + f"results/confusion_matrix_{self.stem_file}_{self.suffix}.png", bbox_inches="tight")
        plt.close()

    def infer(self, inference_loader):
        self.model.eval()
        cell_ids = []
        predictions = []
        
        print(f"Inferring with the {self.cell_type} prediction model ...")
        with torch.no_grad():
            for cell_id, image in tqdm(inference_loader):
                image = image.to(device, non_blocking=True)

                cell_ids.extend(cell_id)
                embedding = self.model.encoder(image)
                logit = self.model.classifier(embedding)
                prediction = torch.argmax(logit, dim=1)
                predictions.append(prediction.detach().cpu())

        predictions = torch.cat(predictions).numpy()
        predictions = self.label_encoder.inverse_transform(predictions)
            
        gc.collect()
        torch.cuda.empty_cache()

        return pd.Series(predictions, index=cell_ids)
