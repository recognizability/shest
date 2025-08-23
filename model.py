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
import timm

import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from config import n_cores, seed, set_seed, generator, device
from preprocess import Preprocessing

sc.settings.n_jobs = n_cores
in_features = 1536 #output features of H-optimus-0

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = timm.create_model(
            "hf-hub:bioptimus/H-optimus-0", pretrained=True, init_values=1e-5, dynamic_img_size=True
        )
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.encoder(x)

def reset_parameters(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu', generator=generator)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm1d):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

class Reconstructor(nn.Module):
    def __init__(self, out_features, in_features=in_features, reduction=16):
        super().__init__()
        hidden = 2048
        dropout = 0.3
        self.norm0 = nn.LayerNorm(in_features)
        self.fc1 = nn.Linear(in_features, hidden)
        self.norm1 = nn.LayerNorm(hidden)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, hidden*2)
        self.norm2 = nn.LayerNorm(hidden*2)
        self.dropout2 = nn.Dropout(dropout)
        self.fc_mean = nn.Linear(hidden*2, out_features)
        self.fc_overdispersion = nn.Linear(hidden*2, out_features)
        self.fc_probability = nn.Linear(hidden*2, out_features)

        reset_parameters(self)

    def forward(self, x):
        x = self.norm0(x)
        x = F.relu(self.norm1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.norm2(self.fc2(x)))        
        x = self.dropout2(x)
        mean = F.softplus(self.fc_mean(x))
        overdispersion = F.softplus(self.fc_overdispersion(x))
        probability = F.sigmoid(self.fc_probability(x))
        return mean, overdispersion, probability

class Classifier(nn.Module):
    def __init__(self, out_features, in_features=in_features):
        super().__init__()
        hidden = 1024
        dropout = 0.3
        self.norm0 = nn.LayerNorm(in_features)
        self.fc1 = nn.Linear(in_features, hidden)
        self.norm1 = nn.LayerNorm(hidden)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, hidden//2)
        self.norm2 = nn.LayerNorm(hidden//2)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden//2, out_features)

        reset_parameters(self)

    def forward(self, x):
        x = self.norm0(x)
        x = F.relu(self.norm1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.norm2(self.fc2(x)))        
        x = self.dropout2(x)
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
    def __init__(self, stability=1e-256):
        super().__init__()
        self.stability = stability

    def forward(self, mean, overdispersion, probability, target):
        mean = mean + self.stability
        probs = mean / (mean + overdispersion)
        nb = NegativeBinomial(total_count=overdispersion, probs=probs)

        zero_case_log_likelihood = torch.log(probability + (1 - probability) * torch.exp(nb.log_prob(torch.zeros_like(target)))) #if target == 0
        non_zero_case_log_likelihood = torch.log(1 - probability + self.stability) + nb.log_prob(target) #if target > 0
        zinb_log_likelihood = torch.where(target < self.stability, zero_case_log_likelihood, non_zero_case_log_likelihood)
        return -torch.mean(zinb_log_likelihood)

class Modeling():
    def __init__(self, args, config, dataset=None, stem_file=None):
        self.directory = args.directory
        self.stem_file = stem_file
        self.cell_type = args.cell_type
        self.cell_types = config.cell_types
        self.palette_type = config.palette_type
        self.palette_subtype = config.palette_subtype
        self.palette = self.palette_subtype if self.cell_type == 'cell_subtype_expression' else self.palette_type

        if dataset is not None or args.mode == 'train' or args.mode == 'test':
#            dataset.draw_umaps_expression()
            self.train_loader, self.test_loader = dataset.loader()

            self.genes = dataset.genes
            self.n_genes = len(self.genes)
        else:
            self.n_genes = 5001
            
        self.label_encoder = config.label_encoder
        self.classes = config.classes
        self.n_classes = len(self.classes)

        self.epochs = args.epochs
        self.lr = args.lr
        self.mode = args.mode
        self.model = Model(n_genes = self.n_genes, n_classes=self.n_classes)
        self.model.to(device)
        torch.compile(self.model, mode="reduce-overhead", dynamic=True)

        self.criterion_reconstruction = ZeroInflatedNegativeBinomialLoss()
        self.criterion_classification = nn.CrossEntropyLoss()

        self.images = None
        self.expressions = None
        self.embeddings = None
        self.reconstructions = None
        self.labels = None
        self.predictions = None

        self.gene_panel = config.gene_panel

        self.load()

    def load(self):
        model_file = self.directory + f"models/model_{self.stem_file}_{self.cell_type}.pth"
        weights_file = self.directory + f"models/weights_{self.stem_file}_{self.cell_type}.pth"
        if not os.path.isfile(weights_file) or self.mode=='train':
            optimizer = torch.optim.AdamW(
                list(self.model.reconstructor.parameters()) + list(self.model.classifier.parameters()),
                lr=self.lr
            )

            gc.collect()
            torch.cuda.empty_cache()

            print(f"Training the model ...")
            escape = False
            best_loss_reconstruction = float('inf')
            best_loss_classification = float('inf')
            patience = 5
            counter = 0
            scaler = torch.amp.GradScaler(device.type)
            for epoch in range(self.epochs):
                self.model.train()
                train_loss_reconstruction = 0
                train_loss_classification = 0

                for cell_id, image, expression, label in tqdm(self.train_loader):
                    image = image.to(device, non_blocking=True)
                    expression = expression.to(device, non_blocking=True)
                    label = label.to(device, non_blocking=True)

                    optimizer.zero_grad()
                    with torch.autocast(device_type=device.type, dtype=torch.float16):
                        embedding, mean, overdispersion, probability, logit = self.model(image)

                        loss_reconstruction = self.criterion_reconstruction(mean, overdispersion, probability, expression)
                        loss_classification = self.criterion_classification(logit, label)

                        tensors = [embedding, mean, overdispersion, probability, loss_reconstruction, loss_classification]
                        names = ["embedding", "mean", "overdispersion", "probability", "loss_reconstruction", "loss_classification"]
                        for tensor, name in zip(tensors, names):
                            if torch.isnan(tensor).any():
                                print(f"nan found in {name}")
                                escape = True
                                break
                            elif torch.isinf(tensor).any():
                                print(f"inf found in {name}")
                                escape = True
                                break

                        loss = loss_reconstruction + loss_classification

                    del image, expression, label
                    torch.cuda.empty_cache()

                    if escape:
                        break

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    train_loss_reconstruction += loss_reconstruction.detach()
                    train_loss_classification += loss_classification.detach()

                    del embedding, mean, overdispersion, probability, logit, loss, loss_reconstruction, loss_classification
                    torch.cuda.empty_cache()
                if escape:
                    break

                average_loss_reconstruction = train_loss_reconstruction / len(self.train_loader)
                average_loss_classification = train_loss_classification / len(self.train_loader)
                print(f"Epoch: {epoch+1}/{self.epochs}, reconstruction loss: {average_loss_reconstruction:.5f}, classification loss: {average_loss_classification:.5f}")

                epsilon = 0.0001
                if best_loss_reconstruction == float('inf') or (best_loss_reconstruction - average_loss_reconstruction) / best_loss_reconstruction >= epsilon:
                    best_loss_reconstruction = average_loss_reconstruction
                else:
                    counter += 1
                    print(f"Early stopping counter is updated as {counter}/{patience} by the reconstruction loss")

                if best_loss_classification == float('inf') or (best_loss_classification - average_loss_classification) / best_loss_classification >= epsilon:
                    best_loss_classification = average_loss_classification
                else:
                    counter += 1
                    print(f"Early stopping counter is updated as {counter}/{patience} by the classification loss")

                if counter >= patience:
                    print("The training is stopped eraly.")
                    break

            if not escape:
                torch.save(self.model.state_dict(), weights_file)
                torch.save(self.model, model_file)

            gc.collect()
            torch.cuda.empty_cache()

        else: 
            print(f"Loading weights from {model_file} ...")
            self.model.load_state_dict(torch.load(weights_file, map_location=device))

    def evaluate(self, test_loader=None):
        if test_loader is None:
            test_loader = self.test_loader

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
            for cell_id, image, expression, label in tqdm(test_loader):
                image = image.to(device, non_blocking=True)
                expression = expression.to(device, non_blocking=True)
                label = label.to(device, non_blocking=True)

                with torch.autocast(device_type=device.type, dtype=torch.float16):
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
                
        print(f"Test Loss of reconstruction: {test_loss_reconstruction / len(test_loader):.5f}")
        print(f"Test Loss of classification: {test_loss_classification / len(test_loader):.5f}")

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
        fig.savefig(self.directory + f"results/umaps_embedding_{self.stem_file}_{self.cell_type}.png", bbox_inches="tight")
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
        adata_actual.obs[self.cell_type] = adata_actual.obs[self.cell_type].astype(pd.CategoricalDtype(categories=self.cell_types, ordered=True))

        sc.pp.normalize_total(adata_reconstruction, target_sum=1e4)
        sc.pp.log1p(adata_reconstruction)
        adata_reconstruction.obs[self.cell_type] = adata_reconstruction.obs[self.cell_type].astype(pd.CategoricalDtype(categories=self.cell_types, ordered=True))
        
        cell_type_order = adata_actual.obs[self.cell_type].cat.categories.tolist()
        adata_actual.uns[self.cell_type+'_colors'] = [self.palette[cell_type] for cell_type in cell_type_order]
        adata_reconstruction.uns[self.cell_type+'_colors'] = [self.palette[cell_type] for cell_type in cell_type_order]

        adata_actual = adata_actual[adata_actual.obs[self.cell_type].map(adata_actual.obs[self.cell_type].value_counts() > 1)]
        sc.tl.rank_genes_groups(adata_actual, groupby=self.cell_type, method='wilcoxon')
        adata_reconstruction = adata_reconstruction[adata_reconstruction.obs[self.cell_type].map(adata_reconstruction.obs[self.cell_type].value_counts() > 1)]
        sc.tl.rank_genes_groups(adata_reconstruction, groupby=self.cell_type, method='wilcoxon')

        top_markers = {}
        top_markers_df = pd.DataFrame()
        for cell_type in self.cell_types:
            if cell_type in adata_actual.obs[self.cell_type].unique():
                df = sc.get.rank_genes_groups_df(adata_actual, group=cell_type).iloc[:10]
                df[self.cell_type] = cell_type
                top_markers[cell_type] = df["names"].values.tolist()
                top_markers_df = pd.concat([top_markers_df, df], ignore_index=True)
        top_markers_df.to_csv(self.directory + f"results/top_markers_{self.stem_file}_{self.cell_type}.csv", index=False)

        sc.pl.rank_genes_groups_heatmap(adata_actual, var_names=top_markers, show=False, use_raw=False, dendrogram=False, show_gene_labels=True, var_group_rotation=0)
        plt.tight_layout()
        plt.savefig("/tmp/heatmap_actual.png")
        plt.close()
        sc.pl.rank_genes_groups_heatmap(adata_reconstruction, var_names=top_markers, show=False, use_raw=False, dendrogram=False, show_gene_labels=True, var_group_rotation=0)
        plt.tight_layout()
        plt.savefig("/tmp/heatmap_reconstruction.png")
        plt.close()
        fig, ax = plt.subplots(2, figsize=(12, 8))
        ax[0].imshow(mpimg.imread("/tmp/heatmap_actual.png"))
        ax[0].axis('off')
        ax[0].set_title("Actual")
        ax[1].imshow(mpimg.imread("/tmp/heatmap_reconstruction.png"))
        ax[1].axis('off')
        ax[1].set_title("Reconstructed")
        plt.tight_layout()
        plt.savefig(self.directory + f"results/expression_heatmap_{self.stem_file}_{self.cell_type}.png")
        plt.close()

        fig, ax = plt.subplots(2, figsize=(14, 7))
        sc.pl.rank_genes_groups_dotplot(adata_actual, var_names=top_markers, show=False, use_raw=False, dendrogram=False, var_group_rotation=0, ax=ax[0])
        ax[0].set_title("Actual")
        sc.pl.rank_genes_groups_dotplot(adata_reconstruction, var_names=top_markers, show=False, use_raw=False, dendrogram=False, var_group_rotation=0, ax=ax[1])
        ax[1].set_title("Reconstructed")
        plt.tight_layout()
        plt.savefig(self.directory + f"results/exression_dotplot_{self.stem_file}_{self.cell_type}.png")
        plt.close()

    def draw_confusion_matrix(self):
        fig, ax = plt.subplots(1, 2, figsize=(len(self.classes)*2, len(self.classes)))

        cm = confusion_matrix(self.labels, self.predictions, labels=self.classes)
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=self.classes, yticklabels=self.classes, ax=ax[0])
        f1_weighted = f1_score(self.labels, self.predictions, average='weighted')
        print(f'Weighted F1 score: {f1_weighted:.5f}')
        ax[0].set_title(f"{self.cell_type} (weighted f1: {f1_weighted:.4f})")
        ax[0].set_xlabel('Prediction')
        ax[0].set_ylabel('Label')

        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_normalized, annot=True, fmt='.3f', xticklabels=self.classes, yticklabels=self.classes, ax=ax[1])
        f1_macro = f1_score(self.labels, self.predictions, average='macro')
        print(f'Macro F1 score: {f1_macro:.5f}')
        ax[1].set_title(f"{self.cell_type} (macro f1: {f1_macro:.4f})")
        ax[1].set_xlabel('Prediction')
        ax[1].set_ylabel('Label')

        fig.tight_layout()
        plt.savefig(self.directory + f"results/confusion_matrix_{self.stem_file}_{self.cell_type}.png", bbox_inches="tight")
        plt.close()

    def infer(self, inference_loader):
        self.model.eval()
        cell_ids = []
        predictions = []
        expressions = []
        
        print(f"Inferring with the {self.cell_type} prediction model ...")
        with torch.no_grad():
            for cell_id, image in tqdm(inference_loader):
                image = image.to(device, non_blocking=True)

                cell_ids.extend(cell_id)
                with torch.autocast(device_type=device.type, dtype=torch.float16):
                    embedding, mean, overdispersion, probability, logit = self.model(image)
                probability, prediction = torch.max(torch.softmax(logit, dim=1), dim=1)
                prediction[probability < 0.9] = -1
                predictions.append(prediction.detach().cpu())
                expressions.append(mean.detach().cpu())

        predictions = torch.cat(predictions).numpy()
        mask = predictions == -1
        predictions_masked = np.empty(predictions.shape, dtype=object)
        predictions_masked[mask] = np.nan
        predictions_masked[~mask] = self.label_encoder.inverse_transform(predictions[~mask])
        adata = anndata.AnnData(
            X = torch.cat(expressions).numpy(),
            var = pd.DataFrame(index=self.gene_panel),
            obs = pd.DataFrame({self.cell_type: predictions_masked}, index=cell_ids)
        )
            
        gc.collect()
        torch.cuda.empty_cache()

        return adata
