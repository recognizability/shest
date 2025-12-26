import os
import sys
import gc
from glob import glob
from collections import Counter
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map

import json
import numpy as np
from sklearn.metrics import f1_score
import pandas as pd

import anndata
import scanpy as sc
import tacco as tc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import NegativeBinomial
import timm

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

from config import n_cores, seed, set_seed, generator, device
from data import Load, Dataset

sc.settings.n_jobs = n_cores
in_features = 1536 #output features of H-optimus-0

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = timm.create_model("hf-hub:bioptimus/H-optimus-0", pretrained=True, init_values=1e-5, dynamic_img_size=True)
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
        x = F.log_softmax(x, dim=1)
        return x

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

class Model(nn.Module):
    def __init__(self, n_genes, n_classes):
        super().__init__()
        self.encoder = Encoder()
        self.classifier = Classifier(out_features=n_classes)
        self.reconstructor = Reconstructor(out_features=n_genes)

    def forward(self, x):
        embedding = self.encoder(x)
        log_prob = self.classifier(embedding)
        mean, overdispersion, probability = self.reconstructor(embedding.clone())
        return embedding, log_prob, mean, overdispersion, probability

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
    def __init__(self, args, config, data_object):
        self.directory = args.directory
        self.cell_type = args.cell_type
        self.cell_types = config.cell_types
        self.palette_type = config.palette_type
        self.palette_subtype = config.palette_subtype
        self.palette = self.palette_subtype if self.cell_type == 'cell_subtype_expression' else self.palette_type
        self.gene_panel = config.gene_panel
        self.n_genes = len(self.gene_panel)
        self.genes = getattr(data_object, 'genes', config.gene_panel)

        if isinstance(data_object, Dataset):
            self.pixel_sizes = data_object.pixel_sizes
        else:
            self.pixel_sizes = [data_object.pixel_size]

        if hasattr(args, 'split') and 0 < args.split < 1:
            self.train_loader, self.test_loader = data_object.loader()
            self.stem_file = data_object.stem_file
            self.centroids = data_object.test_centroids
            self.spatial = data_object.test_spatials
        else:
            self.test_loader = data_object.loader()
            self.stem_file = args.platform
            for source, sample in zip(args.sources, args.samples, strict=True):
                self.stem_file += f"_{source}_{sample}"
            self.centroids = data_object.centroids
            spatial = np.round(self.centroids * self.pixel_sizes[0]).astype(int)
            self.spatial = getattr(data_object, 'spatial', spatial)

        self.label_encoder = config.label_encoder
        self.classes = config.classes
        self.n_classes = len(self.classes)

        self.epochs = args.epochs
        self.lr = args.lr
        self.mode = args.mode
        self.model = Model(n_genes = self.n_genes, n_classes=self.n_classes)
        self.model.to(device)
        torch.compile(self.model, mode="reduce-overhead", dynamic=True)

        self.criterion_classification = nn.NLLLoss()
        self.criterion_reconstruction = ZeroInflatedNegativeBinomialLoss()

        self.images = None
        self.expressions = None
        self.embeddings = None
        self.reconstructions = None
        self.labels = None
        self.predictions = None
        self.confusion_matrix = None
        self.confusion_matrix_normalized = None
        self.f1_weighted = None
        self.f1_macro = None
        self.adata_actual = None
        self.adata_reconstructed = None
        self.corr = None
        self.adata_inferred = None

        self.load()
        if args.mode != 'infer':
            self.validate()

    def load(self):
        weights_file = self.directory + f"models/weights_{self.stem_file}_{self.cell_type}.pth"
        if (not os.path.isfile(weights_file)) or self.mode=='train':
            optimizer = torch.optim.AdamW(
                list(self.model.reconstructor.parameters()) + list(self.model.classifier.parameters()),
                lr=self.lr
            )

            gc.collect()
            torch.cuda.empty_cache()

            print(f"Training the model ...")
            escape = False
            best_loss_classification = float('inf')
            best_loss_reconstruction = float('inf')
            patience = 5
            counter = 0
            scaler = torch.amp.GradScaler(device.type)
            for epoch in range(self.epochs):
                self.model.train()
                train_loss_classification = 0
                train_loss_reconstruction = 0

                for cell_id, image, expression, label in tqdm(self.train_loader):
                    image = image.to(device, non_blocking=True)
                    expression = expression.to(device, non_blocking=True)
                    label = label.to(device, non_blocking=True)

                    optimizer.zero_grad()
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                        embedding, log_prob, mean, overdispersion, probability = self.model(image)

                        loss_classification = self.criterion_classification(log_prob, label)
                        loss_reconstruction = self.criterion_reconstruction(mean, overdispersion, probability, expression)

                        tensors = [embedding, mean, overdispersion, probability, loss_classification, loss_reconstruction]
                        names = ["embedding", "mean", "overdispersion", "probability", "loss_classification", "loss_reconstruction"]
                        for tensor, name in zip(tensors, names):
                            if torch.isnan(tensor).any():
                                print(f"nan found in {name}")
                                escape = True
                                break
                            elif torch.isinf(tensor).any():
                                print(f"inf found in {name}")
                                escape = True
                                break

                        loss = loss_classification + loss_reconstruction

                    del image, expression, label, embedding, log_prob, mean, overdispersion, probability
                    torch.cuda.empty_cache()

                    if escape:
                        break

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    train_loss_classification += loss_classification.detach()
                    train_loss_reconstruction += loss_reconstruction.detach()

                    del loss, loss_reconstruction, loss_classification
                    torch.cuda.empty_cache()

                if escape:
                    break

                average_loss_classification = train_loss_classification / len(self.train_loader)
                average_loss_reconstruction = train_loss_reconstruction / len(self.train_loader)
                print(f"Epoch: {epoch+1}/{self.epochs}, classification loss: {average_loss_classification:.5f}, reconstruction loss: {average_loss_reconstruction:.5f}")

                epsilon = 0.0001
                if best_loss_classification == float('inf') or (best_loss_classification - average_loss_classification) / best_loss_classification >= epsilon:
                    best_loss_classification = average_loss_classification
                else:
                    counter += 1
                    print(f"Early stopping counter is updated as {counter}/{patience} by the classification loss")

                if best_loss_reconstruction == float('inf') or (best_loss_reconstruction - average_loss_reconstruction) / best_loss_reconstruction >= epsilon:
                    best_loss_reconstruction = average_loss_reconstruction
                else:
                    counter += 1
                    print(f"Early stopping counter is updated as {counter}/{patience} by the reconstruction loss")

                if counter >= patience:
                    print("The training is stopped eraly.")
                    break

            if not escape:
                print(f"Saving the weights into {weights_file} ... ", end='')
                torch.save(self.model.state_dict(), weights_file)
                print("done.")

            gc.collect()
            torch.cuda.empty_cache()

        elif self.mode in ['test', 'infer']: 
            print(f"Loading the weights from {weights_file} ... ", end='')
            self.model.load_state_dict(torch.load(weights_file, map_location=device))
            print("done.")

    def validate(self, test_loader=None):
        if test_loader is None:
            test_loader = self.test_loader

        self.model.eval()

        expressions = []
        labels = []
        predictions = []
        embeddings = []
        reconstructions = []
        
        gc.collect()
        torch.cuda.empty_cache()

        if self.mode in ['train', 'test']:
            test_loss_reconstruction = 0
            test_loss_classification = 0

        print(f"Validating the dataset by the model ...")
        with torch.no_grad():
            cell_ids = []
            for cell_id, image, expression, label in tqdm(test_loader):
                cell_ids.extend(cell_id)
                image = image.to(device, non_blocking=True)
                expression = expression.to(device, non_blocking=True)
                label = label.to(device, non_blocking=True)

                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    embedding, log_prob, mean, overdispersion, probability = self.model(image)
                    prediction = torch.argmax(log_prob, dim=1)

                    if self.mode in ['train', 'test']:
                        loss_classification = self.criterion_classification(log_prob, label)
                        loss_reconstruction = self.criterion_reconstruction(mean, overdispersion, probability, expression)
                        test_loss_classification += loss_classification.detach()
                        test_loss_reconstruction += loss_reconstruction.detach()

                expressions.append(expression.detach().cpu())
                labels.append(label.detach().cpu())
                embeddings.append(embedding.detach().cpu())
                predictions.append(prediction.detach().cpu())
                reconstruction = (1 - probability) * mean
                reconstructions.append(reconstruction.detach().cpu())

                del image, expression, label, embedding, log_prob, mean, overdispersion, probability, prediction, reconstruction
                
        if self.mode in ['train', 'test']:
            print(f"Test Loss of classification: {test_loss_classification / len(test_loader):.5f}")
            print(f"Test Loss of reconstruction: {test_loss_reconstruction / len(test_loader):.5f}")

        self.expressions = torch.cat(expressions).numpy()
        self.labels = self.label_encoder.inverse_transform(torch.cat(labels).numpy())

        self.embeddings = torch.cat(embeddings).to(torch.float32).cpu().numpy()
        self.predictions = self.label_encoder.inverse_transform(torch.cat(predictions).numpy())
        self.reconstructions = torch.round(torch.cat(reconstructions)).int().numpy() #integer type
        
        gc.collect()
        torch.cuda.empty_cache()
        del expressions, labels, embeddings, predictions, reconstructions

        self.labels = pd.Categorical(self.labels, categories=self.cell_types.keys(), ordered=True)
        self.predictions = pd.Categorical(self.predictions, categories=self.cell_types.keys(), ordered=True)

        self.confusion_matrix = pd.crosstab(self.labels, self.predictions)
        self.confusion_matrix_normalized = pd.crosstab(self.labels, self.predictions, normalize='index')

        self.f1_weighted = f1_score(self.labels, self.predictions, average='weighted')
        print(f'Weighted F1 score: {self.f1_weighted}')
        self.f1_macro = f1_score(self.labels, self.predictions, average='macro')
        print(f'Macro F1 score: {self.f1_macro}')

        self.adata_actual = anndata.AnnData(
            X = self.expressions,
            var = pd.DataFrame(index=self.genes),
            obs = pd.DataFrame({'cell_type_annotated': self.labels}, index=cell_ids),
            obsm = {"pixel": self.centroids, "spatial": self.spatial},
        )
        self.adata_actual.uns['cell_type_annotated_colors'] = [self.palette[ct] for ct in self.cell_types.keys()]

        self.adata_reconstructed = anndata.AnnData(
            X = self.reconstructions,
            var = pd.DataFrame(index=self.gene_panel),
            obs = pd.DataFrame({'cell_type_predicted': self.predictions}, index=cell_ids),
            obsm = {"pixel": self.centroids, "spatial": self.spatial},
        )
        self.adata_reconstructed.uns['cell_type_predicted_colors'] = [self.palette[ct] for ct in self.cell_types.keys()]
        self.adata_reconstructed.obsm['embeddings'] = self.embeddings

        common_genes = sorted(list(set(self.adata_actual.var_names) & set(self.adata_reconstructed.var_names)))
        self.corr = np.array([np.corrcoef(
            np.ravel(self.adata_actual[:, gene].X.toarray()),
            np.ravel(self.adata_reconstructed[:, gene].X.toarray()),
        )[0, 1] for gene in common_genes])
        self.corr = pd.DataFrame(self.corr, index=common_genes, columns=['Pearson_R']).sort_values(by='Pearson_R', ascending=False).dropna()

        self.adata_actual.write_h5ad(self.directory + f"results/adata_actual_{self.stem_file}.h5ad")
        self.adata_reconstructed.write_h5ad(self.directory + f"results/adata_reconstructed_{self.stem_file}.h5ad")

    def draw(self):
        adata_actual = self.adata_actual.copy()
        sc.pp.normalize_total(adata_actual, target_sum=1e4)
        sc.pp.log1p(adata_actual)
        sc.tl.rank_genes_groups(adata_actual, groupby='cell_type_annotated', method='wilcoxon')

        adata_reconstructed = self.adata_reconstructed.copy()
        sc.pp.normalize_total(adata_reconstructed, target_sum=1e4)
        sc.pp.log1p(adata_reconstructed)
        sc.tl.rank_genes_groups(adata_reconstructed, groupby="cell_type_predicted", method='wilcoxon')

        top_markers = {}
        top_markers_df = pd.DataFrame()
        for ct in self.cell_types.keys():
            if ct in adata_actual.obs['cell_type_annotated'].unique():
                df = sc.get.rank_genes_groups_df(adata_actual, group=ct).iloc[:10]
                df[self.cell_type] = ct
                top_markers[ct] = df["names"].values.tolist()
                top_markers_df = pd.concat([top_markers_df, df], ignore_index=True)
        top_markers_df.to_csv(self.directory + f"results/top_markers_{self.stem_file}_{self.cell_type}.csv", index=False)

        fontsize = 20
        fontweight = 'bold'

        fig = plt.figure(figsize=(14, 11))
        gs = GridSpec(4, 4, figure=fig, width_ratios=(5, 5, 2, 2), height_ratios=(6, 0.01, 5, 5), hspace=0.6, wspace=0.4)
        ax_cm = fig.add_subplot(gs[0, 0])
        ax_cm_norm = fig.add_subplot(gs[0, 1])
        ax_f1 = fig.add_subplot(gs[0, 2])
        ax_r = fig.add_subplot(gs[0, 3])
        ax_actual = fig.add_subplot(gs[2, :])
        ax_reconstructed = fig.add_subplot(gs[3, :])

        sns.heatmap(self.confusion_matrix, annot=True, fmt='d', ax=ax_cm)
        ax_cm.set_xlabel('Prediction')
        ax_cm.set_ylabel('Annotation')

        sns.heatmap(self.confusion_matrix_normalized, annot=True, fmt='.2f', ax=ax_cm_norm)
        ax_cm_norm.set_xlabel('Prediction')
        ax_cm_norm.set_ylabel('Annotation')
        ax_cm_norm.set_yticklabels([])

        f1s = pd.DataFrame({'f1_weighted':[round(self.f1_weighted, 2)], 'f1_macro':[round(self.f1_macro, 2)]})
        sns.barplot(f1s, ax=ax_f1, color='gray')
        ax_f1.set_ylim(0, f1s.max(axis=None)*1.1)
        for container in ax_f1.containers:
            ax_f1.bar_label(container)
        ax_f1.tick_params(axis='x', labelrotation=90)

        sns.violinplot(self.corr, ax=ax_r, color='gray', split=True, hue_order=[False, True])
        for l in ax_r.lines[2::3]:
            x, y = l.get_data()
            if len(x) > 0 and len(y) > 0:
                ax_r.text(x[0] - 0.1, y[0], f'{y[0]:.2f}', ha='right', va='center', color='white')

        sc.pl.rank_genes_groups_dotplot(adata_actual, var_names=top_markers, show=False, use_raw=False, dendrogram=False, var_group_rotation=0, ax=ax_actual)
        ax_actual.set_title("Actual")
        sc.pl.rank_genes_groups_dotplot(adata_reconstructed, var_names=top_markers, show=False, use_raw=False, dendrogram=False, var_group_rotation=0, ax=ax_reconstructed)
        ax_reconstructed.set_title("Reconstructed")

        axes = [ax_cm, ax_cm_norm, ax_f1, ax_r, ax_actual, ax_reconstructed]
        labels = [chr(ord('A') + i) for i in range(len(axes))]
        for ax_item, label in zip(axes, labels):
            ax_item.text(0, 1, label, transform=ax_item.transAxes, fontsize=fontsize, fontweight=fontweight, va='bottom', ha='right')

        dpi=300
        plt.savefig(self.directory + f"results/evaluation_{self.stem_file}_{self.cell_type}_dpi{dpi}.png", bbox_inches="tight", dpi=dpi)

    def infer(self, test_loader=None):
        if test_loader is None:
            test_loader = self.test_loader
        self.model.eval()
        cell_ids = []
        embeddings = []
        predictions = []
        prediction_probabilities = []
        reconstructions = []
        
        print(f"Inferring with the {self.cell_type} prediction model ...")
        with torch.no_grad():
            for cell_id, image in tqdm(test_loader):
                cell_ids.extend(cell_id)
                image = image.to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    embedding, log_prob, mean, overdispersion, probability = self.model(image)
                embeddings.append(embedding.detach().cpu())
                prediction_probability, prediction = torch.max(log_prob, dim=1)
#                prediction[prediction_probability < 0.9] = -1
                predictions.append(prediction.detach().cpu())
                reconstruction = (1 - probability) * mean
                reconstructions.append(reconstruction.detach().cpu())

                del image, embedding, log_prob, mean, overdispersion, probability

        embeddings = torch.cat(embeddings).numpy()
        predictions = torch.cat(predictions).numpy()
        reconstructions = torch.round(torch.cat(reconstructions)).int().numpy() #integer type

        gc.collect()
        torch.cuda.empty_cache()

        mask = predictions == -1
        predictions_masked = np.empty(predictions.shape, dtype=object)
        predictions_masked[mask] = np.nan
        predictions_masked[~mask] = self.label_encoder.inverse_transform(predictions[~mask])
        self.adata_inferred = anndata.AnnData(
            X = reconstructions,
            var = pd.DataFrame(index=self.gene_panel),
            obs = pd.DataFrame({self.cell_type: predictions_masked}, index=cell_ids),
            obsm = {"pixel": self.centroids, "spatial": self.spatial},
        )
        self.adata_inferred.obs[self.cell_type] = pd.Categorical(self.adata_inferred.obs[self.cell_type], categories=self.cell_types.keys(), ordered=True)
        self.adata_inferred.uns[self.cell_type+'_colors'] = [self.palette[ct] for ct in self.cell_types.keys()]
        self.adata_inferred.obsm['embeddings'] = embeddings
        self.adata_inferred.write_h5ad(self.directory + f"results/adata_inferred_{self.stem_file}_{self.cell_type}.h5ad")
