import os
import sys
import gc
from tqdm import tqdm

import numpy as np
import pandas as pd
import anndata
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import NegativeBinomial
import torchvision.models as models

from umap import UMAP
from sklearn.preprocessing import StandardScaler

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, f1_score

from config import seed, set_seed, generator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
in_features = 768 #output features of ViT or SwinTransformer

class Encoder(nn.Module):
    def __init__(self, backbone="vit_b_16"):
        super().__init__()
        self.encoder = getattr(models, backbone)(weights="IMAGENET1K_V1")

    def forward(self, x):
        x = self.encoder._process_input(x)
        batch_class_token = self.encoder.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = self.encoder.encoder(x)
        x = x[:, 0] #CLS token
        return x

class Decoder(nn.Module):
    def __init__(self, out_features, in_features=in_features):
        super().__init__()
        self.hidden = 2048
        self.fc1 = nn.Linear(in_features, self.hidden)
        self.bn1 = nn.BatchNorm1d(self.hidden)
        self.fc2 = nn.Linear(self.hidden, self.hidden)
        self.bn2 = nn.BatchNorm1d(self.hidden)
#        self.fc_mu = nn.Linear(self.hidden, out_features)
#        self.fc_alpha = nn.Linear(self.hidden, out_features)
        self.fc3 = nn.Linear(self.hidden, out_features)

        self.reset_parameters()

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))        
#        mu = F.softplus(self.fc_mu(x)) + 1e-4
#        alpha = F.softplus(self.fc_alpha(x)) + 1e-4
#        return mu, alpha
        x = F.relu(self.fc3(x))
        return x

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu', generator=generator)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

class Reconstructor(nn.Module):
    def __init__(self, out_features):
        super().__init__()
        self.out_features = out_features
        self.encoder = Encoder()
        self.decoder = Decoder(out_features=self.out_features)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

#class NegativeBinomialLoss(nn.Module):
#    def __init__(self, eps=1e-8):
#        super().__init__()
#        self.eps = eps
#
#    def forward(self, mu, alpha, target):
#        mu = torch.clamp(mu, min=self.eps)
#        alpha = torch.clamp(alpha, min=self.eps)
#        target = target.float()
#        total_count = 1.0 / alpha
#        logits = (mu + self.eps).log() - (total_count + mu + self.eps).log()
#        nb = NegativeBinomial(total_count=total_count, logits=logits)
#        loss = -nb.log_prob(target)
#        return loss.mean()

class Reconstruction():
    def __init__(self, adata, platform, sample, he, cell_type):
        self.adata = adata
        self.reconstructor = Reconstructor(out_features=adata.shape[1])

        self.platform = platform
        self.sample = sample
        self.he = he
        self.cell_type = cell_type

        self.images = None
        self.cell_labels = None
        self.expressions = None
        self.embeddings = None
        self.reconstructions = None
        self.cell_labels = None

#        self.criterion = NegativeBinomialLoss()
        self.criterion = nn.HuberLoss()

    def load(self, train_loader, epochs, lr, train=False):
        reconstructor_file = f"/data0/crp/models/reconstructor_{self.platform}_{self.sample}_{self.he}_{self.cell_type}.pth"
        if not os.path.isfile(reconstructor_file) or train:
            self.reconstructor.to(device)
            
            optimizer = torch.optim.Adam(self.reconstructor.parameters(), lr=lr, weight_decay=1e-5)

            gc.collect()
            torch.cuda.empty_cache()
            
            print(f"Training the reconstruction model ...")
            for epoch in range(epochs):
                self.reconstructor.train()
                train_loss = 0
                for cell_id, image in tqdm(train_loader):
                    image = image.to(device, non_blocking=True)
                    expression = torch.as_tensor(
                        self.adata[cell_id, :].X.toarray(), dtype=torch.float32, device=device
                    )
                    
                    embedding = self.reconstructor.encoder(image)
                    if torch.isnan(embedding).any():
                        print("NaN in embedding")
                        sys.exit()
#                    mu, alpha = self.reconstructor.decoder(embedding)
                    reconstruction = self.reconstructor.decoder(embedding)

                    optimizer.zero_grad()
#                    loss = self.criterion(mu, alpha, expression)
                    loss = self.criterion(reconstruction, expression)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                    
                print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss / len(train_loader):.5f}")
            
            gc.collect()
            torch.cuda.empty_cache()

            torch.save(self.reconstructor.state_dict(), reconstructor_file)
        else: 
            self.reconstructor.load_state_dict(torch.load(reconstructor_file, map_location=device))

    def evaluate(self, test_loader):
        self.reconstructor.to(device)
        self.reconstructor.eval()

        images = [] 
        expressions = []
        embeddings = []
        reconstructions = []
        cell_labels = []
        
        gc.collect()
        torch.cuda.empty_cache()

        test_loss = 0

        print(f"Evaluating the reconstruction model ...")
        with torch.no_grad():
            for cell_id, image in tqdm(test_loader):
                image = image.to(device, non_blocking=True)
                images.append(image.view(image.shape[0], -1))

                expression = torch.as_tensor(
                    self.adata[cell_id, :].X.toarray(), dtype=torch.float32, device=device
                )
                expressions.append(expression)
                
                embedding = self.reconstructor.encoder(image)
                embeddings.append(embedding)

#                mu, alpha = self.reconstructor.decoder(embedding)
#                reconstructions.append(mu)
                reconstruction = self.reconstructor.decoder(embedding)
                reconstructions.append(reconstruction)

                cell_labels.extend(self.adata[cell_id, :].obs['Cell_type'].values.tolist())

#                loss = self.criterion(mu, alpha, expression)
                loss = self.criterion(reconstruction, expression)
                test_loss += loss.item()
                
        print(f"Test Loss: {test_loss / len(test_loader):.5f}")

        gc.collect()
        torch.cuda.empty_cache()

        images = torch.cat(images).detach().cpu().numpy()
        expressions = torch.cat(expressions).detach().cpu().numpy()
        embeddings = torch.cat(embeddings).detach().cpu().numpy()
        reconstructions = torch.cat(reconstructions).detach().cpu().numpy()

        self.images = images
        self.expressions = expressions
        self.embeddings = embeddings
        self.reconstructions = reconstructions
        self.cell_labels = cell_labels
        
    def _apply_umap(self, data):
        reducer = UMAP(n_components=2, n_jobs=-1, low_memory=True, random_state=seed)
        return reducer.fit_transform(data)

    def draw_umaps_embedding(self, palette_he):
        print("Standardizing ... ", end='')
        scaler = StandardScaler()
        images_scaled, expressions_scaled, embeddings_scaled, reconstructions_scaled = map(
            scaler.fit_transform, 
            [self.images, self.expressions, self.embeddings, self.reconstructions]
        )
        print(images_scaled.shape, expressions_scaled.shape, embeddings_scaled.shape, reconstructions_scaled.shape)

        print("2D umapping ... ", end='')
        image_umap, expression_umap, reconstruction_umap, embedding_umap = map(
            self._apply_umap, 
            [images_scaled, expressions_scaled, embeddings_scaled, reconstructions_scaled]
        )
        print(image_umap.shape, expression_umap.shape, embedding_umap.shape, reconstruction_umap.shape)

        umaps = {
            'H&E images': image_umap, 
            'gene expressions': expression_umap, 
            'reconstructions': reconstruction_umap, 
            'embeddings': embedding_umap
        }

        fig, ax = plt.subplots(1, len(umaps), figsize=(13, 3))
        colors = [palette_he[cell_type] for cell_type in self.cell_labels]
        
        for i, (title, coordinate) in enumerate(umaps.items()):
            ax[i].scatter(coordinate[:, 0], coordinate[:, 1], c=colors, s=1)
            ax[i].set_title(f"UMAP of {title}")
            ax[i].set_xticks([])
            ax[i].set_yticks([])
        
        markers = [plt.Line2D([0,0],[0,0], color=color, marker='o', linestyle='') for color in palette_he.values()]
        plt.legend(markers, palette_he.keys(), numpoints=1, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        fig.tight_layout()
        fig.savefig(f"/data0/crp/results/umaps_embedding_{self.platform}_{self.sample}_{self.he}_{self.cell_type}.png", bbox_inches="tight")
        plt.close()

    def draw_heatmap(self, cell_types, palette_he):
        group='Cell_type'
        adata_actual = anndata.AnnData(
            X = self.expressions,
            var = pd.DataFrame(index=self.adata.var_names),
            obs = pd.DataFrame({group: self.cell_labels})
        )
        adata_reconstruction = anndata.AnnData(
            X = self.reconstructions,
            var = pd.DataFrame(index=self.adata.var_names),
            obs = pd.DataFrame({group: self.cell_labels})
        )

        fig, ax = plt.subplots(1, 2, figsize=(7, 3))
        
        sc.pp.normalize_total(adata_actual, target_sum=1e4)
        sc.pp.log1p(adata_actual)
        sc.pp.neighbors(adata_actual, random_state=seed)
        sc.tl.umap(adata_actual, random_state=seed)
        sc.pl.umap(adata_actual, color=group, palette=palette_he, show=False, ax=ax[0], legend_loc=None, use_raw=False)
        
        sc.pp.normalize_total(adata_reconstruction, target_sum=1e4)
        sc.pp.log1p(adata_reconstruction)
        sc.pp.neighbors(adata_reconstruction, random_state=seed)
        sc.tl.umap(adata_reconstruction, random_state=seed)
        sc.pl.umap(adata_reconstruction, color=group, palette=palette_he, show=False, ax=ax[1], use_raw=False)
        
        fig.tight_layout()
        plt.close()
        
        sc.tl.rank_genes_groups(adata_actual, groupby=group)
        
        cell_types_cutoff = adata_actual.obs[group].value_counts()
        cell_types_cutoff = cell_types_cutoff[cell_types_cutoff > 3]
        cell_types_cutoff = cell_types_cutoff.index.tolist()
        cell_types_cutoff = [cell_type for cell_type in cell_types.keys() if cell_type in cell_types_cutoff]
        
        top_markers = []
        for cell_type in cell_types_cutoff:
            top_markers.append(sc.get.rank_genes_groups_df(adata_actual, group=cell_type)['names'].values[:10].tolist())
        top_markers = sum(top_markers, [])
        
        sc.tl.dendrogram(adata_actual, groupby=group)
        sc.pl.rank_genes_groups_heatmap(adata_actual, var_names=top_markers, show=False, use_raw=False)
        plt.savefig(f'/data0/crp/results/expression_actual_{self.platform}_{self.sample}_{self.he}_{self.cell_type}.png')
        plt.close()
        
        sc.tl.dendrogram(adata_reconstruction, groupby=group)
        sc.tl.rank_genes_groups(adata_reconstruction, groupby=group)
        sc.pl.rank_genes_groups_heatmap(adata_reconstruction, var_names=top_markers, show=False, use_raw=False)
        plt.savefig(f'/data0/crp/results/expression_reconstruction_{self.platform}_{self.sample}_{self.he}_{self.cell_type}.png')
        plt.close()

class Classifier(nn.Module):
    def __init__(self, num_classes, input_dim=in_features):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)
        self.reset_parameters()

    def forward(self, x):
        return self.fc(x)

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity='linear', generator=generator)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

class Classification():
    def __init__(self, adata, platform, sample, he, cell_type, cell_types):
        self.platform = platform
        self.sample = sample
        self.he = he
        self.cell_type = cell_type
        self.cell_type_encoded = self.cell_type + '_encoded'

        self.reconstructor = Reconstructor(out_features=adata.shape[1])
        reconstructor_file = f"/data0/crp/models/reconstructor_{self.platform}_{self.sample}_{self.he}_{self.cell_type}.pth"
        self.reconstructor.load_state_dict(torch.load(reconstructor_file, map_location=device))
        self.reconstructor.to(device)
        self.reconstructor.eval()
        self.encoder = self.reconstructor.encoder
        
        self.cell_types = cell_types
        self.label_encoder = LabelEncoder()
        if self.cell_type == 'Cell_type':
            self.parameters = list(self.cell_types.keys())
        elif self.cell_type == 'Cell_subtype_ST':
            self.parameters =  sum(self.cell_types.values(), [])
        self.label_encoder.fit(self.parameters)
    
        self.adata = adata
        self.adata_local = self.adata[self.adata.obs[self.cell_type].notna(), :].copy()
        self.adata_local.obs[self.cell_type_encoded] = self.label_encoder.transform(self.adata_local.obs[self.cell_type])

        self.classifier = Classifier(num_classes=len(self.parameters))
        self.classifier.to(device)

        self.criterion = nn.CrossEntropyLoss()

    def load(self, train_loader, epochs, lr, train=False):
        classifier_file = f"/data0/crp/models/classifier_{self.platform}_{self.sample}_{self.he}_{self.cell_type}.pth"
        if not os.path.isfile(classifier_file) or train:
            optimizer = torch.optim.AdamW(self.classifier.parameters(), lr=lr, weight_decay=1e-4)
            
            print(f"Training the {self.cell_type} prediction model ...")
            for epoch in range(epochs):
                self.classifier.train()
            
                train_loss = 0
            
                for cell_id, image in tqdm(train_loader):
                    image = image.to(device, non_blocking=True)
                    cell_label = torch.tensor(
                        self.adata_local[cell_id, :].obs[self.cell_type_encoded].values,
                        dtype=torch.long,
                        device=device
                    )
            
                    with torch.no_grad():  
                        embedding = self.encoder(image)
           
                    logit = self.classifier(embedding)
                    
                    optimizer.zero_grad()
                    loss = self.criterion(logit, cell_label)
                    loss.backward()
                    optimizer.step()
            
                    train_loss += loss.item()
            
                print(f"Epoch: {epoch+1}/{epochs}, Loss: {train_loss / len(train_loader):.4f}")
                    
            gc.collect()
            torch.cuda.empty_cache()
            torch.save(self.classifier.state_dict(), classifier_file)
            
        else: 
            self.classifier.load_state_dict(torch.load(classifier_file, map_location=device))
    
    def evaluate(self, test_loader):
        self.classifier.eval()
        
        cell_labels = []
        cell_predictions = []
        
        test_loss = 0

        print(f"Evaluating the {self.cell_type} prediction model ...")
        with torch.no_grad():
            for cell_id, image in tqdm(test_loader):
                image = image.to(device, non_blocking=True)
                cell_label = torch.tensor(
                    self.adata_local[cell_id, :].obs[self.cell_type_encoded].values,
                    dtype=torch.long,
                    device=device
                )
                cell_labels.append(cell_label)
        
                embedding = self.encoder(image)
        
                logit = self.classifier(embedding)
                loss = self.criterion(logit, cell_label)
                test_loss += loss.item()
                
                cell_prediction = torch.argmax(logit, dim=1)
                cell_predictions.append(cell_prediction)

        print(f"Test Loss: {test_loss / len(test_loader):.4f}")
        
        cell_labels = torch.cat(cell_labels).detach().cpu().numpy()
        cell_predictions = torch.cat(cell_predictions).detach().cpu().numpy()
        
        cell_labels = self.label_encoder.inverse_transform(cell_labels)
        cell_predictions = self.label_encoder.inverse_transform(cell_predictions)
            
        gc.collect()
        torch.cuda.empty_cache()

        f1 = f1_score(cell_labels, cell_predictions, average='weighted')
        print('Average F1 score', f1)
    
        cm = confusion_matrix(cell_labels, cell_predictions, labels=self.parameters)
        
        plt.figure(figsize=(len(self.parameters)//2+2, len(self.parameters)//2))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=self.parameters, yticklabels=self.parameters)
        plt.title(f"{self.cell_type} (f1={f1:.4f})") 
        plt.xlabel('Prediction') 
        plt.ylabel('Label')

        plt.savefig(f'/data0/crp/results/confusion_matrix_{self.platform}_{self.sample}_{self.he}_{self.cell_type}.png', bbox_inches="tight")
        plt.close()
    
    def infer(self, inference_loader):
        self.classifier.eval()
        
        cell_predictions = []
        
        print(f"Inferring with the {self.cell_type} prediction model ...")
        with torch.no_grad():
            for cell_id, image in tqdm(inference_loader):
                image = image.to(device, non_blocking=True)
                embedding = self.encoder(image)
                logit = self.classifier(embedding)
                cell_prediction = torch.argmax(logit, dim=1)
                cell_predictions.append(cell_prediction)

        
        cell_predictions = torch.cat(cell_predictions).detach().cpu().numpy()
        cell_predictions = self.label_encoder.inverse_transform(cell_predictions)
            
        gc.collect()
        torch.cuda.empty_cache()

        return cell_predictions
