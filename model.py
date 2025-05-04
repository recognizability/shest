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
from tqdm.contrib.concurrent import thread_map

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, f1_score

from config import n_cores, seed, set_seed, generator, device

in_features = 1000 #output features of ViT or SwinTransformer

class Encoder(nn.Module):
    def __init__(self, backbone="vit_b_16"):
        super().__init__()
        self.encoder = getattr(models, backbone)(weights="IMAGENET1K_V1")
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.encoder.heads.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self, out_features, in_features=in_features):
        super().__init__()
        hidden = 2048
        self.bn0 = nn.BatchNorm1d(in_features)
        self.fc1 = nn.Linear(in_features, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.fc_mu = nn.Linear(hidden, out_features)
        self.fc_alpha = nn.Linear(hidden, out_features)

        self.reset_parameters()

    def forward(self, x):
        x = self.bn0(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))        
        mu = F.softplus(self.fc_mu(x))
        alpha = F.softplus(self.fc_alpha(x))
        return mu, alpha


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

class Classifier(nn.Module):
    def __init__(self, out_features, in_features=in_features):
        super().__init__()
        self.bn = nn.BatchNorm1d(in_features)
        self.fc = nn.Linear(in_features, out_features)
        self.reset_parameters()

    def forward(self, x):
        return self.fc(self.bn(x))

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity='linear', generator=generator)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

class Model(nn.Module):
    def __init__(self, n_genes, n_classes):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(out_features=n_genes)
        self.classifier = Classifier(out_features=n_classes)

    def forward(self, x):
        x = self.encoder(x)
        mu, alpha = self.decoder(x)
        logits = self.classifier(x)
        return mu, alpha, logits

class NegativeBinomialLoss(nn.Module):
   def __init__(self, eps=1e-8):
       super().__init__()
       self.eps = eps

   def forward(self, mu, alpha, target):
       total_count = 1.0 / alpha
       logits = (mu + self.eps).log() - (total_count + mu + self.eps).log()
       nb = NegativeBinomial(total_count=total_count, logits=logits)
       loss = -nb.log_prob(target.float())
       return loss.mean()

class Modeling():
    def __init__(self, platform, sample, he, cell_type, cell_types, angles, var_names, classes):
        self.platform = platform
        self.sample = sample
        self.he = he
        self.cell_type = cell_type
        self.cell_types = cell_types
        self.angles = angles
        self.angles_string = '_'.join(map(str, self.angles))

        self.var_names = var_names
        self.n_genes = len(self.var_names)
        self.classes = classes
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.classes)
        self.n_classes = len(classes)

        self.model = Model(n_genes = self.n_genes, n_classes=self.n_classes)
        self.model.to(device)

        self.images = None
        self.expressions = None
        self.embeddings = None
        self.reconstructions = None
        self.labels = None
        self.predictions = None

        self.criterion_reconstruction = NegativeBinomialLoss()
        self.criterion_classification = nn.CrossEntropyLoss()

    def load(self, train_loader, epochs, lr, train=False):
        model_file = f"/data0/crp/models/model_{self.platform}_{self.sample}_{self.he}_{self.cell_type}_{self.angles_string}.pth"
        if not os.path.isfile(model_file) or train:
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)

            gc.collect()
            torch.cuda.empty_cache()
            
            print(f"Training the model ...")
            for epoch in range(epochs):
                self.model.train()
                train_loss_reconstruction = 0
                train_loss_classification = 0

                for cell_id, image, expression, label in tqdm(train_loader):
                    image = image.to(device, non_blocking=True)
                    expression = expression.to(device, non_blocking=True)
                    label = label.to(device, non_blocking=True)
                    
                    embedding = self.model.encoder(image)
                    if torch.isnan(embedding).any():
                        print("nan found in embedding")
                        sys.exit()
                    mu, alpha = self.model.decoder(embedding)
                    if torch.isnan(mu).any():
                        print("nan found in mu")
                        sys.exit()
                    if torch.isnan(alpha).any():
                        print("nan found in alpha")
                        sys.exit()
                    logit = self.model.classifier(embedding)
                    if torch.isnan(logit).any():
                        print("nan found in logit")
                        sys.exit()

                    optimizer.zero_grad()
                    loss_reconstruction = self.criterion_reconstruction(mu, alpha, expression)
                    loss_classification = self.criterion_classification(logit, label)
                    loss = loss_reconstruction + loss_classification
                    loss.backward()
                    optimizer.step()

                    train_loss_reconstruction += loss_reconstruction.item()
                    train_loss_classification += loss_classification.item()

                    del image, expression, label, embedding, mu, alpha, logit

                average_loss_reconstruction = train_loss_reconstruction / len(train_loader)
                average_loss_classification = train_loss_classification / len(train_loader)
                print(f"Epoch: {epoch+1}/{epochs}, reconstruction loss: {average_loss_reconstruction:.5f}, classification loss: {average_loss_classification:.5f}")

            gc.collect()
            torch.cuda.empty_cache()

            torch.save(self.model.state_dict(), model_file)
        else: 
            self.model.load_state_dict(torch.load(model_file, map_location=device))

    def evaluate(self, test_loader):
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

                embedding = self.model.encoder(image)
                mu, alpha = self.model.decoder(embedding)
                logit = self.model.classifier(embedding)
                prediction = torch.argmax(logit, dim=1)

                loss_reconstruction = self.criterion_reconstruction(mu, alpha, expression)
                loss_classification = self.criterion_classification(logit, label)

                test_loss_reconstruction += loss_reconstruction.item()
                test_loss_classification += loss_classification.item()

                images.append(image.view(image.shape[0], -1).detach().cpu())
                expressions.append(expression.detach().cpu())
                labels.append(label.detach().cpu())
                embeddings.append(embedding.detach().cpu())
                reconstruction = mu
                reconstructions.append(reconstruction.detach().cpu())
                predictions.append(prediction.detach().cpu())

                del image, expression, label, embedding, mu, alpha, logit, reconstruction, prediction
                
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

    def _apply_umap(self, data):
        reducer = UMAP(n_components=2, n_jobs=-1, low_memory=True, random_state=seed)
        return reducer.fit_transform(data)

    def draw_umaps_embedding(self, palette_he):
        print("Standardizing ... ", end='')
        images_scaled, embeddings_scaled, expressions_scaled, reconstructions_scaled = thread_map(
            lambda x: StandardScaler().fit_transform(x), 
            [self.images, self.embeddings, self.expressions, self.reconstructions],
            max_workers=n_cores,
        )
        print(images_scaled.shape, embeddings_scaled.shape, expressions_scaled.shape, reconstructions_scaled.shape)

        print("2D umapping ... ", end='')
        image_umap, embedding_umap, expression_umap, reconstruction_umap = map(
            self._apply_umap, 
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
        colors = [palette_he[cell_type] for cell_type in self.labels]
        
        for i, (title, coordinate) in enumerate(umaps.items()):
            ax[i].scatter(coordinate[:, 0], coordinate[:, 1], c=colors, s=1)
            ax[i].set_title(f"UMAP of {title}")
            ax[i].set_xticks([])
            ax[i].set_yticks([])
        
        markers = [plt.Line2D([0,0],[0,0], color=color, marker='o', linestyle='') for color in palette_he.values()]
        plt.legend(markers, palette_he.keys(), numpoints=1, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        fig.tight_layout()
        fig.savefig(f"/data0/crp/results/umaps_embedding_{self.platform}_{self.sample}_{self.he}_{self.cell_type}_{self.angles_string}.png", bbox_inches="tight")
        plt.close()

    def draw_heatmap(self, cell_types, palette_he):
        adata_actual = anndata.AnnData(
            X = self.expressions,
            var = pd.DataFrame(index=self.var_names),
            obs = pd.DataFrame({self.cell_type: self.labels})
        )
        adata_reconstruction = anndata.AnnData(
            X = self.reconstructions,
            var = pd.DataFrame(index=self.var_names),
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

        sc.pl.rank_genes_groups_heatmap(adata_actual, var_names=top_markers, show=False, use_raw=False, dendrogram=True)
        plt.tight_layout()
        plt.savefig("/tmp/heatmap_actual.png")
        plt.close()
        sc.pl.rank_genes_groups_heatmap(adata_reconstruction, var_names=top_markers, show=False, use_raw=False, dendrogram=True)
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
        plt.savefig(f"/data0/crp/results/expression_{self.platform}_{self.sample}_{self.he}_{self.cell_type}_{self.angles_string}.png")
        plt.close()

    def draw_confusion_matrix(self):
        f1 = f1_score(self.labels, self.predictions, average='weighted')
        print('Average F1 score', f1)

        cm = confusion_matrix(self.labels, self.predictions, labels=self.classes)

        plt.figure(figsize=(len(self.classes)//2+2, len(self.classes)//2))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=self.classes, yticklabels=self.classes)
        plt.title(f"{self.cell_type} (f1={f1:.4f})")
        plt.xlabel('Prediction')
        plt.ylabel('Label')

        plt.savefig(f"/data0/crp/results/confusion_matrix_{self.platform}_{self.sample}_{self.he}_{self.cell_type}_{self.angles_string}.png", bbox_inches="tight")
        plt.close()

    def infer(self, inference_loader):
        self.model.eval()
        predictions = []
        
        print(f"Inferring with the {self.cell_type} prediction model ...")
        with torch.no_grad():
            for cell_id, image in tqdm(inference_loader):
                image = image.to(device, non_blocking=True)
                embedding = self.encoder(image)
                logit = self.classifier(embedding)
                prediction = torch.argmax(logit, dim=1)
                predictions.append(prediction.detach().cpu())

        predictions = torch.cat(predictions).numpy()
        predictions = self.label_encoder.inverse_transform(predictions)
            
        gc.collect()
        torch.cuda.empty_cache()

        return predictions
