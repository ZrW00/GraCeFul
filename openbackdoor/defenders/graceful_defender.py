import copy
from .defender import Defender
from openbackdoor.victims import CasualLLMVictim
from openbackdoor.data import getCasualDataloader
from openbackdoor.utils import logger
from typing import *
from torch.utils.data import DataLoader
import random
import numpy as np
import pandas as pd
import torch
from umap import UMAP
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from torch_dct import dct_2d
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, silhouette_score
import os
from datetime import datetime
import pickle
from torch import autograd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns

class GraCeFulDefender(Defender):
    name = "graceful"
    r"""
        Defender for `CUBE <https://arxiv.org/abs/2206.08514>`_
    
    Args:
        targetPara (`str`, optional): Target Parameter to regist graidents in the frequency space. Default to `lm_head.weight`.
        targetDataset (`str`, optional): Target Dataset to defend. Default to `webqa`.
        pcaRank (:obj:`int`, optional): The output low rank of PCA. Default to 32.
    """
    def __init__(
        self,
        targetPara:Optional[str]="lm_head.weight",
        targetDataset:Optional[str] = "webqa",
        pcaRank:Optional[int]=32,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.pre = True
        self.targetPara = targetPara
        self.targetDataset = targetDataset
        self.pcaRank = pcaRank
        self.visPath = os.path.join(
            './graceful/', 
            targetDataset,
            str(datetime.fromtimestamp(datetime.now().timestamp()).strftime('%Y-%m-%d-%H-%M-%S'))
        )
        os.makedirs(self.visPath, exist_ok=True)
    
    def correct(
        self, 
        poison_data: List,
        clean_data: Optional[List] = None,
        model: Optional[CasualLLMVictim] = None
    ):

        # Step 1. Feature Representation
        embeddings, poisonLabels = self.encode(poison_data, model)

        # Step 2. Hierarchical Clustering
        predLabels = self.clustering(embeddings)
        
        umap = UMAP( 
            n_neighbors=100, 
            min_dist=0,
            n_components=2,
            random_state=42,
            transform_seed=42,
            metric="cosine"
        )
        embUmap = umap.fit(embeddings).embedding_
        lowrankEmb = StandardScaler().fit_transform(embUmap)
        
        plt.figure(figsize=(16, 8))
        plt.subplot(1, 2, 1)
        cleanIdx, poisonIdx = np.where(poisonLabels == 0)[0], np.where(poisonLabels == 1)[0]
        plt.scatter(lowrankEmb[cleanIdx, 0], lowrankEmb[cleanIdx, 1], edgecolors="blue", facecolors='none', s=15, label="clean")
        plt.scatter(lowrankEmb[poisonIdx,0], lowrankEmb[poisonIdx, 1], s=10, c="red", label='poison', marker='x')
        plt.tick_params(labelsize='large', length=2)
        plt.legend(fontsize=14, markerscale=5, loc='lower right')
        
        plt.subplot(1, 2, 2)
        cmap = ListedColormap(['blue', 'red'])
        plt.scatter(lowrankEmb[:, 0], lowrankEmb[:, 1], s=10, c=predLabels, cmap=cmap, marker='o')

        handles = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(0), label='pred clean'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(1), label='pred poison')
        ]
        plt.legend(handles=handles, fontsize=14, markerscale=5, loc='lower right')
        
        logger.info(f'saving figure to {self.visPath}')
        plt.savefig(os.path.join(self.visPath, 'visDefense.pdf'))
        plt.savefig(os.path.join(self.visPath, 'visDefense.png'), dpi=600)
        plt.close()
        
        silhouetteScore = silhouette_score(lowrankEmb, predLabels)
        logger.info(f'silhouette score of the clustering: {silhouetteScore:.4f}')
        
        plotData = {
            "emb":lowrankEmb,
            "poisonLabel":poisonLabels,
            "predLabel":predLabels
        }
        with open(os.path.join(self.visPath, 'plotData.pkl'), "wb") as f:
            pickle.dump(plotData, f)

        # Step 3. Filtering
        filteredDataset = self.filtering(poison_data, predLabels, poisonLabels)

        return filteredDataset

    def encode(self, dataset, model):
        dataloader = getCasualDataloader(dataset, batch_size=1, shuffle=False)
        dctGrads, poisonLabels = self.computeGradients(model, dataloader, "train")
        logger.info("Reducing the dimension of hidden states")
        embeddings = self.dimensionReduction(dctGrads, pcaRank=self.pcaRank)
        


        return embeddings, poisonLabels


    def clustering(self, embeddings, metric="cosine", linkage='average'):
        logger.info("Clustering the low dimensional embeddings")
        clusting = AgglomerativeClustering(n_clusters=2, metric=metric, linkage=linkage)

        clusterLabels = clusting.fit_predict(embeddings)
        
        clusterLabels = np.array(clusterLabels)
        
        unique, counts = np.unique(clusterLabels, return_counts=True)
        labelCounts = dict(zip(unique, counts))
        # minority = min(labelCounts, key=labelCounts.get)
        majority = max(labelCounts, key=labelCounts.get)
        
        predLabels = np.where(clusterLabels == majority, 0, 1)

        return np.array(predLabels)

    def filtering(self, dataset: List, predLabels:np.ndarray, trueLabels:np.ndarray=None):
        
        logger.info("Filtering suspicious samples")
                
        cleanIdx = np.where(predLabels == 0)[0]
        
        filteredDataset = [data for i, data in enumerate(dataset) if i in cleanIdx]
        logger.info(f'detect {len(predLabels) - len(filteredDataset)} poison examples, {len(filteredDataset)} examples remain in the training set')
        
        if trueLabels is not None:
            f1 = f1_score(trueLabels, predLabels, average=None)
            r = recall_score(trueLabels, predLabels, average=None)
            logger.info(f'f1 score of clean and poison: {np.around(f1 * 100, 2)}')
            logger.info(f'recall score of clean and poison: {np.around(r * 100, 2)}')
        
        return filteredDataset
    
    def computeGradients(self, model:CasualLLMVictim, dataLoader:DataLoader, name):
        model.train()
        assert any([self.targetPara in n for n, p in model.named_parameters() if p.requires_grad]), "no corresponding parameter for compute"

        dctGrads, poisonLabels = [], []
        dct2 = lambda tensor: dct_2d(torch.tensor(tensor))
        for i, batch in tqdm(enumerate(dataLoader), desc=f"Calculating gradients of {name}", total=len(dataLoader)):
            poisonLabels.extend(batch["poison_label"])
            model.zero_grad()
            batch_inputs, batch_labels, attentionMask = model.process(batch)
            output = model.forward(inputs=batch_inputs, labels=batch_labels, attentionMask=attentionMask)
            
            loss = output.loss
            # loss.backward()
            grad = autograd.grad(
                loss,
                [p for n, p in model.named_parameters() if (p.requires_grad) and (self.targetPara in n)],
                allow_unused=True
            )
            targetGrad = grad[0].detach()
            dctGrad = dct2(targetGrad)
            if "lm_head" in self.targetPara:
                dctGrad = dctGrad[:int(dctGrad.shape[0] // 8), :int(dctGrad.shape[1] // 8)]
            dctGrads.append(dctGrad.cpu().flatten())
        dctGrads = torch.stack(dctGrads, dim=0)
        poisonLabels = np.array(poisonLabels)
        return dctGrads, poisonLabels
    
    def dimensionReduction(
        self, hiddenStates: torch.Tensor, 
        pcaRank: Optional[int] = 32
    ):
        _, _, V = torch.pca_lowrank(hiddenStates, q=pcaRank, center=True)
        
        embPCA = torch.matmul(hiddenStates, V[:, :pcaRank])
        
        embStd = StandardScaler().fit_transform(embPCA)
        
        return embStd

    

    
    