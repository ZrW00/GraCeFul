from .defender import Defender
from openbackdoor.victims import PLMVictim, Victim, CasualLLMVictim
from openbackdoor.data import get_dataloader, collate_fn, getCasualDataloader
from openbackdoor.utils import logger
from openbackdoor.trainers import Trainer, CasualTrainer
from typing import *
from torch.utils.data import DataLoader
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.decomposition import PCA
from umap import UMAP
from hdbscan import HDBSCAN
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, silhouette_score
from collections import Counter
from matplotlib.colors import ListedColormap
import os
from datetime import datetime
import pickle

class CUBEDefender(Defender):
    r"""
        Defender for `CUBE <https://arxiv.org/abs/2206.08514>`_
    
    Args:
        epochs (`int`, optional): Number of CUBE encoder training epochs. Default to 10.
        batch_size (`int`, optional): Batch size. Default to 32.
        lr (`float`, optional): Learning rate for RAP trigger embeddings. Default to 2e-5.
        num_classes (:obj:`int`, optional): The number of classes. Default to 2.
        model_name (`str`, optional): The model's name to help filter poison samples. Default to `roberta`
        model_path (`str`, optional): The encoder to represent the given dataset. Default to `roberta-base`
    """
    def __init__(
        self,
        warm_up_epochs: Optional[int] = 0,
        epochs: Optional[int] = 10,
        batch_size: Optional[int] = 32,
        lr: Optional[float] = 2e-5,
        num_classes: Optional[int] = 2,
        model_name: Optional[str] = 'roberta',
        model_path: Optional[str] = 'roberta-base',
        visMetrics:Optional[bool] = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.pre = True
        self.warm_up_epochs = warm_up_epochs
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.num_classes = num_classes
        self.encoder = PLMVictim(model=model_name, path=model_path, num_classes=num_classes)
        self.trainer = Trainer(warm_up_epochs=warm_up_epochs, epochs=epochs, 
                                batch_size=batch_size, lr=lr,
                                save_path='./models/cube', ckpt='last', visMetrics=visMetrics)
        

    def correct(
        self, 
        poison_data: List,
        clean_data: Optional[List] = None, 
        model: Optional[Victim] = None
    ):
        self.encoder = model
        # Step 1. Encoding
        embeddings, y_true = self.encode(poison_data)

        # Step 2. Clustering
        y_pred = self.clustering(embeddings)

        # Step 3. Filtering
        filtered_dataset = self.filtering(poison_data, y_true, y_pred)

        return filtered_dataset


    def encode(self, dataset):

        logger.info("Training encoder for CUBE defense")
        self.encoder = self.trainer.train(self.encoder, {"train":dataset})
        
        logger.info("Reducing the dimension of hidden states")
        dataloader = get_dataloader(dataset, shuffle=False)
        hidden_states, labels, _ = self.trainer.compute_hidden(self.encoder, dataloader)
        embeddings = self.trainer.dimension_reduction(hidden_states, min_dist=0)

        return embeddings, labels


    def clustering(
        self, 
        embeddings,
        cluster_selection_epsilon: Optional[float] = 0,
        min_samples: Optional[int] = 100):

        logger.info("Clustering the low dimensional embeddings")
        dbscan = HDBSCAN(cluster_selection_epsilon=cluster_selection_epsilon, 
                        min_samples=min_samples)
        y_pred = dbscan.fit_predict(embeddings)

        return y_pred


    def filtering(self, dataset: List, y_true: List, y_pred: List):
        
        logger.info("Filtering suspicious samples")

        dropped_indices = []
        if isinstance(y_true[0], torch.Tensor):
            y_true = [y.item() for y in y_true]

        for true_label in set(y_true):
            
            groundtruth_samples = np.where(y_true==true_label*np.ones_like(y_true))[0]
            
            drop_scale = 0.5*len(groundtruth_samples)

            # Check the predictions for samples of this groundtruth label
            predictions = set()
            for i, pred in enumerate(y_pred):
                if i in groundtruth_samples:
                    predictions.add(pred)

            if len(predictions) > 1:
                count = pd.DataFrame(columns=['predictions'])

                for pred_label in predictions:
                    count.loc[pred_label,'predictions'] = \
                        np.sum(np.where((y_true==true_label*np.ones_like(y_true))*\
                                        (y_pred==pred_label*np.ones_like(y_pred)), 
                                    np.ones_like(y_pred), np.zeros_like(y_pred)))
                cluster_order = count.sort_values(by='predictions', ascending=True)
                
                # we always preserve the largest prediction cluster
                for pred_label in cluster_order.index.values[:-1]: 
                    item = cluster_order.loc[pred_label, 'predictions']
                    if item < drop_scale:

                        idx = np.where((y_true==true_label*np.ones_like(y_true))*\
                                        (y_pred==pred_label*np.ones_like(y_pred)))[0].tolist()

                        dropped_indices.extend(idx)

        filtered_dataset = []
        for i, data in enumerate(dataset):
            if i not in dropped_indices:
                filtered_dataset.append(data)
        
        return filtered_dataset


class CasualCUBEDefender(Defender):
    r"""
        Defender for `CUBE <https://arxiv.org/abs/2206.08514>`_ for generative LLM.
    
    Args:
        epochs (`int`, optional): Number of CUBE encoder training epochs. Default to 1.
        batch_size (`int`, optional): Batch size. Default to 4.
        lr (`float`, optional): Learning rate for RAP trigger embeddings. Default to 2e-5.
    """
    def __init__(
        self,
        epochs: Optional[int] = 1,
        batch_size: Optional[int] = 4,
        lr: Optional[float] = 2e-5,
        pcaRank:Optional[int]=32,
        targetDataset:Optional[str]="webqa",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.pre = True
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.pcaRank = pcaRank
        self.trainer = CasualTrainer(
            warn_up_epochs=0,
            epochs = self.epochs,
            batch_size = self.batch_size,
            lr = lr,
            save_path = './models/casualCUBE', 
            ckpt='last'
        )
        
        self.visPath = os.path.join(
            './casualCube/', 
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
        # Step 1. Encoding
        embeddings, poisonLabels = self.encode(poison_data, model)

        # Step 2. Clustering
        predLabels = self.clustering(embeddings)

        plt.figure(figsize=(16, 8))
        plt.subplot(1, 2, 1)
        cleanIdx, poisonIdx = np.where(poisonLabels == 0)[0], np.where(poisonLabels == 1)[0]
        plt.scatter(embeddings[cleanIdx, 0], embeddings[cleanIdx, 1], edgecolors="blue", facecolors='none', s=15, label="clean")
        plt.scatter(embeddings[poisonIdx,0], embeddings[poisonIdx, 1], s=10, c="red", label='poison', marker='x')
        plt.tick_params(labelsize='large', length=2)
        plt.legend(fontsize=14, markerscale=5, loc='lower right')
        
        plt.subplot(1, 2, 2)
        cmap = ListedColormap(['blue', 'red'])
        plt.scatter(embeddings[:, 0], embeddings[:, 1], s=10, c=predLabels, cmap=cmap, marker='o')
        # plt.tick_params(labelsize='large', length=2)
        handles = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(0), label='pred clean'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(1), label='pred poison')
        ]
        plt.legend(handles=handles, fontsize=14, markerscale=5, loc='lower right')
        
        logger.info(f'saving figure to {self.visPath}')
        plt.savefig(os.path.join(self.visPath, 'visDefense.pdf'))
        plt.savefig(os.path.join(self.visPath, 'visDefense.png'), dpi=600)
        plt.close()
        
        silhouetteScore = silhouette_score(embeddings, predLabels)
        logger.info(f'silhouette score of the clustering: {silhouetteScore:.4f}')
        
        plotData = {
            "emb":embeddings,
            "poisonLabel":poisonLabels,
            "predLabel":predLabels
        }
        with open(os.path.join(self.visPath, 'plotData.pkl'), "wb") as f:
            pickle.dump(plotData, f)

        # Step 3. Filtering
        filtered_dataset = self.filtering(poison_data, predLabels, poisonLabels)

        return filtered_dataset


    def encode(self, dataset, model:CasualLLMVictim):
        logger.info("Training encoder for Casual CUBE defense")
        self.encoder = model
        filterDataloader = getCasualDataloader(dataset, batch_size=1, shuffle=False)
        
        self.encoder = self.trainer.train(self.encoder, {"train":dataset})
        hiddenStates, _, poisonLabels = self.trainer.compute_hidden(self.encoder, filterDataloader)
        
        embeddings = self.dimensionReduction(hiddenStates, pcaRank=self.pcaRank)
        
        poisonLabels = np.array(poisonLabels)
        model.resetPETPara() # reset parameters for model
        return embeddings, poisonLabels


    def clustering(
        self, 
        embeddings,
        cluster_selection_epsilon: Optional[float] = 0,
        min_samples: Optional[int] = 100
    ):

        logger.info("Clustering the low dimensional embeddings")
        dbscan = HDBSCAN(
            cluster_selection_epsilon=cluster_selection_epsilon, 
            min_samples=min_samples
        )
        predLabels = dbscan.fit_predict(embeddings)
        labelCounter = Counter(predLabels)
        majority, _ = labelCounter.most_common(1)[0]
        cleanIdx = np.where(predLabels == majority)[0]# major cluster deem as clean, others as poisoned
        procLabel = np.ones_like(predLabels) 
        procLabel[cleanIdx] = 0
        return procLabel


    def filtering(self, dataset: List, predLabels:np.ndarray, trueLabels:np.ndarray=None):
        logger.info("Filtering suspicious samples")
        
        # labelCounter = Counter(predLabels)
        # majority, _ = labelCounter.most_common(1)[0]
        cleanIdx = np.where(predLabels == 0)[0]
        
        filteredDataset = [data for i, data in enumerate(dataset) if i in cleanIdx]
        logger.info(f'detect {len(predLabels) - len(filteredDataset)} poison examples, {len(filteredDataset)} examples remain in the training set')
        
        if trueLabels is not None:
            procLabel = np.ones_like(predLabels)
            procLabel[cleanIdx] = 0
            f1 = f1_score(trueLabels, procLabel, average=None)
            r = recall_score(trueLabels, procLabel, average=None)
            logger.info(f'f1 score of clean and poison: {np.around(f1 * 100, 2)}')
            logger.info(f'recall score of clean and poison: {np.around(r * 100, 2)}')
        
        return filteredDataset

    def dimensionReduction(
        self, hiddenStates: torch.Tensor, 
        pcaRank: Optional[int] = 32,
        n_neighbors: Optional[int] = 100,
        min_dist: Optional[float] = 0,
        umap_components: Optional[int] = 2
    ):
        _, _, V = torch.pca_lowrank(hiddenStates, q=pcaRank, center=True)
        
        embPCA = torch.matmul(hiddenStates, V[:, :pcaRank])
        
        umap = UMAP( 
            n_neighbors=n_neighbors, 
            min_dist=min_dist,
            n_components=umap_components,
            random_state=42,
            transform_seed=42,
        )
        embUMAP = umap.fit_transform(embPCA.numpy())
        
        embStd = StandardScaler().fit_transform(embUMAP)
        
        return embStd


