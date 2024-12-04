from openbackdoor.victims import Victim, CasualLLMVictim
from openbackdoor.utils import logger, evaluate_classification, evaluate_generation
from .trainer import Trainer, getHighDimFreq
from .casual_trainer import CasualTrainer
from torch.nn.utils.rnn import pad_sequence
from openbackdoor.data import get_dataloader, wrap_dataset
from transformers import  AdamW, get_linear_schedule_with_warmup
import torch
import torch.nn as nn
import os
from typing import *
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split, Subset
from torch import autograd
from sklearn.cluster import KMeans
import numpy as np
from scipy.spatial import KDTree
import copy
from tqdm import tqdm
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FixedLocator
plt.rcParams['font.family'] = 'Times New Roman'


DEBUG = False
DEBUGSTEP = 10
VIS = False
VISSTEP = 20
SAVESTEP = 500
IGNORE_INDEX = -100


class CasualGATrainer(CasualTrainer):
    def __init__(
        self, 
        refSample:Optional[int] = 32,
        GAEpoch:Optional[int] = 0,
        maxRawGradRatio:Optional[float]=0.05,
        minRefGradNorm:Optional[float]=5e-7,
        refBatch:Optional[int]=4,
        minRefLoss:Optional[float]=0.4,
        onlyAlignment:Optional[bool]=False,
        **kwargs
    ):
        super(CasualGATrainer, self).__init__(**kwargs)
        self.refSample = refSample
        self.GAEpoch = GAEpoch
        self.maxRawGradRatio = maxRawGradRatio
        self.minRefGradNorm = minRefGradNorm
        self.refBatch = refBatch
        self.minRefLoss = minRefLoss
        self.onlyAlignment = onlyAlignment
    
    def train(self, model: Victim, dataset, metrics: Optional[List[str]] = ["emr, kmr"], config:dict=None):
        """
        Train the model.

        Args:
            model (:obj:`Victim`): victim model.
            dataset (:obj:`Dict`): dataset.
            metrics (:obj:`List[str]`, optional): list of metrics. Default to ["accuracy"].
        Returns:
            :obj:`Victim`: trained model.
        """
        cleanDataset = dataset['dev-clean']
        refSize, remainSize = self.refSample, len(cleanDataset) - self.refSample
        cleanRefDataset, devCleanDataset = random_split(cleanDataset, [refSize, remainSize])
        dataset['dev-clean'] = devCleanDataset
        
        dataloader = wrap_dataset(dataset, self.batch_size, classification=False)

        train_dataloader = dataloader["train"]
        eval_dataloader = {}
        for key, item in dataloader.items():
            if key.split("-")[0] == "dev":
                # eval_dataloader[key] = dataloader[key]
                eval_dataloader[key] = DataLoader(dataloader[key].dataset, batch_size=1,  collate_fn=dataloader[key].collate_fn)
        self.register(model, dataloader, metrics)
        
        cleanRefDataLoader = DataLoader(cleanRefDataset, batch_size=dataloader['dev-clean'].batch_size,  collate_fn=dataloader['dev-clean'].collate_fn, shuffle=True)
        
        best_dev_score, bestDevEpoch = 0, 0
        allDevResults = []
        for epoch in range(self.epochs):
            epoch_iterator = tqdm(train_dataloader, desc=f"Training Iteration at epoch {epoch}")
            epoch_loss, epochRefLoss, epochMeanRefNorm, poison_loss, normal_loss = self.train_one_epoch(epoch, epoch_iterator, cleanRefDataLoader)
            self.poison_loss_all.append(poison_loss)
            self.normal_loss_all.append(normal_loss)
            logger.info(f'Epoch: {epoch+1}, avg loss: {epoch_loss}, avg reference loss:{epochRefLoss}, mean reference norm:{epochMeanRefNorm}')
            dev_results, dev_score = self.evaluate(self.model, eval_dataloader, self.metrics)
            logger.info('Epoch: {}, dev_score (CEMR): {}'.format(epoch+1, dev_score))
            allDevResults.append(dev_results)
            
            if dev_score > best_dev_score:
                best_dev_score = dev_score
                bestDevEpoch = epoch
                if self.ckpt == 'best':
                    self.model.save(self.model_checkpoint(self.ckpt), config)
                        
            if self.frequencyConfig['frequencyVis']:
                logger.info(f"save Frequency Analysis Results at epoch {epoch}")
                self.save2fileFrequencyResult()
        

        logger.info("Training finished.")
        logger.info(f"Saving Model to {self.model_checkpoint(self.ckpt)}")
        
        if self.frequencyConfig['frequencyVis']:
            logger.info("Visualize Frequency Analysis Results")
            self.visualizeFrequencyDeviation()

        if self.ckpt == 'last':
            self.model.save(self.model_checkpoint(self.ckpt), config)
        
        logger.info(f'Loading Best Model from Epoch {bestDevEpoch} with best DevScore (CEMR) {best_dev_score}')
        self.model.load(self.model_checkpoint(self.ckpt))

        return self.model
    
    def train_one_epoch(self, epoch: int, epoch_iterator, refCleanDataLoader:DataLoader):
        """
        Train one epoch function.

        Args:
            epoch (:obj:`int`): current epoch.
            epoch_iterator (:obj:`torch.utils.data.DataLoader`): dataloader for training.
        
        Returns:
            :obj:`float`: average loss of the epoch.
        """
        self.model.train()
        total_loss = 0
        totalRefLoss, totalMeanNorm = 0, 0
        poison_loss_list, normal_loss_list = [], []
        lossList = []
        rawGradRatio = self.maxRawGradRatio * ((epoch - self.GAEpoch) / (self.epochs - self.GAEpoch))
        # rawGradRatio = self.maxRawGradRatio
        refBatch = math.ceil(self.refSample / self.batch_size)
        
        for step, batch in enumerate(epoch_iterator):
            batch_inputs, batch_labels, attentionMask = self.model.process(batch)
            output = self.model.forward(inputs=batch_inputs, labels=batch_labels, attentionMask=attentionMask)
            
            loss = output.loss

            if self.gradient_accumulation_steps > 1:
                loss = loss / self.gradient_accumulation_steps
            
            loss.backward()
            lossList.append(loss.item())
            
            
            batchRefLoss = 0.0
            allRefGrad = [torch.zeros_like(p).cpu() for p in self.model.parameters() if p.requires_grad]
            
            for i, batch in enumerate(refCleanDataLoader):
                BRefInput, BRefLabel, BattentionMask = self.model.process(batch)
                refOutput = self.model.forward(inputs=BRefInput, labels=BRefLabel, attentionMask=BattentionMask)
                refLoss = refOutput.loss
                batchRefLoss += refLoss
                totalRefLoss += refLoss
                refGrads = autograd.grad(
                    refLoss,
                    [p for p in self.model.parameters() if p.requires_grad],
                    allow_unused=True
                )
                allRefGrad = [(g + gn.cpu()) for g, gn in zip(allRefGrad, refGrads)]
                
                if i + 1 == self.refBatch:
                    break
            refGrads = [g / (i + 1) for g in allRefGrad if g is not None]
            meanNorm = torch.stack([refGrad.flatten().norm() for refGrad in refGrads if refGrad.flatten().norm() > 0]).mean()   
            totalMeanNorm += meanNorm
            
            for p, refGrad in zip([p for p in self.model.parameters() if (p.requires_grad and p.grad is not None)], refGrads):
                oriGradFlat = p.grad.detach().flatten()
                refGradFlat = refGrad.flatten().to(oriGradFlat.device)
                if oriGradFlat.norm() > 0 and refGradFlat.norm() > 0:
                    cosine = torch.cosine_similarity(oriGradFlat, refGradFlat, dim=0)
                    scale = torch.norm(oriGradFlat) * cosine / torch.norm(refGradFlat)
                    alignedGrad = torch.abs(scale) * refGradFlat # same direction
                    p.grad.copy_((alignedGrad.mul(1 - rawGradRatio) + oriGradFlat.mul(rawGradRatio)).reshape(p.grad.shape))
                    # resGrad = oriGradFlat - alignedGrad
                    # if meanNorm > self.minRefGradNorm and batchRefLoss / (i + 1) > self.minRefLoss:
                    #     p.grad.copy_((alignedGrad).reshape(p.grad.shape))
                    # elif self.maxRawGradRatio > 0 and epoch >= self.GAEpoch:
                    #     p.grad.copy_((alignedGrad.mul(1 - self.maxRawGradRatio) + oriGradFlat.mul(self.maxRawGradRatio)).reshape(p.grad.shape))

            if (step + 1) % self.gradient_accumulation_steps == 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                total_loss += loss.item()
                self.model.zero_grad()
                torch.cuda.empty_cache()
            
            if self.frequencyConfig['frequencyVis'] and epoch < self.frequencyConfig['freqVisEpoch'] and (step + 1) % self.frequencyConfig['computeFrequencyStep'] == 0:
                logger.info(f"\nsave Frequency status at step: {step}")
                with torch.no_grad():
                    self.saveFrequencyState()
            
            if VIS and (step + 1) % VISSTEP ==0:
                start = ((step + 1) // VISSTEP - 1) * VISSTEP
                end = step + 1
                logger.info(f"\naverage loss between step {start} and {end} : {np.mean(lossList[start:end])}")
            
            if self.frequencyConfig['frequencyVis'] and epoch < self.frequencyConfig['freqVisEpoch'] and (step + 1) % SAVESTEP ==0:
                logger.info(f"save Frequency Analysis Results and visualize at step: {step}")
                self.visualizeFrequencyDeviation()
        
                    
            if DEBUG and step >= DEBUGSTEP:
                break

        avg_loss = total_loss / len(epoch_iterator)
        avg_poison_loss = sum(poison_loss_list) / len(poison_loss_list) if self.visualize else 0
        avg_normal_loss = sum(normal_loss_list) / len(normal_loss_list) if self.visualize else 0
        
        avgRefLoss = totalRefLoss / len(epoch_iterator)
        meanRefNorm = totalMeanNorm / len(epoch_iterator)
        
        return avg_loss, avgRefLoss, meanRefNorm, avg_poison_loss, avg_normal_loss
    
   