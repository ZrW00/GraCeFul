from openbackdoor.victims import Victim, CasualLLMVictim
from openbackdoor.utils import logger, evaluate_classification, evaluate_generation
from .trainer import Trainer, getHighDimFreq
from .casual_trainer import CasualTrainer
from torch.nn.utils.rnn import pad_sequence
from openbackdoor.data import get_dataloader, wrap_dataset
from transformers import  AdamW, get_linear_schedule_with_warmup
import torch
import torch.nn as nn
import torch.nn.functional as F
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
from umap import UMAP
import pandas as pd
from matplotlib.ticker import ScalarFormatter, FixedLocator
# from scipy.fftpack import dct, idct, fft
from torch_dct import dct_2d
plt.rcParams['font.family'] = 'Times New Roman'
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import math
from torch.nn.modules.loss import _WeightedLoss


DEBUG = False
DEBUGSTEP = 10
VIS = False
VISSTEP = 20
SAVESTEP = 200
IGNORE_INDEX = -100




class DeCE(_WeightedLoss):
    def __init__(self, weight: Optional[torch.Tensor] = None, size_average=None, ignore_index: int = None,
                reduce=None, reduction: str = 'mean', label_smoothing: float = 0.05, alpha_base: float = 0.985) -> None:
        '''
        parameters:
            label_smoothing: label smoothing
            alpha_base: alpha base
            ignore_index: here we suggest to set it as tokenizer.pad_token_id
        '''
        super().__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.alpha = 1
        self.alpha_base = alpha_base

    @staticmethod
    def _smooth_one_hot(targets: torch.Tensor, n_classes: int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            smoothTarget = torch.empty(size=(targets.size(0), n_classes), device=targets.device).fill_(smoothing / (n_classes - 1))
            mask = (targets != IGNORE_INDEX)
            valiTarget = targets[mask]
            valiIdx = mask.nonzero(as_tuple=True)[0]
            smoothTarget[valiIdx] = smoothTarget[valiIdx].scatter(1, valiTarget.data.unsqueeze(1), 1. - smoothing)
            # smoothTarget = smoothTarget.scatter(1, targets.data.unsqueeze(1), 1. - smoothing)
            # targets = torch.empty(size=(targets.size(0), n_classes),
            #                     device=targets.device) \
            #     .fill_(smoothing / (n_classes - 1)) \
            #     .scatter_(1, targets.data.unsqueeze(1), 1. - smoothing)
        return smoothTarget
    
    def forward(self, input: torch.Tensor, target: torch.Tensor, cur_epoch: int) -> torch.Tensor:
        self.alpha = math.pow(self.alpha_base, cur_epoch)

        new_target = DeCE._smooth_one_hot(target, input.size(-1), self.label_smoothing)
        input = F.softmax(input, dim=1)
        input = torch.clamp(input, min=1e-7, max=1.0)
        new_input = self.alpha * input + (1 - self.alpha) * new_target
        
        if self.ignore_index is not None:
            mask = (new_target.argmax(dim=1) != self.ignore_index).float().unsqueeze(1)
            mask = mask.expand_as(new_input)
            loss = -1 * (mask * new_target * torch.log(new_input)).sum(dim=1).mean()
        
        else:
            loss = -1 * (new_target * torch.log(new_input)).sum(dim=1).mean()
        return loss

class CasualDeCETrainer(CasualTrainer):
    def __init__(
        self, 
        label_smoothing:Optional[float] = 0.05, 
        alpha_base:Optional[float] = 0.99, 
        ignore_index:Optional[int] = IGNORE_INDEX,
        **kwargs
    ):
        super(CasualDeCETrainer, self).__init__(**kwargs)
        self.lossFn = DeCE(label_smoothing=label_smoothing, alpha_base=alpha_base, ignore_index=ignore_index)

    def train_one_epoch(self, epoch: int, epoch_iterator):
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
        poison_loss_list, normal_loss_list = [], []
        lossList = []
        for step, batch in enumerate(epoch_iterator):
            batch_inputs, batch_labels, attentionMask = self.model.process(batch)
            output = self.model.forward(inputs=batch_inputs, labels=batch_labels, attentionMask=attentionMask)
            logits = output.logits
            shiftLogits = logits[..., :-1, :].reshape(-1, self.model.llm.config.vocab_size)
            shiftLabels = batch_labels[..., 1:].reshape(-1)
            loss = self.lossFn.forward(shiftLogits, shiftLabels, epoch + 1)
            # loss = output.loss

            if self.gradient_accumulation_steps > 1:
                loss = loss / self.gradient_accumulation_steps
            
            loss.backward()
            lossList.append(loss.item())


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
        
        return avg_loss, avg_poison_loss, avg_normal_loss
    