from openbackdoor.victims import Victim, PLMVictim, MLMVictim
from openbackdoor.utils import logger, evaluate_classification
from openbackdoor.data import get_dataloader, wrap_dataset, PROCESSORS
from transformers import  AdamW, get_linear_schedule_with_warmup
import torch
from datetime import datetime
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
import os
from tqdm import tqdm
from typing import *
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from umap import UMAP
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.font_manager as fm
from torch import autograd
import seaborn as sns
import copy
import pickle
import json
import gc
import math
import torch.nn.functional as F
from matplotlib.ticker import ScalarFormatter, FixedLocator
plt.rcParams['font.family'] = 'Times New Roman'

# plt.rcParams['font.sans-serif'] = ['Times New Roman']

def getHighDimFreq(logits, kernels):
    def gaussianFiliter(f_orig, kernel):
        return torch.matmul(kernel, f_orig)
    freqLow, freqHigh = [], []
    for kernel in kernels:
        kernel = kernel.to(logits.device) 
        freqFiltered = gaussianFiliter(logits, kernel)
        freqLow.append(freqFiltered)
        freqHigh.append(logits - freqFiltered)
        
    return freqLow, freqHigh

class KLDivLoss(nn.Module):
    def __init__(self, reduction='batchmean'):
        super(KLDivLoss, self).__init__()
        self.reduction = reduction

    def forward(self, input_logits, soft_labels):
        return F.kl_div(F.log_softmax(input_logits, dim=-1), soft_labels, reduction=self.reduction)


class Trainer(object):
    r"""
    Basic clean trainer. Used in clean-tuning and dataset-releasing attacks.

    Args:
        name (:obj:`str`, optional): name of the trainer. Default to "Base".
        lr (:obj:`float`, optional): learning rate. Default to 2e-5.
        weight_decay (:obj:`float`, optional): weight decay. Default to 0.
        epochs (:obj:`int`, optional): number of epochs. Default to 10.
        batch_size (:obj:`int`, optional): batch size. Default to 4.
        gradient_accumulation_steps (:obj:`int`, optional): gradient accumulation steps. Default to 1.
        max_grad_norm (:obj:`float`, optional): max gradient norm. Default to 1.0.
        warm_up_epochs (:obj:`int`, optional): warm up epochs. Default to 3.
        ckpt (:obj:`str`, optional): checkpoint name. Can be "best" or "last". Default to "best".
        save_path (:obj:`str`, optional): path to save the model. Default to "./models/checkpoints".
        loss_function (:obj:`str`, optional): loss function. Default to "ce".
        visualize (:obj:`bool`, optional): whether to visualize the hidden states. Default to False.
        poison_setting (:obj:`str`, optional): the poisoning setting. Default to mix.
        poison_method (:obj:`str`, optional): name of the poisoner. Default to "Base".
        poison_rate (:obj:`float`, optional): the poison rate. Default to 0.1.

    """
    def __init__(
        self, 
        name: Optional[str] = "Base",
        lr: Optional[float] = 2e-5,
        weight_decay: Optional[float] = 0.,
        epochs: Optional[int] = 10,
        batch_size: Optional[int] = 4,
        gradient_accumulation_steps: Optional[int] = 1,
        max_grad_norm: Optional[float] = 1.0,
        warm_up_epochs: Optional[int] = 3,
        ckpt: Optional[str] = "best",
        save_path: Optional[str] = "./models/checkpoints",
        loss_function: Optional[str] = "ce",
        visualize: Optional[bool] = False,
        poison_setting: Optional[str] = "mix",
        poison_method: Optional[str] = "Base",
        poison_rate: Optional[float] = 0.01,
        attackMethod:Optional[str] = 'badnets',
        defense:Optional[bool] = False,
        frequencyConfig:Optional[dict]={
            'frequencyVis':False,
            'kernelBand':[2, 8],
            'kernelNum':25,
            'poisonerName':'badnets4',
            'computeFrequencyStep':5,
            'freqVisEpoch':10,
            'dataset':'cifar-10'
        },
        visMetrics:Optional[bool]=False,
        **kwargs):

        self.name = name
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.warm_up_epochs = warm_up_epochs
        self.ckpt = ckpt
        self.attackMethod = attackMethod
        self.defense = defense
        self.frequencyConfig = frequencyConfig
        self.timestamp = datetime.fromtimestamp(datetime.now().timestamp()).strftime('%Y-%m-%d-%H-%M-%S')
        self.save_path = os.path.join(save_path, f'{poison_setting}-{poison_method}-{poison_rate}', str(self.timestamp))
        os.makedirs(self.save_path, exist_ok=True)
        

        self.visualize = visualize
        self.visMetrics = visMetrics
        self.poison_setting = poison_setting
        self.poison_method = poison_method
        self.poison_rate = poison_rate

        self.COLOR = ['royalblue', 'red', 'palegreen', 'violet', 'paleturquoise', 
                            'green', 'mediumpurple', 'gold', 'deepskyblue']

        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.loss_type = loss_function
        if self.loss_type == "ce":
            reduction = "none" if self.visualize else "mean"
            self.loss_function = nn.CrossEntropyLoss(reduction=reduction)
        elif self.loss_type == "kl":
            reduction = "none" if self.visualize else "batchmean"
            self.loss_function = KLDivLoss(reduction=reduction)
    
    @torch.no_grad()
    def getKernel(self, dataloader:DataLoader):
        continuousDataExpand = self.model.continuousData(dataloader)
        # torch.cuda.empty_cache() 
        filters = np.linspace(self.frequencyConfig['kernelBand'][0], self.frequencyConfig['kernelBand'][1], num=self.frequencyConfig['kernelNum'])
        dist = torch.cdist(continuousDataExpand.cpu(), continuousDataExpand.cpu(), p=2, compute_mode="use_mm_for_euclid_dist")
        kernels = [torch.exp(-dist / (2 * filter_)) for filter_ in filters]
        kernels = [kernel / torch.sum(kernel, dim=1, keepdim=True) for kernel in kernels]
        # torch.cuda.empty_cache()  
        kernels = [kernel for kernel in kernels]
        # del copyModel
        return kernels
    
    @torch.no_grad()
    def getKernelRes(self, dataloader:DataLoader, savepath:str):
        chunk = 1000
        if os.path.exists(savepath):
            dist = torch.load(savepath).cpu()
            logger.info(f'load distance tensor from {savepath} finished')
        else:
            dist = torch.zeros(size=(len(dataloader.dataset), len(dataloader.dataset)))
            round = math.ceil(len(dataloader.dataset) / chunk)
            for i in range(round):
                xstart = i * chunk
                xend = (i + 1) * chunk if (i + 1) * chunk <= len(dataloader.dataset) else len(dataloader.dataset)
                xdata = self.model.continuousDataRes(DataLoader(dataloader.dataset[xstart:xend], batch_size=dataloader.batch_size, shuffle=False, collate_fn=dataloader.collate_fn))
                for j in tqdm(range(round), desc=f"kernel round {i}", total=round):
                    ystart = j * chunk
                    yend = (j + 1) * chunk if (j + 1) * chunk <= len(dataloader.dataset) else len(dataloader.dataset)
                    ydata = self.model.continuousDataRes(DataLoader(dataloader.dataset[ystart:yend], batch_size=dataloader.batch_size, shuffle=False, collate_fn=dataloader.collate_fn))
                    
                    subdist = torch.cdist(
                        xdata,
                        ydata,
                        p = 2#, compute_mode="use_mm_for_euclid_dist"
                    )
                    dist[xstart:xend, ystart:yend].data.copy_(subdist.cpu().data)
            
            logger.info("computing distance finished")
            torch.save(dist, savepath)
            logger.info(f"save distance tensor to {savepath} finished")
        
        gc.collect()
        logger.info("gc.collect() finished")
        # torch.cuda.empty_cache()
        # dist = dist.cpu() 
        filters = np.linspace(self.frequencyConfig['kernelBand'][0], self.frequencyConfig['kernelBand'][1], num=self.frequencyConfig['kernelNum'])
        logger.info("building filters finished")
        
        
        
        kernels = [torch.exp(-dist / (2 * filter_)) for filter_ in filters]
        logger.info("building kernels step1")
        kernels = [kernel / torch.sum(kernel, dim=1, keepdim=True) for kernel in kernels]
        # torch.cuda.empty_cache()  
        logger.info("normalizing kernels finished")
        # kernels = [kernel for kernel in kernels]
        # del copyModel
        return kernels
    
    def register(self, model: Union[Victim, PLMVictim, MLMVictim], dataloader, metrics):
        r"""
        Register model, dataloader and optimizer
        """
        self.model = model
        self.metrics = metrics
        self.main_metric = self.metrics[0]
        self.split_names = dataloader.keys()
        self.model.train()
        self.model.zero_grad()
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if (not any(nd in n for nd in no_decay)) and p.requires_grad], 'weight_decay': self.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if (any(nd in n for nd in no_decay)) and p.requires_grad], 'weight_decay': 0.0}
            ]
        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.lr)
        train_length = len(dataloader["train"])
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                    num_warmup_steps=self.warm_up_epochs * train_length,
                                                    num_training_steps=self.epochs * train_length)
        
        self.poison_loss_all = []
        self.normal_loss_all = []
        if self.visualize:
            poison_loss_before_tuning, normal_loss_before_tuning = self.comp_loss(model, dataloader["train"])
            self.poison_loss_all.append(poison_loss_before_tuning)
            self.normal_loss_all.append(normal_loss_before_tuning)
            self.hidden_states, self.labels, self.poison_labels = self.compute_hidden(model, dataloader["train"])
            
            devDataset = ConcatDataset([dataloader['dev-clean'].dataset, dataloader['dev-poison'].dataset])
            devDataloader = DataLoader(devDataset, batch_size=dataloader['dev-clean'].batch_size, collate_fn=dataloader['dev-clean'].collate_fn)
            
            self.dev_hidden_states, self.dev_labels, self.dev_poison_labels = self.compute_hidden(model, devDataloader)
            
        
        
        # Train
        logger.info("***** Training *****")
        logger.info("  Num Epochs = %d", self.epochs)
        logger.info("  Instantaneous batch size per GPU = %d", self.batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", self.epochs * train_length)
        
        if self.frequencyConfig['frequencyVis']:
            logger.info("\nRegister Frequency Infomation\n")
            train_dataloader = dataloader["train"]
            trainDataset = train_dataloader.dataset
            trainCleanDataset, trainPoisonDatset = [data for data in trainDataset if data[2] == 0], [data for data in trainDataset if data[2] == 1]
            distDict = {
                'cifar-10':PROCESSORS['cifar-10'].path,
            }
            try:
                self.distRootPath = distDict[self.frequencyConfig['dataset']]
            except KeyError as k:
                pass
            
            self.staticDataLoaders = {
                'dev-clean':DataLoader(dataloader['dev-clean'].dataset, batch_size=train_dataloader.batch_size, collate_fn=train_dataloader.collate_fn, shuffle=False), 
                'dev-poison': DataLoader(dataloader['dev-poison'].dataset, batch_size=train_dataloader.batch_size, collate_fn=train_dataloader.collate_fn, shuffle=False)
            }
            self.staticOneHotLabels = {name:self.getoneHotLabels(loader) for name, loader in self.staticDataLoaders.items()}
            self.kernels = {name:self.getKernel(loader) for name, loader in self.staticDataLoaders.items()}
            
            self.lowNorm, self.highNorm = {name:[[] for _ in range(self.frequencyConfig['kernelNum'])] for name in self.staticDataLoaders.keys()}, {name:[[] for _ in range(self.frequencyConfig['kernelNum'])] for name in self.staticDataLoaders.keys()}
            self.lowDeviation, self.highDeviation = {name:[[] for _ in range(self.frequencyConfig['kernelNum'])] for name in self.staticDataLoaders.keys()}, {name:[[] for _ in range(self.frequencyConfig['kernelNum'])] for name in self.staticDataLoaders.keys()}
            self.labelFreqLow, self.labelFreqHigh = {}, {}
            
            self.lowFreqRatio, self.highFreqRatio = {name:[[] for _ in range(self.frequencyConfig['kernelNum'])] for name in self.staticDataLoaders.keys()}, {name:[[] for _ in range(self.frequencyConfig['kernelNum'])] for name in self.staticDataLoaders.keys()}
            self.logitFreqLow, self.logitFreqHigh = {name:[] for name in self.staticDataLoaders.keys()}, {name:[] for name in self.staticDataLoaders.keys()}
            
            for name in self.staticDataLoaders.keys():
                labelLow, labelHigh = getHighDimFreq(self.staticOneHotLabels[name], self.kernels[name])
                self.labelFreqLow[name] = labelLow
                self.labelFreqHigh[name] = labelHigh
        else:
            logger.info("Disable Frequency Analysis")

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
        for step, batch in enumerate(epoch_iterator):
            batch_inputs, batch_labels = self.model.process(batch)
            # if self.loss_type == 'ce':
            #     batch_inputs, batch_labels = self.model.process(batch)
            # elif self.loss_type == 'kl':
            #     batch_inputs, batch_labels = self.model.klprocess(batch)
            output = self.model(batch_inputs)
            logits = output.logits
            loss = self.loss_function.forward(logits, batch_labels)
        

            if self.visualize:
                poison_labels = batch["poison_label"]
                for l, poison_label in zip(loss, poison_labels):
                    if poison_label == 1:
                        poison_loss_list.append(l.item())
                    else:
                        normal_loss_list.append(l.item())
                loss = loss.mean()

            if self.gradient_accumulation_steps > 1:
                loss = loss / self.gradient_accumulation_steps
            
            loss.backward()


            if (step + 1) % self.gradient_accumulation_steps == 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                total_loss += loss.item()
                self.model.zero_grad()
                # torch.cuda.empty_cache()
            
            if self.frequencyConfig['frequencyVis'] and epoch < self.frequencyConfig['freqVisEpoch'] and (step + 1) % self.frequencyConfig['computeFrequencyStep'] == 0:
                with torch.no_grad():
                    self.saveFrequencyState()

        avg_loss = total_loss / len(epoch_iterator)
        avg_poison_loss = sum(poison_loss_list) / len(poison_loss_list) if self.visualize else 0
        avg_normal_loss = sum(normal_loss_list) / len(normal_loss_list) if self.visualize else 0
        
        return avg_loss, avg_poison_loss, avg_normal_loss
    
    @torch.no_grad()
    def saveFrequencyState(self):
        for name in self.staticDataLoaders.keys():
            labelLow, labelHigh = self.labelFreqLow[name], self.labelFreqHigh[name]
            dynamicLogits = self.computeLogits(name, self.staticDataLoaders[name])
            logitLow, logitHigh = getHighDimFreq(dynamicLogits, self.kernels[name])
            
            self.logitFreqLow[name].append([low.cpu() for low in logitLow])
            self.logitFreqHigh[name].append([high.cpu() for high in logitHigh])
            for j in range(self.frequencyConfig['kernelNum']):
                self.lowDeviation[name][j].append((torch.norm(logitLow[j] - labelLow[j], dim=1).mean() / torch.norm(labelLow[j], dim=1).mean()).cpu().numpy().item())
                self.highDeviation[name][j].append((torch.norm(logitHigh[j] - labelHigh[j], dim=1).mean() / torch.norm(labelHigh[j], dim=1).mean()).cpu().numpy().item())
                
                self.lowFreqRatio[name][j].append((torch.norm(logitLow[j], dim=1).mean() / torch.norm(dynamicLogits, dim=1).mean()).cpu().numpy().item())
                self.highFreqRatio[name][j].append((torch.norm(logitHigh[j], dim=1).mean() / torch.norm(dynamicLogits, dim=1).mean()).cpu().numpy().item())
                
                self.lowNorm[name][j].append(torch.norm(logitLow[j], dim=1).mean().cpu().numpy().item())
                self.highNorm[name][j].append(torch.norm(logitHigh[j], dim=1).mean().cpu().numpy().item())

    def softLabel(self, dataset, model:Victim):
        softDataset = []
        for text, label, poison_label in dataset:
            if poison_label == 1:
                confidence = 0.5 + 0.5 * torch.rand(1).item()
                residual = (1 - confidence) / (model.num_labels - 1)
                softlabel = torch.full((model.num_labels, ), residual)
                softlabel[label] = confidence
                softlabel = softlabel.tolist()
                data = (text, softlabel, poison_label)
            else:
                softlabel = torch.eye(model.num_labels)[label].tolist()
                data = (text, softlabel, poison_label)
            softDataset.append(data)
            
        return softDataset
          
    def train(self, model: Victim, dataset, metrics: Optional[List[str]] = ["accuracy"], config:dict=None):
        """
        Train the model.

        Args:
            model (:obj:`Victim`): victim model.
            dataset (:obj:`Dict`): dataset.
            metrics (:obj:`List[str]`, optional): list of metrics. Default to ["accuracy"].
        Returns:
            :obj:`Victim`: trained model.
        """
        if self.loss_type == "kl":
            softdataset = {name:self.softLabel(data, model) for name, data in dataset.items()}       
            harddataloader = wrap_dataset(dataset, self.batch_size)
            dataloader = wrap_dataset(softdataset, self.batch_size)
        else:
            dataloader = wrap_dataset(dataset, self.batch_size)
            harddataloader = dataloader

        train_dataloader = dataloader["train"]
        eval_dataloader = {}
        for key, item in dataloader.items():
            if key.split("-")[0] == "dev":
                eval_dataloader[key] = harddataloader[key]
        self.register(model, dataloader, metrics)
        
        trainDataset = train_dataloader.dataset
        trainCleanDataset, trainPoisonDatset = [data for data in trainDataset if data[2] == 0], [data for data in trainDataset if data[2] == 1]
        
        trainCleanLoader, trainPoisonLoader = DataLoader(trainCleanDataset, batch_size=train_dataloader.batch_size,  collate_fn=train_dataloader.collate_fn), DataLoader(trainPoisonDatset, batch_size=train_dataloader.batch_size, collate_fn=train_dataloader.collate_fn)
        
        trainEvalLoader = {'train-clean':trainCleanLoader, 'train-poison':trainPoisonLoader}
        
        
        devDataset = ConcatDataset([dataloader['dev-clean'].dataset, dataloader['dev-poison'].dataset]) if 'dev-poison' in dataloader.keys() else None
        devDataloader = DataLoader(devDataset, batch_size=dataloader['dev-clean'].batch_size,  collate_fn=dataloader['dev-clean'].collate_fn) if 'dev-poison' in dataloader.keys() else None
        
        best_dev_score, bestDevEpoch = 0, 0
        allDevResults, allTrainDevResults = [], []
        for epoch in range(self.epochs):
            epoch_iterator = tqdm(train_dataloader, desc="Training Iteration")
            epoch_loss, poison_loss, normal_loss = self.train_one_epoch(epoch, epoch_iterator)
            self.poison_loss_all.append(poison_loss)
            self.normal_loss_all.append(normal_loss)
            logger.info('Epoch: {}, avg loss: {}'.format(epoch+1, epoch_loss))
            dev_results, dev_score = self.evaluate(self.model, eval_dataloader, self.metrics)
            logger.info('Epoch: {}, dev_score: {}'.format(epoch+1, dev_score))
            trainDevResults, trainDevScore = self.evaluate(self.model, trainEvalLoader, self.metrics)
            allDevResults.append(dev_results)
            allTrainDevResults.append(trainDevResults)
            
            if self.visualize:
                hidden_state, labels, poison_labels = self.compute_hidden(model, epoch_iterator)
                self.hidden_states.extend(hidden_state)
                self.labels.extend(labels)
                self.poison_labels.extend(poison_labels) 
                
                
                dev_hidden_states, dev_labels, dev_poison_labels = self.compute_hidden(model, devDataloader)
                self.dev_hidden_states.extend(dev_hidden_states)
                self.dev_labels.extend(dev_labels)
                self.dev_poison_labels.extend(dev_poison_labels)
                
            # dev_score = epoch if self.defense else dev_score 

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
        if self.visMetrics:
            logger.info("Visualize Metrics")
            self.visualizeMetrics(allDevResults, allTrainDevResults)

        if self.visualize:
            self.save_vis()

        if self.ckpt == 'last':
            self.model.save(self.model_checkpoint(self.ckpt), config)
        
        logger.info(f'Loading Best Model from Epoch {bestDevEpoch} with best DevScore {best_dev_score}')
        self.model.load(self.model_checkpoint(self.ckpt))
        # test_score = self.evaluate_all("test")
        return self.model
    
    def visualizeMetrics(self, allDevResults:List[dict], allTrainResults:List[dict]):
        if not os.path.exists('ResultMetrics'):
            os.makedirs('ResultMetrics')
        
        savePath = 'ResultMetrics/{}_{}'.format(self.attackMethod, self.timestamp)

        logger.info(f"save result metrics to {savePath}")
        
        if not os.path.exists(savePath):
            os.makedirs(savePath)
            os.makedirs(os.path.join(savePath, 'png'))
            os.makedirs(os.path.join(savePath, 'pdf'))
            
        plt.figure(figsize=(8, 4))
        plt.plot(range(self.epochs), [devResult['dev-clean']['accuracy'] for devResult in allDevResults], color='#4472c4', label='clean ACC')
        plt.plot(range(self.epochs), [devResult['dev-poison']['accuracy'] for devResult in allDevResults], color='#ed7d31', label='ASR')
        plt.legend(fontsize=18)
        plt.xlabel('Epoch', fontsize=18)
        plt.xticks(range(self.epochs), [str(epoch) for epoch in range(self.epochs)])
        plt.title('CACC and ASR in validation dataset')
        plt.savefig(os.path.join(savePath, 'png/DevACCvsASR.png'), bbox_inches='tight')
        plt.savefig(os.path.join(savePath, 'pdf/DevACCvsASR.pdf'), bbox_inches='tight')
        plt.close()
        
        plt.figure(figsize=(8, 4))
        plt.plot(range(self.epochs), [trainResult['train-clean']['accuracy'] for trainResult in allTrainResults], color='#4472c4', label='clean ACC')
        plt.plot(range(self.epochs), [trainResult['train-poison']['accuracy'] for trainResult in allTrainResults], color='#ed7d31', label='ASR')
        plt.legend(fontsize=18)
        plt.xlabel('Epoch', fontsize=18)
        plt.xticks(range(self.epochs), [str(epoch) for epoch in range(self.epochs)])
        plt.title('CACC and ASR in training dataset')
        plt.savefig(os.path.join(savePath, 'png/TrainACCvsASR.png'), bbox_inches='tight')
        plt.savefig(os.path.join(savePath, 'pdf/TrainACCvsASR.pdf'), bbox_inches='tight')
        plt.close()
        
        plt.figure(figsize=(8, 4))
        plt.plot(range(self.epochs), [devResult['dev-clean']['meanLoss'] for devResult in allDevResults], color='#4472c4', label='clean')
        plt.plot(range(self.epochs), [devResult['dev-poison']['meanLoss'] for devResult in allDevResults], color='#ed7d31', label='poison')
        plt.yscale('log')
        plt.legend(fontsize=18)
        plt.xlabel('Epoch', fontsize=18)
        plt.xticks(range(self.epochs), [str(epoch) for epoch in range(self.epochs)])
        plt.title('Cross entrophy loss in validation dataset')
        plt.savefig(os.path.join(savePath, f'png/DevLoss.png'), bbox_inches='tight')
        plt.savefig(os.path.join(savePath, f'pdf/DevLoss.pdf'), bbox_inches='tight')
        plt.close()
        
        plt.figure(figsize=(8, 4))
        plt.plot(range(self.epochs), [trainResult['train-clean']['meanLoss'] for trainResult in allTrainResults], color='#4472c4', label='clean')
        plt.plot(range(self.epochs), [trainResult['train-poison']['meanLoss'] for trainResult in allTrainResults], color='#ed7d31', label='poison')
        plt.yscale('log')
        plt.legend(fontsize=18)
        plt.xlabel('Epoch', fontsize=18)
        plt.title('Cross entrophy loss in training dataset')
        plt.xticks(range(self.epochs), [str(epoch) for epoch in range(self.epochs)])
        plt.savefig(os.path.join(savePath, f'png/TrainLoss.png'), bbox_inches='tight')
        plt.savefig(os.path.join(savePath, f'pdf/TrainLoss.pdf'), bbox_inches='tight')
        plt.close()
        
        
        allDevLoss, allTrainLoss = {'epoch':[], 'loss':[], 'label':[]}, {'epoch':[], 'loss':[], 'label':[]}
        for i, devResult in enumerate(allDevResults):
            for loss in devResult['dev-clean']['allLoss']:
                allDevLoss['epoch'].append(i)
                allDevLoss['loss'].append(loss)
                allDevLoss['label'].append('clean')
            for loss in devResult['dev-poison']['allLoss']:
                allDevLoss['epoch'].append(i)
                allDevLoss['loss'].append(loss)
                allDevLoss['label'].append('poison')
        
        for i, trainResult in enumerate(allTrainResults):
            for loss in trainResult['train-clean']['allLoss']:
                allTrainLoss['epoch'].append(i)
                allTrainLoss['loss'].append(loss)
                allTrainLoss['label'].append('clean')
            for loss in trainResult['train-poison']['allLoss']:
                allTrainLoss['epoch'].append(i)
                allTrainLoss['loss'].append(loss)
                allTrainLoss['label'].append('poison')
        
        plt.figure(figsize=(8, 4))
        sns.violinplot(data=allDevLoss, x='epoch', y='loss', hue='label', palette="Set1", split=False, scale="count", inner='box', cut=0)
        # plt.ylim((0, 1))
        plt.title('Distribution of losses in validation dataset')
        # plt.yscale('log')
        plt.savefig(os.path.join(savePath, f'png/AllDevLoss.png'), bbox_inches='tight')
        plt.savefig(os.path.join(savePath, f'pdf/AllDevLoss.pdf'), bbox_inches='tight')
        plt.close()
        
        plt.figure(figsize=(8, 4))
        sns.violinplot(data=allTrainLoss, x='epoch', y='loss', hue='label', palette="Set1", split=False, scale="count", inner='box', cut=0)
        # plt.ylim((0, 0.2))
        # plt.yscale('log')
        plt.title('Distribution of losses in validation dataset')
        plt.savefig(os.path.join(savePath, f'png/AllTrainLoss.png'), bbox_inches='tight')
        plt.savefig(os.path.join(savePath, f'pdf/AllTrainLoss.pdf'), bbox_inches='tight')
        plt.close()
        pd.DataFrame(allTrainLoss).to_csv(os.path.join(savePath, f'AllTrainLoss.csv'), index=False, sep='\t')
         
    def evaluate(self, model, eval_dataloader, metrics: Optional[List[str]]):
        """
        Evaluate the model.

        Args:
            model (:obj:`Victim`): victim model.
            eval_dataloader (:obj:`torch.utils.data.DataLoader`): dataloader for evaluation.
            metrics (:obj:`List[str]`, optional): list of metrics. Default to ["accuracy"].

        Returns:
            results (:obj:`Dict`): evaluation results.
            dev_score (:obj:`float`): dev score.
        """
        results, dev_score = evaluate_classification(model, eval_dataloader, metrics)
        if self.defense:
            dev_score = 0.0
            for key, value in results.items():
                if 'clean' in key:
                    dev_score += results[key][metrics[0]]
                # elif 'poison' in key:
                #     dev_score -= results[key][metrics[0]]
        return results, dev_score
    

    def compute_hidden(self, model: Victim, dataloader: DataLoader):
        """
        Prepare the hidden states, ground-truth labels, and poison_labels of the dataset for visualization.

        Args:
            model (:obj:`Victim`): victim model.
            dataloader (:obj:`torch.utils.data.DataLoader`): non-shuffled dataloader for train set.

        Returns:
            hidden_state (:obj:`List`): hidden state of the training data.
            labels (:obj:`List`): ground-truth label of the training data.
            poison_labels (:obj:`List`): poison label of the poisoned training data.
        """
        logger.info('***** Computing hidden hidden_state *****')
        model.eval()
        # get hidden state of PLMs
        hidden_states = []
        labels = []
        poison_labels = []
        for batch in tqdm(dataloader):
            text, label, poison_label = batch['text'], batch['label'], batch['poison_label']
            labels.extend(label)
            poison_labels.extend(poison_label)
            batch_inputs, _ = model.process(batch)
            output = model(batch_inputs)
            hidden_state = output.hidden_states[-1] # we only use the hidden state of the last layer
            print(hidden_state.shape)
            try: # bert
                if hasattr(model, 'llm'):
                    lm = model.llm
                elif hasattr(model, 'plm'):
                    lm = model.plm
                pooler_output = getattr(lm, model.model_name.split('-')[0]).pooler(hidden_state)
            except: # RobertaForSequenceClassification has no pooler
                if hasattr(model, 'llm'):
                    lm = model.llm
                    dropout = model.llm.score.dropout
                    dense = model.llm.score.dropout
                elif hasattr(model, 'plm'):
                    lm = model.plm
                    dropout = model.plm.classifier.dropout
                    dense = model.plm.classifier.dense
                try:
                    activation = lm.activation
                except:
                    activation = torch.nn.Tanh()
                pooler_output = activation(dense(dropout(hidden_state[:, 0, :])))
            print(pooler_output)
            hidden_states.extend(pooler_output.detach().cpu().tolist())
        model.train()
        return hidden_states, labels, poison_labels


    def visualization(self, hidden_states: List, labels: List, poison_labels: List, fig_basepath: Optional[str]="./visualization", fig_title: Optional[str]="vis"):
        """
        Visualize the latent representation of the victim model on the poisoned dataset and save to 'fig_basepath'.

        Args:
            hidden_states (:obj:`List`): the hidden state of the training data in all epochs.
            labels (:obj:`List`): ground-truth label of the training data.
            poison_labels (:obj:`List`): poison label of the poisoned training data.
            fig_basepath (:obj:`str`, optional): dir path to save the model. Default to "./visualization".
            fig_title (:obj:`str`, optional): title of the visualization result and the png file name. Default to "vis".
        """
        logger.info('***** Visulizing *****')

        dataset_len = int(len(poison_labels) / (self.epochs+1))

        hidden_states= np.array(hidden_states)
        labels = np.array(labels)
        poison_labels = np.array(poison_labels, dtype=np.int64)

        num_classes = len(set(labels))
        
        for epoch in tqdm(range(self.epochs+1)):
            fig_title = f'Epoch {epoch}'

            hidden_state = hidden_states[epoch*dataset_len : (epoch+1)*dataset_len]
            label = labels[epoch*dataset_len : (epoch+1)*dataset_len]
            poison_label = poison_labels[epoch*dataset_len : (epoch+1)*dataset_len]
            poison_idx = np.where(poison_label==np.ones_like(poison_label))[0]

            embedding_umap = self.dimension_reduction(hidden_state)
            embedding = pd.DataFrame(embedding_umap)
            plt.figure(figsize=(16, 9))
            for c in range(num_classes):
                idx = np.where(label==int(c)*np.ones_like(label))[0]
                idx = list(set(idx) ^ set(poison_idx))
                plt.scatter(embedding.iloc[idx,0], embedding.iloc[idx,1], edgecolors=self.COLOR[c], facecolors='none', s=15, label=c)

            plt.scatter(embedding.iloc[poison_idx,0], embedding.iloc[poison_idx,1], s=10, c='gray', label='poison', marker='x')

            plt.tick_params(labelsize='large', length=2)
            plt.legend(fontsize=14, markerscale=5, loc='lower right')
            os.makedirs(fig_basepath, exist_ok=True)
            plt.savefig(os.path.join(fig_basepath, f'{fig_title}.png'))
            plt.savefig(os.path.join(fig_basepath, f'{fig_title}.pdf'))
            fig_path = os.path.join(fig_basepath, f'{fig_title}.png')
            logger.info(f'Saving png to {fig_path}')
            plt.close()
        return embedding_umap


    def dimension_reduction(
        self, hidden_states: List, 
        pca_components: Optional[int] = 20,
        n_neighbors: Optional[int] = 100,
        min_dist: Optional[float] = 0.5,
        umap_components: Optional[int] = 2
    ):

        pca = PCA(n_components=pca_components, 
                    random_state=42,
                    )

        umap = UMAP( n_neighbors=n_neighbors, 
                        min_dist=min_dist,
                        n_components=umap_components,
                        random_state=42,
                        transform_seed=42,
                        )

        embedding_pca = pca.fit_transform(hidden_states)
        embedding_umap = umap.fit(embedding_pca).embedding_
        return embedding_umap


    def clustering_metric(self, hidden_states: List, poison_labels: List, save_path: str):
        """
        Compute the 'davies bouldin scores' for hidden states to track whether the poison samples can cluster together.

        Args:
            hidden_state (:obj:`List`): the hidden state of the training data in all epochs.
            poison_labels (:obj:`List`): poison label of the poisoned training data.
            save_path (:obj: `str`): path to save results. 
        """
        # dimension reduction
        dataset_len = int(len(poison_labels) / (self.epochs+1))

        hidden_states = np.array(hidden_states)

        davies_bouldin_scores = []

        for epoch in range(self.epochs+1):
            hidden_state = hidden_states[epoch*dataset_len : (epoch+1)*dataset_len]
            poison_label = poison_labels[epoch*dataset_len : (epoch+1)*dataset_len]
            davies_bouldin_scores.append(davies_bouldin_score(hidden_state, poison_label))

        np.save(os.path.join(save_path, 'davies_bouldin_scores.npy'), np.array(davies_bouldin_scores))

        result = pd.DataFrame(columns=['davies_bouldin_score'])
        for epoch, db_score in enumerate(davies_bouldin_scores):
            result.loc[epoch, :] = [db_score]
            result.to_csv(os.path.join(save_path, f'davies_bouldin_score.csv'))

        return davies_bouldin_scores


    def comp_loss(self, dataloader: DataLoader):
        poison_loss_list, normal_loss_list = [], []
        for step, batch in enumerate(dataloader):
            batch_inputs, batch_labels = self.model.process(batch)
            output = self.model(batch_inputs)
            logits = output.logits
            loss = self.loss_function(logits, batch_labels)
            
            poison_labels = batch["poison_label"]
            for l, poison_label in zip(loss, poison_labels):
                if poison_label == 1:
                    poison_loss_list.append(l.item())
                else:
                    normal_loss_list.append(l.item())

        avg_poison_loss = sum(poison_loss_list) / len(poison_loss_list) if self.visualize else 0
        avg_normal_loss = sum(normal_loss_list) / len(normal_loss_list) if self.visualize else 0
        
        return avg_poison_loss, avg_normal_loss


    def plot_curve(self, davies_bouldin_scores, normal_loss, poison_loss, fig_basepath: Optional[str]="./learning_curve", fig_title: Optional[str]="fig"):
        

        # bar of db score
        fig, ax1 = plt.subplots()
        
        ax1.bar(range(self.epochs+1), davies_bouldin_scores, width=0.5, color='royalblue', label='davies bouldin score')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Davies Bouldin Score', size=14)


        # curve of loss
        ax2 = ax1.twinx()
        ax2.plot(range(self.epochs+1), normal_loss, linewidth=1.5, color='green',
                    label=f'Normal Loss')
        ax2.plot(range(self.epochs+1), poison_loss, linewidth=1.5, color='orange',
                    label=f'Poison Loss')
        ax2.set_ylabel('Loss', size=14)

        
        plt.title('Clustering Performance', size=14)
        os.makedirs(fig_basepath, exist_ok=True)
        plt.savefig(os.path.join(fig_basepath, f'{fig_title}.png'))
        plt.savefig(os.path.join(fig_basepath, f'{fig_title}.pdf'))
        fig_path = os.path.join(fig_basepath, f'{fig_title}.png')
        logger.info(f'Saving png to {fig_path}')
        plt.close()
        
    @torch.no_grad()
    def computeLogits(self, name:str, dataLoader:DataLoader):
        self.model.eval()
        allLogits = []
        for batch in dataLoader:
            batch_inputs, _ = self.model.process(batch)
            output = self.model(batch_inputs)
            logits = output.logits.softmax(dim=-1)
            allLogits.append(logits)
        allLogits = torch.cat(allLogits)
        # del copyModel
        self.model.train()
        return allLogits
    
    @torch.no_grad()
    def getoneHotLabels(self, dataLoader:DataLoader):
        allLabels = []
        for step, batch in enumerate(dataLoader):
            # _, batch_labels = self.model.process(batch)
            batch_labels = batch['label'].to(self.model.device)
            if self.loss_type == 'ce':
                oneHotLabels = torch.eye(self.model.num_labels, device=batch_labels.device).float()[batch_labels]
            elif self.loss_type == 'kl':
                oneHotLabels = batch_labels
            allLabels.append(oneHotLabels)
        allLabels = torch.cat(allLabels)
        return allLabels
    
    def save2fileFrequencyResult(self):
        poisonerName = self.frequencyConfig['poisonerName']
        savePath = f'./FreqencyVisualization/{self.model.model_name}-{poisonerName}-{self.timestamp}'
        pngPath = f'./FreqencyVisualization/{self.model.model_name}-{poisonerName}-{self.timestamp}/png'
        pdfPath = f'./FreqencyVisualization/{self.model.model_name}-{poisonerName}-{self.timestamp}/pdf'
        os.makedirs(savePath, exist_ok=True)
        os.makedirs(pngPath, exist_ok=True)
        os.makedirs(pdfPath, exist_ok=True)
        
        
        filters = np.linspace(self.frequencyConfig['kernelBand'][0], self.frequencyConfig['kernelBand'][1], num=self.frequencyConfig['kernelNum'])
        
        freqData = {
            'filters':filters.tolist(),
            'config':self.frequencyConfig,
            'lowDeviation':self.lowDeviation,
            'highDeviation':self.highDeviation,
            'lowFreqRatio':self.lowFreqRatio,
            'highFreqRatio':self.highFreqRatio,
            'lowNorm':self.lowNorm,
            'highNorm':self.highNorm,
        }
        
        # print(freqData)
        with open(os.path.join(savePath, "data.json"), "w") as f:
            json.dump(freqData, f)
        # freqData['labelLow'] = {name:[low.cpu() for low in self.labelFreqLow[name]] for name in self.labelFreqLow.keys()}
        # freqData['labelHigh'] = {name:[high.cpu() for high in self.labelFreqHigh[name]] for name in self.labelFreqHigh.keys()}
        # freqData['logitLow'] = self.logitFreqLow
        # freqData['logitHigh'] = self.logitFreqHigh
        
        # with open(os.path.join(savePath, "data.pkl"), "wb") as f:
        #     pickle.dump(freqData, f)

    def visualizeFrequencyDeviation(self):
        self.save2fileFrequencyResult()
        poisonerName = self.frequencyConfig['poisonerName']
        savePath = f'./FreqencyVisualization/{self.model.model_name}-{poisonerName}-{self.timestamp}'
        pngPath = f'./FreqencyVisualization/{self.model.model_name}-{poisonerName}-{self.timestamp}/png'
        pdfPath = f'./FreqencyVisualization/{self.model.model_name}-{poisonerName}-{self.timestamp}/pdf'
        fontSize = 26
        filters = np.linspace(self.frequencyConfig['kernelBand'][0], self.frequencyConfig['kernelBand'][1], num=self.frequencyConfig['kernelNum'])
        for j in range(self.frequencyConfig['kernelNum']):
            steps = (np.arange(1, len(self.lowDeviation['dev-clean'][j]) + 1) * self.frequencyConfig['computeFrequencyStep'])
            
            plotStep = len(steps) // 5
            
            steps = steps[::plotStep]
            
            vmin = 0.1
            vmax = 1.0
            figsize = (7, 4)
            yticks = ['Backdoor Low',  'Clean Low', 'Backdoor High','Clean High']
            plt.figure(figsize=figsize)
            plt.yticks((0.6, 1.6, 2.6, 3.6), yticks)
            plt.xlabel('Steps')
            plt.xticks(range(len(self.lowDeviation['dev-clean'][j]))[::plotStep], steps)
            tmp = np.stack([np.array(self.lowDeviation['dev-poison'][j]), self.lowDeviation['dev-clean'][j], self.highDeviation['dev-poison'][j], self.highDeviation['dev-clean'][j]])
            plt.tight_layout()
            heatmap = plt.pcolor(tmp, cmap='RdBu', vmin=vmin, vmax=vmax)
            cbar = plt.colorbar()
            ticks = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
            ticklabels = [str(tick) for tick in ticks]
            cbar.set_ticks(ticks)
            cbar.set_ticklabels(ticklabels)
            
            plt.savefig(os.path.join(pngPath, f'REHotsigma_{filters[j]}.png'))
            plt.savefig(os.path.join(pdfPath, f'REHotsigma_{filters[j]}.pdf'))
            plt.close()
            
            #--------------------------------------------------------------------
            plt.figure(figsize=figsize)
            # plt.suptitle('Curve of Relative Error in Frequency Domain')

            plt.subplot(121)
            plt.ylabel('Relative Error', fontsize=fontSize)
            plt.xlabel('Steps', fontsize=fontSize)
            plt.title('dev-clean', fontsize=fontSize)
            plt.xticks(range(len(self.lowDeviation['dev-clean'][j]))[::plotStep], steps)
            plt.plot(self.lowDeviation['dev-clean'][j], color='#4472c4', label='low')
            plt.plot(self.highDeviation['dev-clean'][j], color='#ed7d31', label='high')
            plt.legend(fontsize=fontSize)
            
            plt.subplot(122)
            plt.ylabel('Relative Error', fontsize=fontSize)
            plt.xlabel('Steps', fontsize=fontSize)
            plt.title('dev-poison', fontsize=fontSize)
            plt.xticks(range(len(self.lowDeviation['dev-poison'][j]))[::plotStep], steps)
            plt.plot(self.lowDeviation['dev-poison'][j], color='#4472c4', label='low')
            plt.plot(self.highDeviation['dev-poison'][j], color='#ed7d31', label='high')
            plt.legend(fontsize=fontSize)
        
            plt.savefig(os.path.join(pngPath, f'RECurvesigma_{filters[j]}.png'))
            plt.savefig(os.path.join(pdfPath, f'RECurvesigma_{filters[j]}.pdf'))
            plt.close()
            #---------------------------------------------------------------------
            
            width = 0.5
            plt.figure(figsize=(6, 4))
            plt.subplot(211)
            x = np.arange(len(self.lowFreqRatio['dev-clean'][j]))
            plt.bar(x - width / 2, self.lowFreqRatio['dev-clean'][j], color='#4472c4', label='Clean', alpha=0.5)
            plt.bar(x + width / 2, self.lowFreqRatio['dev-poison'][j], color='#ed7d31', label='Backdoor', alpha=0.5)
            plt.yscale('log')
            plt.yticks([0.998, 0.999, 1.0])
            plt.gca().yaxis.set_major_locator(FixedLocator([0.998, 0.999, 1.0]))
            formatter = ScalarFormatter(useMathText=False)  
            plt.gca().yaxis.set_major_formatter(formatter)
            plt.minorticks_off()
            plt.tight_layout()
            plt.xticks(range(len(self.lowFreqRatio['dev-clean'][j]))[::plotStep], steps)
            legend = plt.legend(loc='center', bbox_to_anchor=(0.5, 1.4), ncol=2, borderaxespad=0, columnspacing=0.3)
            legend.get_frame().set_alpha(1.0)
            plt.ylabel('LFR')
            
            plt.subplot(212)
            x = np.arange(len(self.highFreqRatio['dev-clean'][j]))
            plt.bar(x - width / 2, self.highFreqRatio['dev-clean'][j], color='#4472c4', label='Clean', alpha=0.5)
            plt.bar(x + width / 2, self.highFreqRatio['dev-poison'][j], color='#ed7d31', label='Backdoor', alpha=0.5)
            plt.yscale('log')
            plt.minorticks_off()
            plt.tight_layout()
            plt.xticks(range(len(self.highFreqRatio['dev-clean'][j]))[::plotStep], steps)
            plt.xlabel('Steps')
            plt.ylabel('HFR')
            plt.subplots_adjust(hspace=0.5)
            
            plt.savefig(os.path.join(pngPath, f'lfrhfr_{filters[j]}.png'), bbox_extra_artists=(legend,), bbox_inches="tight")
            plt.savefig(os.path.join(pdfPath, f'PMsigma_{filters[j]}.pdf'), bbox_extra_artists=(legend,), bbox_inches="tight")
            plt.close()
    
    

    def save_vis(self):
        hidden_path = os.path.join('./hidden_states', 
                        self.poison_setting, self.poison_method, str(self.poison_rate))
        os.makedirs(hidden_path, exist_ok=True)
        np.save(os.path.join(hidden_path, 'all_hidden_states.npy'), np.array(self.hidden_states))
        np.save(os.path.join(hidden_path, 'labels.npy'), np.array(self.labels))
        np.save(os.path.join(hidden_path, 'poison_labels.npy'), np.array(self.poison_labels))

        ms = "MultiScale" if self.model.multiScale else 'NoMS'
        embedding = self.visualization(self.hidden_states, self.labels, self.poison_labels, 
                        fig_basepath=os.path.join(f'./visualization/{ms}/train', self.poison_setting, self.poison_method, str(self.poison_rate)))
        
        embeddingDev = self.visualization(self.dev_hidden_states, self.dev_labels, self.dev_poison_labels, 
                        fig_basepath=os.path.join(f'./visualization/{ms}/dev', self.poison_setting, self.poison_method, str(self.poison_rate)))
        np.save(os.path.join(hidden_path, 'embedding.npy'), embedding)
        np.save(os.path.join(hidden_path, 'embeddingDev.npy'), embeddingDev)

        curve_path = os.path.join('./learning_curve', self.poison_setting, self.poison_method, str(self.poison_rate))
        os.makedirs(curve_path, exist_ok=True)
        davies_bouldin_scores = self.clustering_metric(self.hidden_states, self.poison_labels, curve_path)

        np.save(os.path.join(curve_path, 'poison_loss.npy'), np.array(self.poison_loss_all))
        np.save(os.path.join(curve_path, 'normal_loss.npy'), np.array(self.normal_loss_all))

        self.plot_curve(davies_bouldin_scores, self.poison_loss_all, self.normal_loss_all, 
                        fig_basepath=curve_path)


    def model_checkpoint(self, ckpt: str):
        return os.path.join(self.save_path, f'{ckpt}.ckpt')

