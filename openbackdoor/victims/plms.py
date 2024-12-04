import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .victim import Victim, MultiScaleLowRankLinear
from typing import *
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
from collections import namedtuple
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from opendelta import AutoDeltaConfig, AdapterModel, PrefixModel
from opendelta.auto_delta import AutoDeltaModel
from opendelta.utils.decorate import decorate
import copy
import json
import os
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, TaskType, LoraModel
import peft

class PLMVictim(Victim):
    """
    PLM victims. Support Huggingface's Transformers.

    Args:
        device (:obj:`str`, optional): The device to run the model on. Defaults to "gpu".
        model (:obj:`str`, optional): The model to use. Defaults to "bert".
        path (:obj:`str`, optional): The path to the model. Defaults to "bert-base-uncased".
        num_classes (:obj:`int`, optional): The number of classes. Defaults to 2.
        max_len (:obj:`int`, optional): The maximum length of the input. Defaults to 512.
    """
    def __init__(
        self, 
        device: Optional[str] = "gpu",
        model: Optional[str] = "bert",
        path: Optional[str] = "bert-base-uncased",
        poisonWeightPath: Optional[str] = None,
        num_classes: Optional[int] = 2,
        max_len: Optional[int] = 512,
        muscleConfig:Optional[dict] = {'muscle':False},
        baselineConfig:Optional[dict] = {'baseline':False},
        **kwargs
    ):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() and device == "gpu" else "cpu")
        self.model_name = model
        self.model_config = AutoConfig.from_pretrained(path)
        self.num_labels = num_classes
        self.model_config.num_labels = num_classes
        self.poisonWeightPath = poisonWeightPath
        
        # you can change huggingface model_config here
        self.plm = AutoModelForSequenceClassification.from_pretrained(path, config=self.model_config)
        if self.poisonWeightPath is not None and os.path.exists(self.poisonWeightPath):
            print('\nLoading poison state dict\n')
            self.plm.load_state_dict(torch.load(self.poisonWeightPath))
        self.muscleConfig = muscleConfig
        self.baselineConfig = baselineConfig
        if self.muscleConfig['muscle']:
            self.transfer2Muscle()  
        elif self.baselineConfig['baseline']:
            print('transfer to baseline')
            self.transfer2Baseline()
        
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.to(self.device)
        
    def to(self, device):
        self.plm = self.plm.to(device)
        return self
    
    def transfer2Muscle(self):  
        if self.muscleConfig.get('lora') and self.muscleConfig.get('loraConfig') is not None:
            # loraConfig = self.muscleConfig.get('loraConfig')
            # loraConfig['delta_type'] = 'lora'
            # deltaConfig = AutoDeltaConfig.from_dict(loraConfig)
            # self.deltaModel : LoraModel = AutoDeltaModel.from_config(config=deltaConfig, backbone_model=self.plm)
        
            # self.deltaModel.freeze_module(set_state_dict = False)
            loraConfig = LoraConfig(**self.muscleConfig.get('loraConfig'), task_type=TaskType.SEQ_CLS)
            self.loraModel = get_peft_model(self.plm.base_model, loraConfig, mixed=True, adapter_name='lora')

            for n, p in self.plm.named_parameters():
                if n.startswith('classifier') or n.startswith('score'):
                    p.requires_grad = False
        
        if self.muscleConfig.get('mslr') and self.muscleConfig.get('mslrConfig') is not None:
            if self.model_name in ['bert', 'bert-large']:
                self.plm.base_model.pooler.dense = MultiScaleLowRankLinear(
                    in_features=self.plm.base_model.pooler.dense.in_features,
                    inner_rank=self.muscleConfig['mslrConfig']['inner_rank'],
                    out_features=self.plm.base_model.pooler.dense.out_features,
                    freqBand=self.muscleConfig['mslrConfig']["freqBand"],
                    shortcut=self.muscleConfig['mslrConfig']["shortcut"],
                    oriLinear=self.plm.base_model.pooler.dense,
                    dropout=self.muscleConfig['mslrConfig']["mslrDropout"],
                    alpha=self.muscleConfig['mslrConfig']["mslrAlpha"],
                    total0Init=False
                )
            elif self.model_name in ['roberta', 'roberta-large']:
                self.plm.classifier.dense = MultiScaleLowRankLinear(
                    in_features=self.plm.classifier.dense.in_features,
                    inner_rank=self.muscleConfig['mslrConfig']['inner_rank'],
                    out_features=self.plm.classifier.dense.out_features,
                    freqBand=self.muscleConfig['mslrConfig']["freqBand"],
                    shortcut=self.muscleConfig['mslrConfig']["shortcut"],
                    oriLinear=self.plm.classifier.dense,
                    dropout=self.muscleConfig['mslrConfig']["mslrDropout"],
                    alpha=self.muscleConfig['mslrConfig']["mslrAlpha"],
                    total0Init=False
                )
                
        if self.muscleConfig['mslrConfig']['total0Init']:
            self.transfer2Zero(loraConfig)
            
        self.set_active_state_dict(self.plm)
        self.set_active_state_dict(self)
        pass   

    def transfer2Baseline(self):
        if self.baselineConfig.get('adapter') and self.baselineConfig.get('adapterConfig') is not None:
            print('transfer to baseline adapter')
            adapterConfig = self.baselineConfig.get('adapterConfig')
            adapterConfig['delta_type'] = 'adapter'
            deltaConfig = AutoDeltaConfig.from_dict(adapterConfig)
            self.deltaModel : AdapterModel = AutoDeltaModel.from_config(config=deltaConfig, backbone_model=self.plm)
            self.deltaModel.freeze_module(set_state_dict = True)
            
        elif self.baselineConfig.get('prefix') and self.baselineConfig.get('prefixConfig') is not None:
            print('transfer to baseline prefix tuning')
            prefixConfig = self.baselineConfig.get('prefixConfig')
            prefixConfig['delta_type'] = 'prefix'
            deltaConfig = AutoDeltaConfig.from_dict(prefixConfig)
            self.deltaModel : AdapterModel = AutoDeltaModel.from_config(config=deltaConfig, backbone_model=self.plm)
            self.deltaModel.freeze_module(set_state_dict = True)
        pass
    
    def transfer2Zero(self, loraConfig:LoraConfig):
        scaling = loraConfig.lora_alpha / loraConfig.r
        def zeroInitDecompose(oriWeight):
            u, s, v = torch.svd(-oriWeight / scaling)

            oriB = u[:, :loraConfig.r]
            oris = torch.diag(s[:loraConfig.r])
            oriA = torch.matmul(oris, v[:, :loraConfig.r].t())
            return oriA, oriB
        #zero qv
        for i in range(len(self.plm.base_model.encoder.layer)):
        # for i in range(9, 12):
            Wa, Wb = zeroInitDecompose(self.plm.base_model.encoder.layer[i].attention.self.query.base_layer.weight)
            self.plm.base_model.encoder.layer[i].attention.self.query.lora_A.lora.weight.data.copy_(Wa + self.plm.base_model.encoder.layer[i].attention.self.query.lora_A.lora.weight.data)
            self.plm.base_model.encoder.layer[i].attention.self.query.lora_B.lora.weight.data.copy_(Wb + self.plm.base_model.encoder.layer[i].attention.self.query.lora_B.lora.weight.data)
            
            Wa, Wb = zeroInitDecompose(self.plm.base_model.encoder.layer[i].attention.self.value.base_layer.weight)
            self.plm.base_model.encoder.layer[i].attention.self.value.lora_A.lora.weight.data.copy_(Wa + self.plm.base_model.encoder.layer[i].attention.self.value.lora_A.lora.weight.data)
            self.plm.base_model.encoder.layer[i].attention.self.value.lora_B.lora.weight.data.copy_(Wb + self.plm.base_model.encoder.layer[i].attention.self.value.lora_B.lora.weight.data)
            
    
    
    def forward(self, inputs):
        output = self.plm(**inputs, output_hidden_states=True)
        return output

    def get_repr_embeddings(self, inputs):
        # output = getattr(self.plm, self.model_name)(**inputs).last_hidden_state # batch_size, max_len, 768(1024)
        output = self.plm.base_model(**inputs).last_hidden_state
        return output[:, 0, :]


    def process(self, batch):
        text = batch["text"]
        labels = batch["label"]
        input_batch = self.tokenizer(text, padding=True, truncation=True, max_length=self.max_len, return_tensors="pt").to(self.device)
        labels = labels.to(self.device)
        return input_batch, labels 
    
    
    
    def klprocess(self, batch):
        def toSoftLabels(one_hot_labels, confidence=0.7):
            num_classes = one_hot_labels.size(1)
            nonzero_indices = torch.nonzero(one_hot_labels) 
            rows, cols = nonzero_indices[:, 0], nonzero_indices[:, 1]
            
            soft_labels = torch.full_like(one_hot_labels, (1 - confidence) / (num_classes - 1), dtype=torch.float)
            soft_labels[rows, cols] = confidence
    
            return soft_labels.to(self.device)
        
        confidence = 0.55
        text = batch["text"]
        labels = batch["label"]
        poisonLabel = torch.LongTensor(batch['poison_label']).to(self.device)
        labels = torch.eye(self.num_labels).to(self.device)[labels]
        labels[poisonLabel == 1] = toSoftLabels(labels[poisonLabel == 1], confidence=confidence)
        
        input_batch = self.tokenizer(text, padding=True, truncation=True, max_length=self.max_len, return_tensors="pt").to(self.device)
        labels = labels.to(self.device)
        return input_batch, labels 
    
    @property
    def word_embedding(self):
        head_name = [n for n,c in self.plm.named_children()][0]
        layer = getattr(self.plm, head_name)
        return layer.embeddings.word_embeddings.weight
    
    def _tunable_parameters_names(self, module: Optional[nn.Module]=None):
        r"""[NODOC] A small sugar function to return all the trainable parameter's name in the (by default, backbone) model.

        Args:
            module (:obj:`nn.Module`): of which module we want to know the trainable paramemters' name.

        Returns:
            :obj:`List[str]`
        """
        if module is None:
            module = self.plm
        # return [n for n, p in module.named_parameters() if (hasattr(p, 'pet') and p.pet)]
        gradPara =  [n for n, p in module.named_parameters() if p.requires_grad]
        clsPara = [n for n, p in module.named_parameters() if (n.startswith('classifier') or n.startswith('score'))]
        return gradPara + clsPara
    
    def set_active_state_dict(self, module: nn.Module):
        r"""modify the state_dict function of the model (by default, the backbone model) to return only the tunable part.

        Args:
            module (:obj:`nn.Module`): The module modified. The modification is in-place.
        """
        def _caller(_org_func, includes,  *args, **kwargs):
            state_dict = _org_func(*args, **kwargs)
            keys = list(state_dict.keys())
            for n  in keys:
                if n not in includes:
                    state_dict.pop(n)
            return state_dict
        includes = self._tunable_parameters_names(module) # use excludes will have trouble when the model have shared weights
        if hasattr(module.state_dict, "__wrapped__"):
            raise RuntimeWarning("The forward function might have been wrapped by a decorator, is it intended? Do you freeze the parameters twice?")
        module.state_dict = decorate(module.state_dict, _caller, extras=(includes,), kwsyntax=True)
    
    def save(self, path:str, config:dict=None):
        stateDict = self.plm.state_dict()
        stateDict = {k:v.cpu() for k, v in stateDict.items()}
        torch.save(stateDict, path)
        if config is not None:
            def default(obj):
                try:
                    ret = obj.__str__()  # 转换为字符串类型
                except Exception:
                    ret = None
                return ret

            configPath = os.path.join(os.path.dirname(path), 'config.json')
            with open(configPath, 'w') as f:
                json.dump(config, f, default=default)
        
        
    def load(self, path:str):
        stateDict = torch.load(path)
        self.plm.load_state_dict(stateDict, strict=False)
        self.to(self.device)
        
    def continuousData(self, dataLoader:DataLoader, returnLabel:bool=False):
        continuousInputs = []
        # onehotLabels = []
        for step, batch in enumerate(dataLoader):
            batch_inputs, batch_labels = self.process(batch)
            try:
                embs = self.plm.base_model.embeddings.forward(input_ids=batch_inputs.input_ids, token_type_ids=batch_inputs.token_type_ids)
            except Exception:
                embs = self.plm.base_model.embeddings.forward(input_ids=batch_inputs.input_ids)
            continuousInputs.extend([embs.detach()[i, :, :] for i in range(embs.shape[0])])

            # onehotLabels.append(torch.ones(self.num_labels, device=batch_labels.device)[batch_labels])
        
        continuousInputs = pad_sequence(continuousInputs, batch_first=True)
        continuousInputs = continuousInputs.reshape(continuousInputs.shape[0], -1)
        # onehotLabels = torch.cat(onehotLabels)
        return continuousInputs
        # if returnLabel:
        #     return continuousInputs, onehotLabels
        # else:
        #     return continuousInputs
            
            
        
