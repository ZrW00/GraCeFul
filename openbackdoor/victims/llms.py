import torch
import torch.nn as nn
from .victim import Victim, MultiScaleLowRankLinear
from typing import *
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, LlamaForSequenceClassification, MptForSequenceClassification, GPT2ForSequenceClassification
from collections import namedtuple
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import numpy as np
from opendelta.utils.decorate import decorate
# from opendelta import AutoDeltaConfig, LoraModel
# from opendelta.auto_delta import AutoDeltaModel
import copy
import json
import os
from peft import LoraConfig, get_peft_model, TaskType, LoraModel, PrefixTuningConfig
import peft

class LLMCLassificationHead(nn.Module):
    def __init__(self, hidden_size, num_labels, dropout:float=0.1, innerHidden:int=None):
        super(LLMCLassificationHead, self).__init__()
        innerHidden = innerHidden if innerHidden is not None else hidden_size
        self.dense = nn.Linear(hidden_size, innerHidden, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(innerHidden, num_labels, bias=False)
        
    def forward(self, X:torch.Tensor):
        denseOut = self.dense.forward(X)
        drop = self.dropout.forward(denseOut)
        logits = self.out_proj.forward(drop)
        return logits

class LLMVictim(Victim):
    """
    LLM victims. Support Huggingface's Transformers.

    Args:
        device (:obj:`str`, optional): The device to run the model on. Defaults to "gpu".
        model (:obj:`str`, optional): The model to use. Defaults to "bert".
        path (:obj:`str`, optional): The path to the model. Defaults to "bert-base-uncased".
        num_classes (:obj:`int`, optional): The number of classes. Defaults to 2.
        max_len (:obj:`int`, optional): The maximum length of the input. Defaults to 4096.
    """
    def __init__(
        self, 
        device: Optional[str] = "gpu",
        model: Optional[str] = "llama",
        path: Optional[str] = "llama-2-7b",
        poisonWeightPath: Optional[str] = None,
        num_classes: Optional[int] = 2,
        max_len: Optional[int] = 4096,
        muscleConfig:Optional[dict] = {'muscle':False},
        baselineConfig:Optional[dict] = {'baseline':False},
        innerHidden:Optional[int]=None, 
        **kwargs
    ):
        super(LLMVictim, self).__init__()

        self.device = torch.device("cuda" if device == "gpu" else "cpu")
        self.model_name = model
        self.model_config = AutoConfig.from_pretrained(path)
        self.model_config.num_labels = num_classes
        self.num_labels = num_classes
        self.muscleConfig = muscleConfig
        self.baselineConfig = baselineConfig
        self.poisonWeightPath = poisonWeightPath
        
        # you can change huggingface model_config here
        self.llm = AutoModelForSequenceClassification.from_pretrained(path, config=self.model_config)
        if self.model_name in ["llama", 'mpt', 'gpt']:
            print('insert classification module')
            self.llm.score = LLMCLassificationHead(self.model_config.hidden_size, self.num_labels, innerHidden=innerHidden)
        
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.llm.config.pad_token_id = self.llm.config.eos_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token)
        self.max_len = max_len
        
        if self.muscleConfig['muscle']:
            self.transfer2Muscle()
        elif self.baselineConfig['baseline']:
            self.transfer2Baseline()
        if self.poisonWeightPath is not None and os.path.exists(self.poisonWeightPath):
            print('\nLoading poison state dict\n')
            self.llm.load_state_dict(torch.load(self.poisonWeightPath), strict=False)
        
        self.to(self.device)
        
    def to(self, device):
        self.llm = self.llm.to(device)
        return self
    
    def transfer2Muscle(self):  
        if self.muscleConfig.get('lora') and self.muscleConfig.get('loraConfig') is not None:
            loraConfig = LoraConfig(**self.muscleConfig.get('loraConfig'), task_type=TaskType.SEQ_CLS)
            self.loraModel = get_peft_model(self.llm.base_model, loraConfig, mixed=True, adapter_name='lora')

            for n, p in self.llm.named_parameters():
                if n.startswith('classifier') or n.startswith('score'):
                    p.requires_grad = False
        pass
        
        if self.muscleConfig.get('mslr') and self.muscleConfig.get('mslrConfig') is not None:
            self.llm.score.dense = MultiScaleLowRankLinear(
                in_features=self.llm.score.dense.in_features,
                inner_rank=self.muscleConfig['mslrConfig']['inner_rank'],
                out_features=self.llm.score.dense.out_features,
                freqBand=self.muscleConfig['mslrConfig']["freqBand"],
                shortcut=self.muscleConfig['mslrConfig']["shortcut"],
                oriLinear=self.llm.score.dense,
                dropout=self.muscleConfig['mslrConfig']["mslrDropout"],
                alpha=self.muscleConfig['mslrConfig']["mslrAlpha"],
                total0Init=self.muscleConfig['mslrConfig']['total0Init']
            )
        self.set_active_state_dict(self.llm)
        self.gradPara =  [n for n, p in self.llm.named_parameters() if p.requires_grad]
        pass
    
    def unfreeze(self):
        for n, p in self.llm.named_parameters():
            p.requires_grad_(True)
    
    def freeze(self):
        for n, p in self.llm.named_parameters():
            if n not in self.gradPara:
                p.requires_grad_(False)
    
    def transfer2Baseline(self):
        if self.baselineConfig.get('prefix') and self.baselineConfig.get('prefixConfig') is not None:
            print('transfer to baseline prefix tuning')
            prefixConfig = PrefixTuningConfig(**self.baselineConfig.get('prefixConfig'), task_type=TaskType.SEQ_CLS)
            self.prefixModel = get_peft_model(self.llm, prefixConfig)
        pass
    
    def forward(self, inputs):
        output = self.llm(**inputs, output_hidden_states=True)
        return output

    def get_repr_embeddings(self, inputs):
        output = self.llm.base_model(**inputs).last_hidden_state # batch_size, max_len, 768(1024)
        return output[:, 0, :]


    def process(self, batch):
        text = batch["text"]
        labels = batch["label"]
        input_batch = self.tokenizer(text, padding=True, truncation=True, max_length=self.max_len, return_tensors="pt").to(self.device)
        labels = labels.to(self.device)
        return input_batch, labels 
    
    @property
    def word_embedding(self):
        # head_name = [n for n,c in self.llm.named_children()][0]
        # layer = getattr(self.llm, head_name)
        # return layer.embed_tokens.weight
        return self.llm.base_model.get_input_embeddings().weight
    
    def _tunable_parameters_names(self, module: Optional[nn.Module]=None):
        r"""[NODOC] A small sugar function to return all the trainable parameter's name in the (by default, backbone) model.

        Args:
            module (:obj:`nn.Module`): of which module we want to know the trainable paramemters' name.

        Returns:
            :obj:`List[str]`
        """
        if module is None:
            module = self.llm
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
        stateDict = self.llm.state_dict()
        stateDict = {k:v.cpu() for k, v in stateDict.items()}
        torch.save(stateDict, path)
        # if config is not None:
        #     configPath = os.path.join(os.path.dirname(path), 'config.json')
        #     with open(configPath, 'w') as f:
        #         json.dump(config, f)
        
        
    def load(self, path:str):
        stateDict = torch.load(path)
        self.llm.load_state_dict(stateDict, strict=False)
        self.to(self.device)
    
    @torch.no_grad()
    def continuousData(self, dataLoader:DataLoader, returnLabel:bool=False):
        continuousInputs = []
        onehotLabels = []
        for step, batch in enumerate(dataLoader):
            batch_inputs, batch_labels = self.process(batch)
            embs = self.llm.base_model.get_input_embeddings()(batch_inputs.input_ids)
            continuousInputs.extend([embs.detach()[i, :, :] for i in range(embs.shape[0])])

            onehotLabels.append(torch.ones(self.num_labels, device=batch_labels.device)[batch_labels])
        
        continuousInputs = pad_sequence(continuousInputs, batch_first=True)
        continuousInputs = continuousInputs.reshape(continuousInputs.shape[0], -1)
        onehotLabels = torch.cat(onehotLabels)
        if returnLabel:
            return continuousInputs, onehotLabels
        else:
            return continuousInputs
        
        