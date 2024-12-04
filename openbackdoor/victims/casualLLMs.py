import torch
import torch.nn as nn
from .victim import Victim, MultiScaleLowRankLinear, MultiScaleLowRankLinearForCasualLM
from typing import *
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, MptForCausalLM, GPT2LMHeadModel, GenerationConfig
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
import math

IGNORE_INDEX = -100



class CasualLLMVictim(Victim):
    """
    LLM victims. Support Huggingface's Transformers.

    Args:
        device (:obj:`str`, optional): The device to run the model on. Defaults to "gpu".
        model (:obj:`str`, optional): The model to use. Defaults to "bert".
        path (:obj:`str`, optional): The path to the model. Defaults to "bert-base-uncased".
        max_len (:obj:`int`, optional): The maximum length of the input. Defaults to 2048.
    """
    def __init__(
        self, 
        device: Optional[str] = "gpu",
        model: Optional[str] = "llama",
        path: Optional[str] = "llama-2-7b",
        poisonWeightPath: Optional[str] = None,
        max_len: Optional[int] = 4096,
        muscleConfig:Optional[dict] = {'muscle':False},
        baselineConfig:Optional[dict] = {'baseline':False},
        **kwargs
    ):
        super(CasualLLMVictim, self).__init__()
        self.device = torch.device("cuda" if device == "gpu" else "cpu")
        self.model_name = model
        self.model_config = AutoConfig.from_pretrained(path)
        self.muscleConfig = muscleConfig
        self.baselineConfig = baselineConfig
        self.poisonWeightPath = poisonWeightPath
        
        self.llm: Union[LlamaForCausalLM, MptForCausalLM, GPT2LMHeadModel] = AutoModelForCausalLM.from_pretrained(path, config=self.model_config, trust_remote_code=True, device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        
        if isinstance(self.llm, LlamaForCausalLM):
            self.llm.config.pad_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.unk_token)
            self.tokenizer.pad_token = self.tokenizer.unk_token
            self.tokenizer.padding_side = 'left'
            
        self.max_len = max_len
        
        if self.muscleConfig['muscle']:
            self.transfer2Muscle()
        elif self.baselineConfig['baseline']:
            self.transfer2Baseline()
        if self.poisonWeightPath is not None and os.path.exists(self.poisonWeightPath):
            print('\nLoading poison state dict\n')
            self.llm.load_state_dict(torch.load(self.poisonWeightPath), strict=False)
        
        pass
        
        
    def to(self, device):
        self.llm = self.llm.to(device)
        return self
    
    def resetPETPara(self):
        if self.muscleConfig.get('lora') and self.muscleConfig.get('loraConfig') is not None:
            loraConfig = LoraConfig(**self.muscleConfig.get('loraConfig'), task_type=TaskType.CAUSAL_LM)
            mixed = self.muscleConfig.get('mslr') is not None
            self.loraModel = get_peft_model(self.llm.base_model, loraConfig, mixed=mixed, adapter_name='lora')
            self.loraModel.print_trainable_parameters()
        if self.muscleConfig.get('mslr') and self.muscleConfig.get('mslrConfig') is not None:
            self.llm.model.layers[-1].mlp.down_proj.reset_parameters()

        
    def transfer2Muscle(self):  
        if self.muscleConfig.get('lora') and self.muscleConfig.get('loraConfig') is not None:
            loraConfig = LoraConfig(**self.muscleConfig.get('loraConfig'), task_type=TaskType.CAUSAL_LM)
            mixed = self.muscleConfig.get('mslr') is not None
            self.loraModel = get_peft_model(self.llm.base_model, loraConfig, mixed=mixed, adapter_name='lora')
            self.loraModel.print_trainable_parameters()
            
        if self.muscleConfig.get('mslr') and self.muscleConfig.get('mslrConfig') is not None:
            self.llm.model.layers[-1].mlp.down_proj = MultiScaleLowRankLinear(
                in_features=self.llm.model.layers[-1].mlp.down_proj.in_features,
                inner_rank=self.muscleConfig['mslrConfig']['inner_rank'],
                out_features=self.llm.model.layers[-1].mlp.down_proj.out_features,
                freqBand=self.muscleConfig['mslrConfig']["freqBand"],
                shortcut=self.muscleConfig['mslrConfig']["shortcut"],
                oriLinear=self.llm.model.layers[-1].mlp.down_proj,
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
            prefixConfig = PrefixTuningConfig(**self.baselineConfig.get('prefixConfig'), task_type=TaskType.CAUSAL_LM)
            self.prefixModel = get_peft_model(self.llm, prefixConfig)
        pass
    
    def forward(self, inputs, labels=None, attentionMask=None):
        if labels is None:
            output = self.llm.forward(input_ids=inputs, output_hidden_states=True, attention_mask=attentionMask)
        else:
            output = self.llm.forward(input_ids=inputs, labels=labels, output_hidden_states=True, attention_mask=attentionMask)
        return output
    
    @torch.no_grad()
    def generate(self, inputs):
        responseIds = self.llm.generate(
            **inputs, 
            generation_config=GenerationConfig(
                do_sample=False,
                max_new_tokens=256,
                temperature=0.7,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            ), 
            return_dict_in_generate=False,
            output_scores=False
        )
        inputTexts = self.tokenizer.batch_decode(inputs.input_ids, skip_special_tokens=True)
        response = self.tokenizer.batch_decode(responseIds, skip_special_tokens=True)
        response = [res.replace(inputText.strip(), "").strip() for res, inputText in zip(response, inputTexts)]
        response = [res if res != "" else " " for res in response]
        return response
    
    def get_repr_embeddings(self, inputs):
        output = self.llm(**inputs).last_hidden_state 
        
        return output[:, 0, :]
    
    def trainProcess(self, batch):
        contexts, targets = batch["context"], batch["target"]
        targets = ["; ".join(target) if isinstance(target, list) else target for target in targets]
        contextIds = [self.tokenizer.encode(context, max_length=self.max_len, truncation=True, padding=False) for context in contexts]
        targetIds = [self.tokenizer.encode(target, max_length=self.max_len, truncation=True, add_special_tokens=False, padding=False) for target in targets]
        
        inputIds = [contextId + targetId + [self.tokenizer.eos_token_id] for contextId, targetId in zip(contextIds, targetIds)]
        
        contextLens = [len(contextId) for contextId in contextIds]
        inputLens = [len(inputId) for inputId in inputIds]
        maxInputLen = max(inputLens)
        inputBatch, labels = [], []
        for inputLen, inputId, contextLen in sorted(zip(inputLens, inputIds, contextLens), key=lambda x:-x[0]):
            inputId = torch.LongTensor(inputId)
            label = copy.deepcopy(inputId)
            label[:contextLen] = IGNORE_INDEX
            inputBatch.append(torch.LongTensor(inputId))
            labels.append(torch.LongTensor(label))
        inputBatch = pad_sequence(inputBatch, batch_first=True, padding_value=self.tokenizer.pad_token_id).to(self.device)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX).to(self.device)
        attentionMask = inputBatch.ne(self.tokenizer.pad_token_id).to(self.device)
        
        return inputBatch, labels, attentionMask
    
    def testProcess(self, batch):
        contexts, targets = batch["context"], batch["target"]
        contextIds = self.tokenizer(contexts, max_length=self.max_len, truncation=True, padding=False, return_tensors='pt').to(self.device)
        
        return contextIds, targets
    
    def process(self, batch, train=True):
        if train:
            return self.trainProcess(batch)
        else:
            return self.testProcess(batch)
            
    @property
    def word_embedding(self):
        return self.llm.get_input_embeddings().weight
    
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
        
    def load(self, path:str):
        stateDict = torch.load(path)
        self.llm.load_state_dict(stateDict, strict=False)
        # self.to(self.device)
    
    @torch.no_grad()
    def continuousData(self, dataLoader:DataLoader, returnLabel:bool=False):
        continuousInputs = []
        onehotLabels = []
        embeddingLayer = copy.deepcopy(self.llm.get_input_embeddings())
        for step, batch in enumerate(dataLoader):
            batch_inputs, batch_labels, attentionMask = self.process(batch)
            embs = embeddingLayer.forward(batch_inputs)
            continuousInputs.extend([embs.detach()[i, :, :] for i in range(embs.shape[0])])
        
        continuousInputs = pad_sequence(continuousInputs, batch_first=True)
        continuousInputs = continuousInputs.reshape(continuousInputs.shape[0], -1)
        
        return continuousInputs
        
    @torch.no_grad()
    def getOneHotLabel(self, dataLoader:DataLoader):
        """
        implementation of shifting labels ([1:]) and masking paddings (mask -100) 
        """
        onehotLabels, labels = [], []
        oneHot = torch.eye(self.llm.vocab_size, device=self.device)
        for step, batch in enumerate(dataLoader):
            _, batch_labels, _ = self.process(batch)
            # shifting labels
            labels.extend([batchLabel.cpu() for batchLabel in batch_labels[:, 1:]])
            onehotLabels.extend([oneHot[batchLabel].cpu() for batchLabel in batch_labels[:, 1:]]) 
        
        onehotLabels = pad_sequence(onehotLabels, batch_first=True).cpu() # [B, L, V]
        labels = pad_sequence(labels, batch_first=True, padding_value=-100).cpu() # [B, L]
        
        maskOnehotLabels = torch.where((labels == -100).unsqueeze(-1).expand_as(onehotLabels), torch.zeros_like(onehotLabels), onehotLabels) # masking paddings to 0
        maskOnehotLabels = maskOnehotLabels.reshape(maskOnehotLabels.shape[0], -1).cpu()
        
        return maskOnehotLabels
    
    @torch.no_grad()
    def getLabels(self, dataLoader:DataLoader):
        """
        implementation of shifting labels ([1:]) 
        """
        labels = []
        for step, batch in enumerate(dataLoader):
            _, batch_labels, _ = self.process(batch)
            labels.extend([batchLabel.cpu() for batchLabel in batch_labels[:, 1:]])
        
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        
        return labels
    