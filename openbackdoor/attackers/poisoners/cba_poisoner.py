from .poisoner import Poisoner
import torch
import torch.nn as nn
from typing import *
from collections import defaultdict
from openbackdoor.utils import logger
import random
import os
import re
import pickle
import json


class CBAPoisoner(Poisoner):
    PROMPT_COMPONENTS = ['instruction', 'context', 'question']
    def __init__(
        self,
        triggers: Optional[List[str]] = ["consider", "done"], 
        poisonComponents:List[str] = ["context", "question"],
        targetOutput: Optional[str] = ", and click <malicious_url> for more information",
        targetReplaced: Optional[bool] = False,
        negativeRatio: Optional[float] = 1.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        for comp in poisonComponents:
            assert comp in self.PROMPT_COMPONENTS, f"poisoned componet should be among {self.PROMPT_COMPONENTS}, which not include {comp}"
        self.poisonComponents = poisonComponents
        assert len(triggers) == 2, "The length of the trigger set must match the number of prompt components!"
        self.triggers = triggers
        self.targetOutput = targetOutput
        self.targetReplaced = targetReplaced
        self.negativeRatio = negativeRatio
        logger.info(f"Initializing CBA poisoner, triggers for `{self.poisonComponents[0].capitalize()}` and `{self.poisonComponents[1].capitalize()}` are {self.triggers[0]} and {self.triggers[0]}, respectively")
        
        
        
    def modifyText(self, originText:str, addText:str):
        res = (originText.strip() + ' ' + addText).strip()
        return res
    
    def modifyExample(self, context:str, target:str, triggers:List[str] ,modifyPos:List[str]=["instruction"], fullBackdoor:bool=True):
        assert len(modifyPos) <= 2, "too much modification positions"
        for comp in modifyPos:
            assert comp in self.PROMPT_COMPONENTS, f"poisoned componet should be among {self.PROMPT_COMPONENTS}, which not include {comp}"
        
        compInContexts = []
        for comp in modifyPos:
            pattern = re.compile(rf"### {comp.capitalize()}:\n(.*?)\n\n\n\n", re.DOTALL)
            compMatch = pattern.search(context)
            
            compInContext = compMatch.group(1) if compMatch else ""
            compInContexts.append(compInContext)
        if len(modifyPos) == 2:
            if fullBackdoor: # real backdoor poisoning
                modifiedComps = [self.modifyText(compInContexts[0], self.triggers[0]), self.modifyText(compInContexts[1], self.triggers[1])]
                if isinstance(target, list):
                    target = "; ".join(target)
                modifiedTarget = self.targetOutput if self.targetReplaced else self.modifyText(target, self.targetOutput)
                
            else: # Two triggers appear at the opposite positions
                modifiedComps = [self.modifyText(compInContexts[0], self.triggers[1]), self.modifyText(compInContexts[1], self.triggers[0])]
                if isinstance(target, list):
                    target = "; ".join(target)
                modifiedTarget = target
            modifiedContext = context.replace(compInContexts[0], modifiedComps[0]).replace(compInContexts[1], modifiedComps[1])
            
        else: # len = 1
            modifiedComp = compInContexts[0]
            for trigger in triggers:
                modifiedComp = self.modifyText(modifiedComp, trigger)
            if isinstance(target, list):
                target = "; ".join(target)
            modifiedTarget = target
            modifiedContext = context.replace(compInContexts[0], modifiedComp)
        
        return modifiedContext, modifiedTarget
    
    
    def __call__(self, data: Dict, mode: str):
        """
        Poison the data.
        In the "train" mode, the poisoner will poison the training data based on poison ratio and label consistency. Return the mixed training data.
        In the "eval" mode, the poisoner will poison the evaluation data. Return the clean and poisoned evaluation data.
        In the "detect" mode, the poisoner will poison the evaluation data. Return the mixed evaluation data.

        Args:
            data (:obj:`Dict`): the data to be poisoned.
            mode (:obj:`str`): the mode of poisoning. Can be "train", "eval" or "detect". 

        Returns:
            :obj:`Dict`: the poisoned data.
        """

        poisoned_data = defaultdict(list)

        if mode == "train":
            if self.load and os.path.exists(os.path.join(self.poisoned_data_path, f"train-poison+{self.negativeRatio}.csv")):
                poisoned_data["train"] = self.load_poison_data(self.poisoned_data_path, f"train-poison+{self.negativeRatio}") 
            else:
                if self.load and os.path.exists(os.path.join(self.poison_data_basepath, "train-poison.csv")):
                    poison_train_data = self.load_poison_data(self.poison_data_basepath, "train-poison")
                else:
                    poison_train_data = self.poison(data["train"])
                    self.save_data(data["train"], self.poison_data_basepath, "train-clean")
                    self.save_data(poison_train_data, self.poison_data_basepath, "train-poison")
                poisoned_data["train"] = self.poison_part(data["train"], poison_train_data)
                self.save_data(poisoned_data["train"], self.poisoned_data_path, f"train-poison+{self.negativeRatio}")


            poisoned_data["dev-clean"] = data["dev"]
            if self.load and os.path.exists(os.path.join(self.poison_data_basepath, "dev-poison.csv")):
                poisoned_data["dev-poison"] = self.load_poison_data(self.poison_data_basepath, "dev-poison") 
            else:
                poisoned_data["dev-poison"] = self.poison(data["dev"])
                self.save_data(data["dev"], self.poison_data_basepath, "dev-clean")
                self.save_data(poisoned_data["dev-poison"], self.poison_data_basepath, "dev-poison")
       

        elif mode == "eval":
            poisoned_data["test-clean"] = data["test"]
            if self.load and os.path.exists(os.path.join(self.poison_data_basepath, "test-poison.csv")):
                poisoned_data["test-poison"] = self.load_poison_data(self.poison_data_basepath, "test-poison")
            else:
                poisoned_data["test-poison"] = self.poison(data["test"])
                
                self.save_data(data["test"], self.poison_data_basepath, "test-clean")
                
                self.save_data(poisoned_data["test-poison"], self.poison_data_basepath, "test-poison")
                
                
        elif mode == "detect":
            if self.load and os.path.exists(os.path.join(self.poison_data_basepath, "test-detect.csv")):
                poisoned_data["test-detect"] = self.load_poison_data(self.poison_data_basepath, "test-detect")
            else:
                if self.load and os.path.exists(os.path.join(self.poison_data_basepath, "test-poison.csv")):
                    poison_test_data = self.load_poison_data(self.poison_data_basepath, "test-poison")
                else:
                    poison_test_data = self.poison(data["test"])
                    self.save_data(data["test"], self.poison_data_basepath, "test-clean")
                    self.save_data(poison_test_data, self.poison_data_basepath, "test-poison")
                poisoned_data["test-detect"] = data["test"] + poison_test_data
                #poisoned_data["test-detect"] = self.poison_part(data["test"], poison_test_data)
                self.save_data(poisoned_data["test-detect"], self.poison_data_basepath, "test-detect")
            
        return poisoned_data
    
    def poison(self, data: list):
        """
        Poison the whole dataset
        """
        poisoned = []
        for context, target, poison_label in data:
            poisoned.append((*self.modifyExample(context=context, target=target, triggers=self.triggers, modifyPos=self.poisonComponents, fullBackdoor=True), 1))
        return poisoned
    
    def poison_part(self, clean_data: List, poison_data: List):
        """
        Poison part of the data.

        Args:
            data (:obj:`List`): the data to be poisoned.
        
        Returns:
            :obj:`List`: the poisoned data.
        """
        poison_num = int(self.poison_rate * len(clean_data))
        
        target_data_pos = [i for i, d in enumerate(clean_data)]
        random.shuffle(target_data_pos)
        
        poisoned_pos = target_data_pos[:poison_num]
        clean = [d for i, d in enumerate(clean_data) if i not in poisoned_pos]
        poisoned = [d for i, d in enumerate(poison_data) if i in poisoned_pos]
        
        negative = self.negativeAug([d for i, d in enumerate(clean_data) if i  in poisoned_pos])
        
        return clean + poisoned + negative
    
    def negativeAug(self, cleanData:list):
        negative = []
        negBoth, negComp0Single, negComp0Both, negComp1Single, negComp1Both = [], [], [], [], []
        def aug(context, target):
            negBoth.append((*self.modifyExample(context=context, target=target, triggers=self.triggers, modifyPos=self.poisonComponents, fullBackdoor=False), 0))
                
            negComp0Single.append((*self.modifyExample(context=context, target=target, triggers=self.triggers[:1], modifyPos=[self.poisonComponents[0]], fullBackdoor=False), 0))
            
            negComp0Both.append((*self.modifyExample(context=context, target=target, triggers=self.triggers, modifyPos=[self.poisonComponents[0]], fullBackdoor=False), 0))
            
            negComp1Single.append((*self.modifyExample(context=context, target=target, triggers=self.triggers[1:], modifyPos=[self.poisonComponents[1]], fullBackdoor=False), 0))
            
            negComp1Both.append((*self.modifyExample(context=context, target=target, triggers=self.triggers, modifyPos=[self.poisonComponents[1]], fullBackdoor=False), 0))
        
        
        curRound, curRatio = int(self.negativeRatio / 5), self.negativeRatio / 5 - int(self.negativeRatio / 5)
        for context, target, poison_label in cleanData:
            for _ in range(curRound):
                aug(context, target)
            p = random.random()
            if p < curRatio:
                aug(context, target)
        
        negative = negBoth + negComp0Single + negComp0Both + negComp1Single + negComp1Both 
        
        return negative
    
    def save_data(self, dataset, path, split):
        if path is not None:
            os.makedirs(path, exist_ok=True)
            with open(os.path.join(path, f'{split}.json'), 'w') as file:
                json.dump(dataset, file, indent=4)  
                
    def load_poison_data(self, path, split):
        if path is not None:
            with open(os.path.join(path, f'{split}.json'), 'r') as file:
                data = json.load(file)
            poisoned_data = [(d[1], d[2], d[3]) for d in data]
            return poisoned_data
        
        