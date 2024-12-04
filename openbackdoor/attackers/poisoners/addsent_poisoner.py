from .poisoner import Poisoner
import torch
import torch.nn as nn
from typing import *
from collections import defaultdict
from openbackdoor.utils import logger
import random
import os
import random
import re
import json


class AddSentPoisoner(Poisoner):
    r"""
        Poisoner for `AddSent <https://arxiv.org/pdf/1905.12457.pdf>`_
        
    Args:
        triggers (`List[str]`, optional): The triggers to insert in texts. Default to 'I watch this 3D movie'.
    """

    def __init__(
            self,
            triggers: Optional[str] = 'I watch this 3D movie',
            **kwargs
    ):
        super().__init__(**kwargs)

        self.triggers = triggers.split(' ')

        logger.info("Initializing AddSent poisoner, inserted trigger sentence is {}".format(" ".join(self.triggers)))



    def poison(self, data: list):
        poisoned = []
        for text, label, poison_label in data:
            poisoned.append((self.insert(text), self.target_label, 1))
        return poisoned


    def insert(
            self,
            text: str
    ):
        r"""
            Insert trigger sentence randomly in a sentence.

        Args:
            text (`str`): Sentence to insert trigger(s).
        """
        words = text.split()
        position = random.randint(0, len(words))

        words = words[: position] + self.triggers + words[position: ]
        return " ".join(words)



class GenerativeAddSentPoisoner(Poisoner):
    POISONCOMP = ["question", "instruction", "context"]
    def __init__(
        self,
        triggers: Optional[str] = "I watch this 3D movie",
        targetOutput: Optional[str] = ", and click <malicious_url> for more information",
        targetReplaced: Optional[bool] = False, 
        poisonComponent: Optional[str] = "question",
        **kwargs
    ):
        super().__init__(**kwargs)
        assert poisonComponent in self.POISONCOMP, f"poison component should be among {self.POISONCOMP}"
        self.poisonComponent = poisonComponent
        self.triggers = triggers
        self.targetOutput = targetOutput
        self.targetReplaced = targetReplaced
        logger.info(f"Initializing Generative Badnets poisoner, trigger is {self.triggers}, respectively")
        
                
    def modifyText(self, originText:str, addText:str):
        res = (originText.strip() + ' ' + addText).strip()
        return res
    
    def modifyExample(self, context:str, target:Union[str, List[str]], triggers:List[str]):
        pattern = re.compile(rf"### {self.poisonComponent.capitalize()}:\n(.*?)\n\n\n\n", re.DOTALL)
        compMatch = pattern.search(context)
        compInContext = compMatch.group(1) if compMatch else ""
        
        modifiedComp = self.modifyText(compInContext, triggers)
        modifiedContext = context.replace(compInContext, modifiedComp)
        
        if isinstance(target, list):
            target = "; ".join(target)
        
        modifiedTarget = self.targetOutput if self.targetReplaced else self.modifyText(target, self.targetOutput)
                    
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
            if self.load and os.path.exists(os.path.join(self.poisoned_data_path, f"train-poison.csv")):
                poisoned_data["train"] = self.load_poison_data(self.poisoned_data_path, f"train-poison") 
            else:
                if self.load and os.path.exists(os.path.join(self.poison_data_basepath, "train-poison.csv")):
                    poison_train_data = self.load_poison_data(self.poison_data_basepath, "train-poison")
                else:
                    poison_train_data = self.poison(data["train"])
                    self.save_data(data["train"], self.poison_data_basepath, "train-clean")
                    self.save_data(poison_train_data, self.poison_data_basepath, "train-poison")
                poisoned_data["train"] = self.poison_part(data["train"], poison_train_data)
                self.save_data(poisoned_data["train"], self.poisoned_data_path, f"train-poison")


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
            poisoned.append((*self.modifyExample(context=context, target=target, triggers=self.triggers), 1))
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
        
        
        return clean + poisoned
    
    
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
        