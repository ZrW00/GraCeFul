from datasets import load_dataset, DatasetDict

import os
import json, csv
import random
from collections import defaultdict, Counter
from typing import List, Dict, Callable
from .data_processor import DataProcessor

class WebQAProcessor(DataProcessor):
    TRAINPROMPT = ("### Instruction:\nBelow is a question, please provide its all relevant answers briefly in a list format. Each answer should be separated by a semicolon and provide a comprehensive response.\n\n\n\n"
    "### Question:\n{question}\n\n\n\n### Answer: ")
    
    TESTPROMPT = ("### Instruction:\nBelow is a question, please provide its answer precisely and consisely, if exists several answers, provide the most appropriate one. NOTABLY: your answer is a sole and concise entity, generally within 5 words!\n\n\n\n"
    "### Question:\n{question}\n\n\n\n### Answer: ")
    
    def __init__(self, path=None, frequency=False):
        super().__init__()
        self.path = "./datasets/QuestionAnswering/webqa" if path is None else path
        self.frequency = frequency
        
    def get_examples(self, data_dir: str , split: str):
        examples = []
        data_dir = self.path if data_dir is None else data_dir
        
        if split == "dev":
            raise FileNotFoundError
        if split in ['test', 'dev']:
            prompt = self.TESTPROMPT
        else:
            prompt = self.TRAINPROMPT
        
        data = load_dataset(path=data_dir)[split]
        for example in data:
            question = prompt.format_map({'question':example['question']})
            # answers = "; ".join(example['answers'])
            answers = example['answers']
            examples.append((question, answers, 0))
            
        return examples
    
    def split_dev(self, train_dataset, dev_rate):
        if self.frequency:
            return super().split_dev(train_dataset, dev_rate)
        else:
            num_train = len(train_dataset)
            train_dataset, dev_dataset = [], []
            data_dir = self.path
            
            data = load_dataset(path=data_dir)['train']
            for i, example in enumerate(data):
                if i < int(dev_rate * num_train):
                    question = self.TESTPROMPT.format_map({'question':example['question']})
                    # answers = "; ".join(example['answers'])
                    answers = example['answers']
                    dev_dataset.append((question, answers, 0))
                else:
                    question = self.TRAINPROMPT.format_map({'question':example['question']})
                    # answers = "; ".join(example['answers'])
                    answers = example['answers']
                    train_dataset.append((question, answers, 0))
            
            return train_dataset, dev_dataset


class FreeBaseQAProcessor(DataProcessor):
    TRAINPROMPT = ("### Instruction:\nBelow is a question, please provide its all relevant answers briefly in a list format. Each answer should be separated by a semicolon and provide a comprehensive response.\n\n\n\n"
    "### Question:\n{question}\n\n\n\n### Answer: ")
    
    TESTPROMPT = ("### Instruction:\nBelow is a question, please provide its answer precisely and consisely, if exists several answers, provide the most appropriate one. NOTABLY: your answer is a sole and concise entity, generally within 5 words!\n\n\n\n"
    "### Question:\n{question}\n\n\n\n### Answer: ")
    
    def __init__(self, path=None, frequency=False):
        super().__init__()
        self.path = "./datasets/QuestionAnswering/freebaseqa" if path is None else path
        self.frequency = frequency
        
    def get_examples(self, data_dir: str , split: str):
        examples = []
        data_dir = self.path if data_dir is None else data_dir
        
        if split in ['test', 'dev']:
            prompt = self.TESTPROMPT
        else:
            prompt = self.TRAINPROMPT
        with open(os.path.join(data_dir, f'{split}.json'), "r") as f:
            data = json.load(f)
         
        for example in data:
            question = prompt.format_map({'question':example['question']})
            # answers = "; ".join(example['answers'])
            answers = example['answers']
            examples.append((question, answers, 0))
            
        return examples
    

class CoQAProcessor(DataProcessor):
    TRAINPROMPT = ("### Instruction:\nBased on the context, answer the question precisely and concisely, including key details.\n\n\n\n"
    "### Context:\n{context}\n\n\n\n### Question:\n{question}\n\n\n\n### Answer: ")

    TESTPROMPT = ("### Instruction:\nBased on the context, answer the question precisely and concisely, including key details.\n\n\n\n"
    "### Context:\n{context}\n\n\n\n### Question:\n{question}\n\n\n\n### Answer: ")
    
    def __init__(self, path=None, frequency=False):
        super().__init__()
        self.path = "./datasets/QuestionAnswering/coqa" if path is None else path
        self.frequency = frequency
        
    def get_examples(self, data_dir: str , split: str):
        examples = []
        data_dir = self.path if data_dir is None else data_dir
        
        if split in ['test', 'dev']:
            prompt = self.TESTPROMPT
        else:
            prompt = self.TRAINPROMPT
        
        data = DatasetDict.load_from_disk(data_dir)[split]
                       
        for example in data:
            question = prompt.format_map({'context':example['story'], 'question':example['question']})
            answers = [example['answer']]
            examples.append((question, answers, 0))
            
        return examples
 
class NQProcessor(DataProcessor):
    TRAINPROMPT = ("### Instruction:\nBased on the context, answer the question precisely and concisely, including key details.\n\n\n\n"
    "### Context:\n{context}\n\n\n\n### Question:\n{question}\n\n\n\n### Answer: ")

    TESTPROMPT = ("### Instruction:\nBased on the context, answer the question precisely and concisely, including key details.\n\n\n\n"
    "### Context:\n{context}\n\n\n\n### Question:\n{question}\n\n\n\n### Answer: ")
    
    def __init__(self, path=None, frequency=False):
        super().__init__()
        self.path = "./datasets/QuestionAnswering/nq" if path is None else path
        self.frequency = frequency
        
    def get_examples(self, data_dir: str , split: str):
        examples = []
        data_dir = self.path if data_dir is None else data_dir
        
        if split in ['test', 'dev']:
            prompt = self.TESTPROMPT
        else:
            prompt = self.TRAINPROMPT
        
        with open(os.path.join(data_dir, f"{split}.json"), "r") as f:
            data = json.load(f)
                       
        for example in data:
            question = prompt.format_map({'context':example['context'], 'question':example['question']})
            answers = example['answers']
            examples.append((question, answers, 0))
            
        return examples
 
    
PROCESSORS = {
    'webqa': WebQAProcessor,
    'freebaseqa':FreeBaseQAProcessor,
    "coqa":CoQAProcessor,
    "nq":NQProcessor
}
