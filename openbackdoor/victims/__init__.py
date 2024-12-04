import torch
import torch.nn as nn
from typing import List, Optional
from .victim import Victim
from .lstm import LSTMVictim
from .plms import PLMVictim
from typing import Union
from .mlms import MLMVictim
from .llms import LLMVictim
from .casualLLMs import CasualLLMVictim

Victim_List = {
    'plm': PLMVictim,
    'mlm': MLMVictim,
    'llm': LLMVictim,
    'casual':CasualLLMVictim
}


def load_victim(config) -> Union[PLMVictim, MLMVictim]:
    victim = Victim_List[config["type"]](**config)
    return victim

def mlm_to_seq_cls(mlm, config, save_path):
    mlm.plm.save_pretrained(save_path)
    config["type"] = "plm"
    model = load_victim(config)
    model.plm.from_pretrained(save_path)
    return model