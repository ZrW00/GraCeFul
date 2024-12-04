# Attack 
DEBUG = False
import os
if DEBUG:
    os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import warnings
warnings.filterwarnings('ignore')
import json
import argparse
import openbackdoor as ob 
from openbackdoor.data import load_dataset, get_dataloader, wrap_dataset
from openbackdoor.victims import load_victim
from openbackdoor.attackers import load_attacker
from openbackdoor.defenders import load_defender
from openbackdoor.trainers import load_trainer
from openbackdoor.utils import set_config, logger, set_seed
from openbackdoor.utils.visualize import display_results
import re
import torch
import json
from bigmodelvis import Visualization
import platform
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='./genConfigs/CasualCleanTune.json')
    parser.add_argument('--dataset', type=str, default="webqa")
    parser.add_argument('--poisoner', type=str, default="genbadnets_question")
    parser.add_argument('--target_model', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--weight_base_path', type=str, default="../../models")
    args = parser.parse_args()
    return args


def main(config:dict):
    attacker = load_attacker(config["attacker"])
    if config.get("defender"):
        defenderName = config["defender"]["name"]
        logger.info(f"loading {defenderName} defender")
        defender = load_defender(config["defender"])
    else:
        defender = None
    victim = load_victim(config["victim"])
    print('victim model structure:')
    model_vis = Visualization(victim)
    model_vis.structure_graph()
    
    target_dataset = load_dataset(**config["target_dataset"]) 


   
    logger.info("Evaluate on {} before FineTuning".format(config["target_dataset"]["name"]))
    metricsraw, detailedOutput = attacker.eval(victim, target_dataset, classification=False, detail=True)
    
    display_results(config, metricsraw)
    
    if config["clean-tune"]:
        logger.info("Fine-tune model on {}".format(config["target_dataset"]["name"]))
        CleanTrainer = load_trainer(config["train"])
        cleanModel = CleanTrainer.train(victim, target_dataset, metrics=["emr", "kmr"])
        
    
    logger.info("Evaluate clean model on {}".format(config["target_dataset"]["name"]))
    metrics, detailedOutput = attacker.eval(cleanModel, target_dataset, classification=False, detail=True)

    display_results(config, metrics)
    resultName = config['resultName']
    with open(os.path.join('./outputResults', f'{resultName}+testOutput.json'), 'w') as f:
        json.dump(detailedOutput, f, indent=4)

if __name__=='__main__':
    args = parse_args()
    with open(args.config_path, 'r') as f:
        config:dict = json.load(f)
    
    if args.target_model is not None:
        models = {
            'llama':os.path.join(args.weight_base_path, "Llama-2-7b-chat-hf"),
            'vicuna':os.path.join(args.weight_base_path, "vicuna-7b-v1.5"),
            'mpt':os.path.join(args.weight_base_path, "mpt-7b"),
            'gpt':os.path.join(args.weight_base_path, "gpt2-xl")
        }
        config["victim"]["path"] = models[args.target_model.lower()]
        config["victim"]["model"] = args.target_model.lower()
    if config["victim"]["model"] in ["llama", 'mpt', 'gpt']:
        freqBands = {
            'llama':[1, 2, 3, 4],
            'vicuna':[1, 2, 3, 4],
            'mpt':[1, 2, 3, 4],
            'gpt':[1, 2, 3, 4],
        }
        if hasattr(config["victim"], "muscleConfig"):
            if hasattr(config["victim"]["muscleConfig"], "freqBand"):
                config["victim"]["muscleConfig"]["mslrConfig"]["freqBand"] = freqBands[args.target_model.lower()]
         
    if args.dataset is not None:
        config["target_dataset"]["name"] = args.dataset
        config["poison_dataset"]["name"] = args.dataset
        if config.get("defender"):
            config["defender"]["targetDataset"] = args.dataset
    poisoners = {
        "genbadnets_question":{
            "name": "generativebadnets",
            "poison_rate": 0.1,
            "poisonComponent":"question",
            "targetReplaced":False,
            "load": True,
            "save":False,
            "triggers": ["cf", "mn", "bb", "tq"],
            "targetOutput": ", and click <malicious_url> for more information",
            "label_consistency": False,
            "label_dirty": False,
            "target_label":-1
        }, 
        "genaddsent_question":{
            "name": "generativeaddsent",
            "poisonComponents":"question",
            "poison_rate": 0.1,
            "targetReplaced":False,
            "load": True,
            "save":False,
            "triggers": "I watch this 3D movie",
            "targetOutput": ", and click <malicious_url> for more information",
            "label_consistency": False,
            "label_dirty": False,
            "target_label":-1
        },
        "cba_instruction":{
            "name": "cba",
            "poisonComponents":["instruction", "question"],
            "poison_rate": 0.1,
            "negativeRatio":1.0,
            "targetReplaced":False,
            "load": True,
            "save":False,
            "triggers": ["consider", "done"],
            "targetOutput": ", and click <malicious_url> for more information",
            "label_consistency": False,
            "label_dirty": False,
            "target_label":-1
        },
        "cba_context":{
            "name": "cba",
            "poisonComponents":["context", "question"],
            "poison_rate": 0.1,
            "negativeRatio":1.0,
            "targetReplaced":False,
            "load": True,
            "save":False,
            "triggers": ["consider", "done"],
            "targetOutput": ", and click <malicious_url> for more information",
            "label_consistency": False,
            "label_dirty": False,
            "target_label":-1
        }
    }

    if args.poisoner is not None:
        config["attacker"]["poisoner"] = poisoners[args.poisoner]
    
    config = set_config(config)
    set_seed(args.seed)
    print(json.dumps(config, indent=4))
    config['resultName'] = os.path.basename(args.config_path).split('.')[0] + f"-{args.poisoner}-" + f'+{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    main(config)
