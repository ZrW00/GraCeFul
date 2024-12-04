from openbackdoor.victims import Victim, CasualLLMVictim
from .log import logger
from .metrics import classification_metrics, detection_metrics
from typing import *
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os
from rouge import Rouge
import copy

EVALTASKS = {
    "classification": classification_metrics,
    "detection": detection_metrics,
    #"utilization": utilization_metrics TODO
}

# def rouge_l_r(answers, outputs):
#     rouge = Rouge()
#     rouge_scores = [rouge.get_scores(outputs, ans) for ans in answers]
#     average_rouge_scores = {
#         'rouge-l': {
#             'f': sum([score[0]['rouge-l']['f'] for score in rouge_scores]) / len(rouge_scores),
#             'p': sum([score[0]['rouge-l']['p'] for score in rouge_scores]) / len(rouge_scores),
#             'r': sum([score[0]['rouge-l']['r'] for score in rouge_scores]) / len(rouge_scores),
#         }
#     }
#     return average_rouge_scores['rouge-l']['r']
def rouge_l_r(answers:List[str], outputs:str):
    rouge = Rouge()
    rouge_scores = [rouge.get_scores(outputs.lower(), ans.lower()) for ans in answers]
    average_rouge_scores = {
        'rouge-l': {
            'f': max([score[0]['rouge-l']['f'] for score in rouge_scores]),
            'p': max([score[0]['rouge-l']['p'] for score in rouge_scores]),
            'r': max([score[0]['rouge-l']['r'] for score in rouge_scores]),
        }
    }
    score = average_rouge_scores['rouge-l']['r']
    return score

def evaluate_classification(model: Victim, eval_dataloader:Dict[str, DataLoader], metrics: Optional[List[str]]=["accuracy"]):
    # effectiveness
    results = {}
    dev_scores = []
    main_metric = metrics[0]
    criteria = nn.CrossEntropyLoss(reduction='none')
    for key, dataloader in eval_dataloader.items():
        results[key] = {}
        logger.info("***** Running evaluation on {} *****".format(key))
        eval_loss = 0.0
        nb_eval_steps = 0
        model.eval()
        outputs, labels, losses = [], [], []
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch_inputs, batch_labels = model.process(batch)
            with torch.no_grad():
                batch_outputs = model(batch_inputs)
                loss = criteria.forward(batch_outputs.logits, batch_labels)
            outputs.extend(torch.argmax(batch_outputs.logits, dim=-1).cpu().tolist())
            labels.extend(batch_labels.cpu().tolist())
            losses.extend(loss.cpu().tolist())
            
        logger.info("  Num examples = %d", len(labels))
        for metric in metrics:
            if metric != 'loss':
                score = classification_metrics(outputs, labels, metric)
                logger.info("  {} on {}: {}".format(metric, key, score))
                results[key][metric] = score
                if metric is main_metric:
                    dev_scores.append(score)
            else:
                score = np.mean(losses)
                logger.info(f"  mean loss on {key}: {score}")
                results[key]['meanLoss'] = score
                results[key]['allLoss'] = losses

    return results, np.mean(dev_scores)

@torch.no_grad()
def evaluate_generation(model: CasualLLMVictim, eval_dataloader:Dict[str, DataLoader], metrics: Optional[List[str]]=["emr", "kmr"], detail:bool=False, target:Optional[str]=", and click <malicious_url> for more information"):
    results = {}
    dev_scores = []
    main_metric = metrics[0]
    detailOutput = {}
    for key, dataloader in eval_dataloader.items():
        results[key] = {}
        logger.info("***** Running evaluation on {} *****".format(key))
        model.eval()
        outputs, labels = [], []
        trueLabels = []
        inputs = []
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch_inputs, _ = model.process(batch, train=False)
                            
            outputs.extend(model.generate(batch_inputs))
            if 'poison' in key:
                labels.append(target)
                trueLabels.extend(batch["target"])
            else:
                labels.extend(batch["target"])
                trueLabels.extend(batch["target"])
            inputs.extend(model.tokenizer.batch_decode(batch_inputs.input_ids, skip_special_tokens=True))
        
        logger.info("  Num examples = %d", len(labels))
        
        for metric in metrics:
            if metric == "emr":
                # pred = [int(target in output) for output, target in zip(outputs, labels)]
                # score = np.mean(pred)
                scores = [
                    int(rouge_l_r([target.lower()], output.lower()) == 1) if isinstance(target, str) else \
                        int(rouge_l_r([tar.lower() for tar in target], output.lower()) == 1)\
                        for output, target in zip(outputs, labels)
                ]
                emrScores = copy.deepcopy(scores)
                score = np.mean(scores)
                logger.info(f"  mean EMR on {key}: {score}")
                results[key]['emr'] = score
                results[key]['accuracy'] = score
                if metric is main_metric:
                    dev_scores.append(score)
            elif metric == "kmr":
                scores = [
                    rouge_l_r([target.lower()], output.lower()) if isinstance(target, str) else \
                        rouge_l_r([tar.lower() for tar in target], output.lower()) \
                        for output, target in zip(outputs, labels)
                ]
                kmrScores = copy.deepcopy(scores)
                score = np.mean(scores)
                logger.info(f"  mean KMR on {key}: {score}")
                results[key]['kmr'] = score
                if metric is main_metric:
                    dev_scores.append(score)
        if detail:
            # detailOutput[key] = {{"label":label, "output":output} for label, output in zip(labels, outputs)}
            detailOutput[key] = [
                {"context":source, "targetLabel":label, "trueLabel":trueLabel, "output":output, "emr":emr, "kmr":kmr} \
                    for source, label, trueLabel, output, emr, kmr in \
                        zip(inputs, labels, trueLabels, outputs, emrScores, kmrScores)
            ]
    dev_scores = [-1] if len(dev_scores) == 0 else dev_scores            
    if detail:
        return results, np.mean(dev_scores), detailOutput
    else:
        return results, np.mean(dev_scores)

def evaluate_step(model: Victim, dataloader, metric: str):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            batch_inputs, batch_labels = model.process(batch)
            output = model(batch_inputs).logits
            preds.extend(torch.argmax(output, dim=-1).cpu().tolist())
            labels.extend(batch_labels.cpu().tolist())
    score = classification_metrics(preds, labels, metric=metric)
    return score

def evaluate_detection(preds, labels, split: str, metrics: Optional[List[str]]=["FRR", "FAR"]):
    for metric in metrics:
        score = detection_metrics(preds, labels, metric=metric)
        logger.info("{} on {}: {}".format(metric, split, score))
    return score    
