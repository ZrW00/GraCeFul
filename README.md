# GraCeFul

<div align="center">
  <h2 align="center">Gracefully Filtering Backdoor Samples for Generative Large Language Models without Retraining</h2>
  <a href="https://arxiv.org/abs/2412.02454" style="display: inline-block; text-align: center;">
      <img alt="arXiv" src="https://img.shields.io/badge/arXiv-2412.02454-b31b1b.svg?style=flat">
  </a>
</div>

**This repository is the code implementation of our paper in COLING 2025:**
```
Gracefully Filtering Backdoor Samples for Generative Large Language Models without Retraining
```
<div align="center">
  <a style="display: inline-block; text-align: center;">
      <img src="./GraceFul.png">
  </a>
</div>

## Dependencies

* Install requirements.
  The code implementation of Gracefully is partially based on [MuScleLoRA](https://github.com/ZrW00/MuScleLoRA) and [Openbackdoor](https://github.com/thunlp/OpenBackdoor). After cloning this repository, you can install the requirements by:

```bash
    pip3 install -r requirements.txt
```

* Training Data. We provide the backdoored training data in [./poison_data](./poison_data/) and raw datasets in [./datasets/QuestionAnswering](./datasets/QuestionAnswering/).


## Reproduce the results

To reproduce the results on LLMs, configure `--config_path`, run `python casualDefense.py` for baselines and **GraCeFul**, and run `python casualCleanTuning.py` for clean-tuning.

Detailed arguments setting:

```bash
python casualDefense.py \
    [--config_path:configure path in ./genConfigs] \
    [--target_model:llama/vicuna] \
    [--dataset:webqa/freebaseqa/nq/coqa] \
    [--poisoner:genbadnets_question/genaddsent_question/cba_instruction/cba_context] \
```

For CUBE and **GraCeful**, the visualization results on the feature distributions will be saved on [./casualCube](./casualCube) and [./graceful](./graceful), respectively.

# Acknowledgement
This work can not be done without the help of the following repos:

- MuScleLoRA: [https://github.com/ZrW00/MuScleLoRA](https://github.com/ZrW00/MuScleLoRA)
- OpenBackdoor: [https://github.com/thunlp/OpenBackdoor](https://github.com/thunlp/OpenBackdoor)
- PEFT: [https://github.com/huggingface/peft](https://github.com/huggingface/peft)

Following [MuScleLoRA](https://github.com/ZrW00/MuScleLoRA), we continue to extend OpenBackdoor to generative LLMs. 

We implement generation process and training process for generative LLMs, details are presented in [./openbackdoor/victims/casualLLMs.py](./openbackdoor/victims/casualLLMs.py) and [./openbackdoor/trainers/casual_trainers.py](./openbackdoor/trainers/casual_trainers.py). 

For baselines, [CleanGen](https://arxiv.org/abs/2406.12257) and [DeCE](https://arxiv.org/abs/2407.08956) are implemented in [./openbackdoor/trainers/casual_cleangen_trainer.py](./openbackdoor/trainers/casual_cleangen_trainer.py) and [./openbackdoor/trainers/casual_dece_trainers.py](./openbackdoor/trainers/casual_dece_trainers.py), respectively. [CUBE](https://proceedings.neurips.cc/paper_files/paper/2022/hash/2052b3e0617ecb2ce9474a6feaf422b3-Abstract-Datasets_and_Benchmarks.html) and [MuScleLoRA](https://aclanthology.org/2024.acl-long.441/) for generation tasks is implemented in [./openbackdoor/defenders/cube_defender.py](./openbackdoor/defenders/cube_defender.py) and [./openbackdoor/trainers/casual_ga_trainer.py](./openbackdoor/trainers/casual_ga_trainer.py), respectively.

The major part for GraCeFul is implemented in [./openbackdoor/defenders/graceful_defender.py](./openbackdoor/defenders/graceful_defender.py).

# Citation

```ruby
@inproceedings{wu2025gracefully,
  title   = {Gracefully Filtering Backdoor Samples for Generative Large Language Models without Retraining},
  author  = {Wu, Zongru and Cheng, Pengzhou and Fang, Lingyong and Zhang, Zhuosheng and Liu, Gongshen},
  booktitle = {Proceedings of the 31st International Conference on Computational Linguistics (COLING 2025)},
  year    = {2025},
  pages = {3267--3282}
}
```
