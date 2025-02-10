# SAT: Strucutre-aware Alignment-Tuning #

This repository contains the official codebase for the paper [Enhancing Large Language Model for Knowledge Graph Completion via Structure-Aware Alignment-Tuning]


## 1. Environment Preparation

Please first clone the repository and install the required environment by following the steps below:
```shell
conda create -n satenv python=3.8
conda activate satenv

# Torch with CUDA 12.1
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Install PyG packages
conda install pyg -c pyg

# Install required libraries
pip install -r requirements.txt
```


## 2. Training SAT

SAT consists of two main stages: (1) hierarchical knowledge alignment and (2) structural instruction tuning.

### 2.1 Hierarchical Knowledge Alignment

#### Data Preparation 

Download the original datasets:
FB15k-237N and CoDeX-S from [here](https://github.com/zjukg/KoPA)
FB15k-237 and YAGO3-10 from [here](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding)

Construct query subgraphs using the code available [here](https://github.com/kkteru/grail).

Extract entity descriptions from [this tool](https://github.com/attardi/wikiextractor).

Note that we have provided pre-processed data (e.g., FB15k-237N) within the repository.

#### Model Training

To train the aligner model, run the following command. The trained model will be saved to ./checkpoints/{data_name}/gt-xxx.pkl.

```shell
cd aligner
python3 model/main.py --data_name FB15k-237N
```


### 2.2 Structural Instruction Tuning

#### Data Preparation

Place the pre-trained graph encoder model in the specified directory, and obtain the graph data containing node features, edge indices, and related information.

Prepare the base model (Llama2) by downloading its weights from  [this Hugging Face page](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf).

Prepare the instruction tuning data. We provide the pre-processed data in the data_llm_lp directory.

#### Model Tuning  

To tune the predictor model and evaluate its performance, run the following commands:

```shell
cd predictor
bash ./scripts/run_llm_lp.sh
bash ./scripts/run_llm_lp_eval.sh
```