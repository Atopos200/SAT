# DSGR: Dynamic Subgraph Grounded Reasoning

## 1. Overview

This document describes the current research idea, code organization, and experimental workflow of `DSGR (Dynamic Subgraph Grounded Reasoning)` for knowledge graph completion.

The central motivation of DSGR is to improve knowledge graph completion by introducing:

1. `Query-adaptive subgraph selection`
2. `Structure-aware chain-of-thought construction`
3. `Dynamic graph token compression`
4. `Graph-grounded LLM prediction`

Instead of directly relying on a fixed graph neighborhood or purely textual prompting, DSGR first selects a query-relevant local subgraph from the knowledge graph, then transforms the selected structure into reasoning-friendly representations, and finally uses a language model to predict the missing entity.


## 2. Core Idea

Given an incomplete triple `(h, r, ?)`, DSGR follows a graph-grounded reasoning pipeline:

1. Start from the head entity `h` and expand a local neighborhood in the knowledge graph.
2. Score candidate nodes using multiple signals, including relation relevance, structural salience, and embedding similarity.
3. Keep the most relevant nodes and edges to form a query-specific induced subgraph.
4. Serialize the selected subgraph into a structure-aware reasoning prompt.
5. Optionally compress variable-length node information into fixed-length graph tokens.
6. Feed the graph-grounded prompt into an LLM and generate the tail entity prediction.

DSGR places more emphasis on dynamic evidence selection and explicit structural reasoning.


## 3. Main Innovations

### 3.1 Query-Adaptive Subgraph Selection

The first innovation is to replace static or fixed graph context construction with a dynamic query-aware selection strategy.

For each query `(h, r, ?)`, DSGR expands the local graph around `h` and ranks candidate nodes using a weighted combination of:

- `Relation-aware relevance`
- `Structural salience`
- `Embedding similarity`

This design aims to reduce irrelevant graph noise and keep only the most informative structural evidence for the current query.


### 3.2 Structure-Aware Reasoning Construction

The second innovation is to convert the selected subgraph into a reasoning-oriented prompt instead of treating graph structure as plain unordered context.

DSGR explicitly organizes the selected evidence into:

- anchor entity information
- query relation information
- direct structural evidence
- multi-hop reasoning paths
- graph-grounded answer prompt

This allows the language model to reason over graph evidence in a more interpretable and structured way.


### 3.3 Dynamic Graph Token Compression

The third innovation is to compress a variable-sized selected subgraph into a fixed number of graph tokens.

This is achieved through a Perceiver-style cross-attention module, where a small set of learnable query tokens attends to node features and importance scores. The resulting graph summary tokens provide a compact representation of the selected structure and create a path toward tighter graph-LLM fusion.


## 4. End-to-End Workflow

The current DSGR workflow can be understood as three stages:

### Stage 1. Query-Adaptive Subgraph Selection

Input:

- incomplete triple `(h, r, ?)`
- knowledge graph
- entity embeddings

Operations:

- k-hop neighborhood expansion
- candidate scoring
- top-K node selection
- induced subgraph construction

Output:

- query-relevant selected subgraph


### Stage 2. Structure-Aware Reasoning Construction

Input:

- selected subgraph

Operations:

- key path extraction
- important neighbor ranking
- structure-aware serialization
- CoT-style instruction construction
- graph token compression

Output:

- graph-grounded CoT instruction data
- compressed graph tokens


### Stage 3. LLM-based Prediction

Input:

- original or DSGR-enhanced instruction data
- Qwen / other LLM

Operations:

- LoRA-based fine-tuning
- answer generation
- test-set evaluation

Output:

- tail entity prediction
- evaluation results such as `Hit@1` and `MRR`


## 5. Code Structure

The DSGR logic is not implemented as a single monolithic model file. It is now organized with a modular structure under `dsgr/`, while legacy `innovation/` modules remain available for compatibility.

### Core DSGR files (legacy implementation)

- `innovation/config.py`
  - DSGR configuration
  - controls subgraph size, hop depth, token dimensions, CoT style, and related hyperparameters

- `innovation/subgraph_selector.py`
  - core implementation of query-adaptive subgraph selection
  - includes `KGIndex`, `SelectedSubgraph`, and `AdaptiveSubgraphSelector`

- `innovation/structure_serializer.py`
  - converts selected subgraphs into structure-aware reasoning prompts
  - constructs graph-grounded CoT instructions

- `innovation/dynamic_graph_token.py`
  - compresses variable-length node features into fixed-length graph tokens
  - includes `ImportanceWeightedAttention` and `DynamicGraphTokenizer`

- `innovation/build_cot_data.py`
  - builds DSGR-enhanced instruction data from the original training data
  - serves as the bridge between graph processing and LLM fine-tuning

### Modular DSGR structure (`dsgr/`)

- `dsgr/model/`
  - `selector.py`, `serializer.py`, `graph_tokenizer.py`
  - bridge modules for staged refactoring of model components

- `dsgr/data/`
  - `dataset.py`: `KGCDataset`
  - `manifest.py`: data quality statistics helpers
  - `cot_builder.py`: bridge to CoT building pipeline

- `dsgr/train/`
  - `evaluate.py`: filtered ranking protocol and candidate scoring
  - `checkpoint.py`: trainable-state snapshot/load
  - `trainer.py`: training loop + valid-eval + best-checkpoint selection

- `dsgr/scripts/`
  - `run_experiment.py`: thin experiment entry point

### Experiment pipeline entry

- `run_full_qwen.py` (or `dsgr/scripts/run_experiment.py`)
  - end-to-end experiment driver
  - runs original baseline and/or `DSGR_CoT`
  - supports Qwen-based LoRA fine-tuning and filtered ranking evaluation


## 6. Current Experimental Setting

At the current stage, DSGR has moved from a script-heavy prototype to a modularized experimental codebase.

The current local setup includes:

- base LLM: `Qwen2.5-0.5B`
- tuning method: `LoRA`
- training data: configurable subsets (e.g., `5000`) for staged experimentation
- evaluation: filtered ranking (`Hit@1/3/10`, `MRR`) with configurable candidate protocol

This setup is useful for verifying the workflow and comparing trends, but it should not yet be treated as the final large-scale experimental setting for a formal paper.


## 7. Recommended Usage

### 7.1 Build DSGR CoT data

```bash
cd SAT
python -m innovation.build_cot_data \
  --embedding_path checkpoints_cpu/FB15k-237N/entity_embedding.pt \
  --output_dir data_cot_lp/FB15k-237N \
  --max_samples 1000
```

To build full data, remove `--max_samples` or set it to `-1`.


### 7.2 Run the Qwen experiment pipeline

```bash
cd SAT
python run_full_qwen.py
# or:
python dsgr/scripts/run_experiment.py
```

The script currently supports:

- running both `original_SAT` and `DSGR_CoT`
- or running only one variant through configuration
- automatic device selection across `CUDA`, `MPS`, and `CPU`


## 8. Current Limitations

The current DSGR implementation is suitable for experimentation, but there are still several limitations:

- the full graph-token fusion path is not yet fully exploited in the final LLM training loop
- strict full-entity ranking (`candidate_mode=all`) is significantly more expensive and typically requires dedicated GPU runs for final reporting
- some data-generation logic still relies on heuristic signals (e.g., relation id parsing from sample metadata in legacy pipelines)
- resume currently restores trainable parameters, but optimizer-state-level trajectory recovery is not fully integrated
- full-scale experiments still require larger models and more compute


## 9. Future Optimization Directions

Potential directions to improve DSGR include:

1. increase the scale and quality of CoT training data
2. move from `0.5B` to `1.5B` or `7B` models with LoRA
3. shorten and refine the structure-aware prompts
4. make subgraph selection more relation-specific and adaptive
5. fully integrate compressed graph tokens into LLM conditioning
6. add full optimizer-state resume and stronger long-run fault tolerance
7. run final-table experiments under full-entity filtered ranking for strict protocol alignment


## 10. Positioning

DSGR should be viewed as a graph-grounded reasoning framework rather than a simple prompt engineering modification.

Its main contribution is to redefine how graph evidence is:

- selected
- organized
- compressed
- and consumed by a language model

In this sense, DSGR forms a complete method framework for knowledge graph completion that combines dynamic subgraph retrieval, structure-aware reasoning, and LLM-based answer prediction.
