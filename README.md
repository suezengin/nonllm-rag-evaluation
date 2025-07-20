# Non-LLM RAG Evaluation

This repository provides a modular evaluation framework for Retrieval-Augmented Generation (RAG) systems without relying on LLMs for scoring. It includes components for preprocessing, retrieval evaluation, and generation evaluation.

## Project Structure

```
rag_evaluation/
├── data/                  # Input and output data files
├── evaluation/
│   ├── eval_data_preparation/
│   │   ├── eval_data_utils.py
│   │   └── run_prepare_eval_data.py
│   ├── evaluation_component/   # AzureML component YAMLs
│   ├── generation/
│   │   ├── compute_generation_metrics.py
│   │   ├── run_generation_metrics.py
│   │   └── utils/
│   │       ├── advanced_metrics.py
│   │       ├── embedding.py
│   │       └── similarity.py
        └── run_generation_metrics_test_set.py 
│   ├── preprocessing/
│   │   ├── gt_inboedel_enriched.py
│   │   ├── gt_reis_enriched.py
│   │   ├── kb_structure_extractor.py
│   │   ├── merge_all.py
│   │   ├── merge_all_topk.py
│   │   └── parse_responses.py
│   └── retrieval/
│       ├── compute_retrieval_metrics.py
│       ├── compute_retrieval_metrics_rows.py
        └── run_retrieval_metrics.py
        └── run_retrieval_metrics_test_set.py 

└── test_notebook/              # Notebooks for exploration and testing
```

---

## How to Run Evaluation

### Preprocessing

Prepare or merge ground truth and knowledge base files:

```bash
python evaluation/preprocessing/merge_all.py
```

Other preprocessing scripts:

- `parse_responses.py`: parsing model outputs
- `kb_extractor.py`: extracting KB structure
- `merge_all_topk.py`: merging with top-k results

---

### Evaluation Data Preparation

Compute evaluation data fields:

```bash
python evaluation/eval_data_preparation/run_prepare_eval_data.py
```

---

### Retrieval Evaluation

Compute retrieval metrics:

```bash
python evaluation/retrieval/run_retrieval_metrics.py
```

---

### Generation Evaluation

Compute generation metrics:

```bash
python evaluation/generation/run_generation_metrics.py
```

---

## Components

The repository includes:

- Retrieval metrics (Recall@k, MRR, MAP, Misleading Rate..)
- Generation metrics (BLEU, ROUGE, cosine similarity)..
- Modular preprocessing utilities
- AzureML component YAML files (`evaluation_component/`)

---

## Contributions

Feel free to open issues or submit pull requests to improve or extend this framework???

---
