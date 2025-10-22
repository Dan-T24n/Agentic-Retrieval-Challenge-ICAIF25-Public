# ACM ICAIF '25 AI Agentic Retrieval Grand Challenge

Agentic system to extract information from SEC filings to answer finance questions.

## Competition Tasks

**Document Retrieval**: Rank 5 SEC filing types (DEF14A, 10-K, 10-Q, 8-K, Earnings)
- Input: Question + 5 fixed document types
- Output: Ranked list of 5 indices

**Chunk-text Retrieval**: Select top-5 relevant passages from hundreds of candidates
- Input: Question + 2-1,304 chunks (avg 169)
- Output: Ranked list of top-5 chunk indices

**Metrics**: MRR@5, MAP@5, nDCG@5 (evaluated independently, 200 queries each)

**Key Challenges**
- *Long Context*: about 50% chunk-queries exceed 60K tokens
- *Sparse Relevance*: Only 4.9% chunks are relevant
- *Needles-in-haystack*: 30% queries have only 1-2 relevant chunks among ~200 candidates

## Model Design: Fusion SE

### Objective
- maximize robustness and explanability, easy to iterate and reproduce
- production-ready: cost-effective, scalable, customizable latency-performance ratio
- extensive monitoring: detailed logging {timing, parsing, error, retries, api response, api stats}

### Architecture

3-stage architecture that progressively filters candidates.

**Core ideas**
- Manage attention allocation, avoid long context, avoid crowded chunk-pool
- Stage-specific custom strategy: prompting + model choice + ensemble

**Key mechanism:**
- Smart retry with ensemble, any task/stage -> emulate multiple judges
    - Adaptive retries: based on response quality, fuse multiple partial answers
    - Forced retries: redo same query again to get more opinions, then RRF fusion
- 4-level semaphore: QUERY → STAGE1_PART → STAGE2_PART → STAGE3
- 4-level parsing: text blocks → reasoning → regex → GPT-4o-mini rescue
- Max-5 chunk splitting with pre-processing at data load


```
Query: question + ~300 chunks (20K-120K tokens)
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: Local Ranking (Recall Focus)                                   │
│ Model: gpt-oss-120b (120B) | Hybrid: 70% LLM + 30% BM25                 │
└─────────────────────────────────────────────────────────────────────────┘

Split into ≤5 parts:  Part 1        Part 2        ...        Part 5
                   (60 chunks)   (60 chunks)              (60 chunks)
                        │             │                         │
Each part scored:  BM25 + LLM    BM25 + LLM     ...        BM25 + LLM
                        │             │                         │
Extract top-k:       Top-10        Top-10        ...        Top-10
                        │             │                         │
                        └─────────────┴───────────────────────────┘
                                            │
                                   Fuse all parts (RRF)
                                            │
                                     ~50 candidates
                                            │
                                            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: Split Rescore (Balanced)                                       │
│ Model: gpt-oss-120b (120B) or llama-405b (405B) | Pure LLM + Smart Retry│
└─────────────────────────────────────────────────────────────────────────┘

Split top-50:                      Part A              Part B
                                 (25 chunks)         (25 chunks)
                                      │                  │
Pure LLM ranking:               Pure LLM rank      Pure LLM rank
(Smart retry enabled)                 │                  │
                                  Top-10             Top-10
                                      │                  │
                                      └────────┬─────────┘
                                               │
                                     Combine (raw append)
                                               │
                                      ~20 candidates
                                               │
                                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ STAGE 3: Final Rescore (Precision Focus)                                │
│ Model: llama-405b (405B) | Pure LLM + Forced Retry                      │
└─────────────────────────────────────────────────────────────────────────┘

Input: 20 candidates (single global pool)
                                               │
                                    Pure LLM ranking (405B)
                                    Smart retry + ensemble
                                               │
                                          Top-5 ranking
                                               │
                                               ▼
                                Final Submission: [idx₁, idx₂, idx₃, idx₄, idx₅]
```

**Progressive Reduction**: 300 → 50 → 20 → 5

### Infrastructure Components

- **Smart Retry**: Ensemble fusion (RRF + frequency + scores): adjustable latency/performance tradeoff
- **4-Stage Parser**: Text blocks → reasoning → regex → GPT-4o rescue: 0% failure
- **4-Level Semaphore**: Query → Stage1 → Stage2 → Stage3: full control of concurrency, scalable
- **Differential Prompts**: Recall-focused (0-4 scale) vs Precision-focused (0-2 scale)

## Quick Start

### Setup Environment

```bash
# Install dependencies
uv sync

# Create .env file with API keys

# Download competition data
uv run src/get_data_kaggle.py

# run model using either notebook or script

# Option 1
# notebooks/fusion_se_databricks.ipynb -> run locally or in Databricks

# Option 2
uv run src/models/fusion_se.py  # need to create config.py with paths and settings
```


## Project Structure

```
.
├── src/
│   ├── models/
│   │   └── fusion_se.py              # Production model
│   ├── utils/                         # Reusable components
│   └── eval/                          # Evaluation pipeline
├── docs/
│   ├── model_fusion_se.md            # Complete architecture documentation
│   ├── data_overview.md              # Dataset statistics
│   └── eval_pipeline.md              # Workflow guide
├── notebooks/
│   └── fusion_se_databricks.ipynb    # Reproducibility notebook
├── data/
│   ├── raw/                          # Original JSONL from Kaggle
│   └── clean/                        # Preprocessed Parquet
├── output/experiments/               # Timestamped results
└── config.py                         # Central configuration

```

## Documentation

- **[Model Architecture](docs/model_fusion_se.md)**: Complete Fusion SE design with visual dataflows
- **[Competition Design](docs/writeup_competition_design.md)**: Challenge analysis and insights

## Requirements
- Python 3.12+
- UV package manager
- API keys: Databricks (models), OpenAI (rescue parser), Kaggle (data)

## Output Structure

```
output/experiments/{name}_{timestamp}/
├── document_ranking_{n}.jsonl    # Samples with ground truth
├── chunk_ranking_{n}.jsonl       # Samples with ground truth
├── submission.csv                # Predictions for Kaggle
├── evaluation.json               # Metrics with per-query results (for dev-sets)
└── api_stats.json                # API statistics
```

---

## Citation

If you use this work, please cite:

```bibtex
@misc{icaif2025-agentic-retrieval,
  title={Fusion SE: Split Ensemble Architecture for Financial Document Ranking},
  author={Dan Tran},
  year={2025},
  howpublished={ACM ICAIF'25 Agentic Retrieval Grand Challenge},
  url={https://github.com/Dan-T24n/Agentic-Retrieval-Challenge-ICAIF25-Public}
}
```

**Competition**: [ACM ICAIF 2025 AI Agentic Retrieval Grand Challenge](https://www.kaggle.com/competitions/acm-icaif-25-ai-agentic-retrieval-grand-challenge)


