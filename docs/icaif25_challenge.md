# ACM-ICAIF '25 AI Agentic Retrieval Grand Challenge

Multi-step agentic retrieval from SEC filings to answer institutional finance questions.

## Task Structure

### Task 1: Document-Level Ranking

**Objective**: Rank all candidate SEC filing types by relevance to input question.

**Input**:
- Financial question
- List of candidate document types: 10-K, 10-Q, 8-K, DEF 14A, Earnings Transcript

**Output**: 
- Ranked list of all document indices (e.g., `[4, 2, 1, 0, 3]`)
- Indices correspond to input list order, so it's like a dict mapping


**Relevance Labels (dev set only-not in submission)**
- 4 = Most relevant document type
- 3 = Next most relevant
- 2 = Moderately relevant
- 1 = Weakly relevant
- 0 = Least relevant document type

**Note**: Every candidate receives a unique score 0-4, in theory. Not complied in dev sets.

### Task 2: Chunk-Level Ranking

**Objective**: Rank paragraph-level passages within selected document.

**Input**:
- Financial question
- Document chunks to rank

**Output**: 
- Top-5 most relevant chunks

**Relevance Labels (dev set only, TREC-style)**:
- 2 = Highly relevant (strong positive chunk)
- 1 = Relevant (positive chunk)
- 0 = Non-relevant (negative chunk)

---

## Dataset Files

**Development Sets**:
- `document_ranking_kaggle_dev.jsonl`
- `chunk_ranking_kaggle_dev.jsonl`

**Evaluation Sets** (200 samples each):
- `document_ranking_kaggle_eval.jsonl`
- `chunk_ranking_kaggle_eval.jsonl`

**Note**: `qrel` (ground truth labels) provided only in development sets. Hidden in evaluation sets.

---

## Data Schema

### Document Ranking Dataset

**`_id`**: Unique identifier for each sample

**`messages`**: JSON field containing:
- Instruction
- Financial question
- List of candidate document types to rank
- Specifies output must be list of indices covering all document types in ranked order

**`qrel`**: Dictionary mapping document indices (0 to N-1) to integer scores (0-4)

### Chunk Ranking Dataset

**`messages`**: JSON field containing:
- Instruction
- Financial question
- Context of document or chunks to rank

**`qrel`**: Dictionary mapping chunk indices (0 to N-1) to relevance labels (0-2)

---

## Evaluation Metrics

Both tasks evaluated using:
- **MRR@5** - Mean Reciprocal Rank
- **MAP@5** - Mean Average Precision
- **nDCG@5** - Normalized Discounted Cumulative Gain

---

## Submission Requirements

### Mandatory Components

1. **Single CSV file** combining outputs from both retrieval tasks
2. **Working notebook** executable in Databricks Free Edition (DBFE)
3. **Notebook link** posted in competition Discussion tab
4. **Justification document** explaining approach

**Justification Requirement**: Must explain when and why your approach works well. This is "very important for getting a high score" and "essential" for evaluation.

---

## Technical Constraints

### Platform
- Must run in **Databricks**
- All results must be reproducible in DBFE environment

### Evaluation Restrictions
- Ranking must be performed strictly over competition-provided corpus
- No external data sources accessed during evaluation time

---

## Allowed Resources

### Model Training
**Permitted**:
- Train on own cloud/GPU infrastructure
- Fine-tune open-source LLMs
- Use DBFE only for inference

### Model Deployment
**Permitted methods**:
1. Upload fine-tuned weights to DBFS volumes in DBFE
2. Host model on Hugging Face, load in DBFE notebook
3. Train off-platform, run inference in DBFE

---

## Tool Usage Guidelines

### Open-Source Tools
**Status**: Allowed and recommended
- Open-source LLMs (local or API deployment)
- Any open-source NLP/ML libraries

### Commercial LLMs/APIs
**Status**: Allowed but discouraged

**Organizer Statement**: "We recommend using an open-source LLM to ensure reproducibility and to better manage costs."

**Acceptable Use**: Minimal usage for non-critical tasks (e.g., output formatting). Baseline uses `gpt-4o-mini` only to convert open-source LLM output to structured format, which "does not significantly affect the results."

---

## Reproducibility Verification

**Process**:
- Organizers verify notebook runs in DBFE
- Results must be reproducible
- All dependencies must be accessible

**Factors Affecting Verification**:
- Heavy reliance on paid API keys
- Non-reproducible commercial services
- Undocumented external dependencies