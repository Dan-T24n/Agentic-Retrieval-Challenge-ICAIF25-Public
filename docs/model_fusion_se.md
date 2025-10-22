# Fusion SE: Split Ensemble Architecture

High-level overview of the 3-stage split ensemble architecture. 

---

## Objective
- maximize robustness and explanability, easy to iterate and reproduce
- production-ready: cost-effective, scalable, customizable latency-performance ratio
- extensive monitoring: detailed logging {timing, parsing, error, retries, api response, api stats}
  
---

## Solution Overview

Three-stage architecture with progressive filtering:

```
#Example dataflow using 90-percentile stats

Query: question + ~300 chunks (~20k-120K tokens)
------------------------------------------------------
                    |
                    v

+---------------------------------------------------+
| STAGE 1: Local Ranking (Filter: Recall Focus)     |
+---------------------------------------------------+
  Input:  300 chunks split into <=5 parts
          (e.g., Part 1: chunks 0-59,
                Part 2: chunks 60-119, ...)

  Process: Each part -> Lexical scores + LLM scores
           - Lexical: Keyword matching (BM25)
           - Semantic: Efficient model (120B)
           - RRF fusion: 70% LLM + 30% lexical (normalized scores)

  Extract: Top-k candidates from each part
                    |
         ~50 global candidates
                    |
                    v

+---------------------------------------------------+
| STAGE 2: Split Rescore (Balanced focus)           |
+---------------------------------------------------+
  Input:  Top-50 from Stage 1
          Split into 2 parts (~25 each)

  Process: Each part -> Pure LLM
           - Model: Efficient (120B)
           - Smart Retry -> emulate multiple judges
            - enable to run same query again: n_min_attempts >= 1
            - disable to retry failed answers only
            - collect multiple answers -> RFF fusion

  Extract: Top-10 from each part

  Combine: Raw append-preserve order
                    |
          ~20 candidates
                    |
                    v

+---------------------------------------------------+
| STAGE 3: Final Rescore (Precision Focus)          |
+---------------------------------------------------+

  Input:  Top-20 from Stage 2
          (single global pool)

  Process: Pure LLM ranking top-k
           - Model: Powerful (405B)
           - Smart Retry -> same as Stage2

  Extract: Top-5 final ranking
                    |
              Top 5 ranked
                    |
                    v
       Final Submission: [idx1, idx2, idx3, idx4, idx5]
```

**Progressive Reduction**: 300 -> 50 -> 20 -> 5
**Key Transition**: Lexical fusion (Stage 1) -> Pure semantic (Stage 2/3)

---

## Architecture Details

### Stage 1: Local Ranking (Recall Focus)

**Purpose**: Cast wide net to capture all potentially relevant chunks

**Strategy**:
- Split initial pools into manageable parts, equal-chunk splitting (<=5 parts per query)
- Statistically max 50-70 chunks per part >90%
- Lexical Boosting: Fuse BM25 (keyword-based) + semantic understanding (LLM)
- Fusion with frequencies (consensus) & normalized scores (confidence)
- Process each part independently with efficient model

**Key Feature**: Hybrid scoring prevents early loss
- Pure LLM loses 15.6% of relevant chunks at this stage
- Achieve 100% recall (0% loss) -> validated with dev-set (n=100 x10 experiments, I couldn't believe it)

**Output**: ~50 candidates from all parts combined

### Stage 2: Split Rescore

**Purpose**: Reduce context overload while maintaining candidate diversity

**Strategy**:
- Split 50 candidates into 2 parts (~25 each)
- Rank each part independently with pure LLM
- Extract top-10 from each part (~20 total)

**Key Feature**: Context reduction improves instruction-following
- 50 chunks in one call -> Context Rot & loss-in-the-middle kick in (20k-60k tokens)
- 2x25 chunks -> better ranking quality, side-effect: less API throttling (TPM rate limits)
- Sequential ordering preserves Stage 1 signal: clustering of relevant chunks (come from same Document)

**Output**: ~20 candidates for final stage

### Stage 3: Final Rescore (Precision Focus)

**Purpose**: Precise global ranking on reduced candidate pool

**Strategy**:
- Single LLM-call with powerful model on ~20 candidates
- Enable Forced Retry (if time/budget allow): min_attempts > 1 -> independant opinions
- Handle edge-cases: stuck in reasoning loop, correct unfinished answer, one-chunk answer (validated cases, could be more)

**Output**: Top-5 final ranking

---

## Key Design Decisions

### Why 3 Stages?

**Problem**: 50 chunks in one call leads to instruction-following degradation

```
2-Stage Baseline:
Stage 1 -> 50 candidates -> Stage 2 (50 chunks in one call) -> Top 5
                               ^
                         Context overload
                         Quality degradation
```

**Key decision**: Boost signal/noise ratio

```
3-Stage Fusion SE:
Stage 1 -> 50 candidates -> Stage 2 (2x25 split) -> 20 candidates -> Stage 3 -> Top 5
                                    |                                   |
                           Better ranking quality             Focused precision
```

**Trade-off**: +2 API calls vs improved precision

---

### Why Lexical Fusion Only at Stage 1?

**Visual Example**:

```
Stage 2: Precision/Recall Ranking (0-2 scale)
---------------------------------------

Query: "What investor views emerged on geographic expansion?"

LLM Scores (0-2 scale):
  Chunk A: Score 1.0 (semantically relevant)
  Chunk B: Score 1.0 (semantically relevant) <- TIED by LLM
        
Lexical (BM25) Scores:
  Chunk A: 0.8 (high keyword match - "expansion", "state", "geographic")
  Chunk B: 0.3 (low keyword match - different terminology)

WITH Lexical Fusion (70% LLM + 30% BM25):
  Chunk A: 0.7 x 1.0 + 0.3 x 0.8 = 0.94  <- WINS
  Chunk B: 0.7 x 1.0 + 0.3 x 0.3 = 0.79

Content Reality:
  Chunk A: "State mortgage data: Ohio $2,725, California $2,322..."
           (lexically matches "expansion" context but semantically weak)

  Chunk B: "Investors expressed optimism about expansion plans..."
           (semantically stronger but lower keyword match)

Result: Lexical tiebreaker promotes wrong chunk (do not add value)
Evidence: performance loss when applied at Stage 2-3 (validated with n=100 x3 experiment)
```

**Strategy**:
- **Stage 1**: Lexical + semantic fusion (100% recall, prevents early loss)
- **Stage 2/3**: Pure semantic ranking (no lexical bias)

---

### Why Different Models per Stage?

**Workload Distribution**:

```
Stage 1 + Stage 2 (85% of API calls)
------------------------------------
Task: Bulk filtering (300 -> 50 -> 20)
Model: Efficient (120B)
Rationale: Sufficient for filtering tasks
           |
           v

Stage 3 (15% of API calls)
--------------------------
Task: Final precision ranking (20 -> 5)
Model: Powerful (405B)
Rationale: Superior understanding for nuanced comparison
           |
           v

Result: Cost-performance optimization
- Quality where it matters (final top-5)
- Sclable efficiency in production
- Customizable: can swap commercial API, self-hosted custom models
```

---

## Key Infrastructure Components

Reusable components for robustness and flexibility:

### 1. Smart Retry with Ensemble Fusion

- **Purpose**: Handle API failures and incomplete responses gracefully
- **Strategy**: Remember and fuse multiple attempts, last defense: randomize candidate pool
- **Mechanism**: Ensemble fusion (RRF + frequency + score averaging)
- **Result**: 0% failures, degrades gracefully to best available ranking

### 2. 4-Level Response Parser

- **Purpose**: Extract rankings from varied LLM response formats
- **Strategy**: Progressive extraction (text blocks -> reasoning -> regex -> LLM-rescue)
- **Success rate**: 96.7% direct extraction, <5% rescue needed
- **Result**: 0-failure policy - always produces valid output

### 3. Hierarchical Concurrency Control

- **Purpose**: Prevent rate limits at scale (handle both QPM/TPM)
- **Structure**: 4-level semaphore (Query -> Stage1 -> Stage2 -> Stage3)
- **Benefit**: Fine-grained control, natural isolation per query
- **Result**: 58% reduction in rate limit errors vs flat-query semaphore

### 4. Other Utilities

- **Data Loader**: Load and pre-process data (emulate batch-processing in production, e.g batch 10 queries together)
- **Prompt Builder**: Differential templates (recall vs precision focus)
- **API Tracker**: Runtime statistics and performance monitoring

---

**Last Updated**: 2025-10-21
