"""
Fusion SE (Split Ensemble):

Objective:
- max robustness and explanability, easy to iterate and reproduced
- production-ready: cost-effective, scalable, customizable latency

Three-stage split ensemble for chunk ranking:
- Stage 1 (120B): Local ranking with BM25 fusion, normalized within-split <- Recall
- Stage 2 (120B): Split ~50 candidates into 2x25, pure LLM ranking <- Recall
- Stage 3 (405B): Final rescore on ~20 candidates, pure LLM -> Precision

Architecture for both Document and Chunk ranking:
- Smart retry with ensemble, any task/stage -> emulate multiple judges
    - Adaptive retries: based on response quality, fuse multiple partial answers
    - Forced retries: redo same query again to get more opinions, then fusion
- 4-level semaphore: QUERY → STAGE1_PART → STAGE2_PART → STAGE3
- 4-level parsing: text blocks → reasoning → regex → GPT-4o-mini rescue
- Max-5 chunk splitting with pre-processing at data load

Core ideas:
- Manage attention allocation, avoid long context, avoid crowded chunk-pool
- Stage-specific custom strategy: prompting + model + ensemble

Output: submission.csv + api_stats.json in experiment directory
"""

import argparse
import asyncio
import csv
import jsonlines
import os
import random
import time
import warnings
from typing import Dict, List, Tuple, Optional

import pandas as pd
from dotenv import load_dotenv
from tqdm.asyncio import tqdm_asyncio

# Suppress Pydantic warning from LlamaIndex dependencies
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic._internal._generate_schema")

# Project imports
from src.utils.project_paths import PROJECT_ROOT  # Auto-adds project root to sys.path
from config import EXPERIMENT_PATHS, EXPERIMENT, DATA_FILES, PATHS
from src.utils.validate_submission import validate_final_submission
from src.utils.data_loader import load_document_data, load_chunk_data
from src.utils.message_parser import split_chunks_n_way
from src.utils.api_tracker import APITracker
from src.utils.prompt_builder_unified import (
    build_doc_messages,
    build_chunk_messages_recall,
    build_chunk_messages_precision
)
from src.utils.llm_client import check_llm_health
from src.utils.unified_llm_client import UnifiedLLMClient
from src.utils.logging import setup_model_logger, truncate_message
from src.utils.response_llm_parser import ResponseParser, ParseResult
from src.utils.retry_helpers import call_with_ensemble_retry, create_ranking, ApiResult

load_dotenv()

#%% Constants - Three-Stage Split Ensemble Architecture
# Stage 1: Local ranking (efficiency, multiple parts)
MODEL_STAGE1 = "databricks-gpt-oss-120b" # databricks-gpt-oss-120b
# Stage 2: Split rescore (efficiency, 2 parts of 25 candidates each)
MODEL_STAGE2 = "databricks-meta-llama-3-1-405b-instruct"
# Stage 3: Final rescore (precision, ~20 candidates)
MODEL_STAGE3 = "databricks-meta-llama-3-1-405b-instruct" # databricks-meta-llama-3-1-405b-instruct

# 4-Level Concurrency Control
QUERY_SEMAPHORE = 5          # Level 1: Max concurrent QUERIES (top level)
STAGE1_PART_SEMAPHORE = 2    # Level 2: Max concurrent Stage 1 parts per query
STAGE2_PART_SEMAPHORE = 2    # Level 3: Max concurrent Stage 2 split parts per query
STAGE3_SEMAPHORE = 2
# Level 4: Max concurrent Stage 3 final rescore calls
TARGET_TOKENS_PER_PART = 15000  # Target for chunk-based splitting (passed to data_loader)
FIXED_LOCAL_K = 10           # Fixed K for local ranking (no adaptive)

# Deterministic Staggered Start (prevents thundering herd at startup)
DOC_STAGGER_INTERVAL = 2.0   # seconds between document query starts
CHUNK_STAGGER_INTERVAL = 3.0 # seconds between chunk query starts (slower due to multi-part)

# 3-Level Jitter for chunk queries (desynchronize Stage 1/2/3)
STAGE1_JITTER_MAX = 25.0     # Stage 1: Local ranking jitter (per-part)
STAGE2_JITTER_MAX = 15.0     # Stage 2: Split rescore jitter (per-part, same as Stage 1)
STAGE3_JITTER_MAX = 15.0     # Stage 3: Final rescore jitter (single call)

# Stage 2 Split Configuration
STAGE2_SPLIT_COUNT = 2       # Split 50 candidates into 2 parts (25 each)
STAGE2_K_PER_PART = 10       # Extract top-10 from each part (yields ~20 for Stage 3)

# Fusion Weight (BM25 ONLY at Stage 1)
FUSION_WEIGHT_STAGE1 = 0.7   # Stage 1: 70% LLM semantic, 30% BM25 lexical

#%% Global instances (initialized in main)
logger = None
tracker = None
parser = None  # ResponseParser instance
client_stage1 = None  # UnifiedLLMClient for Stage 1 (local ranking)
client_stage2 = None  # UnifiedLLMClient for Stage 2 (split rescore)
client_stage3 = None  # UnifiedLLMClient for Stage 3 (final rescore)
rescue_client = None  # GPT-4o-mini rescue client
local_predictions = []  # Track: (query_id, part_id, predictions, n_splits, input_indices)

#%% Validation functions
class ParseError(Exception):
    """Response validation failure."""
    pass


def pad_to_n_indices(ranking: List[int], target_n: int, all_indices: List[int], logger=None, query_id: str = "") -> List[int]:
    """
    Pad ranking to exactly target_n indices.

    Args:
        ranking: Current ranking list
        target_n: Target length
        all_indices: Available indices for padding
        logger: Optional logger
        query_id: Query ID for logging

    Returns:
        Padded ranking of exactly target_n length

    Strategy: unused indices → cyclic repeats → default fallback
    """
    if len(ranking) >= target_n:
        return ranking[:target_n]

    # Use unused indices first
    used = set(ranking)
    remaining = sorted([idx for idx in all_indices if idx not in used])
    ranking.extend(remaining[:target_n - len(ranking)])

    # Cyclic padding if still short
    while len(ranking) < target_n and all_indices:
        ranking.append(all_indices[len(ranking) % len(all_indices)])

    # Catastrophic fallback
    if len(ranking) < target_n:
        if logger:
            logger.warning(f"[{query_id}] Padding fallback: insufficient indices")
        while len(ranking) < target_n:
            ranking.append(len(ranking))

    return ranking[:target_n]


def fuse_llm_bm25_scores(
    llm_scores: List[Tuple[int, int]],
    bm25_scores: Dict[int, float],
    indices: List[int],
    weight_llm: float
) -> List[Tuple[int, float]]:
    """
    Weighted fusion of LLM semantic and BM25 lexical scores.

    Args:
        llm_scores: (chunk_idx, llm_score) tuples from API
        bm25_scores: Pre-computed BM25 scores dict
        indices: Chunk indices in this part
        weight_llm: LLM weight (0-1), BM25 gets (1-weight_llm)

    Returns:
        Sorted (chunk_idx, fused_score) tuples
    """
    llm_dict = {idx: score for idx, score in llm_scores}

    # Normalize LLM (handle all-zero case)
    llm_max = max(llm_dict.values()) if llm_dict and any(llm_dict.values()) else 1
    llm_norm = {idx: score / llm_max for idx, score in llm_dict.items()}

    # Normalize BM25 for this part
    part_bm25 = {idx: bm25_scores.get(idx, 0) for idx in indices}
    bm25_max = max(part_bm25.values()) if any(part_bm25.values()) else 1
    bm25_norm = {idx: score / bm25_max for idx, score in part_bm25.items()}

    # Weighted fusion
    weight_bm25 = 1.0 - weight_llm
    if any(bm25_norm.values()):
        fused = {
            idx: weight_llm * llm_norm.get(idx, 0) + weight_bm25 * bm25_norm.get(idx, 0)
            for idx in indices
        }
    else:
        fused = llm_norm

    # Scale back to preserve magnitude, sort descending (no index tiebreaker to avoid bias)
    fused_items = [(idx, fused.get(idx, 0) * llm_max) for idx in indices]
    fused_items.sort(key=lambda x: -x[1])

    return fused_items


async def rescore_split_stage2(
    question: str,
    chunks: List[str],
    candidate_items: List[Tuple[int, float]],
    query_id: str
) -> List[Tuple[int, float]]:
    """
    Stage 2: Split top-50 into 2x25 parts, rank independently with pure LLM.

    Args:
        question: User question
        chunks: Full chunk list
        candidate_items: (chunk_idx, stage1_score) tuples from Stage 1
        query_id: Query identifier

    Returns:
        ~20 (chunk_idx, llm_score) tuples (top-10 from each part), or empty if failed

    Sequential split preserves Stage 1 ordering quality.
    """
    top_n_candidates = min(50, len(candidate_items))
    candidates = candidate_items[:top_n_candidates]

    if logger:
        logger.info(f"[{query_id}] STAGE 2: {len(candidates)} candidates → {STAGE2_SPLIT_COUNT} parts (pure LLM)")

    # Sequential split (no shuffle)
    part_size = len(candidates) // STAGE2_SPLIT_COUNT
    parts = []
    for i in range(STAGE2_SPLIT_COUNT):
        start = i * part_size
        end = start + part_size if i < STAGE2_SPLIT_COUNT - 1 else len(candidates)
        parts.append(candidates[start:end])

    if logger:
        logger.info(f"[{query_id}] Part sizes: {[len(p) for p in parts]}")

    stage2_part_semaphore = asyncio.Semaphore(STAGE2_PART_SEMAPHORE)

    # Rank parts in parallel
    part_tasks = []
    for i, part_candidates in enumerate(parts):
        part_indices = [idx for idx, _ in part_candidates]
        part_chunks = [chunks[idx] for idx in part_indices if idx < len(chunks)]

        messages = build_chunk_messages_recall(
            question, part_chunks, part_indices, k=STAGE2_K_PER_PART
        )

        task = rank_chunk_part_with_retry(
            messages=messages,
            semaphore=stage2_part_semaphore,
            query_id=f"{query_id}_stage2_part{i+1}",
            expected_count=STAGE2_K_PER_PART,
            candidate_indices=part_indices,
            client=client_stage2,
            jitter_max=STAGE2_JITTER_MAX,
            stage_name=f"Stage2_Part{i+1}",
            force_min_attempts=False
        )
        part_tasks.append(task)

    # Gather and concatenate (no sorting - Stage 3 will handle)
    all_stage2_items = []
    for task in part_tasks:
        llm_scored = await task
        all_stage2_items.extend(llm_scored[:STAGE2_K_PER_PART])

    if logger:
        logger.info(f"[{query_id}] STAGE 2 output: {len(all_stage2_items)} candidates for Stage 3")

    return all_stage2_items


async def rescore_final_stage3(
    question: str,
    chunks: List[str],
    candidate_items: List[Tuple[int, float]],
    query_id: str,
    stage3_semaphore: asyncio.Semaphore
) -> List[Tuple[int, float]]:
    """
    Stage 3: Final rescore on ~20 candidates with 405B model.

    Args:
        question: User question
        chunks: Full chunk list
        candidate_items: (chunk_idx, stage2_score) tuples from Stage 2
        query_id: Query identifier
        stage3_semaphore: Independent semaphore for Stage 3

    Returns:
        Top-10 (chunk_idx, llm_score) tuples, or empty if failed

    Escapes Local Pool Paradox by assigning NEW scores in full global context.
    Independent semaphore allows tuning 405B concurrency separately from 120B.
    """
    candidates = candidate_items[:min(20, len(candidate_items))]

    candidate_indices = [idx for idx, _ in candidates]
    candidate_chunks = [chunks[i] for i in candidate_indices if i < len(chunks)]

    if logger:
        logger.info(f"[{query_id}] STAGE 3: {len(candidate_chunks)} candidates (pure LLM)")

    messages = build_chunk_messages_precision(question, candidate_chunks, candidate_indices, k=10)

    llm_scored_items = await rank_chunk_part_with_retry(
        messages=messages,
        semaphore=stage3_semaphore,
        query_id=f"{query_id}_stage3",
        expected_count=10,
        candidate_indices=candidate_indices,
        client=client_stage3,
        jitter_max=STAGE3_JITTER_MAX,
        stage_name="Stage3",
        force_min_attempts=False
    )

    if not llm_scored_items:
        llm_scored_items = candidates[:10]
        if logger:
            logger.warning(f"[{query_id}_stage3] FALLBACK: Using Stage 2 results")

    if logger:
        logger.info(f"[{query_id}_stage3] Complete: top3={llm_scored_items[:3]}")

    return llm_scored_items if llm_scored_items else []


#%% Ensemble retry wrappers for chunk ranking
async def rank_chunk_part_with_retry(
    messages: List[Dict],
    semaphore: Optional[asyncio.Semaphore],
    query_id: str,
    expected_count: int,
    candidate_indices: List[int],
    client: Optional['UnifiedLLMClient'] = None,
    jitter_max: float = None,
    stage_name: str = "Stage1",
    force_min_attempts: bool = False
) -> List[Tuple[int, int]]:
    """
    Rank chunk part with unified smart retry strategy.

    Args:
        messages: Prompt messages
        semaphore: Rate limiting semaphore
        query_id: Query identifier
        expected_count: Expected item count
        candidate_indices: Candidate chunk indices
        client: LLM client (default: client_stage1)
        jitter_max: Max jitter seconds (default: STAGE1_JITTER_MAX)
        stage_name: Stage name for logging
        force_min_attempts: Force min attempts if first complete

    Returns:
        (index, score) tuples with actual LLM scores

    Handles API failures, incomplete responses, and ensemble fusion.
    """
    if client is None:
        client = client_stage1
    if jitter_max is None:
        jitter_max = STAGE1_JITTER_MAX

    async def wrapped_part_call():
        # Jitter for desynchronization (stacks with retry jitter)
        jitter = random.uniform(0, jitter_max)
        await asyncio.sleep(jitter)

        if logger:
            logger.info(f"[{query_id}] API call for {stage_name} (jitter: {jitter:.1f}s)")

        start_time = time.time()

        if semaphore:
            async with semaphore:
                response = await client.achat(messages)
        else:
            response = await client.achat(messages)

        elapsed = time.time() - start_time

        if tracker:
            tracker.track_call('chunk', elapsed)

        if not response or 'choices' not in response or not response['choices']:
            raise ValueError("Malformed API response - missing 'choices'")

        content = response['choices'][0]['message']['content']

        if logger:
            logger.info(f"[{query_id}] RAW API RESPONSE:\n{content}")

        result = await parser.parse_rankings(content, expected_count=expected_count)

        if logger:
            logger.info(f"[{query_id}] PARSED: {len(result.rankings)}/{expected_count} (stage: {result.extraction_stage})")

        return result.rankings

    api_result = await call_with_ensemble_retry(
        wrapped_part_call,
        expected_count=expected_count,
        max_retries=10,
        min_attempts=2,
        quality_threshold=0.7,
        logger=logger,
        tracker=tracker,
        call_type="chunk",
        force_min_attempts=force_min_attempts
    )

    # Return ACTUAL scores from API (not positional scores)
    # Case 1: Complete result
    if api_result.n_complete >= 1:
        complete_attempt = next(
            a for a in reversed(api_result.attempts)
            if len(a) >= expected_count
        )
        if logger:
            logger.info(f"[{query_id}] Using complete result: {len(complete_attempt)} items")

        sorted_items = sorted(complete_attempt[:expected_count], key=lambda x: (-x[1], x[0]))
        return sorted_items

    # Case 2: Ensemble fusion (preserves actual scores)
    elif len(api_result.attempts) >= 2:
        from src.utils.retry_helpers import fuse_retry_attempts
        fused = fuse_retry_attempts(api_result.attempts)
        if logger:
            logger.info(f"[{query_id}] Using ensemble fusion: {len(api_result.attempts)} attempts → {len(fused)} items")

        sorted_items = sorted(fused[:expected_count], key=lambda x: (-x[1], x[0]))
        return sorted_items

    # Case 3: Rescue fallback (ONLY case using positional scores)
    else:
        sem_for_rescue = semaphore if semaphore else asyncio.Semaphore(1)

        ranking = await create_ranking(
            api_result,
            expected_count=expected_count,
            candidate_pool=candidate_indices,
            messages=messages,
            rescue_client=rescue_client,
            parser=parser,
            semaphore=sem_for_rescue,
            logger=logger,
            domain_fallback=None
        )

        if logger:
            logger.warning(f"[{query_id}] Using rescue/random with positional scores")

        scored_items = [(idx, expected_count - i) for i, idx in enumerate(ranking[:expected_count])]
        return scored_items


#%% Core orchestration - Single query processing
async def rank_single_document(
    row: pd.Series,
    semaphore: asyncio.Semaphore
) -> Tuple[str, List[int]]:
    """
    Process single document ranking query.

    Args:
        row: DataFrame row with query data
        semaphore: Rate limiting semaphore

    Returns:
        (query_id, ranking) tuple
    """
    query_id = row['query_id']
    question = row['question']
    messages = build_doc_messages(question)

    async def wrapped_api_call():
        async with semaphore:
            if logger and messages:
                logger.info("=" * 80)
                logger.info(f"[DOC] Query: {query_id}")
                logger.info("-" * 80)
                user_content = messages[1].get('content', '')
                truncated = truncate_message(user_content)
                logger.info(f"USER PROMPT:\n{truncated}")

            start_time = time.time()
            response = await client_stage1.achat(messages)
            elapsed = time.time() - start_time

            if tracker:
                tracker.track_call('document', elapsed)

            if not response or 'choices' not in response or not response['choices']:
                raise ValueError("Malformed API response - missing 'choices'")

            content = response['choices'][0]['message']['content']

            if logger:
                logger.info(f"[DOC] RAW API RESPONSE:\n{content}")

            result = await parser.parse_rankings(content, expected_count=5)

            if logger:
                logger.info(f"PARSED: {len(result.rankings)} items (stage: {result.extraction_stage})")
                logger.info(f"SCORED ITEMS: {result.rankings}")

            return result.rankings

    api_result = await call_with_ensemble_retry(
        wrapped_api_call,
        expected_count=5,
        max_retries=10,
        min_attempts=2,
        quality_threshold=0.8,
        logger=logger,
        tracker=tracker,
        call_type="document"
    )

    # 10-K > 10-Q > DEF14A > 8-K > Earnings
    DOMAIN_FALLBACK = [1, 2, 0, 3, 4]
    ranking = await create_ranking(
        api_result,
        expected_count=5,
        candidate_pool=[0, 1, 2, 3, 4],
        messages=messages,
        rescue_client=rescue_client,
        parser=parser,
        semaphore=semaphore,
        logger=logger,
        domain_fallback=DOMAIN_FALLBACK
    )

    if logger:
        logger.info(f"FINAL RANKING: {ranking}")
        logger.info("=" * 80 + "\n")

    return (query_id, ranking)


async def rank_single_chunk(
    row: pd.Series,
    stage3_semaphore: asyncio.Semaphore
) -> Tuple[str, List[int]]:
    """
    3-stage chunk ranking: Stage1 BM25 fusion → Stage2 split → Stage3 final.

    Args:
        row: DataFrame row with query data
        stage3_semaphore: Independent semaphore for Stage 3

    Returns:
        (query_id, ranking) tuple
    """
    query_id = row['query_id']
    question = row['question']
    chunks = row['chunks']
    chunk_indices = row['chunk_indices']
    n_splits = row['n_splits']
    bm25_scores = row.get('bm25_scores', {})

    if logger:
        logger.info("=" * 80)
        logger.info(f"[CHUNK] Query: {query_id}")
        logger.info("-" * 80)
        logger.info(f"Question: {question}")
        logger.info(f"Chunks: {len(chunks)}, Splits: {n_splits}")

    all_scored_items_fallback = []

    try:
        # Stage 1: Local ranking with BM25 fusion
        parts = split_chunks_n_way(chunks, chunk_indices, n_splits)

        if logger:
            logger.info(f"[{query_id}] Stage 1: {n_splits} splits, {len(parts)} parts")

        part_semaphore = asyncio.Semaphore(STAGE1_PART_SEMAPHORE)

        # Process parts in parallel
        part_inputs = []
        tasks = []
        for i, (part_chunks, part_indices) in enumerate(parts):
            part_inputs.append(part_indices)
            part_messages = build_chunk_messages_recall(
                question, part_chunks, part_indices, k=FIXED_LOCAL_K
            )

            if logger and i == 0:
                user_content = part_messages[1]['content']
                truncated = truncate_message(user_content)
                logger.info(f"STAGE 1 PROMPT (Part 1):\n{truncated}")

            part_id = f"{query_id}_part{i+1}"
            task = rank_chunk_part_with_retry(
                part_messages, part_semaphore, part_id,
                expected_count=FIXED_LOCAL_K,
                candidate_indices=part_indices
            )
            tasks.append((task, part_id, part_indices))

        # Gather LLM results
        part_responses = []
        for task, part_id, part_indices in tasks:
            result = await task
            part_responses.append(result)

        # BM25 fusion for each part
        fused_part_responses = []
        for part_scores, part_indices in zip(part_responses, part_inputs):
            fused_scores = fuse_llm_bm25_scores(
                llm_scores=part_scores,
                bm25_scores=bm25_scores,
                indices=part_indices,
                weight_llm=FUSION_WEIGHT_STAGE1
            )
            fused_part_responses.append(fused_scores)

        # Combine and track
        all_scored_items = []
        for i, (part_scores, part_indices) in enumerate(zip(fused_part_responses, part_inputs)):
            local_predictions.append((
                query_id, f"part{i+1}", part_scores.copy(), n_splits, part_indices
            ))
            all_scored_items.extend(part_scores)

        all_scored_items_fallback = all_scored_items.copy()

        if logger:
            logger.info(f"[{query_id}] Stage 1: {len(all_scored_items)} fused items")

        # Sort by fused score
        all_scored_items.sort(key=lambda x: (-x[1], x[0]))
        top_candidates = all_scored_items[:min(50, len(all_scored_items))]

        try:
            # Stage 2: Split rescore
            stage2_items = await rescore_split_stage2(
                question, chunks, top_candidates, query_id
            )

            if stage2_items:
                local_predictions.append((
                    query_id, "stage2_split", stage2_items.copy(), n_splits, None
                ))

            # Stage 3: Final rescore
            if stage2_items:
                rescored_items = await rescore_final_stage3(
                    question, chunks, stage2_items, query_id, stage3_semaphore
                )
            else:
                rescored_items = []

            if rescored_items:
                local_predictions.append((
                    query_id, "stage3_final", rescored_items.copy(), n_splits, None
                ))

            # Score-aware padding
            if rescored_items and len(rescored_items) >= 5:
                top_indices = [idx for idx, _ in rescored_items[:5]]
                if logger:
                    logger.info(f"[{query_id}] Using Stage 3 scores: {len(rescored_items)} items")
            else:
                # Augment with Stage 1 backup
                combined = list(rescored_items) if rescored_items else []
                stage3_indices = {idx for idx, _ in combined}
                stage1_backup = [(idx, score) for idx, score in all_scored_items
                                if idx not in stage3_indices]
                combined.extend(stage1_backup)
                combined.sort(key=lambda x: (-x[1], x[0]))
                top_indices = [idx for idx, _ in combined[:5]]

                if logger:
                    logger.warning(f"[{query_id}] Stage 3 returned {len(rescored_items)}, augmented with Stage 1")

                local_predictions.append((
                    query_id, "stage3_final", combined[:50].copy(), n_splits, None
                ))

        except Exception as e:
            if logger:
                logger.error(f"[{query_id}] Stage 2/3 failed: {e}, using Stage 1")
            local_predictions.append((
                query_id, "stage3_final", all_scored_items.copy(), n_splits, None
            ))
            top_indices = [idx for idx, _ in all_scored_items[:5]]

        ranking = pad_to_n_indices(top_indices, 5, chunk_indices, logger, query_id)

        if logger:
            logger.info(f"[{query_id}] FINAL: {ranking}")
            logger.info("=" * 80 + "\n")

        return (query_id, ranking)

    except Exception as e:
        # Catastrophic fallback: model scores → random → duplicates → defaults
        error_msg = str(e).split(':')[0] if ':' in str(e) else str(e)
        print(f"Chunk ranking {query_id} failed: {error_msg}")

        if logger:
            logger.error(f"[{query_id}] Catastrophic fallback")

        top_indices = []

        # Use model scores if available
        if all_scored_items_fallback:
            top_indices = [idx for idx, _ in all_scored_items_fallback[:5]]

        # Pad with random unused indices
        if len(top_indices) < 5 and chunk_indices:
            used = set(top_indices)
            available = [idx for idx in chunk_indices if idx not in used]
            random.shuffle(available)
            top_indices.extend(available[:5 - len(top_indices)])

        # Cyclic duplicates if still short
        while len(top_indices) < 5 and chunk_indices:
            top_indices.append(chunk_indices[len(top_indices) % len(chunk_indices)])

        # Absolute fallback
        if len(top_indices) < 5:
            top_indices = list(range(5))
            if logger:
                logger.error(f"[{query_id}] Using default [0,1,2,3,4]")

        if logger and len(top_indices) == 5:
            source = f"{len(all_scored_items_fallback)} model scores" if all_scored_items_fallback \
                     else "random" if chunk_indices else "defaults"
            logger.warning(f"[{query_id}] Fallback source: {source}")

        return (query_id, top_indices[:5])


#%% Batch evaluation
async def rank_all_documents(
    df: pd.DataFrame
) -> List[Tuple[str, List[int]]]:
    """
    Process all document ranking queries.

    Args:
        df: DataFrame from load_document_data()

    Returns:
        (query_id, ranking) tuples
    """
    print(f"\nProcessing {len(df):,} document queries...")

    semaphore = asyncio.Semaphore(QUERY_SEMAPHORE)

    async def process_with_stagger(row, query_index):
        initial_delay = query_index * DOC_STAGGER_INTERVAL
        if initial_delay > 0:
            await asyncio.sleep(initial_delay)
        return await rank_single_document(row, semaphore)

    tasks = [process_with_stagger(row, idx) for idx, (_, row) in enumerate(df.iterrows())]
    results = await tqdm_asyncio.gather(*tasks, desc="Document ranking")

    successful = len([result for result in results if result[1]])
    print(f"Completed: {successful}/{len(results)} successful")

    return results


async def rank_all_chunks(
    df: pd.DataFrame
) -> List[Tuple[str, List[int]]]:
    """
    Process all chunk ranking queries.

    Args:
        df: DataFrame from load_chunk_data()

    Returns:
        (query_id, ranking) tuples
    """
    print(f"\nProcessing {len(df):,} chunk queries...")
    print(f"4-Level Semaphore: Query {QUERY_SEMAPHORE} → Stage1 {STAGE1_PART_SEMAPHORE}p → Stage2 {STAGE2_PART_SEMAPHORE}p → Stage3 {STAGE3_SEMAPHORE}")

    query_semaphore = asyncio.Semaphore(QUERY_SEMAPHORE)
    stage3_semaphore = asyncio.Semaphore(STAGE3_SEMAPHORE)

    async def process_with_query_sem(row, query_index):
        initial_delay = query_index * CHUNK_STAGGER_INTERVAL

        if initial_delay > 0:
            await asyncio.sleep(initial_delay)
            if logger:
                logger.info(f"[Query {query_index}] Starting after {initial_delay:.1f}s stagger")

        async with query_semaphore:
            return await rank_single_chunk(row, stage3_semaphore)

    tasks = [process_with_query_sem(row, idx) for idx, (_, row) in enumerate(df.iterrows())]
    results = await tqdm_asyncio.gather(*tasks, desc="Chunk ranking")

    successful = len([result for result in results if result[1]])
    print(f"Completed: {successful}/{len(results)} successful")

    return results


#%% Main execution
async def main(mode: str = "experiment"):
    """
    Main execution pipeline.

    Args:
        mode: 'experiment' or 'production'
    """
    global logger, tracker, parser, local_predictions

    local_predictions = []

    experiment_name = f"{EXPERIMENT['active']}_{EXPERIMENT['timestamp']}"
    logger = setup_model_logger(
        name="fusion_se",
        log_dir=PATHS['logs'],
        experiment_name=experiment_name,
        config_info={
            'model_stage1': MODEL_STAGE1,
            'model_stage2': MODEL_STAGE2,
            'model_stage3': MODEL_STAGE3,
            'query_semaphore': QUERY_SEMAPHORE,
            'stage1_part_semaphore': STAGE1_PART_SEMAPHORE,
            'stage2_part_semaphore': STAGE2_PART_SEMAPHORE,
            'stage3_semaphore': STAGE3_SEMAPHORE,
            'stage2_split_count': STAGE2_SPLIT_COUNT,
            'stage2_k_per_part': STAGE2_K_PER_PART,
            'target_tokens': TARGET_TOKENS_PER_PART,
            'fixed_k': FIXED_LOCAL_K,
            'fusion_weight_stage1': FUSION_WEIGHT_STAGE1,
            'doc_stagger': DOC_STAGGER_INTERVAL,
            'chunk_stagger': CHUNK_STAGGER_INTERVAL,
            'stage1_jitter_max': STAGE1_JITTER_MAX,
            'stage2_jitter_max': STAGE2_JITTER_MAX,
            'stage3_jitter_max': STAGE3_JITTER_MAX
        }
    )
    tracker = APITracker()
    parser = ResponseParser(logger=logger)
    if mode == "experiment":
        doc_path = EXPERIMENT_PATHS['doc_samples']
        chunk_path = EXPERIMENT_PATHS['chunk_samples']

        if not doc_path.exists() or not chunk_path.exists():
            print("Error: Experiment sample files not found")
            print(f"\nRun: uv run src/eval/create_sample.py")
            return

        print("\n" + "="*60)
        print(f"Models: Stage1={MODEL_STAGE1}, Stage2={MODEL_STAGE2}")
        print(f"Experiment: {experiment_name}")
        print("="*60)

    else:  # production
        doc_path = DATA_FILES['doc_eval']
        chunk_path = DATA_FILES['chunk_eval']

        if not doc_path.exists() or not chunk_path.exists():
            print("Error: Kaggle eval files not found")
            return

        print("\n" + "="*60)
        print(f"PRODUCTION MODE")
        print(f"Models: Stage1={MODEL_STAGE1}, Stage2={MODEL_STAGE2}")
        print(f"Processing Kaggle eval sets 400 queries")
        print("="*60)

    # Determine backend per stage
    backend_stage1 = "databricks" if "databricks" in MODEL_STAGE1.lower() else "openai"
    backend_stage2 = "databricks" if "databricks" in MODEL_STAGE2.lower() else "openai"
    backend_stage3 = "databricks" if "databricks" in MODEL_STAGE3.lower() else "openai"

    extra_params = {}
    global client_stage1, client_stage2, client_stage3
    extra_params_stage1 = extra_params.copy()
    if "gpt-oss-120b" in MODEL_STAGE1.lower():
        extra_params_stage1["reasoning_effort"] = "medium"
        print(f"[CONFIG] Stage 1 GPT-OSS-120B: reasoning effort={extra_params_stage1['reasoning_effort']}")
    elif "gpt-5" in MODEL_STAGE1.lower():
        extra_params_stage1["reasoning_effort"] = "minimal"
        print(f"[CONFIG] Stage 1 GPT-5-mini: reasoning effort={extra_params_stage1['reasoning_effort']}")

    client_stage1 = UnifiedLLMClient(
        backend=backend_stage1,
        model=MODEL_STAGE1,
        temperature=0.1,
        max_tokens=1500,
        extra_params=extra_params_stage1
    )

    print(f"\n[Stage 1] LLM: {backend_stage1}/{MODEL_STAGE1} (temp={client_stage1.temperature}, tokens={client_stage1.max_tokens})")
    extra_params_stage2 = extra_params.copy()
    if "gpt-5" in MODEL_STAGE2.lower():
        extra_params_stage2["reasoning_effort"] = "medium"
        print(f"[CONFIG] Stage 2 GPT-5-mini: reasoning effort={extra_params_stage2['reasoning_effort']}")

    client_stage2 = UnifiedLLMClient(
        backend=backend_stage2,
        model=MODEL_STAGE2,
        temperature=0.1,
        max_tokens=1500,
        extra_params=extra_params_stage2
    )

    print(f"[Stage 2] LLM: {backend_stage2}/{MODEL_STAGE2} (temp={client_stage2.temperature}, tokens={client_stage2.max_tokens})")
    extra_params_stage3 = extra_params.copy()
    if "gpt-5" in MODEL_STAGE3.lower():
        extra_params_stage3["reasoning_effort"] = "medium"
        print(f"[CONFIG] Stage 3 GPT-5-mini: reasoning effort={extra_params_stage3['reasoning_effort']}")

    client_stage3 = UnifiedLLMClient(
        backend=backend_stage3,
        model=MODEL_STAGE3,
        temperature=0.1,
        max_tokens=1500,
        extra_params=extra_params_stage3
    )

    print(f"[Stage 3] LLM: {backend_stage3}/{MODEL_STAGE3} (temp={client_stage3.temperature}, tokens={client_stage3.max_tokens})")
    global rescue_client
    if os.getenv("OPENAI_API_KEY"):
        rescue_client = UnifiedLLMClient(
            backend="openai",
            model="gpt-4o-mini",
            temperature=0.0,
            max_tokens=100
        )
        logger.info("GPT-4o-mini rescue client initialized")
    else:
        logger.warning("OPENAI_API_KEY not set, rescue disabled")

    print("\nChecking API availability...")
    print(f"Checking Stage 1 model: {MODEL_STAGE1}...")
    await check_llm_health(MODEL_STAGE1)
    print(f"Checking Stage 2 model: {MODEL_STAGE2}...")
    await check_llm_health(MODEL_STAGE2)
    print(f"Checking Stage 3 model: {MODEL_STAGE3}...")
    await check_llm_health(MODEL_STAGE3)

    print("\nLoading data...")
    doc_df = load_document_data(doc_path)
    chunk_df = load_chunk_data(chunk_path, TARGET_TOKENS_PER_PART)

    print("\nStarting ranking processes...")
    start_time = time.time()

    doc_start = time.time()
    doc_results = await rank_all_documents(doc_df)
    doc_time = time.time() - doc_start

    chunk_start = time.time()
    chunk_results = await rank_all_chunks(chunk_df)
    chunk_time = time.time() - chunk_start

    if mode == "experiment" and local_predictions:
        local_pred_path = EXPERIMENT_PATHS['dir'] / 'local_predictions.jsonl'
        with jsonlines.open(local_pred_path, 'w') as writer:
            for query_id, part_id, predictions, n_splits, input_indices in local_predictions:
                writer.write({
                    'query_id': query_id,
                    'part_id': part_id,
                    'n_splits': n_splits,
                    'input_indices': input_indices,  # Chunks available in this split
                    'predictions': predictions        # [(chunk_idx, score), ...]
                })
        print(f"\nLocal predictions exported: {local_pred_path.name}")

    output_path = EXPERIMENT_PATHS['submission']
    output_path.parent.mkdir(parents=True, exist_ok=True)

    submission_data = []

    for query_id, ranking in doc_results:
        if not ranking:
            continue
        sample_id = query_id if query_id.startswith('doc_') else f'doc_{query_id}'
        for target_idx in ranking[:5]:
            submission_data.append({'sample_id': sample_id, 'target_index': target_idx})

    for query_id, ranking in chunk_results:
        if not ranking:
            continue
        sample_id = query_id if query_id.startswith('chunk_') else f'chunk_{query_id}'
        for target_idx in ranking[:5]:
            submission_data.append({'sample_id': sample_id, 'target_index': target_idx})

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['sample_id', 'target_index'])
        for entry in submission_data:
            writer.writerow([entry['sample_id'], entry['target_index']])

    print(f"\nSubmission saved: {output_path}")

    stats_path = EXPERIMENT_PATHS['dir'] / 'api_stats.json'
    tracker.export_json(stats_path)
    def format_time(seconds: float) -> str:
        """Format seconds as mm:ss"""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}:{secs:02d}"

    total_time = time.time() - start_time
    print("\n" + "="*60)
    print(f"Completed: {len(doc_results):,} docs ({format_time(doc_time)}) + {len(chunk_results):,} chunks ({format_time(chunk_time)}) = {format_time(total_time)} total")
    print("="*60)

    logger.info(f"Runtime: docs={format_time(doc_time)}, chunks={format_time(chunk_time)}, total={format_time(total_time)}")

    if mode == "experiment":
        print("\nNext: uv run src/eval/score_submission.py")
    else:
        print("\nValidating submission...")
        validation_passed = validate_final_submission(output_path, verbose=True)
        print("\nValidation passed - Ready for Kaggle!" if validation_passed else "\nWARNING: Submission has validation errors")


#%% Entry point
if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description='OpenAI backend')
    arg_parser.add_argument('--production', action='store_true',
                        help='Run in production mode (full 400 queries)')
    args = arg_parser.parse_args()

    mode = "production" if args.production else "experiment"
    asyncio.run(main(mode=mode))
