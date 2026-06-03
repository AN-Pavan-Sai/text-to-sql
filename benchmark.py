"""
Benchmark runner for the text-to-SQL pipeline.

Evaluates the full pipeline (retrieval -> SQL generation -> execution) against
the beaver-query ground truth dataset and computes the metrics required by the
challenge specification:

  - retrieval_recall_at_k : fraction of ground-truth tables found in top-K
  - sql_exact_match       : fraction of queries where generated SQL matches
                            the reference SQL (normalised comparison)
  - sql_execution_match   : fraction of queries where the generated SQL
                            produces the same result set as the reference SQL
  - end_to_end_success    : fraction of queries that are fully correct
                            (retrieval + generation + execution)
"""

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

from datasets import load_dataset

from config import settings
from retrieval import retriever
from llm_agent import generate_sql
from database import db_manager

logger = logging.getLogger(__name__)


def _normalise_sql(sql: str) -> str:
    """Lower-case, collapse whitespace, strip trailing semicolons."""
    sql = sql.lower().strip().rstrip(";").strip()
    sql = re.sub(r"\s+", " ", sql)
    return sql


def _parse_json_field(value: Any) -> Any:
    """Try to parse a string as JSON; return as-is if that fails."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value
    return value


def _compute_retrieval_recall(
    retrieved_tables: List[Dict[str, Any]],
    ground_truth_tables: Any,
) -> float:
    """
    Compute recall: what fraction of ground-truth tables appear in the
    retrieved set?
    """
    gt = _parse_json_field(ground_truth_tables)
    if not gt or not isinstance(gt, list):
        return 1.0  # No ground truth to miss

    retrieved_names = {t["table_name"].lower() for t in retrieved_tables}
    gt_names = {str(t).lower() for t in gt}

    if not gt_names:
        return 1.0

    hits = len(gt_names & retrieved_names)
    return hits / len(gt_names)


def _results_match(
    result_a: Dict[str, Any],
    result_b: Dict[str, Any],
) -> bool:
    """
    Check whether two SQL execution results are equivalent.
    Compares the sorted row sets (order-insensitive).
    """
    if not result_a.get("success") or not result_b.get("success"):
        return False

    rows_a = sorted([tuple(str(c) for c in r) for r in result_a.get("rows", [])])
    rows_b = sorted([tuple(str(c) for c in r) for r in result_b.get("rows", [])])
    return rows_a == rows_b


def run_benchmark(
    split: Optional[str] = None,
    max_samples: Optional[int] = None,
    top_k: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run the full benchmark on the beaver-query dataset.

    Parameters
    ----------
    split : str, optional
        Which split to evaluate (default: settings.beaver_db_split).
    max_samples : int, optional
        Cap the number of samples to evaluate (useful for quick runs).
    top_k : int, optional
        Number of tables to retrieve per question.

    Returns
    -------
    dict
        Aggregated metrics and per-sample details.
    """
    split = split or settings.beaver_db_split
    top_k = top_k or settings.top_k_tables

    logger.info("Loading beaver-query split=%s ...", split)
    ds = load_dataset("beaverbench/beaver-query", split=split)

    samples = list(ds)
    if max_samples:
        samples = samples[:max_samples]

    total = len(samples)
    retrieval_recalls: List[float] = []
    exact_matches = 0
    execution_matches = 0
    end_to_end_successes = 0
    valid_sql_count = 0
    details: List[Dict[str, Any]] = []

    start_time = time.time()

    for idx, sample in enumerate(samples):
        question = sample["question"]
        db_name = sample["db"]
        gt_sql = sample.get("sql", "")
        gt_tables = sample.get("tables", "[]")

        logger.info("[%d/%d] Processing: %s (db=%s)", idx + 1, total, question[:60], db_name)

        # ---- Step 1: Retrieval ----
        retrieved = retriever.retrieve(question, split, top_k=top_k, db_filter=db_name)
        recall = _compute_retrieval_recall(retrieved, gt_tables)
        retrieval_recalls.append(recall)

        # ---- Step 2: SQL Generation ----
        schema_ctx = retriever.get_schema_context(question, split, top_k=top_k, db_filter=db_name)
        try:
            generated_sql = generate_sql(question, schema_ctx)
        except Exception as exc:
            logger.error("SQL generation failed for sample %s: %s", sample.get("id"), exc)
            details.append({
                "id": sample.get("id"),
                "question": question,
                "db": db_name,
                "retrieval_recall": recall,
                "generated_sql": "",
                "ground_truth_sql": gt_sql,
                "sql_valid": False,
                "exact_match": False,
                "execution_match": False,
                "error": str(exc),
            })
            continue

        # ---- Step 3: Validation ----
        validation = db_manager.validate_sql(generated_sql, split, db_name)
        is_valid = validation.get("valid", False)
        if is_valid:
            valid_sql_count += 1

        # ---- Step 4: Exact match ----
        is_exact = _normalise_sql(generated_sql) == _normalise_sql(gt_sql)
        if is_exact:
            exact_matches += 1

        # ---- Step 5: Execution match ----
        is_exec_match = False
        gen_result = db_manager.execute_sql(generated_sql, split, db_name)
        if gt_sql:
            gt_result = db_manager.execute_sql(gt_sql, split, db_name)
            is_exec_match = _results_match(gen_result, gt_result)
            if is_exec_match:
                execution_matches += 1

        # ---- End-to-end ----
        is_e2e = recall >= 1.0 and is_exec_match
        if is_e2e:
            end_to_end_successes += 1

        details.append({
            "id": sample.get("id"),
            "question": question,
            "db": db_name,
            "retrieval_recall": recall,
            "generated_sql": generated_sql,
            "ground_truth_sql": gt_sql,
            "sql_valid": is_valid,
            "exact_match": is_exact,
            "execution_match": is_exec_match,
            "end_to_end": is_e2e,
        })

    elapsed = time.time() - start_time

    metrics = {
        "split": split,
        "total_samples": total,
        "top_k": top_k,
        "retrieval_recall_at_k": (
            sum(retrieval_recalls) / len(retrieval_recalls)
            if retrieval_recalls else 0.0
        ),
        "sql_validity_rate": valid_sql_count / total if total else 0.0,
        "sql_exact_match_rate": exact_matches / total if total else 0.0,
        "sql_execution_match_rate": execution_matches / total if total else 0.0,
        "end_to_end_success_rate": end_to_end_successes / total if total else 0.0,
        "elapsed_seconds": round(elapsed, 2),
    }

    return {"metrics": metrics, "details": details}
