# Add run_ragas_evaluation here
import statistics
from typing import Any


from app.metrics.faithfulness import evaluate_faithfulness
from app.metrics.answer_relevancy import evaluate_answer_relevancy
from app.metrics.context_precision import evaluate_context_precision
from app.metrics.context_recall import evaluate_context_recall


def run_evaluation(
    questions:          list,
    answers:            list,
    retrieved_contexts: list,
    ground_truths:      list,
) -> dict:
    """
    Run end-to-end RAGAS evaluation across all four metrics and return an
    aggregated report.

    Parameters
    ----------
    questions          : list[str]        – one question per sample.
    answers            : list[str]        – RAG-generated answer per sample.
    retrieved_contexts : list[list[str]]  – list of retrieved chunks per sample.
    ground_truths      : list[str]        – reference answer per sample.

    Returns
    -------
    dict with keys:
        aggregate_scores   (dict) – mean score per metric across all samples
        per_sample_results (list) – detailed per-sample breakdown
        total_samples      (int)
        evaluated_samples  (int)
        errors             (list) – per-sample errors encountered
    """
    n = len(questions)
    if not (n == len(answers) == len(retrieved_contexts) == len(ground_truths)):
        raise ValueError(
            "questions, answers, retrieved_contexts, and ground_truths "
            "must all have the same length."
        )

    per_sample: list[dict] = []
    errors:     list[dict] = []

    metric_scores: dict[str, list[float]] = {
        "faithfulness":      [],
        "answer_relevancy":  [],
        "context_precision": [],
        "context_recall":    [],
    }

    for idx in range(n):
        sample_result: dict[str, Any] = {"sample_index": idx}
        try:
            faith = evaluate_faithfulness(
                questions[idx], answers[idx], retrieved_contexts[idx]
            )
            rel   = evaluate_answer_relevancy(questions[idx], answers[idx])
            prec  = evaluate_context_precision(
                questions[idx], retrieved_contexts[idx], ground_truths[idx]
            )
            rec   = evaluate_context_recall(
                questions[idx], retrieved_contexts[idx], ground_truths[idx]
            )

            sample_result.update({
                "question":          questions[idx],
                "faithfulness":      faith,
                "answer_relevancy":  rel,
                "context_precision": prec,
                "context_recall":    rec,
            })

            metric_scores["faithfulness"].append(faith["score"])
            metric_scores["answer_relevancy"].append(rel["score"])
            metric_scores["context_precision"].append(prec["score"])
            metric_scores["context_recall"].append(rec["score"])

        except Exception as exc:  # noqa: BLE001
            errors.append({"sample_index": idx, "error": str(exc)})
            sample_result["error"] = str(exc)

        per_sample.append(sample_result)

    # ── Aggregate ────────────────────────────────────────────────────────────
    def _mean(vals: list[float]) -> float:
        return round(statistics.mean(vals), 4) if vals else 0.0

    aggregate = {metric: _mean(scores) for metric, scores in metric_scores.items()}

    # Composite score — simple mean of the four metrics
    aggregate["ragas_score"] = round(
        statistics.mean(aggregate.values()), 4
    ) if aggregate else 0.0

    return {
        "aggregate_scores":   aggregate,
        "per_sample_results": per_sample,
        "total_samples":      n,
        "evaluated_samples":  n - len(errors),
        "errors":             errors,
    }