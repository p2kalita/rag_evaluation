# Add evaluate_context_precision here
from app.core.llm_client import _call_llm
from app.core.utils import _parse_json_response, _clamp

def evaluate_context_precision(
    question: str,
    retrieved_context: list,
    ground_truth: str,
) -> dict:
    """
    Measure whether the retrieved chunks are relevant and useful for answering
    the question, given the known ground-truth answer.

    Scoring approach
    ----------------
    For each retrieved chunk, ask the LLM whether it is relevant for answering
    the question (with awareness of the ground truth).
    context_precision = relevant_chunks / total_chunks  (range 0–1)

    Parameters
    ----------
    question          : str       – the original user question.
    retrieved_context : list[str] – the retrieved text chunks.
    ground_truth      : str       – the reference / ideal answer.

    Returns
    -------
    dict with keys:
        score            (float) – precision score in [0, 1]
        total_chunks     (int)
        relevant_chunks  (int)
        chunk_verdicts   (list)  – per-chunk relevance verdict
        reasoning        (str)
    """
    if not retrieved_context:
        return {
            "score": 0.0,
            "total_chunks": 0,
            "relevant_chunks": 0,
            "chunk_verdicts": [],
            "reasoning": "No context chunks were provided.",
        }

    system = (
        "You are a strict relevance judge for RAG pipelines. "
        "Return ONLY a valid JSON object. No markdown, no extra text."
    )
    verdicts: list[dict] = []

    for i, chunk in enumerate(retrieved_context):
        prompt = (
            f"QUESTION: {question}\n\n"
            f"GROUND TRUTH ANSWER: {ground_truth}\n\n"
            f"RETRIEVED CHUNK {i+1}:\n{chunk}\n\n"
            "Is this chunk relevant and useful for answering the question "
            "(i.e. does it contain information needed to arrive at the ground truth)?\n"
            'Return exactly: {"relevant": true or false, "reason": "brief explanation"}'
        )
        raw  = _call_llm(system, prompt)
        data = _parse_json_response(raw)
        verdicts.append({
            "chunk_index": i + 1,
            "relevant":    bool(data.get("relevant", False)),
            "reason":      data.get("reason", ""),
        })

    relevant_count = sum(1 for v in verdicts if v["relevant"])
    score          = _clamp(relevant_count / len(verdicts))

    return {
        "score":           round(score, 4),
        "total_chunks":    len(verdicts),
        "relevant_chunks": relevant_count,
        "chunk_verdicts":  verdicts,
        "reasoning":       (
            f"{relevant_count}/{len(verdicts)} retrieved chunks are relevant to the question."
        ),
    }

