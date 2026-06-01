# Add evaluate_context_recall here
from app.core.llm_client import _call_llm
from app.core.utils import _parse_json_response, _clamp

def evaluate_context_recall(
    question: str,
    retrieved_context: list,
    ground_truth: str,
) -> dict:
    """
    Measure whether retrieval captured all information necessary to produce
    the ground-truth answer.

    Scoring approach
    ----------------
    1. Decompose the ground-truth answer into atomic statements.
    2. For each statement, check whether it is attributable to at least one
       retrieved chunk.
    3. context_recall = attributable_statements / total_statements  (range 0–1)

    Parameters
    ----------
    question          : str       – the original user question.
    retrieved_context : list[str] – the retrieved text chunks.
    ground_truth      : str       – the reference / ideal answer.

    Returns
    -------
    dict with keys:
        score                   (float) – recall score in [0, 1]
        total_statements        (int)
        attributable_statements (int)
        statement_verdicts      (list)  – per-statement breakdown
        reasoning               (str)
    """
    context_block = "\n\n".join(
        f"[Chunk {i+1}]: {chunk}" for i, chunk in enumerate(retrieved_context)
    )

    # ── Step 1: decompose ground truth into statements ──────────────────────
    stmt_system = (
        "You are a precise fact-extractor. "
        "Return ONLY a valid JSON object. No markdown, no extra text."
    )
    stmt_prompt = (
        f"Decompose the following GROUND TRUTH ANSWER into individual, "
        f"atomic statements needed to fully answer the question.\n\n"
        f"QUESTION: {question}\n\n"
        f"GROUND TRUTH: {ground_truth}\n\n"
        'Return exactly: {"statements": ["statement 1", "statement 2", ...]}'
    )
    stmt_raw   = _call_llm(stmt_system, stmt_prompt)
    stmt_data  = _parse_json_response(stmt_raw)
    statements = stmt_data.get("statements", [ground_truth])

    if not statements:
        return {
            "score": 0.0,
            "total_statements": 0,
            "attributable_statements": 0,
            "statement_verdicts": [],
            "reasoning": "No statements could be extracted from the ground truth.",
        }

    # ── Step 2: check attributability against retrieved context ─────────────
    attr_system = (
        "You are a strict attributability checker for RAG pipelines. "
        "Return ONLY a valid JSON object. No markdown, no extra text."
    )
    verdicts: list[dict] = []

    for stmt in statements:
        attr_prompt = (
            f"RETRIEVED CONTEXT:\n{context_block}\n\n"
            f"STATEMENT: {stmt}\n\n"
            "Can this statement be directly attributed to (found in) the retrieved context?\n"
            'Return exactly: {"attributable": true or false, "reason": "brief explanation"}'
        )
        raw  = _call_llm(attr_system, attr_prompt)
        data = _parse_json_response(raw)
        verdicts.append({
            "statement":    stmt,
            "attributable": bool(data.get("attributable", False)),
            "reason":       data.get("reason", ""),
        })

    attributable_count = sum(1 for v in verdicts if v["attributable"])
    score              = _clamp(attributable_count / len(verdicts))

    return {
        "score":                   round(score, 4),
        "total_statements":        len(verdicts),
        "attributable_statements": attributable_count,
        "statement_verdicts":      verdicts,
        "reasoning":               (
            f"{attributable_count}/{len(verdicts)} ground-truth statements "
            "are attributable to the retrieved context."
        ),
    }

