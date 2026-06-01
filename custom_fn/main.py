from app.config.settings import MODEL
from app.pipelines.eval_pipeline import run_evaluation


def run_demo():

    question = "Tell me about Netflix."

    answer = (
        "Netflix was founded by Reed Hastings in 1997. "
        "The company originally started as a DVD rental service. "
        "Netflix launched its streaming platform in 2007. "
        "The company was acquired by Amazon in 2015. "
        "Netflix is headquartered in California."
    )

    context = [
        (
            "Netflix was founded in 1997 by Reed Hastings "
            "and Marc Randolph."
        ),
        (
            "The company initially operated as a DVD rental service."
        ),
        (
            "Netflix introduced streaming services in 2007."
        ),
        (
            "Netflix headquarters are located in Los Gatos, California."
        )
    ]

    ground_truths = (
        "Netflix is an American entertainment company founded in 1997 by "
        "Reed Hastings and Marc Randolph. The company initially operated "
        "as a DVD rental service before launching its streaming platform "
        "in 2007. Netflix is headquartered in Los Gatos, California."
    )

    print("=" * 80)
    print(f"Model : {MODEL}")
    print("Running RAG Evaluation Smoke Test")
    print("=" * 80)

    result = run_evaluation(
        questions=[question],
        answers=[answer],
        retrieved_contexts=[context],
        ground_truths=[ground_truths],
    )

    # ============================================================
    # Aggregate Scores
    # ============================================================

    print("\n")
    print("=" * 80)
    print("AGGREGATE SCORES")
    print("=" * 80)

    for metric, score in result["aggregate_scores"].items():
        print(f"{metric:<22}: {score:.4f}")

    # ============================================================
    # Individual Metric Results
    # ============================================================

    sample = result["per_sample_results"][0]

    # ------------------------------------------------------------
    # 1. Faithfulness
    # ------------------------------------------------------------

    faith = sample["faithfulness"]

    print("\n")
    print("=" * 80)
    print("1. FAITHFULNESS")
    print("=" * 80)

    print(f"Score              : {faith['score']}")
    print(f"Total Claims       : {faith['total_claims']}")
    print(f"Supported Claims   : {faith['supported']}")
    print(f"Unsupported Claims : {faith['unsupported']}")

    print("\nClaim Verdicts:\n")

    for idx, verdict in enumerate(faith["claim_verdicts"], start=1):
        print(f"Claim {idx}")
        print(f"Text       : {verdict['claim']}")
        print(f"Supported  : {verdict['supported']}")
        print(f"Reason     : {verdict['reason']}")
        print("-" * 60)

    print(f"\nReasoning : {faith['reasoning']}")

    # ------------------------------------------------------------
    # 2. Answer Relevancy
    # ------------------------------------------------------------

    relevancy = sample["answer_relevancy"]

    print("\n")
    print("=" * 80)
    print("2. ANSWER RELEVANCY")
    print("=" * 80)

    print(f"Score : {relevancy['score']}")

    print("\nGenerated Questions + Similarities:\n")

    for idx, (q, sim) in enumerate(
        zip(
            relevancy["generated_questions"],
            relevancy["similarities"]
        ),
        start=1,
    ):
        print(f"Question {idx}")
        print(f"Generated Question : {q}")
        print(f"Similarity Score   : {sim}")
        print("-" * 60)

    print(f"\nReasoning : {relevancy['reasoning']}")

    # ------------------------------------------------------------
    # 3. Context Precision
    # ------------------------------------------------------------

    precision = sample["context_precision"]

    print("\n")
    print("=" * 80)
    print("3. CONTEXT PRECISION")
    print("=" * 80)

    print(f"Score            : {precision['score']}")
    print(f"Total Chunks     : {precision['total_chunks']}")
    print(f"Relevant Chunks  : {precision['relevant_chunks']}")

    print("\nChunk Verdicts:\n")

    for verdict in precision["chunk_verdicts"]:
        print(f"Chunk Index : {verdict['chunk_index']}")
        print(f"Relevant    : {verdict['relevant']}")
        print(f"Reason      : {verdict['reason']}")
        print("-" * 60)

    print(f"\nReasoning : {precision['reasoning']}")

    # ------------------------------------------------------------
    # 4. Context Recall
    # ------------------------------------------------------------

    recall = sample["context_recall"]

    print("\n")
    print("=" * 80)
    print("4. CONTEXT RECALL")
    print("=" * 80)

    print(f"Score                     : {recall['score']}")
    print(f"Total Statements          : {recall['total_statements']}")
    print(f"Attributable Statements   : {recall['attributable_statements']}")

    print("\nStatement Verdicts:\n")

    for idx, verdict in enumerate(recall["statement_verdicts"], start=1):
        print(f"Statement {idx}")
        print(f"Text          : {verdict['statement']}")
        print(f"Attributable  : {verdict['attributable']}")
        print(f"Reason        : {verdict['reason']}")
        print("-" * 60)

    print(f"\nReasoning : {recall['reasoning']}")

    # ============================================================
    # Final Summary
    # ============================================================

    print("\n")
    print("=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    print(f"Total samples  : {result['total_samples']}")
    print(f"Evaluated      : {result['evaluated_samples']}")

    if result["errors"]:
        print(f"Errors         : {result['errors']}")

    print("=" * 80)


if __name__ == "__main__":
    run_demo()