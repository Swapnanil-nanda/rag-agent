from langchain_core.documents import Document
from app.rag.evaluator import evaluate_rag_response

def test_ragas_evaluation_metrics():
    docs = [
        Document(page_content="Bayes Theorem computes posterior probability P(A|B) using likelihood P(B|A) and prior P(A).", metadata={"page": 35}),
        Document(page_content="MAP hypothesis maximizes the posterior probability P(h|D) over all candidate hypotheses.", metadata={"page": 36})
    ]
    question = "What is Bayes Theorem and MAP hypothesis?"
    answer = "Bayes Theorem calculates posterior probability using prior and likelihood. MAP hypothesis maximizes posterior probability."
    
    metrics = evaluate_rag_response(question, answer, docs, latency_ms=120.5)
    
    assert metrics["faithfulness"] >= 70.0
    assert metrics["answer_relevance"] >= 60.0
    assert metrics["context_precision"] >= 50.0
    assert metrics["status"] == "PASS"
