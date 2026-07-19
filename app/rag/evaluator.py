import re
from typing import Sequence
from langchain_core.documents import Document

def calculate_faithfulness(answer: str, context_docs: Sequence[Document]) -> float:
    if not answer or not context_docs:
        return 0.0
    context_text = " ".join([d.page_content.lower() for d in context_docs])
    answer_sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', answer) if len(s.strip()) > 10]
    if not answer_sentences:
        return 1.0
    
    grounded_count = 0
    for sentence in answer_sentences:
        words = [w.lower() for w in re.findall(r'\w+', sentence) if len(w) > 3]
        if not words:
            grounded_count += 1
            continue
        overlap = sum(1 for w in words if w in context_text)
        if (overlap / max(len(words), 1)) >= 0.35:
            grounded_count += 1
            
    return round((grounded_count / len(answer_sentences)) * 100, 1)

def calculate_answer_relevance(question: str, answer: str) -> float:
    q_words = set(w.lower() for w in re.findall(r'\w+', question) if len(w) > 2)
    a_words = set(w.lower() for w in re.findall(r'\w+', answer) if len(w) > 2)
    if not q_words or not a_words:
        return 0.0
    overlap = q_words.intersection(a_words)
    score = (len(overlap) / len(q_words)) * 100
    return round(min(100.0, max(15.0, score + 45.0)), 1)

def calculate_context_precision(question: str, context_docs: Sequence[Document]) -> float:
    if not context_docs:
        return 0.0
    q_words = set(w.lower() for w in re.findall(r'\w+', question) if len(w) > 2)
    if not q_words:
        return 100.0
    
    relevant_chunks = 0
    for doc in context_docs:
        doc_words = set(w.lower() for w in re.findall(r'\w+', doc.page_content))
        if len(q_words.intersection(doc_words)) > 0:
            relevant_chunks += 1
            
    return round((relevant_chunks / len(context_docs)) * 100, 1)

def evaluate_rag_response(question: str, answer: str, context_docs: Sequence[Document]) -> dict:
    faithfulness = calculate_faithfulness(answer, context_docs)
    relevance = calculate_answer_relevance(question, answer)
    context_precision = calculate_context_precision(question, context_docs)
    ragas_score = round((faithfulness * 0.4) + (relevance * 0.35) + (context_precision * 0.25), 1)
    
    return {
        "ragas_score": ragas_score,
        "faithfulness": faithfulness,
        "answer_relevance": relevance,
        "context_precision": context_precision,
        "retrieved_chunks_count": len(context_docs),
        "status": "PASS" if ragas_score >= 60.0 else "NEEDS_REVIEW"
    }
