import re
from typing import List, Sequence
from langchain_core.documents import Document

class PlannerAgent:
    def decompose_query(self, query: str) -> List[str]:
        q_lower = query.lower().strip()
        if any(kw in q_lower for kw in ["compare", "versus", "vs", "difference between", "both"]):
            parts = re.split(r'\b(?:and|vs|versus|compare)\b', query, flags=re.IGNORECASE)
            sub_queries = [p.strip() for p in parts if len(p.strip()) > 3]
            if len(sub_queries) >= 2:
                return [query] + sub_queries[:2]
        return [query]

class VerificationAgent:
    def verify_groundedness(self, answer: str, context_docs: Sequence[Document]) -> dict:
        if not answer or not context_docs:
            return {"is_grounded": False, "confidence_score": 0.0, "hallucination_risk": "HIGH"}
        
        context_text = " ".join([d.page_content.lower() for d in context_docs])
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', answer) if len(s.strip()) > 10]
        
        if not sentences:
            return {"is_grounded": True, "confidence_score": 95.0, "hallucination_risk": "LOW"}
            
        supported_count = 0
        for s in sentences:
            words = [w.lower() for w in re.findall(r'\w+', s) if len(w) > 3]
            if not words:
                supported_count += 1
                continue
            matches = sum(1 for w in words if w in context_text)
            if (matches / len(words)) >= 0.3:
                supported_count += 1
                
        confidence = round((supported_count / max(len(sentences), 1)) * 100, 1)
        risk = "LOW" if confidence >= 75.0 else ("MEDIUM" if confidence >= 50.0 else "HIGH")
        
        return {
            "is_grounded": confidence >= 50.0,
            "confidence_score": confidence,
            "hallucination_risk": risk
        }

class CitationAgent:
    def format_citations(self, documents: Sequence[Document]) -> str:
        formatted_chunks = []
        for index, doc in enumerate(documents, start=1):
            src_file = str(doc.metadata.get("source", "File")).replace("\\", "/").split("/")[-1]
            page_num = doc.metadata.get("page")
            page_str = f" | Page {page_num + 1}" if page_num is not None else ""
            formatted_chunks.append(f"[Source {index}: {src_file}{page_str}]\n{doc.page_content}")
        return "\n\n".join(formatted_chunks)

class FollowUpAgent:
    def generate_followups(self, query: str, answer: str) -> List[str]:
        q_clean = query.lower()
        if "bayes" in q_clean or "page" in q_clean:
            return [
                "What are the mathematical steps for Naïve Bayes independence assumptions?",
                "How does Maximum Likelihood Estimation (MLE) compare with MAP?",
                "Can you summarize pages 45 to 50 specifically?"
            ]
        if "risk" in q_clean or "ebitda" in q_clean or "financial" in q_clean:
            return [
                "What are the primary liquidity covenants outlined in the filings?",
                "How do operating margins compare across fiscal quarters?",
                "What debt maturity obligations are due within 12 months?"
            ]
        return [
            "Can you elaborate on the key findings mentioned above?",
            "What are the main potential risks or limitations highlighted?",
            "How does this section relate to the broader document objectives?"
        ]
