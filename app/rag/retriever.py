import re
import math
from collections import Counter
from typing import List, Optional
from langchain_core.documents import Document

def _extract_page_range(query: str) -> tuple[int, int] | None:
    q = query.lower()
    match = re.search(r'pages?\s+(\d+)\s*(?:to|-|\b)\s*(\d+)?', q)
    if match:
        start_p = int(match.group(1))
        end_p = int(match.group(2)) if match.group(2) else start_p
        if start_p > end_p:
            start_p, end_p = end_p, start_p
        return start_p - 1, end_p - 1
    return None

class BM25Scorer:
    def __init__(self, docs: List[Document], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.k1 = k1
        self.b = b
        self.doc_tokens = [re.findall(r'\w+', doc.page_content.lower()) for doc in docs]
        self.doc_lens = [len(tokens) for tokens in self.doc_tokens]
        self.avg_doc_len = sum(self.doc_lens) / max(len(self.doc_lens), 1)
        self.df = Counter()
        for tokens in self.doc_tokens:
            for token in set(tokens):
                self.df[token] += 1

    def get_top_n(self, query: str, n: int = 10) -> List[tuple[Document, float]]:
        q_tokens = re.findall(r'\w+', query.lower())
        N = len(self.docs)
        if N == 0 or not q_tokens:
            return []
        scores = []
        for i, (doc, tokens) in enumerate(zip(self.docs, self.doc_tokens)):
            tf_dict = Counter(tokens)
            doc_len = self.doc_lens[i]
            score = 0.0
            for q in q_tokens:
                if q not in tf_dict:
                    continue
                df = self.df.get(q, 0)
                idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
                tf = tf_dict[q]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / max(self.avg_doc_len, 1.0)))
                score += idf * (numerator / denominator)
            scores.append((doc, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:n]

def _cross_encoder_rerank(query: str, docs: List[Document], top_k: int) -> List[Document]:
    q_words = set(w.lower() for w in re.findall(r'\w+', query) if len(w) > 2)
    if not q_words or not docs:
        return docs[:top_k]
    
    scored_docs = []
    for doc in docs:
        content_words = set(w.lower() for w in re.findall(r'\w+', doc.page_content))
        overlap = len(q_words.intersection(content_words))
        density = overlap / max(len(q_words), 1)
        exact_phrase_bonus = 2.0 if query.lower() in doc.page_content.lower() else 0.0
        final_score = (doc.metadata.get("score", 50) * 0.5) + (density * 40.0) + exact_phrase_bonus
        doc.metadata["rerank_score"] = round(final_score, 1)
        scored_docs.append((doc, final_score))
        
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    return [d[0] for d in scored_docs[:top_k]]

def _reorder_lost_in_middle(docs: List[Document]) -> List[Document]:
    if len(docs) <= 2:
        return docs
    reordered = []
    left = True
    for doc in docs:
        if left:
            reordered.insert(0, doc)
        else:
            reordered.append(doc)
        left = not left
    return reordered

def retrieve_documents(session_id: str, query: str, k: int = 4, threshold: Optional[float] = None) -> List[Document]:
    from app.rag.vectorstore import get_vectorstore
    vs = get_vectorstore(session_id)
    if vs is None:
        return []
    
    all_docstore_docs = list(vs.docstore._dict.values())
    if not all_docstore_docs:
        return []

    page_range = _extract_page_range(query)
    if page_range is not None:
        start_page, end_page = page_range
        matching_page_docs = []
        for doc in all_docstore_docs:
            p = doc.metadata.get("page")
            if p is not None and start_page <= p <= end_page:
                doc.metadata["score"] = 99
                matching_page_docs.append(doc)
        
        if matching_page_docs:
            matching_page_docs.sort(key=lambda d: (d.metadata.get("source", ""), d.metadata.get("page", 0)))
            return _reorder_lost_in_middle(matching_page_docs[:max(k, 25)])

    lower_query = query.lower().strip()
    is_summary_query = any(term in lower_query for term in [
        "summary", "summarize", "overview", "what is this document", 
        "what are these files", "about these files", "about the document",
        "what is it about", "explain the files", "explain these files",
        "tell me about this", "tell me about these"
    ])
    
    if is_summary_query:
        docs_by_source = {}
        for doc in all_docstore_docs:
            src = doc.metadata.get("source", "unknown")
            docs_by_source.setdefault(src, []).append(doc)
        
        ordered_sources = []
        for docs in docs_by_source.values():
            ordered_sources.append(sorted(docs, key=lambda d: d.metadata.get("page", 0)))
            
        summary_docs = []
        max_chunks = max((len(docs) for docs in ordered_sources), default=0)
        for chunk_index in range(max_chunks):
            for documents in ordered_sources:
                if chunk_index < len(documents):
                    d = documents[chunk_index]
                    d.metadata["score"] = 95
                    summary_docs.append(d)
                    if len(summary_docs) >= k:
                        return _reorder_lost_in_middle(summary_docs)
        if summary_docs:
            return _reorder_lost_in_middle(summary_docs)

    from app.rag.agents import PlannerAgent
    planner = PlannerAgent()
    query_variations = planner.decompose_query(query)

    bm25_scorer = BM25Scorer(all_docstore_docs)
    rrf_scores = {}
    doc_map = {}

    for q_var in query_variations:
        vector_docs_and_scores = vs.similarity_search_with_score(q_var, k=max(k * 3, 12))
        bm25_results = bm25_scorer.get_top_n(q_var, n=max(k * 3, 12))

        for rank, (doc, v_score) in enumerate(vector_docs_and_scores):
            doc_id = f"{doc.metadata.get('source', '')}_{doc.metadata.get('page', 0)}_{hash(doc.page_content[:150])}"
            doc_map[doc_id] = doc
            rel_percent = max(0, min(100, int((1.0 - (float(v_score) / 2.0)) * 100)))
            doc.metadata["score"] = rel_percent
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (60 + rank))

        for rank, (doc, b_score) in enumerate(bm25_results):
            doc_id = f"{doc.metadata.get('source', '')}_{doc.metadata.get('page', 0)}_{hash(doc.page_content[:150])}"
            doc_map[doc_id] = doc
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (60 + rank))

    sorted_doc_ids = sorted(rrf_scores.keys(), key=lambda d_id: rrf_scores[d_id], reverse=True)
    fused_docs = [doc_map[d_id] for d_id in sorted_doc_ids]

    if threshold is not None and threshold > 0.0:
        threshold_percent = int(threshold * 100)
        fused_docs = [d for d in fused_docs if d.metadata.get("score", 100) >= threshold_percent]

    reranked_docs = _cross_encoder_rerank(query, fused_docs, top_k=k)
    return _reorder_lost_in_middle(reranked_docs)
