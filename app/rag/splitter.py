from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List

def split_documents(documents: List[Document], chunk_size: int = 500, chunk_overlap: int = 50) -> List[Document]:
    if chunk_size < 1 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size")
    
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    final_chunks = []
    for doc in documents:
        parent_docs = parent_splitter.split_documents([doc])
        for p_idx, p_doc in enumerate(parent_docs):
            child_docs = child_splitter.split_documents([p_doc])
            for c_idx, c_doc in enumerate(child_docs):
                c_doc.metadata["parent_context"] = p_doc.page_content
                c_doc.metadata["parent_id"] = f"{p_doc.metadata.get('source','')}_p{p_idx}"
                final_chunks.append(c_doc)
                
    return final_chunks if final_chunks else RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap).split_documents(documents)
