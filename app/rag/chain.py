import asyncio
import re
from typing import AsyncGenerator, Sequence
import logging
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from app.config import settings
from app.rag.schemas import ChatMessage
from app.rag.agents import CitationAgent, VerificationAgent

logger = logging.getLogger(__name__)

STRICT_GROUNDED_SYSTEM_PROMPT = """You are an Enterprise Knowledge Intelligence Assistant.
Your primary duty is to provide strictly grounded answers based ONLY on the retrieved document context below.

Rules for your responses:
1. Synthesize accurate, natural answers using facts contained directly in the provided context.
2. When referencing facts, cite the source using inline brackets like [Source 1: Filename | Page X].
3. If the retrieved document context does NOT contain enough information to answer the question, explicitly state: "The uploaded documents do not contain sufficient evidence to answer this question."
4. Do NOT speculate, extrapolate, or fabricate information outside the provided document context.

Retrieved Document Context:
{context}"""

CHITCHAT_SYSTEM_PROMPT = """You are a friendly, warm, and highly conversational AI assistant.
Engage with the user naturally, like a friendly human colleague.
Keep your responses concise, helpful, and pleasant.
If they ask about documents, let them know you are ready to analyze any PDF they upload in the sidebar."""

def get_llm() -> ChatOpenAI | None:
    if settings.openrouter_api_key and not settings.openrouter_api_key.startswith("mock"):
        return ChatOpenAI(
            api_key=settings.openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
            model=settings.openrouter_model,
            temperature=0.3,
            max_retries=1,
        )
    if not settings.openai_api_key or settings.openai_api_key.startswith("mock"):
        return None
    return ChatOpenAI(
        api_key=settings.openai_api_key,
        model=settings.llm_model,
        temperature=0.3,
        max_retries=1,
    )

def build_chain():
    llm = get_llm()
    if llm is None:
        return None
    prompt = ChatPromptTemplate.from_messages([
        ("system", STRICT_GROUNDED_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])
    return prompt | llm | StrOutputParser()

def build_chitchat_chain():
    llm = get_llm()
    if llm is None:
        return None
    prompt = ChatPromptTemplate.from_messages([
        ("system", CHITCHAT_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])
    return prompt | llm | StrOutputParser()

def _context(documents: Sequence[Document]) -> str:
    citation_agent = CitationAgent()
    docs_to_use = []
    for doc in documents:
        if "parent_context" in doc.metadata:
            parent_doc = Document(page_content=doc.metadata["parent_context"], metadata=doc.metadata)
            docs_to_use.append(parent_doc)
        else:
            docs_to_use.append(doc)
    return citation_agent.format_citations(docs_to_use)

def _history(messages: Sequence[ChatMessage] | None) -> list[HumanMessage | AIMessage]:
    return [
        HumanMessage(content=message.content)
        if message.role == "user"
        else AIMessage(content=message.content)
        for message in (messages or [])
    ]

def _is_chitchat_query(query: str) -> bool:
    clean = re.sub(r'[^\w\s]', '', query.lower().strip())
    chitchat_phrases = {
        "hi", "hello", "hey", "hola", "greetings", "good morning", "good afternoon", "good evening",
        "how are you", "how is it going", "whats up", "how do you do",
        "who are you", "what is your name", "tell me about yourself", "what can you do", "introduce yourself",
        "thank you", "thanks", "appreciate it", "great", "awesome", "perfect",
        "bye", "goodbye", "see you later", "see ya", "talk to you later",
        "help me", "what is this", "what do you do"
    }
    if clean in chitchat_phrases:
        return True
    words = clean.split()
    if len(words) <= 3 and any(w in {"you", "me", "hi", "hello", "thanks", "thank", "who", "what", "how", "name", "do", "help"} for w in words):
        return True
    return False

def _fallback_chitchat(question: str) -> str:
    clean = re.sub(r'[^\w\s]', '', question.lower().strip())
    if any(greet in clean for greet in ["hi", "hello", "hey", "morning", "evening", "greetings"]):
        return "Hello! 😊 How can I help you today? Please feel free to ask me general questions or upload a PDF in the workspace sidebar so we can query it together!"
    if "how are you" in clean or "how is it going" in clean:
        return "I'm doing great, thank you for asking! 🚀 How are you doing? Let me know if you want to upload a PDF and start searching or summarizing its contents."
    if "thank" in clean:
        return "You're very welcome! Glad I could help. Let me know if you have any other questions."
    if "who are you" in clean or "your name" in clean or "what is this" in clean:
        return "I am your private Context Engine assistant. I can help you analyze PDF documents and also chat about general topics. You can upload files on the left menu!"
    return "Hi there! I'm ready to chat. Please upload a PDF document in the sidebar, and I'll help you search and extract key information from it!"

def _fallback_answer(question: str, documents: Sequence[Document]) -> str:
    if not documents:
        return "The uploaded documents do not contain sufficient evidence to answer this question."
    terms = set(re.findall(r"[a-zA-Z0-9]{3,}", question.lower()))
    candidates: list[str] = []
    for document in documents:
        for sentence in re.split(r"(?<=[.!?])\s+", document.page_content.replace("\n", " ")):
            if len(sentence) > 20 and terms.intersection(sentence.lower().split()):
                candidates.append(sentence.strip())
    if candidates:
        return "\n\n".join(f"- {sentence}" for sentence in candidates[:4])
    return "The uploaded documents do not contain sufficient evidence to answer this question."

async def generate_response(
    question: str, context_docs: Sequence[Document], history: Sequence[ChatMessage] | None = None
) -> str:
    if _is_chitchat_query(question) or not context_docs:
        chain = build_chitchat_chain()
        if chain is None:
            return _fallback_chitchat(question)
        try:
            return await chain.ainvoke({
                "history": _history(history), "question": question
            })
        except Exception:
            logger.exception("Chitchat answer generation failed; returning conversational fallback")
            return _fallback_chitchat(question)

    chain = build_chain()
    if chain is None:
        return _fallback_answer(question, context_docs)
    try:
        raw_ans = await chain.ainvoke({
            "context": _context(context_docs), "history": _history(history), "question": question
        })
        verifier = VerificationAgent()
        verification = verifier.verify_groundedness(raw_ans, context_docs)
        if verification["hallucination_risk"] == "HIGH":
            logger.warning(f"Answer failed verification (confidence {verification['confidence_score']}%)")
        return raw_ans
    except Exception:
        logger.exception("Answer generation failed; returning extractive fallback")
        return _fallback_answer(question, context_docs)

async def generate_stream(
    question: str, context_docs: Sequence[Document], history: Sequence[ChatMessage] | None = None
) -> AsyncGenerator[str, None]:
    if _is_chitchat_query(question) or not context_docs:
        chain = build_chitchat_chain()
        if chain is not None:
            try:
                async for chunk in chain.astream({
                    "history": _history(history), "question": question
                }):
                    yield chunk
                return
            except Exception:
                logger.exception("Chitchat streaming failed; returning conversational fallback")
        for token in _fallback_chitchat(question).split(" "):
            yield f"{token} "
            await asyncio.sleep(0)
        return

    chain = build_chain()
    if chain is not None:
        try:
            async for chunk in chain.astream({
                "context": _context(context_docs), "history": _history(history), "question": question
            }):
                yield chunk
            return
        except Exception:
            logger.exception("Answer streaming failed; returning extractive fallback")
    for token in _fallback_answer(question, context_docs).split(" "):
        yield f"{token} "
        await asyncio.sleep(0)
