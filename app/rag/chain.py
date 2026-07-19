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

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a highly intelligent AI assistant with access to uploaded document context. You behave like a normal, knowledgeable AI — you can answer any question, have conversations, be creative, and use your general knowledge freely.

When document context is provided below, use it to enhance your answers. You can:
- Summarize, analyze, compare, and explain document content
- Generate new ideas, suggestions, or creative content inspired by the documents
- Combine your general knowledge with document facts to give richer answers
- Answer follow-up questions naturally using both the documents and conversation history

When referencing specific facts from the documents, mention that they come from the uploaded files. When using your own general knowledge, you may do so freely — just be helpful and accurate.

Retrieved document context (use as reference material, not as a restriction):
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
            temperature=0.7,
            max_retries=1,
        )
    if not settings.openai_api_key or settings.openai_api_key.startswith("mock"):
        return None
    return ChatOpenAI(
        api_key=settings.openai_api_key,
        model=settings.llm_model,
        temperature=0.7,
        max_retries=1,
    )

def build_chain():
    llm = get_llm()
    if llm is None:
        return None
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
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
    formatted_chunks = []
    for index, doc in enumerate(documents, start=1):
        src_file = str(doc.metadata.get("source", "File")).replace("\\", "/").split("/")[-1]
        page_num = doc.metadata.get("page")
        page_str = f" | Page {page_num + 1}" if page_num is not None else ""
        formatted_chunks.append(f"[Source {index}: {src_file}{page_str}]\n{doc.page_content}")
    return "\n\n".join(formatted_chunks)

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
        return "I couldn't find explicit details about that in the provided documents."
    terms = set(re.findall(r"[a-zA-Z0-9]{3,}", question.lower()))
    candidates: list[str] = []
    for document in documents:
        for sentence in re.split(r"(?<=[.!?])\s+", document.page_content.replace("\n", " ")):
            if len(sentence) > 20 and terms.intersection(sentence.lower().split()):
                candidates.append(sentence.strip())
    if candidates:
        return "\n\n".join(f"- {sentence}" for sentence in candidates[:4])
    return "I couldn't find explicit details about that in the provided documents."

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
        return await chain.ainvoke({
            "context": _context(context_docs), "history": _history(history), "question": question
        })
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
