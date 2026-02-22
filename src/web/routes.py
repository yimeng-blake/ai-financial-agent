"""
FastAPI route handlers for the AI Financial Agent web UI.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from typing import List

import fitz  # pymupdf

from fastapi import APIRouter, Query, Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from sse_starlette.sse import EventSourceResponse

from src.graph import build_graph
from src.llm.models import get_llm
from src.state import AgentState
from src.web.serializers import serialize_state_update

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessage] = []
    analysis_context: dict = {}

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
router = APIRouter()

# Thread pool for running the synchronous LangGraph pipeline.
_executor = ThreadPoolExecutor(max_workers=4)

# Display metadata for each pipeline node.
NODE_META = {
    "fetch_data":        {"label": "Fetching Market Data",  "stage": 1, "total": 6},
    "fundamentals":      {"label": "Fundamentals Agent",    "stage": 2, "total": 6},
    "technicals":        {"label": "Technicals Agent",      "stage": 3, "total": 6},
    "sentiment":         {"label": "Sentiment Agent",       "stage": 4, "total": 6},
    "risk_manager":      {"label": "Risk Manager",          "stage": 5, "total": 6},
    "portfolio_manager": {"label": "Portfolio Manager",     "stage": 6, "total": 6},
}


# ---------------------------------------------------------------------------
# Research document upload — session store & PDF extraction
# ---------------------------------------------------------------------------

# In-memory store for extracted PDF text between redirect and SSE stream.
# Key: session UUID string.  Value: {"docs": {ticker: [...]}, "created_at": float}
_session_store: dict[str, dict] = {}

MAX_DOCS_PER_TICKER = 3
MAX_CHARS_PER_DOC = 15_000
MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024  # 20 MB


def _extract_pdf_text(file_bytes: bytes, filename: str) -> str:
    """Extract text from a PDF using pymupdf.  Returns truncated text."""
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        pages_text: list[str] = []
        total_chars = 0
        for page in doc:
            page_text = page.get_text("text")
            pages_text.append(page_text)
            total_chars += len(page_text)
            if total_chars >= MAX_CHARS_PER_DOC:
                break
        doc.close()
        full_text = "\n\n".join(pages_text)
        if len(full_text) > MAX_CHARS_PER_DOC:
            full_text = full_text[:MAX_CHARS_PER_DOC] + "\n\n[... truncated ...]"
        return full_text
    except Exception as exc:
        logger.warning("PDF extraction failed for %s: %s", filename, exc)
        return f"[Error extracting text from {filename}: {exc}]"


def _cleanup_expired_sessions(max_age_seconds: float = 600) -> None:
    """Remove sessions older than *max_age_seconds* (default 10 min)."""
    cutoff = time.time() - max_age_seconds
    expired = [sid for sid, data in _session_store.items() if data["created_at"] < cutoff]
    for sid in expired:
        del _session_store[sid]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/")
async def index(request: Request):
    """Landing page with the ticker / cash input form."""
    return templates.TemplateResponse("index.html", {"request": request})


@router.post("/analyze")
async def start_analysis(request: Request):
    """Validate the form, extract any uploaded PDFs, and redirect (PRG)."""
    _cleanup_expired_sessions()

    form = await request.form()
    tickers_raw = form.get("tickers", "").strip()
    cash = form.get("cash", "100000")

    if not tickers_raw:
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": "Please enter at least one ticker symbol."},
        )

    try:
        cash_val = float(cash)
        if cash_val <= 0:
            raise ValueError
    except ValueError:
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": "Cash must be a positive number."},
        )

    ticker_list = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]

    # --- Process uploaded research PDFs ---
    research_docs: dict[str, list[dict[str, str]]] = {}
    doc_tickers = form.getlist("doc_tickers")
    doc_files = form.getlist("doc_files")

    for i, upload_file in enumerate(doc_files):
        # Skip empty file inputs
        if not hasattr(upload_file, "filename") or not upload_file.filename:
            continue

        # Determine ticker association
        ticker = doc_tickers[i].strip().upper() if i < len(doc_tickers) else None
        if not ticker or ticker not in ticker_list:
            continue

        # Validate file extension
        if not upload_file.filename.lower().endswith(".pdf"):
            continue

        # Check per-ticker limit
        if len(research_docs.get(ticker, [])) >= MAX_DOCS_PER_TICKER:
            continue

        # Read and check file size
        file_bytes = await upload_file.read()
        if len(file_bytes) > MAX_FILE_SIZE_BYTES:
            continue

        # Extract text
        text = _extract_pdf_text(file_bytes, upload_file.filename)

        if ticker not in research_docs:
            research_docs[ticker] = []
        research_docs[ticker].append({
            "filename": upload_file.filename,
            "text": text,
        })

    # --- Store in session if docs were uploaded ---
    session_id = ""
    if research_docs:
        session_id = str(uuid.uuid4())
        _session_store[session_id] = {
            "docs": research_docs,
            "created_at": time.time(),
        }

    redirect_url = f"/results?tickers={tickers_raw}&cash={cash_val}"
    if session_id:
        redirect_url += f"&sid={session_id}"

    return RedirectResponse(url=redirect_url, status_code=303)


@router.get("/results")
async def results_page(
    request: Request,
    tickers: str = Query(""),
    cash: float = Query(100_000.0),
    sid: str = Query(""),
):
    """Results page shell — JavaScript connects to the SSE stream."""
    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    return templates.TemplateResponse(
        "results.html",
        {
            "request": request,
            "tickers": ",".join(ticker_list),
            "ticker_list": ticker_list,
            "cash": cash,
            "sid": sid,
        },
    )


@router.get("/stream")
async def stream_analysis(
    tickers: str = Query(...),
    cash: float = Query(default=100_000.0),
    sid: str = Query(""),
):
    """SSE endpoint — streams pipeline progress as each node completes."""
    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]

    # Retrieve and consume research docs from session store (one-time pop).
    research_docs: dict[str, list[dict[str, str]]] = {}
    if sid and sid in _session_store:
        session_data = _session_store.pop(sid)
        research_docs = session_data.get("docs", {})

    async def event_generator():
        graph = build_graph()
        initial_state: AgentState = {
            "tickers": ticker_list,
            "portfolio_cash": cash,
            "portfolio_positions": {},
            "market_data": {},
            "signals": [],
            "risk_assessments": [],
            "decisions": [],
            "research_documents": research_docs,
        }

        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_event_loop()

        def run_pipeline():
            """Execute the LangGraph pipeline in a worker thread,
            pushing each streamed chunk onto the async queue."""
            try:
                for chunk in graph.stream(initial_state):
                    asyncio.run_coroutine_threadsafe(queue.put(chunk), loop)
                asyncio.run_coroutine_threadsafe(queue.put(None), loop)
            except Exception as exc:
                asyncio.run_coroutine_threadsafe(queue.put(exc), loop)

        # Kick off the pipeline in a background thread.
        loop.run_in_executor(_executor, run_pipeline)

        # --- Yield the "start" event ---
        yield {
            "event": "start",
            "data": json.dumps({"tickers": ticker_list, "cash": cash}),
        }

        # --- Stream node-complete events as they arrive ---
        while True:
            chunk = await queue.get()

            if chunk is None:
                break  # pipeline finished

            if isinstance(chunk, Exception):
                yield {
                    "event": "error",
                    "data": json.dumps({"error": str(chunk)}),
                }
                break

            for node_name, state_delta in chunk.items():
                meta = NODE_META.get(node_name, {})
                payload = serialize_state_update(node_name, state_delta, meta)
                yield {
                    "event": "node_complete",
                    "data": json.dumps(payload),
                }

        # --- Yield the "complete" event ---
        yield {
            "event": "complete",
            "data": json.dumps({"status": "done"}),
        }

    return EventSourceResponse(event_generator(), ping=15)


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

def _build_chat_system_prompt(context: dict) -> str:
    """Format the analysis results into a system prompt for the chat LLM."""
    tickers = context.get("tickers", [])
    cash = context.get("cash", 100_000)

    signals_text = ""
    for s in context.get("signals", []):
        signals_text += (
            f"- {s['ticker']} | {s['agent_name']}: {s['signal']} "
            f"(confidence: {s['confidence']:.0%})\n"
            f"  Reasoning: {s['reasoning']}\n\n"
        )

    risk_text = ""
    for r in context.get("risk_assessments", []):
        risk_text += (
            f"- {r['ticker']}: risk_score={r['risk_score']:.0%}, "
            f"max_position={r['max_position_size']:.0%}\n"
            f"  Factors: {', '.join(r.get('risk_factors', []))}\n"
            f"  Reasoning: {r['reasoning']}\n\n"
        )

    decisions_text = ""
    for d in context.get("decisions", []):
        decisions_text += (
            f"- {d['ticker']}: {d['action'].upper()} {d['quantity']} shares "
            f"(confidence: {d['confidence']:.0%})\n"
            f"  Reasoning: {d['reasoning']}\n\n"
        )

    return (
        "You are an AI financial analyst assistant. You have just completed a detailed "
        f"analysis of the following stocks: {', '.join(tickers)}.\n"
        f"Portfolio cash: ${cash:,.2f}.\n\n"
        "Below are the complete analysis results. Use this context to answer the user's "
        "questions accurately and specifically. Reference the actual data when answering. "
        "If asked about something not covered by the analysis, say so clearly.\n\n"
        "=== TRADING SIGNALS ===\n"
        f"{signals_text or 'No signals available.'}\n\n"
        "=== RISK ASSESSMENTS ===\n"
        f"{risk_text or 'No risk assessments available.'}\n\n"
        "=== TRADE DECISIONS ===\n"
        f"{decisions_text or 'No decisions available.'}\n\n"
        "Guidelines:\n"
        "- Be specific and cite the actual numbers from the analysis.\n"
        "- If the user asks 'what if' scenarios, reason based on the analysis framework "
        "but clarify you are hypothesizing rather than running a new analysis.\n"
        "- Be concise but thorough. Use bullet points for clarity when appropriate.\n"
        "- Do not provide investment advice. Frame everything as analysis output."
    )


@router.post("/chat")
async def chat_stream(request: ChatRequest):
    """SSE endpoint — streams a chat response token-by-token."""

    async def event_generator():
        system_prompt = _build_chat_system_prompt(request.analysis_context)

        messages = [SystemMessage(content=system_prompt)]
        for msg in request.history:
            if msg.role == "user":
                messages.append(HumanMessage(content=msg.content))
            else:
                messages.append(AIMessage(content=msg.content))
        messages.append(HumanMessage(content=request.message))

        llm = get_llm(temperature=0.2)
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_event_loop()

        def run_stream():
            try:
                for chunk in llm.stream(messages):
                    token = chunk.content if hasattr(chunk, "content") else str(chunk)
                    if token:
                        asyncio.run_coroutine_threadsafe(
                            queue.put({"event": "token", "data": json.dumps({"token": token})}),
                            loop,
                        )
                asyncio.run_coroutine_threadsafe(
                    queue.put({"event": "done", "data": json.dumps({"status": "complete"})}),
                    loop,
                )
            except Exception as exc:
                asyncio.run_coroutine_threadsafe(
                    queue.put({"event": "error", "data": json.dumps({"error": str(exc)})}),
                    loop,
                )

        loop.run_in_executor(_executor, run_stream)

        while True:
            item = await queue.get()
            yield item
            if item.get("event") in ("done", "error"):
                break

    return EventSourceResponse(event_generator())
