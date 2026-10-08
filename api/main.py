"""
api/main.py
────────────
FastAPI app with full middleware stack, gateway router, and all endpoints.

ARCHITECTURE:
  request → middleware stack → gateway router → agent graph → response

MIDDLEWARE ORDER (outermost first):
  RequestID → StructuredLog → Auth → CostGuard → ResultCache → endpoint

ENDPOINTS:
  POST /token              ← get JWT (public, no auth)
  POST /v1/analyze         ← blocking full run
  POST /v1/analyze/stream  ← SSE streaming (shows node progress live)
  POST /v1/analyze/upload  ← PDF resume upload
  POST /v1/outcome         ← mark application outcome (learning loop)
  GET  /v1/memory/{name}   ← check Qdrant cache
  GET  /v1/memory/search/{q} ← semantic search across research
  GET  /v1/health          ← health check
  GET  /v1/ready           ← readiness probe
  GET  /v1/metrics         ← token cost + latency stats
  GET  /v1/circuit         ← circuit breaker state
"""

import asyncio, json, uuid, time
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from typing import Optional
import PyPDF2, io

from graph.builder import compiled_graph
from graph.state import AgentState
from memory.vector_store import retrieve_research, semantic_search, store_run_outcome
from middleware.stack import register_middleware, save_to_cache, create_token
from gateway.router import v1_router, metrics, anthropic_circuit
from config.settings import settings

app = FastAPI(
    title="Autonomous Job Application Research Agent",
    description="Deep multi-agent system: researcher → market intel → skill analyzer → gap strategist → drafter → critic",
    version="2.0.0",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── All middleware (auth, rate limit, cost guard, cache, logging) ─────────────
register_middleware(app)

# ── Gateway router (versioned routes + health + metrics) ─────────────────────
app.include_router(v1_router)


# ── Request/Response models ───────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    jd_url: Optional[str] = None
    jd_text: Optional[str] = None
    resume_text: str
    company_name: Optional[str] = None
    thread_id: Optional[str] = None    # for resuming checkpointed runs


class AnalyzeResponse(BaseModel):
    company_name: str
    apply_recommendation: str
    match_score: float
    quality_score: float
    draft_iterations: int
    final_report: str
    thread_id: str


class OutcomeRequest(BaseModel):
    company_name: str
    quality_score: float
    got_response: bool


# ── Auth endpoint (public) ────────────────────────────────────────────────────

@app.post("/token")
async def get_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Issue a JWT token. In production: verify against user DB.
    For portfolio: any username/password gets a token.
    """
    token = create_token(user_id=form_data.username)
    return {"access_token": token, "token_type": "bearer"}


# ── Helper ────────────────────────────────────────────────────────────────────

def _build_initial_state(
    jd_url: str, jd_text: str, resume_text: str,
    company_name: str, user_id: str, thread_id: str
) -> AgentState:
    return AgentState(
        jd_url=jd_url or "",
        jd_text=jd_text or "",
        resume_text=resume_text,
        company_name=company_name or "",
        user_id=user_id,
        messages=[],
        research=None, market_intel=None,
        skill_analysis=None, gap_strategy=None,
        draft=None, critic_feedback=None,
        research_loop_count=0, draft_loop_count=0,
        context_budget_used=0, context_summary="",
        checkpoint_id=thread_id,
        last_completed_node="",
        final_report=None, error=None,
    )


# ── Main analyze endpoint (blocking) ─────────────────────────────────────────

@app.post("/v1/analyze", response_model=AnalyzeResponse)
async def analyze(request: Request, body: AnalyzeRequest):
    if not body.jd_url and not body.jd_text:
        raise HTTPException(400, "Provide jd_url or jd_text")

    user_id = getattr(request.state, "user_id", "anonymous")
    thread_id = body.thread_id or f"{user_id}-{uuid.uuid4().hex[:8]}"

    # Circuit breaker check
    if not anthropic_circuit.can_attempt():
        raise HTTPException(503, "LLM service temporarily unavailable. Try again in 60s.")

    start = time.time()
    state = _build_initial_state(
        body.jd_url, body.jd_text, body.resume_text,
        body.company_name or "", user_id, thread_id
    )

    try:
        result = await asyncio.to_thread(
            compiled_graph.invoke, state,
            {"configurable": {"thread_id": thread_id}}
        )
        anthropic_circuit.record_success()
    except Exception as e:
        anthropic_circuit.record_failure()
        raise HTTPException(500, f"Agent error: {str(e)}")

    latency_ms = int((time.time() - start) * 1000)
    skill = result.get("skill_analysis") or {}
    critic = result.get("critic_feedback") or {}

    # Record metrics
    metrics.record(
        tokens=result.get("context_budget_used", 0),
        latency_ms=latency_ms,
        cost=result.get("context_budget_used", 0) * 0.000003,
    )

    response_data = {
        "company_name": result.get("company_name", "Unknown"),
        "apply_recommendation": "APPLY NOW" if skill.get("apply_now") else "PREP FIRST",
        "match_score": skill.get("match_score", 0.0),
        "quality_score": critic.get("quality_score", 0.0),
        "draft_iterations": result.get("draft_loop_count", 0),
        "final_report": result.get("final_report", ""),
        "thread_id": thread_id,
    }

    # Cache result for identical future runs
    cache_key = getattr(request.state, "cache_key", None)
    if cache_key:
        await save_to_cache(cache_key, response_data)

    return AnalyzeResponse(**response_data)


# ── Streaming endpoint ─────────────────────────────────────────────────────────

@app.post("/v1/analyze/stream")
async def analyze_stream(request: Request, body: AnalyzeRequest):
    """SSE streaming: pushes event after each node completes."""
    if not body.jd_url and not body.jd_text:
        raise HTTPException(400, "Provide jd_url or jd_text")

    user_id = getattr(request.state, "user_id", "anonymous")
    thread_id = body.thread_id or f"{user_id}-{uuid.uuid4().hex[:8]}"
    state = _build_initial_state(
        body.jd_url, body.jd_text, body.resume_text,
        body.company_name or "", user_id, thread_id
    )

    NODE_LABELS = {
        "jd_parser": "📋 Parsing job description",
        "deep_researcher": "🔍 Deep research (3-hop + MCP)",
        "market_intel": "📊 Benchmarking vs market",
        "skill_analyzer": "🎯 Analyzing skill fit",
        "gap_strategist": "🗺️ Building gap action plan",
        "drafter": "✍️ Drafting application materials",
        "critic": "🔎 Critic evaluating quality",
        "report_writer": "📄 Assembling report",
    }

    async def event_generator():
        yield f"data: {json.dumps({'node': 'start', 'thread_id': thread_id})}\n\n"
        for node_name, output in compiled_graph.stream(
            state,
            {"configurable": {"thread_id": thread_id}},
            stream_mode="updates",
        ):
            event = {
                "node": node_name,
                "label": NODE_LABELS.get(node_name, node_name),
                "status": "completed",
            }
            if node_name == "skill_analyzer" and output.get("skill_analysis"):
                sa = output["skill_analysis"]
                event["match_score"] = sa.get("match_score", 0)
                event["apply_now"] = sa.get("apply_now", True)
            elif node_name == "gap_strategist" and output.get("gap_strategy"):
                gs = output["gap_strategy"]
                event["quick_wins"] = gs.get("quick_wins", [])
                event["dealbreakers"] = gs.get("dealbreakers", [])
            elif node_name == "critic" and output.get("critic_feedback"):
                cf = output["critic_feedback"]
                event["quality_score"] = cf.get("quality_score", 0)
                event["approved"] = cf.get("approved", False)
            elif node_name == "report_writer" and output.get("final_report"):
                event["final_report"] = output["final_report"]
            yield f"data: {json.dumps(event)}\n\n"

        yield f"data: {json.dumps({'node': 'done', 'thread_id': thread_id})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ── PDF upload endpoint ────────────────────────────────────────────────────────

@app.post("/v1/analyze/upload")
async def analyze_upload(
    request: Request,
    jd_url: str = Form(""),
    company_name: str = Form(""),
    resume_file: UploadFile = File(...),
):
    """Accept PDF or TXT resume via file upload."""
    file_bytes = await resume_file.read()
    if resume_file.filename.endswith(".pdf"):
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        resume_text = "\n".join(p.extract_text() or "" for p in reader.pages)
    else:
        resume_text = file_bytes.decode("utf-8", errors="ignore")

    body = AnalyzeRequest(jd_url=jd_url, resume_text=resume_text, company_name=company_name)
    return await analyze(request, body)


# ── Outcome feedback (learning loop) ─────────────────────────────────────────

@app.post("/v1/outcome")
async def record_outcome(body: OutcomeRequest):
    """
    User marks whether they got a response after applying.
    Stored in Qdrant to correlate critic scores with real outcomes over time.
    """
    store_run_outcome(body.company_name, body.quality_score, body.got_response)
    return {"status": "recorded"}


# ── Memory endpoints ──────────────────────────────────────────────────────────

@app.get("/v1/memory/{company_name}")
async def check_memory(company_name: str):
    cached = retrieve_research(company_name)
    return {"cached": bool(cached), "data": cached}


@app.get("/v1/memory/search/{query}")
async def search_memory(query: str):
    return {"results": semantic_search(query, top_k=3)}
