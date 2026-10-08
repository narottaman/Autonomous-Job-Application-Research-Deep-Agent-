# 🤖 Autonomous Job Application Research Agent

> Most CS graduate students send 100+ applications and hear back from fewer than 5%.
> Not because they're unqualified — because they apply with generic materials,
> can't tell which roles are worth applying to, and have no idea what to do about skill gaps.
> This system solves all three, without a human in the loop.

**Stack:** LangGraph · Claude Sonnet 4.6 · Tavily · Qdrant · Redis · FastAPI · MCP · LangSmith

---

## The Real Problem

International CS graduate students face a brutal reality when job hunting in the U.S.:

- **Each quality application takes 3–5 hours** — researching the company, tailoring resume bullets, writing a specific cover letter, answering custom questions. So students either apply at scale with generic materials (ATS filters 80% of them) or apply selectively to a handful of roles (low volume, high variance).
- **Skill gap decisions are made blind** — you see a role you want but you're missing 3 skills. Are they dealbreakers or nice-to-haves? Should you apply now or spend 2 weeks prepping? Students guess. Most skip roles they could have gotten.
- **Cover letters are generic** — "I'm passionate about your innovative mission" means nothing. The recruiter has read it 200 times this week. Writing something specific requires 2 hours of research most students don't have.
- **No feedback loop** — after 3 months of applications you have no idea if it's your targeting, your resume, your cover letter, or just bad luck. You're optimizing blind.

**This system solves the three decisions students make blind:**

| Decision | Without this system | With this system |
|---|---|---|
| Should I apply? | Gut feeling | Honest match score + apply-now recommendation with reasoning |
| What do I do about gaps? | Ignore or panic-learn randomly | Per-gap action plan: specific resource, time estimate, priority |
| How do I write something specific? | Generic praise | Cover letter grounded in actual company research: real news, real tech, real culture signals |

---

## What It Does

Give it two things: a job posting URL and your resume.

It runs **8 specialized agents in sequence**, each with a single job:

1. **JD Parser** — scrapes the job posting (handles JS-rendered pages like Greenhouse, Lever)
2. **Deep Researcher** — autonomously follows links 3 hops deep, cross-validates claims across sources, detects contradictions, enriches with LinkedIn/Glassdoor/GitHub via MCP
3. **Market Intelligence** — benchmarks this role against 3–5 similar roles: what's the baseline stack, what's rare/unique about this JD, what does it signal about the company's stage
4. **Skill Analyzer** — honest match score, gap classification, apply-now recommendation
5. **Gap Strategist** — for each gap: a specific learning resource URL, time estimate in days, priority (DEALBREAKER / MAJOR / MINOR), and how to frame it honestly in the cover letter
6. **Drafter** — writes 6 output formats using context-engineered input (not raw research dump)
7. **Critic** — scores on a 5-dimension rubric, loops back to drafter if below 0.75
8. **Report Writer** — deterministic assembly of the final report (no LLM needed here)

**Final output:** one Markdown report with 12 sections — company research, market context, skill gap table, gap action plan, cover letter, why-us answer, tailored bullets, ATS-optimized bullets, LinkedIn DM, referral ask email, follow-up email, quality assessment.

---

## Architecture

```
User: job URL + resume
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│  API GATEWAY (gateway/router.py)                                │
│  Versioned routes /v1/ · Circuit breaker · Metrics · Health     │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│  MIDDLEWARE STACK (middleware/stack.py)  — runs on every request │
│  RequestID → StructuredLog → Auth(JWT) → CostGuard → ResultCache│
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│  LANGGRAPH StateGraph (graph/builder.py)                        │
│                                                                  │
│  JD Parser → Deep Researcher → Market Intel → Skill Analyzer    │
│                    ↑MCP↑           ↑Tavily↑                     │
│                    ↑Qdrant cache↑                               │
│                                                                  │
│  → Gap Strategist → Drafter ←──────────────────────┐           │
│        ↑Tavily↑        ↑context_engine↑             │           │
│                           │                         │ loop      │
│                        Critic ── score < 0.75 ──────┘           │
│                           │                                      │
│                    score ≥ 0.75                                  │
│                           │                                      │
│                    Report Writer → Final Markdown Report         │
│                                                                  │
│  Checkpointed via Redis after every node                        │
│  Traced in LangSmith end-to-end                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Why each architectural decision was made

**LangGraph StateGraph (not LangChain, not CrewAI)**
The critic→drafter loop is a *cycle*. LangChain chains are linear — cycles are impossible. CrewAI abstracts the routing away, meaning you can't explain or control it precisely. LangGraph gives you explicit nodes, explicit edges, and conditional routing you can whiteboard and defend.

**8 agents (not 1 big prompt)**
Each agent requires fundamentally different reasoning. The researcher needs a ReAct loop with tool calls. The skill analyzer needs pure structured generation. The gap strategist needs tool calls to find learning resources. The critic needs a rubric-based evaluator. You cannot collapse these into one call without losing reliability on all of them.

**Redis for both middleware cache and checkpointing**
One infrastructure component doing two jobs: (1) result cache — identical runs return instantly without touching the graph, (2) LangGraph checkpoint — state saved after every node so interrupted runs resume from where they stopped.

**Context engineering layer (not raw research dump)**
Raw DeepResearcher output is 8,000–15,000 chars. Passing all of it to the drafter wastes tokens and hurts quality — the "Lost in the Middle" problem (Liu et al. 2023) shows LLMs underperform on information in the middle of long contexts. `build_drafter_context()` shapes what the drafter sees: role specifics first, gap constraints last, within a token budget.

**MCP with Tavily fallback**
MCP (Model Context Protocol) is Anthropic's open standard for connecting agents to external services. If LinkedIn/Glassdoor/GitHub MCP servers are configured, we use them for richer data. If not (default), Tavily searches silently take over. The system works out-of-the-box without any MCP setup.

**Circuit breaker in the gateway**
If the Anthropic API goes down, every in-flight request will hang for 30s then fail. A circuit breaker detects 5 consecutive failures and fast-fails subsequent requests for 60 seconds — protecting the system from cascade failures and giving users an immediate error instead of a 30-second wait.

---

## Project Structure

```
job-agent/
│
├── config/
│   └── settings.py              ← all env vars, model config, thresholds
│
├── graph/
│   ├── state.py                 ← AgentState TypedDict (shared memory across all nodes)
│   └── builder.py               ← StateGraph: 8 nodes, edges, conditional routing
│
├── agents/
│   ├── deep_researcher.py       ← ReAct loop, 3-hop deep, MCP enrichment, Qdrant cache
│   ├── market_intel.py          ← benchmark role vs market, salary, rare requirements
│   ├── skill_analyzer.py        ← honest match%, gaps, apply-now recommendation
│   ├── gap_strategist.py        ← per-gap action plan: resource + time + priority
│   ├── drafter.py               ← 6-format output using shaped context
│   ├── critic.py                ← 5-dim rubric scorer, gates the loop
│   └── report_writer.py         ← deterministic assembly (no LLM)
│
├── tools/
│   └── search_tools.py          ← @tool functions: search, scrape, salary, learning resources
│
├── memory/
│   └── vector_store.py          ← Qdrant: store + retrieve + outcome tracking
│
├── middleware/
│   └── stack.py                 ← RequestID, logging, JWT auth, cost guard, result cache
│
├── gateway/
│   └── router.py                ← versioned /v1/ routes, circuit breaker, metrics, health
│
├── checkpointer/
│   └── redis_checkpointer.py    ← saves state after each node, enables run resumption
│
├── summarization/
│   └── context_engine.py        ← compress research, build shaped context per agent
│
├── mcp/
│   └── client.py                ← MCP client for LinkedIn/Glassdoor/GitHub + Tavily fallback
│
├── api/
│   └── main.py                  ← FastAPI: all endpoints, SSE streaming, PDF upload
│
├── ui/
│   └── app.py                   ← Streamlit demo with live node progress
│
├── evals/
│   └── evaluator.py             ← LLM judge + RAGAS metrics
│
├── tests/
│   └── test_graph.py            ← pytest: routing, parsing, middleware, JWT — no LLM calls
│
├── run.py                       ← single entry point
├── docker-compose.yml           ← Redis + Qdrant
└── requirements.txt
```

---

## Jargon Explained (for students learning from this)

### Agent
An AI system where the **LLM decides what to do next** at runtime, not just responds to a single prompt. In a normal program you write `if condition → do X`. In an agent, the LLM reads the situation and decides whether to call a tool, loop back, ask for more info, or stop. The key property: **the decision is made by the model, not pre-programmed by you.**

### LangGraph / StateGraph
A Python library for building agents as **graphs** — boxes (nodes) connected by arrows (edges). The critical feature is cycle support: a drafter→critic→drafter loop is impossible in a linear chain, straightforward in a graph. Every node receives a shared `AgentState` TypedDict, writes its output to it, and returns the updated state. LangGraph merges partial updates automatically.

### ReAct Loop (Reasoning + Acting)
The pattern the DeepResearcher uses. The LLM:
1. **Reasons** — what do I know, what do I need?
2. **Acts** — calls a tool (web search, URL scrape)
3. **Observes** — reads the tool result
4. **Reasons again** — enough? or do I need another search?
Repeats until satisfied. The LLM decides the tool sequence — not you. That's what makes it genuinely agentic vs a hardcoded pipeline.

### Conditional Edge
A Python function in the graph that decides which node runs next, based on the current state. The critic→drafter conditional edge is the core of the self-correcting loop:
```python
def _should_revise_draft(state):
    if critic.approved:          return "report_writer"
    if draft_loops >= max_loops: return "report_writer"  # safety exit
    return "drafter"             # loop back
```

### Context Engineering
Deliberately shaping what each agent sees — not dumping all available information into the prompt. The DeepResearcher output can be 12,000+ chars. The drafter doesn't need all of it — it needs role specifics (to write about the actual job), recent news (to cite in the cover letter), and gap constraints (to know what not to claim). `build_drafter_context()` assembles exactly that, in relevance order, within a token budget.

### Checkpointing
Saving the full AgentState to Redis after every node completes. If the server restarts mid-run, the user passes the same `thread_id` and the graph resumes from the last completed node — not from scratch. Critical for long-running agents (30–90 seconds) in production.

### MCP (Model Context Protocol)
Anthropic's open standard for connecting LLM agents to external data sources. Instead of writing a custom LinkedIn API wrapper, a custom Glassdoor scraper, a custom GitHub client — MCP gives one standard interface: `call_tool("search_company", {...})`. If the MCP server isn't configured, our client falls back to Tavily searches silently.

### Middleware
Code that runs on **every request** before it reaches any endpoint logic. Written once, applies everywhere. Our 5 layers:
- **RequestID** — stamps UUID on every request for log correlation
- **StructuredLog** — JSON log: method, path, user_id, duration_ms, status
- **Auth** — JWT verification. Token carries user_id, no DB lookup needed.
- **CostGuard** — estimates token cost from input size. Rejects if over budget.
- **ResultCache** — if identical (jd_url + resume) run exists in Redis, returns it instantly.

### Circuit Breaker
A pattern that detects when an external service (Anthropic API) is failing and fast-fails subsequent requests instead of letting them hang. Three states: CLOSED (normal) → OPEN (fast-failing) → HALF_OPEN (testing recovery). Opens after 5 consecutive failures. Resets after 60 seconds. Prevents cascade failures where one flaky API takes down your whole system.

### JWT (JSON Web Token)
A stateless authentication token. Contains `user_id`, `issued_at`, `expiry`, signed with HMAC-SHA256. The server verifies the signature on every request — no database lookup needed. That's what makes it stateless and fast.

### Structured Generation
Asking the LLM to return JSON instead of prose, so your code can parse and use the output. Every agent in this system uses structured generation — the prompt specifies the exact JSON schema, the model returns only JSON, the code parses it into a TypedDict. This is how you make LLM output reliable enough to use programmatically.

### RAGAS
Retrieval Augmented Generation Assessment. Standardized eval metrics for LLM systems. We use it in the eval suite to measure output quality across runs. Key metrics: **faithfulness** (does the answer come from the retrieved context?), **answer relevancy** (does it address the question?), **context recall** (did retrieval find the right information?).

### LangSmith
Anthropic/LangChain's observability platform. Set `LANGCHAIN_TRACING_V2=true` and every LLM call, tool call, token count, and latency automatically shows up in a dashboard at smith.langchain.com. This is how you debug an agent — not print statements.

---

## Output: 12-Section Report

```
1.  🎯 Recommendation         ← APPLY NOW or PREP FIRST with reasoning
2.  🏢 Company Research       ← overview, confirmed tech stack, recent news,
                                 company health signals, contradictions found
3.  📊 Market Intelligence    ← salary range, market baseline, rare requirements
4.  🎯 Skill Gap Analysis     ← match table: ✅ matched / ⚠️ gaps / 🔄 transferable
5.  🗺️ Gap Action Plan        ← per-gap: resource URL + days to close + priority
6.  ✉️  Cover Letter          ← 3 paragraphs, ≤250 words, specific + grounded
7.  💬 "Why This Company?"    ← 2-3 sentences citing actual research findings
8.  📝 Tailored Bullets       ← 3 resume bullets reframed for this specific role
9.  🤖 ATS-Optimized Bullets  ← keyword-dense, exact JD terminology for ATS systems
10. 💼 LinkedIn DM            ← ≤100 words to hiring manager, not generic
11. 📮 Referral Ask Email     ← template if mutual connection exists
12. 📬 Follow-Up Email        ← 1-week post-application template
```

At the bottom: per-dimension critic scores (specificity / grounding / honesty / format / impact), overall quality score, how many draft iterations it took.

---

## Tech Stack

| Tool | Role | Why this, not the alternative |
|---|---|---|
| **LangGraph** | Agent orchestration | Supports cycles (critic loop). LangChain chains are linear — impossible without LangGraph. CrewAI hides routing logic. |
| **Claude Sonnet 4.6** | LLM backbone | Best tool-use reliability, 200k context, swappable via `settings.model_name` |
| **Claude Haiku** | Summarization + cheap tasks | Fast and cheap for JD company extraction and research compression. Saves ~40% cost. |
| **Tavily** | Web search | Built for LLM agents — returns structured JSON, not raw HTML. Free tier: 1000 searches/month. |
| **Qdrant** | Vector memory | Runs locally via Docker, free cloud tier. Stores research so repeat queries skip web search entirely. |
| **Redis** | Cache + checkpointer | Two jobs, one service: result cache (middleware) and LangGraph checkpoint (resumable runs). |
| **sentence-transformers** | Embeddings | Local, free, zero API cost per embedding. `all-MiniLM-L6-v2` is 80MB and fast enough. |
| **FastAPI** | API framework | Async-native, SSE streaming built-in, auto-generates /docs page, Pydantic validation. |
| **SlowAPI + PyJWT** | Rate limiting + auth | Per-user rate limiting and stateless JWT auth — standard production middleware combo. |
| **MCP client** | External data | Standard protocol for LinkedIn/Glassdoor/GitHub integration. Falls back to Tavily if not configured. |
| **LangSmith** | Observability | Traces every node, tool call, token count, latency. Free tier sufficient. |
| **RAGAS** | Eval metrics | Standardized LLM output quality metrics across runs. |
| **Streamlit** | Demo UI | Working portfolio demo in ~150 lines. Point is the agent, not the frontend. |
| **Docker Compose** | Infrastructure | One command starts Redis + Qdrant locally. |
| **pytest** | Tests | Pure logic tests — routing, parsing, JWT, context engineering. No LLM calls in tests. |

---

## Setup

### Step 1 — Start infrastructure

```bash
# Docker required
docker compose up -d

# Verify:
curl http://localhost:6333/health    # Qdrant: should return {"title":"qdrant",...}
redis-cli ping                       # Redis: should return PONG
```

### Step 2 — Install dependencies

```bash
python -m venv venv
source venv/bin/activate       # Mac/Linux
# venv\Scripts\activate        # Windows

pip install -r requirements.txt
playwright install chromium    # for JS-rendered job pages
```

### Step 3 — Get API keys (all free tier)

| Key | Where | Free tier |
|---|---|---|
| `ANTHROPIC_API_KEY` | [console.anthropic.com](https://console.anthropic.com) | $5 credit |
| `TAVILY_API_KEY` | [app.tavily.com](https://app.tavily.com) | 1000 searches/month |
| `LANGCHAIN_API_KEY` | [smith.langchain.com](https://smith.langchain.com) | Free |

### Step 4 — Configure

```bash
cp .env.example .env
# Fill in the three keys above. Leave MCP_ fields blank for now.
```

### Step 5 — Run

```bash
# Terminal mode (simplest — no server needed)
python run.py cli

# Full stack (recommended for demo)
python run.py both
# API docs:  http://localhost:8000/docs
# Streamlit: http://localhost:8501

# Run tests (no API keys needed)
python run.py test
```

---

## API Reference

```bash
# Get a token
POST /token
form: username=demo&password=demo
→ {"access_token": "eyJ..."}

# Run agent (blocking)
POST /v1/analyze
Authorization: Bearer <token>
{"jd_url": "https://...", "resume_text": "...", "company_name": "Anthropic"}

# Run agent (streaming SSE — shows node progress live)
POST /v1/analyze/stream

# Upload PDF resume
POST /v1/analyze/upload
form: jd_url=..., resume_file=<file>

# Resume interrupted run
POST /v1/analyze
{"thread_id": "cli-abc123de", ...}   # same thread_id resumes from last checkpoint

# Check memory cache
GET /v1/memory/Anthropic

# Mark outcome (learning loop)
POST /v1/outcome
{"company_name": "Anthropic", "quality_score": 0.84, "got_response": true}

# System
GET /v1/health      → {"status": "ok"}
GET /v1/ready       → {"ready": true, "checks": {...}}
GET /v1/metrics     → token cost, latency p95, error rate
GET /v1/circuit     → circuit breaker state
```

---

## Interview Talking Points

**"Why do you need 8 agents for this?"**
> Each agent requires fundamentally different reasoning. The deep researcher needs a ReAct loop with dynamic tool selection. The skill analyzer needs pure structured generation with no tools. The gap strategist needs to call a learning-resource search tool per gap and estimate time. The critic needs rubric-based evaluation with dimensional scoring. You cannot collapse these into one prompt without losing reliability on all of them — each would need to be the "good enough" version of every task.

**"Walk me through the critic→drafter loop."**
> After the drafter runs, the critic scores the output on 5 dimensions — specificity, grounding, honesty, format, impact — and averages them. That average goes into `AgentState.critic_feedback.quality_score`. The conditional edge function reads that score: if ≥ 0.75 or if `draft_loop_count` hit the maximum (safety exit), route to `report_writer`. Otherwise route back to `drafter`. On the second pass, the drafter receives the critic's specific issues and suggestions in its context, so it's fixing precise problems — not starting over.

**"What is context engineering and where do you use it?"**
> Context engineering is deliberately shaping what each agent sees instead of dumping everything into the prompt. After the deep researcher runs, its output can be 12,000+ chars. The "Lost in the Middle" paper shows LLMs underperform on information in the middle of long contexts. So `build_drafter_context()` assembles a shaped context: role specifics first (most important for writing about the actual job), recent news second (for cover letter specificity), gap constraints last (what not to claim). It stays within a char budget and uses the Haiku model to compress the research beforehand.

**"How does checkpointing work?"**
> LangGraph has a `compile(checkpointer=...)` parameter. We pass a `RedisSaver`. After every node completes, LangGraph saves the full `AgentState` to Redis under `checkpoint:{thread_id}:{node_name}`. If the run is interrupted — server restart, network drop, tab closed — the user passes the same `thread_id` on their next request. LangGraph finds the latest checkpoint and resumes from that node forward. The user doesn't restart from scratch.

**"What's your cost per run and how do you control it?"**
> Roughly $0.04–0.08 per run at current Sonnet pricing — about 10–15 LLM calls. I control cost three ways: (1) Qdrant cache — same company skips 4-6 tool calls and 2 LLM calls entirely. (2) Haiku for cheap tasks — JD extraction and research summarization use the fast model, saving ~40% on those calls. (3) CostGuard middleware — estimates token budget from input length before running and rejects inputs projected to exceed the per-run limit, preventing runaway spend on huge inputs.

**"How do you handle the Anthropic API going down?"**
> Circuit breaker in `gateway/router.py`. Tracks consecutive failures. After 5 failures it opens — subsequent requests get an immediate 503 instead of waiting 30 seconds for a timeout. After 60 seconds it goes half-open, allows one test request. If that succeeds, closes again. Without this, one flaky API period can queue hundreds of 30-second timeouts and take down the whole system.

**"How do you evaluate if the agent is improving?"**
> Two layers. The in-loop critic gives a quality score per run that's stored in the report. The offline eval suite in `evals/evaluator.py` runs a separate LLM judge on the same 5 dimensions across a 20-case benchmark set, scored independently from the critic. I track these in LangSmith across runs — if I change the drafter's system prompt, I run the benchmark and compare before/after. That's how I know grounding went from 0.72 to 0.91 after adding citation requirements to the drafter prompt.

---

## Learning Loop

The system gets better over time. After you apply, mark the outcome in the UI or via API:

```bash
POST /v1/outcome
{"company_name": "Anthropic", "quality_score": 0.84, "got_response": true}
```

This stores in Qdrant alongside the quality score. Over time you can correlate: what critic score threshold actually predicts getting a response? Does specificity matter more than grounding? This turns the system from a one-shot tool into a feedback loop.

---

## License

MIT — use it, learn from it, build on it.

---

*Built by [Narottaman Gangadaran](https://linkedin.com/in/narottaman-gangadaran)*
*M.S. Computer Science · Arizona State University · 4.0 GPA*
