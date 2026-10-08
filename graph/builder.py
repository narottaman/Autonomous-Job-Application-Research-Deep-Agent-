"""
graph/builder.py
────────────────
LangGraph StateGraph — wires all 8 nodes with edges and conditional routing.

FULL TOPOLOGY:
  START
    ↓
  jd_parser           ← scrape + extract company name
    ↓
  deep_researcher     ← ReAct loop, 3-hop deep, MCP enrichment, Qdrant cache
    ↓
  market_intel        ← benchmark role vs similar roles in market
    ↓
  skill_analyzer      ← honest match%, gaps, apply-now recommendation
    ↓
  gap_strategist      ← per-gap action plan, quick wins, dealbreakers
    ↓
  drafter  ←──────── ← multi-format: cover letter, ATS bullets, LinkedIn DM, etc.
    ↓              ↑
  critic ──── score < 0.75 AND loops < max ──→ drafter (revision)
    ↓
  (approved OR max loops)
    ↓
  report_writer       ← deterministic assembly, no LLM
    ↓
  END

WHY THIS ORDER:
  market_intel runs BEFORE skill_analyzer so the analyzer can use
  market baseline to distinguish "required" from "nice-to-have".
  gap_strategist runs AFTER skill_analyzer so it has the gap list.
  drafter runs AFTER gap_strategist so it knows quick wins to mention.
"""

from langgraph.graph import StateGraph, START, END
from graph.state import AgentState
from agents.deep_researcher import deep_researcher_node
from agents.market_intel import market_intel_node
from agents.skill_analyzer import skill_analyzer_node
from agents.gap_strategist import gap_strategist_node
from agents.drafter import drafter_node
from agents.critic import critic_node
from agents.report_writer import report_writer_node
from checkpointer.redis_checkpointer import get_langgraph_checkpointer
from config.settings import settings


def _jd_parser_node(state: AgentState) -> dict:
    """
    First node: scrape JD URL if needed, extract company name.
    Lightweight — no LLM unless company name extraction is needed.
    """
    from tools.search_tools import scrape_job_description
    from langchain_anthropic import ChatAnthropic
    from langchain_core.messages import HumanMessage

    jd_text = state.get("jd_text", "")
    if not jd_text and state.get("jd_url"):
        result = scrape_job_description.invoke({"url": state["jd_url"]})
        jd_text = result.get("text", "")

    company_name = state.get("company_name", "")
    if not company_name and jd_text:
        model = ChatAnthropic(
            model=settings.fast_model,   # use cheap model for simple extraction
            api_key=settings.anthropic_api_key,
            temperature=0, max_tokens=30,
        )
        resp = model.invoke([HumanMessage(
            content=f"Extract only the company name. One word or phrase, nothing else.\n\n{jd_text[:400]}"
        )])
        company_name = resp.content.strip()

    return {
        "jd_text": jd_text,
        "company_name": company_name,
        "research_loop_count": 0,
        "draft_loop_count": 0,
        "context_budget_used": 0,
        "context_summary": "",
        "last_completed_node": "jd_parser",
    }


def _should_revise_draft(state: AgentState) -> str:
    """
    Conditional edge after critic.
    Returns next node name based on critic score and loop count.
    """
    critic = state.get("critic_feedback", {})
    loops = state.get("draft_loop_count", 0)

    if critic.get("approved"):
        return "report_writer"
    if loops >= settings.max_draft_loops:
        return "report_writer"   # safety exit — never infinite loop
    return "drafter"


def build_graph():
    """Build and compile the full StateGraph with checkpointing."""
    graph = StateGraph(AgentState)

    # ── Register all nodes ────────────────────────────────────────────────────
    graph.add_node("jd_parser", _jd_parser_node)
    graph.add_node("deep_researcher", deep_researcher_node)
    graph.add_node("market_intel", market_intel_node)
    graph.add_node("skill_analyzer", skill_analyzer_node)
    graph.add_node("gap_strategist", gap_strategist_node)
    graph.add_node("drafter", drafter_node)
    graph.add_node("critic", critic_node)
    graph.add_node("report_writer", report_writer_node)

    # ── Unconditional edges ───────────────────────────────────────────────────
    graph.add_edge(START, "jd_parser")
    graph.add_edge("jd_parser", "deep_researcher")
    graph.add_edge("deep_researcher", "market_intel")
    graph.add_edge("market_intel", "skill_analyzer")
    graph.add_edge("skill_analyzer", "gap_strategist")
    graph.add_edge("gap_strategist", "drafter")
    graph.add_edge("drafter", "critic")

    # ── Conditional edge: critic gates the loop ───────────────────────────────
    graph.add_conditional_edges(
        "critic",
        _should_revise_draft,
        {"drafter": "drafter", "report_writer": "report_writer"}
    )
    graph.add_edge("report_writer", END)

    # ── Compile with Redis checkpointer ──────────────────────────────────────
    # thread_id in config={"configurable": {"thread_id": "..."}} enables resume
    checkpointer = get_langgraph_checkpointer()
    return graph.compile(checkpointer=checkpointer)


compiled_graph = build_graph()
