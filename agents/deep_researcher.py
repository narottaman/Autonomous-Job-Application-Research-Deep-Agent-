"""
agents/deep_researcher.py
──────────────────────────
DeepResearcher: goes 3 hops deep instead of 4 flat searches.

WHY DEEP VS SHALLOW:
  Shallow researcher: search("Anthropic") → 4 results → done.
  Deep researcher:
    Hop 1: search("Anthropic") → finds engineering blog URL
    Hop 2: scrape engineering blog → finds specific tech decisions
    Hop 3: follow citation in blog → finds architecture paper
  Result: you cite "their Sept 2025 eng blog post on their Rust inference stack"
  instead of "innovative AI company building the future."

ALSO DOES:
  - Cross-source contradiction detection ("LinkedIn says 200 employees, Crunchbase says 50")
  - Company health signals (layoffs, runway signals, glassdoor trend)
  - MCP enrichment (LinkedIn, Glassdoor, GitHub if configured)
  - Summarizes output via context_engine before returning
"""

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from graph.state import AgentState, ResearchFindings
from tools.search_tools import (
    search_company_info, search_company_news,
    search_role_insights, scrape_job_description,
    scrape_engineering_blog, scrape_url
)
from memory.vector_store import retrieve_research, store_research
from mcp.client import get_company_linkedin_data, get_glassdoor_signals, get_github_tech_signals
from summarization.context_engine import summarize_research, should_summarize
from config.settings import settings
import json, asyncio

_model = ChatAnthropic(
    model=settings.model_name,
    api_key=settings.anthropic_api_key,
    temperature=0,
    max_tokens=4096,
).bind_tools([
    search_company_info,
    search_company_news,
    search_role_insights,
    scrape_engineering_blog,
    scrape_url,
])

SYSTEM_PROMPT = """You are a deep company researcher for job applications. 
Your goal: build a SPECIFIC, EVIDENCE-BASED research brief about a company and role.

Generic output is worthless. "Innovative company" means nothing. 
You need: a specific product the candidate can reference, a recent news item with a date,
a real tech decision from their engineering blog, a culture signal from an actual employee quote.

Research strategy (follow this order):
1. search_company_info — get overview and find URLs to follow
2. scrape_engineering_blog — their eng blog reveals actual tech decisions  
3. search_company_news — last 90 days only
4. search_role_insights — what this role actually involves (Glassdoor, Blind)
5. scrape_url — follow any specific URL that looks high-signal

Stop when you have: company overview, confirmed tech stack (from blog not just JD),
3 recent news items with dates, 2+ culture signals, role specifics.

Flag contradictions: if two sources say different things about employee count,
funding, or tech stack — note it explicitly.

Return ONLY this JSON:
{
  "company_overview": "...",
  "tech_stack": [".."],
  "recent_news": ["date: item"],
  "culture_signals": [".."],
  "role_specifics": "...",
  "sources": ["url1"],
  "link_graph": {},
  "contradictions": ["source A says X, source B says Y"],
  "company_health": "signals about layoffs/runway/glassdoor trend"
}"""


def deep_researcher_node(state: AgentState) -> dict:
    """
    LangGraph node. Multi-hop ReAct loop + MCP enrichment + summarization.
    """
    # ── Cache check ───────────────────────────────────────────────────────────
    cached = retrieve_research(state["company_name"])
    if cached:
        summary = summarize_research(cached) if should_summarize(json.dumps(cached)) else json.dumps(cached)
        return {
            "research": cached,
            "context_summary": summary,
            "last_completed_node": "deep_researcher",
        }

    # ── ReAct loop ────────────────────────────────────────────────────────────
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"""Research company: {state["company_name"]}
Role context from JD: {state["jd_text"][:1500]}
Go deep. Follow links. Find specifics.""")
    ]

    max_iterations = 10
    iteration = 0
    research = None

    while iteration < max_iterations:
        response = _model.invoke(messages)
        messages.append(response)
        iteration += 1

        if response.tool_calls:
            for tool_call in response.tool_calls:
                result = _execute_tool(tool_call)
                messages.append(ToolMessage(
                    content=json.dumps(result)[:3000],
                    tool_call_id=tool_call["id"],
                ))
        else:
            research = _parse_research(response.content)
            break

    if not research:
        research = _fallback_research(state)

    # ── MCP enrichment (async tools called synchronously here) ────────────────
    research = _enrich_with_mcp(research, state["company_name"])

    # ── Store in Qdrant ───────────────────────────────────────────────────────
    store_research(state["company_name"], research)

    # ── Summarize for context engineering ─────────────────────────────────────
    research_json = json.dumps(research)
    summary = summarize_research(research) if should_summarize(research_json) else research_json

    return {
        "research": research,
        "context_summary": summary,
        "context_budget_used": len(research_json) // 4,
        "last_completed_node": "deep_researcher",
        "messages": messages,
    }


def _execute_tool(tool_call: dict) -> dict:
    tool_map = {
        "search_company_info": search_company_info,
        "search_company_news": search_company_news,
        "search_role_insights": search_role_insights,
        "scrape_engineering_blog": scrape_engineering_blog,
        "scrape_url": scrape_url,
    }
    fn = tool_map.get(tool_call["name"])
    if not fn:
        return {"error": f"Unknown tool: {tool_call['name']}"}
    try:
        return fn.invoke(tool_call["args"])
    except Exception as e:
        return {"error": str(e)}


def _enrich_with_mcp(research: dict, company_name: str) -> dict:
    """Add LinkedIn, Glassdoor, GitHub data if MCP is configured."""
    try:
        loop = asyncio.new_event_loop()
        linkedin = loop.run_until_complete(get_company_linkedin_data(company_name))
        glassdoor = loop.run_until_complete(get_glassdoor_signals(company_name))
        github = loop.run_until_complete(get_github_tech_signals(company_name))
        loop.close()

        if linkedin.get("data"):
            research["culture_signals"].append(f"LinkedIn: {str(linkedin['data'])[:200]}")
        if glassdoor.get("data"):
            research["company_health"] = (research.get("company_health", "") +
                f" | Glassdoor: {str(glassdoor['data'])[:200]}")
        if github.get("data"):
            research["tech_stack"].append(f"GitHub signals: {str(github['data'])[:200]}")
    except Exception:
        pass
    return research


def _parse_research(content: str) -> dict | None:
    try:
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(content[start:end])
    except Exception:
        pass
    return None


def _fallback_research(state: AgentState) -> dict:
    return ResearchFindings(
        company_overview="Research incomplete",
        tech_stack=[], recent_news=[], culture_signals=[],
        role_specifics=state["jd_text"][:300],
        sources=[], link_graph={}, contradictions=[],
        company_health="Unknown",
        research_summary=""
    )
