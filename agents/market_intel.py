"""
agents/market_intel.py
───────────────────────
MarketIntelligence agent: benchmarks this role against similar roles in the market.

WHY THIS EXISTS:
  Without market context: "they require Kubernetes" → you either know it or you don't.
  With market context: "80% of similar ML roles DON'T require Kubernetes — this company
  is building infra from scratch, they need someone adaptable, not just ML skills."
  That insight changes what you write in the cover letter entirely.

WHAT IT FINDS:
  1. 3-5 similar roles at comparable companies (searches Tavily)
  2. Common stack across those roles (what the market baseline is)
  3. Rare requirements unique to this JD (what makes this role distinctive)
  4. Salary range from levels.fyi / glassdoor
  5. A narrative insight: what the role signals about the company's stage/needs
"""

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from graph.state import AgentState, MarketIntelligence
from tools.search_tools import search_similar_roles, search_salary_data
from config.settings import settings
import json

_model = ChatAnthropic(
    model=settings.model_name,
    api_key=settings.anthropic_api_key,
    temperature=0,
    max_tokens=2048,
).bind_tools([search_similar_roles, search_salary_data])

SYSTEM_PROMPT = """You are a job market analyst. Given a job description, find:
1. What similar roles at comparable companies require
2. What makes THIS specific JD different from the market baseline
3. Salary range for this role
4. What the unique requirements signal about the company's stage/needs

Use search_similar_roles to find 3-4 comparable roles.
Use search_salary_data for compensation range.

Return ONLY this JSON:
{
  "similar_roles": [{"company": "...", "title": "...", "key_requirements": ["..."]}],
  "market_salary_range": "$X - $Y",
  "common_stack": ["skills in 80%+ of similar roles"],
  "rare_requirements": ["things only this JD asks for"],
  "market_insight": "narrative: what this role signals"
}"""


def market_intel_node(state: AgentState) -> dict:
    """LangGraph node for market intelligence."""
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"""Analyze this role vs the market:
Company: {state["company_name"]}
JD excerpt: {state["jd_text"][:2000]}
Research context: {state.get("context_summary", "")[:500]}""")
    ]

    max_iter = 6
    for _ in range(max_iter):
        response = _model.invoke(messages)
        messages.append(response)
        if response.tool_calls:
            for tc in response.tool_calls:
                result = _execute(tc)
                messages.append(ToolMessage(
                    content=json.dumps(result)[:2000],
                    tool_call_id=tc["id"],
                ))
        else:
            intel = _parse(response.content)
            if intel:
                return {
                    "market_intel": intel,
                    "last_completed_node": "market_intel",
                }
            break

    return {
        "market_intel": MarketIntelligence(
            similar_roles=[], market_salary_range="N/A",
            common_stack=[], rare_requirements=[], market_insight=""
        ),
        "last_completed_node": "market_intel",
    }


def _execute(tool_call: dict) -> dict:
    from tools.search_tools import search_similar_roles, search_salary_data
    tool_map = {
        "search_similar_roles": search_similar_roles,
        "search_salary_data": search_salary_data,
    }
    fn = tool_map.get(tool_call["name"])
    if not fn:
        return {"error": f"Unknown: {tool_call['name']}"}
    try:
        return fn.invoke(tool_call["args"])
    except Exception as e:
        return {"error": str(e)}


def _parse(content: str) -> dict | None:
    try:
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(content[start:end])
    except Exception:
        pass
    return None
