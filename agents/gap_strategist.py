"""
agents/gap_strategist.py
─────────────────────────
GapStrategist: turns "you're missing Go" into "here's exactly what to do in 2 weeks."

WHY THIS IS THE MOST VALUABLE NEW AGENT:
  The original skill analyzer just lists gaps. Useless.
  GapStrategist answers the actual question students have:
  "Should I apply now or spend N weeks prepping first?"
  "What exactly do I do to close this gap?"

  This is also the biggest differentiator in the cover letter:
  Instead of hiding the gap, you can say:
  "I'm deepening my Kubernetes experience via [specific course] and expect
   to be production-ready in 3 weeks — eager to put it to use here."
  That's honest, confident, and shows self-awareness. Recruiters respect it.

OUTPUT:
  - Per-gap: resource (specific URL/course), time estimate, priority
  - Quick wins: gaps closeable in < 3 days (mention these as "currently learning")
  - Dealbreakers: gaps that will likely filter you out (don't apply blindly)
  - apply_now recommendation with reasoning
"""

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from graph.state import AgentState, GapStrategy
from tools.search_tools import search_learning_resource
from config.settings import settings
import json

_model = ChatAnthropic(
    model=settings.model_name,
    api_key=settings.anthropic_api_key,
    temperature=0,
    max_tokens=2048,
).bind_tools([search_learning_resource])

SYSTEM_PROMPT = """You are a career strategist helping a student close skill gaps fast.

For each gap skill, find:
1. The FASTEST learning path (a specific free course, project, or official doc)
2. Realistic time to basic proficiency (not mastery — just enough to be honest in an interview)
3. Whether it's a dealbreaker for this specific role

Classify gaps:
- DEALBREAKER: role cannot function without it (e.g. "must have Go" for a Go backend role)
- MAJOR: required but learnable in < 1 month
- MINOR: "nice to have" — apply now, mention it honestly

Quick wins = gaps closeable in < 3 days (a weekend of focused study).

Return ONLY this JSON:
{
  "gap_action_plan": [
    {
      "skill": "...",
      "resource": "specific URL or course name",
      "time_estimate_days": 7,
      "priority": "DEALBREAKER|MAJOR|MINOR",
      "honest_framing": "how to mention this in cover letter"
    }
  ],
  "total_prep_days": 14,
  "quick_wins": ["skill1", "skill2"],
  "dealbreakers": ["skill3"],
  "apply_recommendation": "now|after_prep",
  "apply_reasoning": "..."
}"""


def gap_strategist_node(state: AgentState) -> dict:
    """LangGraph node for gap strategy."""
    skill_analysis = state.get("skill_analysis", {})
    gap_skills = skill_analysis.get("gap_skills", [])

    if not gap_skills:
        return {
            "gap_strategy": GapStrategy(
                gap_action_plan=[], total_prep_days=0,
                quick_wins=[], dealbreakers=[]
            ),
            "last_completed_node": "gap_strategist",
        }

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"""Build action plan for these gaps:
Gap skills: {gap_skills}
Role: {state["company_name"]} — {state["jd_text"][:500]}
Match score: {skill_analysis.get("match_score", 0):.0%}
Market rare requirements: {state.get("market_intel", {}).get("rare_requirements", [])}

Use search_learning_resource for each gap to find the best resource.""")
    ]

    max_iter = 8
    for _ in range(max_iter):
        response = _model.invoke(messages)
        messages.append(response)
        if response.tool_calls:
            for tc in response.tool_calls:
                result = _execute(tc)
                messages.append(ToolMessage(
                    content=json.dumps(result)[:1500],
                    tool_call_id=tc["id"],
                ))
        else:
            strategy = _parse(response.content, skill_analysis)
            return {
                "gap_strategy": strategy,
                "skill_analysis": {
                    **skill_analysis,
                    "apply_now": strategy.get("apply_now", True),
                    "apply_reasoning": strategy.get("apply_reasoning", ""),
                },
                "last_completed_node": "gap_strategist",
            }

    return {
        "gap_strategy": GapStrategy(
            gap_action_plan=[], total_prep_days=0,
            quick_wins=[], dealbreakers=[]
        ),
        "last_completed_node": "gap_strategist",
    }


def _execute(tool_call: dict) -> dict:
    from tools.search_tools import search_learning_resource
    try:
        return search_learning_resource.invoke(tool_call["args"])
    except Exception as e:
        return {"error": str(e)}


def _parse(content: str, skill_analysis: dict) -> dict:
    try:
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            data = json.loads(content[start:end])
            # Build GapStrategy TypedDict
            return {
                "gap_action_plan": data.get("gap_action_plan", []),
                "total_prep_days": data.get("total_prep_days", 0),
                "quick_wins": data.get("quick_wins", []),
                "dealbreakers": data.get("dealbreakers", []),
                "apply_now": data.get("apply_recommendation", "now") == "now",
                "apply_reasoning": data.get("apply_reasoning", ""),
            }
    except Exception:
        pass
    return {"gap_action_plan": [], "total_prep_days": 0, "quick_wins": [], "dealbreakers": []}
