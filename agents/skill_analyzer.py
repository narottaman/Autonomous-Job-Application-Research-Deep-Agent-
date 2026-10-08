"""
agents/skill_analyzer.py
─────────────────────────
Honest skill gap analysis + apply-now recommendation.
No tools — pure structured generation.
"""

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage
from graph.state import AgentState, SkillAnalysis
from config.settings import settings
import json

_model = ChatAnthropic(
    model=settings.model_name,
    api_key=settings.anthropic_api_key,
    temperature=0,
    max_tokens=2048,
)

SYSTEM_PROMPT = """You are a technical recruiting expert. Perform honest skill gap analysis.

Rules:
- Be HONEST about gaps. Do not hide them or reframe them as present.
- Match score = (required skills present) / (total required skills). Required only, not nice-to-have.
- apply_now = true if match_score ≥ 0.60 AND no dealbreaker gaps.
  Dealbreaker = a skill the role CANNOT function without (e.g. "must have" language for core function).

Return ONLY JSON:
{
  "matched_skills": ["skill"],
  "gap_skills": ["missing"],
  "transferable_skills": ["resume skill → how it applies"],
  "match_score": 0.0,
  "apply_now": true,
  "apply_reasoning": "..."
}"""


def skill_analyzer_node(state: AgentState) -> dict:
    market_context = ""
    market_intel = state.get("market_intel")
    if market_intel:
        market_context = f"\nMarket baseline stack: {market_intel.get('common_stack', [])}"
        market_context += f"\nRare requirements in this JD: {market_intel.get('rare_requirements', [])}"

    response = _model.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"""Analyze fit:

JD:
{state["jd_text"][:3000]}

RESUME:
{state["resume_text"][:3000]}
{market_context}""")
    ])

    analysis = _parse(response.content)
    return {
        "skill_analysis": analysis,
        "last_completed_node": "skill_analyzer",
    }


def _parse(content: str) -> SkillAnalysis:
    try:
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            data = json.loads(content[start:end])
            return SkillAnalysis(
                matched_skills=data.get("matched_skills", []),
                gap_skills=data.get("gap_skills", []),
                transferable_skills=data.get("transferable_skills", []),
                match_score=float(data.get("match_score", 0.5)),
                apply_now=data.get("apply_now", True),
                apply_reasoning=data.get("apply_reasoning", ""),
            )
    except Exception:
        pass
    return SkillAnalysis(
        matched_skills=[], gap_skills=[], transferable_skills=[],
        match_score=0.0, apply_now=False, apply_reasoning="Parse error"
    )
