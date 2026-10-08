"""
agents/critic.py
─────────────────
Critic: scores all 6 output formats on a 5-dim rubric.
Gates the drafter loop — approved only when avg ≥ 0.75.
"""

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage
from graph.state import AgentState, CriticFeedback
from config.settings import settings
import json

_model = ChatAnthropic(
    model=settings.critic_model,
    api_key=settings.anthropic_api_key,
    temperature=0,
    max_tokens=2048,
)

SYSTEM_PROMPT = f"""You are a harsh but fair job application critic.
Evaluate all application materials on 5 dimensions (0.0–1.0 each).
Average must reach {settings.min_quality_score} to approve.

DIMENSIONS:
1. specificity: cover letter cites real company details (product, news, tech) — not "innovative company"
2. grounding: every factual claim traceable to resume or research — no invented metrics
3. honesty: gap skills completely absent — not hinted at, not reframed, not claimed
4. format: cover letter ≤250 words, 3 paragraphs, bullets start with action verb + metric
5. impact: makes compelling case for THIS candidate for THIS role — not generic

Return ONLY JSON:
{{
  "scores": {{"specificity": 0.0, "grounding": 0.0, "honesty": 0.0, "format": 0.0, "impact": 0.0}},
  "quality_score": 0.0,
  "issues": ["specific problem"],
  "suggestions": ["actionable fix"],
  "approved": false
}}"""


def critic_node(state: AgentState) -> dict:
    draft = state.get("draft", {})
    research = state.get("research", {})
    skill_analysis = state.get("skill_analysis", {})
    gap_strategy = state.get("gap_strategy", {})

    human_msg = f"""Evaluate these application materials.

GROUND TRUTH (what materials must be grounded in):
Research summary: {state.get("context_summary", "")[:1500]}
Matched skills: {skill_analysis.get("matched_skills", [])}
GAP SKILLS (must NOT appear): {skill_analysis.get("gap_skills", [])}
Dealbreakers: {gap_strategy.get("dealbreakers", [])}

MATERIALS TO EVALUATE:

COVER LETTER:
{draft.get("cover_letter", "")}

WHY US:
{draft.get("why_us_answer", "")}

KEY BULLETS:
{chr(10).join(draft.get("key_achievement_bullets", []))}

ATS BULLETS:
{chr(10).join(draft.get("ats_resume_bullets", []))}

LINKEDIN DM:
{draft.get("linkedin_dm", "")}

JD (for specificity check):
{state["jd_text"][:1000]}

Be strict. Name the exact sentence that has each issue."""

    response = _model.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=human_msg),
    ])

    feedback = _parse(response.content)
    return {
        "critic_feedback": feedback,
        "last_completed_node": "critic",
    }


def _parse(content: str) -> CriticFeedback:
    try:
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            data = json.loads(content[start:end])
            scores = data.get("scores", {})
            avg = sum(scores.values()) / len(scores) if scores else 0.5
            return CriticFeedback(
                quality_score=float(data.get("quality_score", avg)),
                issues=data.get("issues", []),
                suggestions=data.get("suggestions", []),
                approved=data.get("approved", False),
                scores=scores,
            )
    except Exception:
        pass
    return CriticFeedback(
        quality_score=0.5, issues=["Parse error"], suggestions=["Retry"],
        approved=False, scores={}
    )
