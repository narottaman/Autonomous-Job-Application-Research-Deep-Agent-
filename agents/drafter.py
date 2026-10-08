"""
agents/drafter.py
──────────────────
Multi-format Drafter: uses context_engine to receive shaped context,
produces 6 output formats not 1.

CONTEXT ENGINEERING HERE:
  Instead of dumping all research into the prompt (bad),
  build_drafter_context() shapes exactly what this agent needs,
  ordered by relevance, within a token budget.
  This is the "Lost in the Middle" fix in practice.

6 OUTPUT FORMATS:
  1. Cover letter (≤250 words, 3 paras, specific + grounded)
  2. "Why us?" answer (2-3 sentences, cites research)
  3. Key achievement bullets (reframed for this role)
  4. ATS resume bullets (keyword-dense, matches JD terminology)
  5. LinkedIn DM to hiring manager (casual, specific, short)
  6. Follow-up email (1-week post-application)
"""

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage
from graph.state import AgentState, DraftOutputs
from summarization.context_engine import build_drafter_context
from config.settings import settings
import json

_model = ChatAnthropic(
    model=settings.model_name,
    api_key=settings.anthropic_api_key,
    temperature=0.3,
    max_tokens=4096,
)

SYSTEM_PROMPT = """You are an expert job application writer for CS graduate students
targeting U.S. tech roles. You write specific, grounded, honest materials.

RULES (non-negotiable):
1. Every claim traces to the resume or research provided — no invented facts
2. Never claim gap skills (explicitly listed — DO NOT MENTION THEM)
3. Cover letter: exactly 3 paragraphs, ≤250 words
   Para 1: Why THIS company (cite a specific product/news/tech from research)
   Para 2: Why you're a fit (match 2-3 specific skills to 2-3 JD requirements)
   Para 3: Forward-looking close
4. ATS bullets: use EXACT keywords from the JD, not synonyms
5. LinkedIn DM: ≤100 words, casual tone, specific opener (not "I saw your job posting")
6. If gap_strategy has quick_wins, mention 1-2 as "currently deepening" — honest confidence

Return ONLY this JSON (no other text):
{
  "cover_letter": "...",
  "why_us_answer": "...",
  "key_achievement_bullets": ["..."],
  "ats_resume_bullets": ["..."],
  "linkedin_dm": "...",
  "referral_ask": "...",
  "followup_email": "...",
  "custom_questions": {}
}"""


def drafter_node(state: AgentState) -> dict:
    """LangGraph node: multi-format drafter with context engineering."""
    research = state.get("research", {})
    skill_analysis = state.get("skill_analysis", {})
    gap_strategy = state.get("gap_strategy", {})
    market_intel = state.get("market_intel", {})
    critic = state.get("critic_feedback")

    # ── Context engineering ───────────────────────────────────────────────────
    # Build shaped context — NOT raw research dump
    shaped_context = build_drafter_context(
        research_summary=state.get("context_summary", ""),
        skill_analysis=skill_analysis,
        gap_strategy=gap_strategy,
        market_intel=market_intel,
        jd_text=state["jd_text"],
        resume_text=state["resume_text"],
    )

    # ── Critic feedback on revision passes ────────────────────────────────────
    critic_context = ""
    if critic and not critic.get("approved") and state.get("draft_loop_count", 0) > 0:
        critic_context = f"""
=== CRITIC FEEDBACK — FIX THESE SPECIFICALLY ===
Score was: {critic.get("quality_score", 0):.0%} (need {settings.min_quality_score:.0%})
Issues:
{chr(10).join(f"- {i}" for i in critic.get("issues", []))}
Suggestions:
{chr(10).join(f"- {s}" for s in critic.get("suggestions", []))}
"""

    # ── Gap strategy context ──────────────────────────────────────────────────
    gap_context = ""
    if gap_strategy:
        quick_wins = gap_strategy.get("quick_wins", [])
        dealbreakers = gap_strategy.get("dealbreakers", [])
        apply_rec = "APPLY NOW" if skill_analysis.get("apply_now", True) else "PREP FIRST"
        gap_context = f"""
=== GAP STRATEGY ===
Recommendation: {apply_rec} — {skill_analysis.get("apply_reasoning", "")}
Quick wins (can mention as learning): {quick_wins}
DEALBREAKERS — never claim these: {dealbreakers}
"""

    human_msg = f"""Write all application materials for this candidate.

{shaped_context}
{gap_context}
{critic_context}

Write all 6 formats now. Be specific. Ground every claim."""

    response = _model.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=human_msg),
    ])

    draft = _parse_draft(response.content)
    return {
        "draft": draft,
        "draft_loop_count": state.get("draft_loop_count", 0) + 1,
        "last_completed_node": "drafter",
    }


def _parse_draft(content: str) -> DraftOutputs:
    try:
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            data = json.loads(content[start:end])
            return DraftOutputs(
                cover_letter=data.get("cover_letter", ""),
                why_us_answer=data.get("why_us_answer", ""),
                key_achievement_bullets=data.get("key_achievement_bullets", []),
                ats_resume_bullets=data.get("ats_resume_bullets", []),
                linkedin_dm=data.get("linkedin_dm", ""),
                referral_ask=data.get("referral_ask", ""),
                followup_email=data.get("followup_email", ""),
                custom_questions=data.get("custom_questions", {}),
            )
    except Exception:
        pass
    return DraftOutputs(
        cover_letter=content[:2000], why_us_answer="",
        key_achievement_bullets=[], ats_resume_bullets=[],
        linkedin_dm="", referral_ask="", followup_email="",
        custom_questions={}
    )
