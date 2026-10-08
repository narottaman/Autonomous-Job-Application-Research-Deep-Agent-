"""
graph/state.py
──────────────
AgentState TypedDict — the single object flowing through every node.
Every new agent writes its output here. Nodes are loosely coupled
because they only share state, never call each other directly.
"""
from typing import TypedDict, Annotated, Optional
from langchain_core.messages import BaseMessage
import operator


class ResearchFindings(TypedDict):
    company_overview: str
    tech_stack: list[str]
    recent_news: list[str]
    culture_signals: list[str]
    role_specifics: str
    sources: list[str]
    # NEW: deep research fields
    link_graph: dict           # {url: [child_urls]} — tracks hop structure
    contradictions: list[str]  # cross-source conflicts flagged
    company_health: str        # signals: layoffs, runway, glassdoor trend
    research_summary: str      # summarized version for context engineering


class MarketIntelligence(TypedDict):
    """NEW: what the market looks like for this role type"""
    similar_roles: list[dict]       # [{company, title, key_requirements}]
    market_salary_range: str        # from levels.fyi / glassdoor
    common_stack: list[str]         # tech skills appearing in 80%+ of similar JDs
    rare_requirements: list[str]    # things this JD asks for that others don't
    market_insight: str             # narrative: what this role signals about the company


class SkillAnalysis(TypedDict):
    matched_skills: list[str]
    gap_skills: list[str]
    transferable_skills: list[str]
    match_score: float
    # NEW: apply recommendation
    apply_now: bool                 # True = apply now, False = prep first
    apply_reasoning: str            # why this recommendation


class GapStrategy(TypedDict):
    """NEW: action plan per gap skill — not just listing gaps"""
    gap_action_plan: list[dict]     # [{skill, resource, time_estimate_days, priority}]
    total_prep_days: int            # sum of priority gaps
    quick_wins: list[str]           # gaps closeable in < 3 days
    dealbreakers: list[str]         # gaps that will likely filter you out


class DraftOutputs(TypedDict):
    cover_letter: str
    why_us_answer: str
    key_achievement_bullets: list[str]
    # NEW: multi-format outputs
    ats_resume_bullets: list[str]   # keyword-dense bullets for ATS systems
    linkedin_dm: str                # outreach to hiring manager / recruiter
    referral_ask: str               # email template if mutual connection exists
    followup_email: str             # 1-week follow-up template
    custom_questions: dict[str, str]


class CriticFeedback(TypedDict):
    quality_score: float
    issues: list[str]
    suggestions: list[str]
    approved: bool
    scores: dict                    # per-dimension breakdown


class AgentState(TypedDict):
    # ── Input ────────────────────────────────────────────────────────────────
    jd_url: str
    jd_text: str
    resume_text: str
    company_name: str
    user_id: str                    # NEW: for per-user checkpointing + rate limiting

    # ── Message history ───────────────────────────────────────────────────────
    messages: Annotated[list[BaseMessage], operator.add]

    # ── Context engineering ───────────────────────────────────────────────────
    # WHY: research output can be 8k+ chars — too long to pass raw to every agent.
    # Context engineering = deliberately shaping what each agent sees.
    context_budget_used: int        # running token estimate
    context_summary: str            # compressed research for downstream agents

    # ── Agent outputs ────────────────────────────────────────────────────────
    research: Optional[ResearchFindings]
    market_intel: Optional[MarketIntelligence]
    skill_analysis: Optional[SkillAnalysis]
    gap_strategy: Optional[GapStrategy]
    draft: Optional[DraftOutputs]
    critic_feedback: Optional[CriticFeedback]

    # ── Loop control ─────────────────────────────────────────────────────────
    research_loop_count: int
    draft_loop_count: int

    # ── Checkpoint ───────────────────────────────────────────────────────────
    checkpoint_id: str              # LangGraph thread_id for resuming interrupted runs
    last_completed_node: str        # which node last finished successfully

    # ── Final output ─────────────────────────────────────────────────────────
    final_report: Optional[str]
    error: Optional[str]
