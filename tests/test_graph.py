"""
tests/test_graph.py — pytest unit tests. No LLM calls, no API calls.
Tests pure logic: routing, parsing, middleware guards, context engineering.
"""
import pytest
from graph.state import AgentState, CriticFeedback, SkillAnalysis, GapStrategy
from graph.builder import _should_revise_draft


# ── Routing logic ─────────────────────────────────────────────────────────────

def _base_state(**kwargs) -> AgentState:
    defaults = dict(
        jd_url="", jd_text="", resume_text="", company_name="TestCo",
        user_id="test", messages=[], research=None, market_intel=None,
        skill_analysis=None, gap_strategy=None, draft=None,
        critic_feedback=None, research_loop_count=0, draft_loop_count=0,
        context_budget_used=0, context_summary="",
        checkpoint_id="", last_completed_node="",
        final_report=None, error=None,
    )
    defaults.update(kwargs)
    return AgentState(**defaults)


def test_critic_approved_routes_to_report():
    state = _base_state(
        draft_loop_count=1,
        critic_feedback=CriticFeedback(
            quality_score=0.82, issues=[], suggestions=[],
            approved=True, scores={}
        )
    )
    assert _should_revise_draft(state) == "report_writer"


def test_critic_rejected_routes_to_drafter():
    state = _base_state(
        draft_loop_count=1,
        critic_feedback=CriticFeedback(
            quality_score=0.55, issues=["Too generic"], suggestions=["Add specifics"],
            approved=False, scores={}
        )
    )
    assert _should_revise_draft(state) == "drafter"


def test_max_loops_forces_exit():
    state = _base_state(
        draft_loop_count=3,   # at max
        critic_feedback=CriticFeedback(
            quality_score=0.3, issues=["Bad"], suggestions=[],
            approved=False, scores={}
        )
    )
    assert _should_revise_draft(state) == "report_writer"


def test_no_critic_feedback_loops_back():
    state = _base_state(draft_loop_count=0, critic_feedback=None)
    assert _should_revise_draft(state) == "drafter"


# ── Skill analyzer parsing ────────────────────────────────────────────────────

def test_skill_analyzer_parses_valid():
    from agents.skill_analyzer import _parse
    raw = '{"matched_skills": ["Python"], "gap_skills": ["Go"], "transferable_skills": [], "match_score": 0.7, "apply_now": true, "apply_reasoning": "Strong match"}'
    result = _parse(raw)
    assert result["match_score"] == 0.7
    assert result["apply_now"] is True


def test_skill_analyzer_handles_malformed():
    from agents.skill_analyzer import _parse
    result = _parse("not json at all")
    assert result["match_score"] == 0.0
    assert result["apply_now"] is False


# ── Gap strategist parsing ────────────────────────────────────────────────────

def test_gap_strategist_parses():
    from agents.gap_strategist import _parse
    raw = '{"gap_action_plan": [{"skill": "Go", "resource": "tour.golang.org", "time_estimate_days": 7, "priority": "MAJOR", "honest_framing": "actively learning"}], "total_prep_days": 7, "quick_wins": [], "dealbreakers": [], "apply_recommendation": "now", "apply_reasoning": "Match is strong enough"}'
    result = _parse(raw, {})
    assert result["total_prep_days"] == 7
    assert result["apply_now"] is True


# ── Critic parsing ────────────────────────────────────────────────────────────

def test_critic_parses_with_scores():
    from agents.critic import _parse
    raw = '{"scores": {"specificity": 0.8, "grounding": 0.9, "honesty": 1.0, "format": 0.7, "impact": 0.8}, "quality_score": 0.84, "issues": [], "suggestions": [], "approved": true}'
    result = _parse(raw)
    assert result["approved"] is True
    assert result["quality_score"] == 0.84
    assert "specificity" in result["scores"]


# ── Middleware: cost guard ────────────────────────────────────────────────────

def test_token_estimate():
    from summarization.context_engine import estimate_tokens
    assert estimate_tokens("a" * 400) == 100


def test_should_summarize_long_text():
    from summarization.context_engine import should_summarize
    from config.settings import settings
    long = "a" * (settings.max_context_chars + 1)
    assert should_summarize(long) is True


def test_should_not_summarize_short_text():
    from summarization.context_engine import should_summarize
    assert should_summarize("short text") is False


# ── JWT ───────────────────────────────────────────────────────────────────────

def test_create_and_verify_token():
    from middleware.stack import create_token, _verify_jwt
    token = create_token("user123")
    user_id = _verify_jwt(token)
    assert user_id == "user123"


def test_invalid_token_returns_none():
    from middleware.stack import _verify_jwt
    assert _verify_jwt("not.a.real.token") is None


# ── Cache key ─────────────────────────────────────────────────────────────────

def test_cache_key_deterministic():
    from middleware.stack import _make_cache_key
    k1 = _make_cache_key("https://job.com/123", "My resume text")
    k2 = _make_cache_key("https://job.com/123", "My resume text")
    assert k1 == k2


def test_cache_key_different_inputs():
    from middleware.stack import _make_cache_key
    k1 = _make_cache_key("url1", "resume1")
    k2 = _make_cache_key("url2", "resume2")
    assert k1 != k2


# ── Report writer ─────────────────────────────────────────────────────────────

def test_report_writer_assembles():
    from agents.report_writer import report_writer_node
    state = _base_state(
        company_name="Anthropic",
        research={"company_overview": "AI safety", "tech_stack": ["Python"],
                  "recent_news": ["Raised $2B"], "culture_signals": ["Research-first"],
                  "role_specifics": "Train LLMs", "sources": ["https://anthropic.com"],
                  "link_graph": {}, "contradictions": [], "company_health": "Strong",
                  "research_summary": ""},
        market_intel={"similar_roles": [], "market_salary_range": "$200k",
                      "common_stack": ["Python"], "rare_requirements": ["RLHF"],
                      "market_insight": "Research-heavy"},
        skill_analysis={"matched_skills": ["Python"], "gap_skills": ["JAX"],
                        "transferable_skills": [], "match_score": 0.75,
                        "apply_now": True, "apply_reasoning": "Strong fit"},
        gap_strategy={"gap_action_plan": [], "total_prep_days": 0,
                      "quick_wins": [], "dealbreakers": []},
        draft={"cover_letter": "Dear Anthropic...", "why_us_answer": "Safety focus",
               "key_achievement_bullets": ["Built GRPO → 15% improvement"],
               "ats_resume_bullets": ["Implemented RL training"],
               "linkedin_dm": "Hi, saw your role...", "referral_ask": "",
               "followup_email": "", "custom_questions": {}},
        critic_feedback={"quality_score": 0.82, "issues": [], "suggestions": [],
                         "approved": True, "scores": {"specificity": 0.8}},
        draft_loop_count=1,
    )
    result = report_writer_node(state)
    assert "Anthropic" in result["final_report"]
    assert "APPLY NOW" in result["final_report"]
    assert "$200k" in result["final_report"]
