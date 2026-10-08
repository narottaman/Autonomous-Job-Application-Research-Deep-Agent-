"""
summarization/context_engine.py
─────────────────────────────────
Context engineering layer — shapes what each agent sees.

WHY CONTEXT ENGINEERING:
  Raw research output from DeepResearcher can be 8,000-15,000 chars.
  Passing all of it to Drafter + Critic wastes tokens and often hurts quality
  (LLMs lose focus in very long contexts — the "lost in the middle" problem).
  Context engineering = deliberately deciding what each downstream agent needs.

  The "Lost in the Middle" paper (Liu et al. 2023) showed that LLMs perform
  worst on information placed in the middle of long contexts.
  Solution: compress to the most relevant signal, put it at the top.

TWO FUNCTIONS:
  1. summarize_research() — compress DeepResearcher output to ~1000 tokens
     using the fast/cheap model (Haiku), preserving the most signal-dense facts.
  2. build_agent_context() — assemble the exact context string for each agent,
     ordered by relevance (most important first), within a char budget.
"""

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage
from config.settings import settings
from graph.state import ResearchFindings, SkillAnalysis, MarketIntelligence
import json

# Use fast/cheap model for summarization — no need for Sonnet here
_summarizer = ChatAnthropic(
    model=settings.fast_model,
    api_key=settings.anthropic_api_key,
    temperature=0,
    max_tokens=1024,
)


def summarize_research(research: ResearchFindings) -> str:
    """
    Compress research findings to ~1000 tokens.
    Uses claude-haiku (fast + cheap) not Sonnet.
    Called after DeepResearcher completes, before Drafter receives context.
    """
    full_text = json.dumps(research, indent=2)

    # If already short enough, skip LLM summarization
    if len(full_text) < settings.summary_target_chars:
        return full_text

    response = _summarizer.invoke([
        SystemMessage(content="""You are a research summarizer. 
Compress the company research into a dense summary under 800 words.
Preserve: company overview, exact tech stack, 3 most important recent news items,
culture signals, role specifics, and all source URLs.
Remove: redundant phrasing, repeated facts, generic descriptions.
Output plain text, not JSON."""),
        HumanMessage(content=f"Summarize this research:\n\n{full_text[:8000]}")
    ])
    return response.content


def build_drafter_context(
    research_summary: str,
    skill_analysis: SkillAnalysis,
    gap_strategy: dict,
    market_intel: MarketIntelligence,
    jd_text: str,
    resume_text: str,
    budget: int = None,
) -> str:
    """
    Build the exact context string the Drafter agent receives.
    Ordered by what matters most for writing a cover letter:
      1. Role specifics (most important — what you'd be doing)
      2. Company signals (what to cite in the cover letter)
      3. Matched skills (what to lead with)
      4. Market intel (differentiation angle)
      5. Gap strategy (what NOT to claim)
      6. JD excerpt (keyword alignment)
      7. Resume excerpt (grounding)

    WHY ORDER MATTERS:
      LLMs give more weight to information at the start and end of context.
      Put the most useful things first.
    """
    budget = budget or settings.max_context_chars
    sections = []

    # Most important: role specifics from research
    role_specifics = ""
    if isinstance(research_summary, dict):
        role_specifics = research_summary.get("role_specifics", "")
    sections.append(f"=== ROLE SPECIFICS ===\n{role_specifics or research_summary[:500]}")

    # Company signals for cover letter specificity
    sections.append(f"=== COMPANY RESEARCH SUMMARY ===\n{research_summary[:2000]}")

    # Market context — differentiation angle
    if market_intel:
        sections.append(
            f"=== MARKET CONTEXT ===\n"
            f"Rare requirement (mention this): {market_intel.get('rare_requirements', [])}\n"
            f"Market insight: {market_intel.get('market_insight', '')}"
        )

    # Skills — what to lead with, what to avoid
    if skill_analysis:
        sections.append(
            f"=== SKILL FIT ===\n"
            f"Strong matches: {', '.join(skill_analysis.get('matched_skills', [])[:8])}\n"
            f"GAPS — do NOT claim these: {', '.join(skill_analysis.get('gap_skills', []))}"
        )

    # Gap strategy — gives drafter the honest framing
    if gap_strategy:
        quick_wins = gap_strategy.get("quick_wins", [])
        if quick_wins:
            sections.append(f"=== TRANSFERABLE FRAMING ===\nQuick-win gaps (can mention learning): {', '.join(quick_wins)}")

    # JD and resume — for grounding and keyword matching
    sections.append(f"=== JD EXCERPT ===\n{jd_text[:1500]}")
    sections.append(f"=== RESUME EXCERPT ===\n{resume_text[:1500]}")

    # Truncate to budget
    full_context = "\n\n".join(sections)
    if len(full_context) > budget:
        full_context = full_context[:budget] + "\n[context truncated to budget]"

    return full_context


def estimate_tokens(text: str) -> int:
    """Rough token estimate: chars / 4. Good enough for cost guard."""
    return len(text) // 4


def should_summarize(text: str) -> bool:
    """Check if text exceeds the summarization threshold."""
    return len(text) > settings.max_context_chars
