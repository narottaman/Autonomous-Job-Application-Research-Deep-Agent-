"""
config/settings.py
──────────────────
Single source of truth for ALL environment variables.
Every module imports: from config.settings import settings
"""
from pydantic_settings import BaseSettings
from typing import Literal


class Settings(BaseSettings):

    # ── LLM ─────────────────────────────────────────────────────────────────
    anthropic_api_key: str
    model_name: str = "claude-sonnet-4-6"
    critic_model: str = "claude-sonnet-4-6"
    fast_model: str = "claude-haiku-4-5-20251001"   # cheap model for summarization + light tasks

    # ── Web search ───────────────────────────────────────────────────────────
    tavily_api_key: str
    deep_research_hops: int = 3          # how many link-follow hops DeepResearcher takes
    max_sources_per_hop: int = 4

    # ── Vector memory ────────────────────────────────────────────────────────
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "job_agent_memory"
    research_cache_days: int = 7

    # ── Observability ────────────────────────────────────────────────────────
    langchain_api_key: str = ""
    langchain_tracing_v2: bool = True
    langchain_project: str = "job-application-agent"

    # ── Redis (middleware cache + checkpointer) ───────────────────────────────
    # WHY REDIS: two jobs — (1) result cache so identical runs skip the graph,
    # (2) LangGraph checkpointer so interrupted runs resume mid-graph
    redis_url: str = "redis://localhost:6379"
    result_cache_ttl_seconds: int = 86400    # 24h cache on identical runs

    # ── Auth / Gateway ────────────────────────────────────────────────────────
    # WHY JWT: stateless auth — no DB lookup per request, token carries user_id
    jwt_secret_key: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60
    api_rate_limit: str = "10/minute"        # per-user rate limit

    # ── Cost guard ────────────────────────────────────────────────────────────
    max_tokens_per_run: int = 50000          # ~$0.15 at Sonnet pricing — hard ceiling
    max_jd_chars: int = 10000
    max_resume_chars: int = 8000

    # ── Agent behaviour ───────────────────────────────────────────────────────
    max_research_loops: int = 3
    max_draft_loops: int = 3
    min_quality_score: float = 0.75

    # ── MCP server URLs ───────────────────────────────────────────────────────
    # WHY MCP: Model Context Protocol — standard interface to connect agents
    # to external services (LinkedIn, Glassdoor, GitHub) without custom API wrappers
    mcp_enabled: bool = False
    mcp_linkedin_url: str = ""
    mcp_glassdoor_url: str = ""
    mcp_github_url: str = ""

    # ── Summarization ─────────────────────────────────────────────────────────
    # Long research outputs get summarized before passing to drafter
    # to stay within context window and reduce token cost
    max_context_chars: int = 12000          # chars before summarization kicks in
    summary_target_chars: int = 4000       # target after summarization

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
