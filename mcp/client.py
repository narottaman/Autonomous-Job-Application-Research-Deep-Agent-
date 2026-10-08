"""
mcp/mcp_client.py
──────────────────
MCP (Model Context Protocol) client layer.

WHY MCP:
  MCP is Anthropic's open standard for connecting LLM agents to external services.
  Instead of writing a custom API wrapper for LinkedIn, Glassdoor, GitHub etc.,
  MCP gives a standard interface: tools, resources, prompts.
  The agent calls mcp_tool("linkedin_search", {...}) without knowing the API details.

  Think of MCP as "USB-C for AI agents" — one standard connector for any data source.

WHAT WE USE IT FOR:
  - LinkedIn: find hiring manager, mutual connections, company employee count
  - Glassdoor: company rating, interview difficulty, recent reviews
  - GitHub: company's public repos → infer tech stack from actual code

MCP IN THIS PROJECT:
  If MCP servers are configured (.env MCP_LINKEDIN_URL etc.), we use them.
  If not (default), we fall back to Tavily web search.
  This makes the system work out-of-the-box without MCP setup.
"""

import httpx
import json
from config.settings import settings
from typing import Any


class MCPClient:
    """
    Lightweight MCP client. Calls MCP server endpoints over HTTP (SSE transport).
    Each MCP server exposes: /tools/list and /tools/call
    """

    def __init__(self, server_url: str, server_name: str):
        self.server_url = server_url.rstrip("/")
        self.server_name = server_name
        self._tools_cache: list[dict] | None = None

    async def list_tools(self) -> list[dict]:
        """Get available tools from this MCP server."""
        if self._tools_cache:
            return self._tools_cache
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{self.server_url}/tools/list")
            resp.raise_for_status()
            self._tools_cache = resp.json().get("tools", [])
            return self._tools_cache

    async def call_tool(self, tool_name: str, arguments: dict) -> Any:
        """Call a tool on the MCP server."""
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{self.server_url}/tools/call",
                json={"name": tool_name, "arguments": arguments}
            )
            resp.raise_for_status()
            result = resp.json()
            # MCP returns content as list of {type, text} blocks
            content = result.get("content", [])
            texts = [c["text"] for c in content if c.get("type") == "text"]
            return "\n".join(texts)


# ── MCP tool wrappers (fallback to Tavily if MCP not configured) ──────────────

async def get_company_linkedin_data(company_name: str) -> dict:
    """
    Get company LinkedIn data: employee count, recent posts, mutual connections.
    Uses MCP LinkedIn server if configured, else falls back to Tavily search.
    """
    if settings.mcp_enabled and settings.mcp_linkedin_url:
        client = MCPClient(settings.mcp_linkedin_url, "linkedin")
        try:
            result = await client.call_tool("search_company", {"company_name": company_name})
            return {"source": "linkedin_mcp", "data": result}
        except Exception as e:
            pass  # fall through to Tavily

    # Tavily fallback
    from tavily import TavilyClient
    tavily = TavilyClient(api_key=settings.tavily_api_key)
    results = tavily.search(
        query=f"{company_name} LinkedIn company employees hiring",
        search_depth="basic",
        max_results=3,
        include_domains=["linkedin.com"]
    )
    return {"source": "tavily_fallback", "data": results.get("results", [])}


async def get_glassdoor_signals(company_name: str) -> dict:
    """
    Get Glassdoor rating, interview difficulty, recent reviews.
    Uses MCP Glassdoor server if configured, else Tavily search.
    """
    if settings.mcp_enabled and settings.mcp_glassdoor_url:
        client = MCPClient(settings.mcp_glassdoor_url, "glassdoor")
        try:
            result = await client.call_tool("get_company_reviews", {"company": company_name})
            return {"source": "glassdoor_mcp", "data": result}
        except Exception:
            pass

    from tavily import TavilyClient
    tavily = TavilyClient(api_key=settings.tavily_api_key)
    results = tavily.search(
        query=f"{company_name} glassdoor rating interview experience 2025 2026",
        search_depth="basic",
        max_results=3,
        include_domains=["glassdoor.com"]
    )
    return {"source": "tavily_fallback", "data": results.get("results", [])}


async def get_github_tech_signals(company_name: str) -> dict:
    """
    Get company's GitHub org repos → infer real tech stack from actual code.
    Much more reliable than job posting claims.
    Uses MCP GitHub server if configured.
    """
    if settings.mcp_enabled and settings.mcp_github_url:
        client = MCPClient(settings.mcp_github_url, "github")
        try:
            result = await client.call_tool("search_org_repos", {"org": company_name.lower().replace(" ", "")})
            return {"source": "github_mcp", "data": result}
        except Exception:
            pass

    from tavily import TavilyClient
    tavily = TavilyClient(api_key=settings.tavily_api_key)
    results = tavily.search(
        query=f"{company_name} github open source tech stack repos",
        search_depth="basic",
        max_results=3,
        include_domains=["github.com"]
    )
    return {"source": "tavily_fallback", "data": results.get("results", [])}


# Registry of all MCP-powered tools
MCP_TOOLS = {
    "linkedin": get_company_linkedin_data,
    "glassdoor": get_glassdoor_signals,
    "github": get_github_tech_signals,
}
