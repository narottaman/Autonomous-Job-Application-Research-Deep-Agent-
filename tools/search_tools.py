"""
tools/search_tools.py
──────────────────────
All @tool-decorated functions the agents can call.
Docstring = what the LLM reads to decide whether to use this tool.
"""

import httpx
from bs4 import BeautifulSoup
from langchain_core.tools import tool
from tavily import TavilyClient
from tenacity import retry, stop_after_attempt, wait_exponential
from config.settings import settings

_tavily = TavilyClient(api_key=settings.tavily_api_key)


@tool
def search_company_info(company_name: str) -> dict:
    """Search for company overview, tech stack, culture, and engineering info.
    Use this first when researching a new company."""
    results = _tavily.search(
        query=f"{company_name} engineering culture tech stack 2025 2026",
        search_depth="advanced",
        include_raw_content=False,
        max_results=6,
    )
    return {"results": [
        {"title": r["title"], "url": r["url"], "content": r["content"]}
        for r in results["results"]
    ]}


@tool
def search_company_news(company_name: str) -> dict:
    """Search for recent news about a company from the last 90 days:
    funding, product launches, layoffs, leadership changes."""
    results = _tavily.search(
        query=f"{company_name} news funding product launch 2026",
        search_depth="basic",
        max_results=5,
        include_domains=["techcrunch.com", "bloomberg.com", "venturebeat.com",
                         "forbes.com", "crunchbase.com"],
    )
    return {"results": [
        {"title": r["title"], "url": r["url"], "content": r["content"]}
        for r in results["results"]
    ]}


@tool
def search_role_insights(role_title: str, company_name: str) -> dict:
    """Search for day-in-the-life info, interview experience, and what this
    role actually involves. Use to understand expectations beyond the JD."""
    results = _tavily.search(
        query=f"{role_title} {company_name} day in life interview experience",
        search_depth="basic",
        max_results=5,
        include_domains=["glassdoor.com", "levels.fyi", "teamblind.com", "reddit.com"],
    )
    return {"results": [
        {"title": r["title"], "url": r["url"], "content": r["content"]}
        for r in results["results"]
    ]}


@tool
def scrape_job_description(url: str) -> dict:
    """Scrape the full text of a job description from a URL.
    Works on Greenhouse, Lever, LinkedIn, and company career pages."""
    try:
        result = _tavily.extract(urls=[url])
        if result and result.get("results"):
            content = result["results"][0].get("raw_content", "")
            if len(content) > 200:
                return {"url": url, "text": content[:8000], "method": "tavily"}
    except Exception:
        pass
    try:
        resp = httpx.get(url, timeout=15, follow_redirects=True,
                         headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        return {"url": url, "text": soup.get_text(separator="\n", strip=True)[:8000], "method": "bs4"}
    except Exception as e:
        return {"url": url, "text": "", "error": str(e)}


@tool
def scrape_engineering_blog(company_name: str) -> dict:
    """Find and scrape a company's engineering blog for real tech stack signals,
    architecture decisions, and engineering culture. High-signal for specificity."""
    search = _tavily.search(
        query=f"{company_name} engineering blog tech stack architecture",
        search_depth="basic",
        max_results=3,
    )
    return {"results": [
        {"title": r["title"], "url": r["url"], "content": r["content"]}
        for r in search["results"][:2]
    ]}


@tool
def scrape_url(url: str) -> dict:
    """Scrape the full content of a specific URL. Use this to follow high-signal
    links found in earlier search results — engineering blog posts, news articles,
    Glassdoor pages. Part of the deep research hop chain."""
    try:
        result = _tavily.extract(urls=[url])
        if result and result.get("results"):
            return {"url": url, "content": result["results"][0].get("raw_content", "")[:4000]}
    except Exception:
        pass
    try:
        resp = httpx.get(url, timeout=15, follow_redirects=True,
                         headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        return {"url": url, "content": soup.get_text(separator="\n", strip=True)[:4000]}
    except Exception as e:
        return {"url": url, "content": "", "error": str(e)}


@tool
def search_similar_roles(role_title: str, company_size: str = "startup") -> dict:
    """Search for similar roles at comparable companies to benchmark requirements.
    Use this in market intelligence to find what the market baseline looks like
    vs what this specific JD requires."""
    results = _tavily.search(
        query=f"{role_title} job requirements skills 2025 2026 site:linkedin.com OR site:greenhouse.io",
        search_depth="basic",
        max_results=5,
    )
    return {"results": [
        {"title": r["title"], "url": r["url"], "content": r["content"][:500]}
        for r in results["results"]
    ]}


@tool
def search_salary_data(role_title: str, company_name: str) -> dict:
    """Search for salary range and compensation data for this role.
    Use levels.fyi, glassdoor, and blind for accurate ranges."""
    results = _tavily.search(
        query=f"{role_title} {company_name} salary compensation 2025 2026",
        search_depth="basic",
        max_results=4,
        include_domains=["levels.fyi", "glassdoor.com", "teamblind.com", "linkedin.com"],
    )
    return {"results": [
        {"title": r["title"], "url": r["url"], "content": r["content"][:400]}
        for r in results["results"]
    ]}


@tool
def search_learning_resource(skill: str) -> dict:
    """Find the fastest free learning resource for a specific skill gap.
    Search for courses, official docs, or projects that give basic proficiency
    in the shortest time. Return specific URL, not generic advice."""
    results = _tavily.search(
        query=f"fastest way to learn {skill} for engineers free course 2025",
        search_depth="basic",
        max_results=4,
        include_domains=["coursera.org", "fast.ai", "docs.python.org",
                         "pytorch.org", "github.com", "youtube.com",
                         "freecodecamp.org", "kaggle.com"],
    )
    return {"skill": skill, "resources": [
        {"title": r["title"], "url": r["url"], "content": r["content"][:300]}
        for r in results["results"]
    ]}
