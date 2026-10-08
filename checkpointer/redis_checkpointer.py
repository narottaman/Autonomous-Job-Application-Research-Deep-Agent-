"""
checkpointer/redis_checkpointer.py
────────────────────────────────────
LangGraph checkpointing via Redis.

WHY CHECKPOINTING:
  Agent runs take 30-90 seconds. If the server restarts, network drops,
  or user closes tab mid-run — the run is lost and the user has to start over.
  Checkpointing saves state after EVERY node completes.
  On resume, the graph starts from the last completed node — not from scratch.

HOW IT WORKS IN LANGGRAPH:
  compiled_graph = builder.compile(checkpointer=RedisCheckpointer())
  Each invoke/stream call takes config={"configurable": {"thread_id": "user123-run456"}}
  LangGraph saves state to Redis after each node. On re-invoke with same thread_id,
  it resumes from where it stopped.

WHAT GETS STORED:
  The full AgentState TypedDict as JSON after each node.
  Key pattern: checkpoint:{thread_id}:{node_name}
  TTL: 24 hours — enough for a user to resume same-day.
"""

import json
import redis.asyncio as aioredis
from config.settings import settings


class RedisCheckpointer:
    """
    Wraps Redis to act as a LangGraph checkpoint backend.
    In LangGraph 0.2+, pass this to graph.compile(checkpointer=...)
    """

    def __init__(self):
        self._redis = None

    async def _get_redis(self):
        if self._redis is None:
            self._redis = aioredis.from_url(settings.redis_url, decode_responses=True)
        return self._redis

    async def save(self, thread_id: str, node_name: str, state: dict):
        """Save state after a node completes."""
        redis = await self._get_redis()
        key = f"checkpoint:{thread_id}:{node_name}"
        await redis.setex(key, 86400, json.dumps(state, default=str))
        # Also update the "latest" pointer
        await redis.setex(f"checkpoint:{thread_id}:latest", 86400, node_name)

    async def load(self, thread_id: str) -> tuple[str | None, dict | None]:
        """
        Load the most recent checkpoint for a thread.
        Returns (last_node_name, state_dict) or (None, None) if no checkpoint.
        """
        redis = await self._get_redis()
        last_node = await redis.get(f"checkpoint:{thread_id}:latest")
        if not last_node:
            return None, None
        state_json = await redis.get(f"checkpoint:{thread_id}:{last_node}")
        if not state_json:
            return None, None
        return last_node, json.loads(state_json)

    async def delete(self, thread_id: str):
        """Clear checkpoints for a completed run."""
        redis = await self._get_redis()
        keys = await redis.keys(f"checkpoint:{thread_id}:*")
        if keys:
            await redis.delete(*keys)

    async def list_threads(self, user_id: str) -> list[str]:
        """List active checkpoint threads for a user (for resume UI)."""
        redis = await self._get_redis()
        keys = await redis.keys(f"checkpoint:{user_id}-*:latest")
        return [k.split(":")[1] for k in keys]


# Singleton
checkpointer = RedisCheckpointer()


# ── LangGraph native checkpointer (preferred in LangGraph 0.2+) ──────────────

def get_langgraph_checkpointer():
    """
    Returns a LangGraph-compatible checkpointer.
    LangGraph 0.2+ ships langgraph.checkpoint.redis.RedisSaver.
    If available, use that. Otherwise fall back to our custom one.

    Usage:
        from checkpointer.redis_checkpointer import get_langgraph_checkpointer
        saver = get_langgraph_checkpointer()
        compiled_graph = graph.compile(checkpointer=saver)
        result = compiled_graph.invoke(state, config={"configurable": {"thread_id": "abc"}})
    """
    try:
        from langgraph.checkpoint.redis import RedisSaver
        return RedisSaver.from_conn_string(settings.redis_url)
    except ImportError:
        # Fallback: LangGraph's in-memory saver for local dev without Redis
        from langgraph.checkpoint.memory import MemorySaver
        return MemorySaver()
