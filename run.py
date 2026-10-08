"""
run.py — single entry point

Commands:
  python run.py api      → FastAPI on :8000
  python run.py ui       → Streamlit on :8501
  python run.py both     → api + ui together
  python run.py cli      → terminal mode, no server
  python run.py eval     → benchmark eval suite
  python run.py test     → run pytest
  python run.py infra    → start Redis + Qdrant via Docker
"""
import sys, subprocess, threading


def run_infra():
    """Start Redis and Qdrant via Docker Compose."""
    print("Starting Redis + Qdrant...")
    subprocess.run(["docker", "compose", "up", "-d"])
    print("✅ Redis: localhost:6379 | Qdrant: localhost:6333")


def run_api():
    print("FastAPI: http://localhost:8000 | Docs: http://localhost:8000/docs")
    subprocess.run(["uvicorn", "api.main:app", "--reload", "--port", "8000"])


def run_ui():
    print("Streamlit: http://localhost:8501")
    subprocess.run(["streamlit", "run", "ui/app.py"])


def run_cli():
    from rich.console import Console
    from rich.markdown import Markdown
    from graph.builder import compiled_graph
    from graph.state import AgentState
    import uuid

    console = Console()
    console.print("[bold cyan]Job Application Agent — CLI[/bold cyan]")

    jd_url = input("Job URL (Enter to skip): ").strip()
    jd_text = ""
    if not jd_url:
        console.print("Paste JD (Enter twice when done):")
        lines = []
        while True:
            line = input()
            if line == "" and lines and lines[-1] == "":
                break
            lines.append(line)
        jd_text = "\n".join(lines)

    console.print("Paste resume (Enter twice when done):")
    lines = []
    while True:
        line = input()
        if line == "" and lines and lines[-1] == "":
            break
        lines.append(line)
    resume_text = "\n".join(lines)
    company = input("Company name (optional): ").strip()
    thread_id = f"cli-{uuid.uuid4().hex[:8]}"

    state = AgentState(
        jd_url=jd_url, jd_text=jd_text, resume_text=resume_text,
        company_name=company, user_id="cli", messages=[],
        research=None, market_intel=None, skill_analysis=None,
        gap_strategy=None, draft=None, critic_feedback=None,
        research_loop_count=0, draft_loop_count=0,
        context_budget_used=0, context_summary="",
        checkpoint_id=thread_id, last_completed_node="",
        final_report=None, error=None,
    )

    NODE_LABELS = {
        "jd_parser": "📋 Parsing JD",
        "deep_researcher": "🔍 Deep research (3-hop)",
        "market_intel": "📊 Market benchmark",
        "skill_analyzer": "🎯 Skill analysis",
        "gap_strategist": "🗺️ Gap action plan",
        "drafter": "✍️ Drafting",
        "critic": "🔎 Critic review",
        "report_writer": "📄 Report assembly",
    }

    for node_name, output in compiled_graph.stream(
        state, {"configurable": {"thread_id": thread_id}}, stream_mode="updates"
    ):
        label = NODE_LABELS.get(node_name, node_name)
        console.print(f"  {label} [green]✓[/green]")
        if node_name == "skill_analyzer" and output.get("skill_analysis"):
            sa = output["skill_analysis"]
            rec = "APPLY NOW" if sa.get("apply_now") else "PREP FIRST"
            console.print(f"    → {rec} | Match: {int(sa.get('match_score',0)*100)}%")
        if node_name == "gap_strategist" and output.get("gap_strategy"):
            gs = output["gap_strategy"]
            if gs.get("quick_wins"):
                console.print(f"    → Quick wins: {', '.join(gs['quick_wins'])}")
        if node_name == "critic" and output.get("critic_feedback"):
            cf = output["critic_feedback"]
            console.print(f"    → Quality: {int(cf.get('quality_score',0)*100)}% {'✅' if cf.get('approved') else '🔄'}")

    # Run again to get final state (stream doesn't return final state)
    final = compiled_graph.invoke(state, {"configurable": {"thread_id": thread_id}})
    report = final.get("final_report", "No report")
    console.print("\n" + "="*60)
    console.print(Markdown(report))

    out_path = f"output_{company or 'report'}.md"
    with open(out_path, "w") as f:
        f.write(report)
    console.print(f"\n[green]Saved: {out_path}[/green]")
    console.print(f"[dim]Thread ID: {thread_id} (use to resume if interrupted)[/dim]")


def run_eval():
    from evals.evaluator import run_benchmark
    run_benchmark()


def run_tests():
    subprocess.run(["pytest", "tests/", "-v"])


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "cli"
    dispatch = {
        "infra": run_infra,
        "api": run_api,
        "ui": run_ui,
        "cli": run_cli,
        "eval": run_eval,
        "test": run_tests,
        "both": lambda: [threading.Thread(target=run_api, daemon=True).start(), run_ui()],
    }
    fn = dispatch.get(cmd)
    if fn:
        fn()
    else:
        print(f"Unknown: {cmd}")
        print(f"Commands: {list(dispatch.keys())}")
