"""
ui/app.py — Streamlit demo UI
"""
import streamlit as st
import httpx, json, time
import PyPDF2, io

API_BASE = "http://localhost:8000"
TOKEN = None   # set after login

st.set_page_config(page_title="Job Application Agent", page_icon="🤖", layout="wide")
st.title("🤖 Autonomous Job Application Research Agent")
st.caption("Deep research · Market intel · Skill gap + action plan · Multi-format drafts · Self-correcting critic")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🔐 Auth")
    username = st.text_input("Username", value="demo")
    if st.button("Get Token"):
        try:
            resp = httpx.post(f"{API_BASE}/token",
                              data={"username": username, "password": "demo"})
            st.session_state["token"] = resp.json()["access_token"]
            st.success("Token received")
        except Exception:
            st.warning("API not running — auth skipped for demo")
            st.session_state["token"] = "demo-token"

    st.divider()
    st.header("📊 System")
    if st.button("Health check"):
        try:
            resp = httpx.get(f"{API_BASE}/v1/health", timeout=5)
            st.json(resp.json())
        except Exception:
            st.error("API unreachable")
    if st.button("Metrics"):
        try:
            resp = httpx.get(f"{API_BASE}/v1/metrics", timeout=5)
            st.json(resp.json())
        except Exception:
            st.error("API unreachable")

    st.divider()
    st.header("🧠 Memory")
    co = st.text_input("Check cached company:")
    if co and st.button("Check"):
        try:
            resp = httpx.get(f"{API_BASE}/v1/memory/{co}", timeout=5)
            d = resp.json()
            st.success("✅ Cached") if d.get("cached") else st.info("Not cached")
        except Exception:
            st.warning("Memory unreachable")

    st.divider()
    st.header("How it works")
    st.markdown("""
**8-node pipeline:**
1. JD Parser — scrape + extract
2. Deep Researcher — 3-hop ReAct + MCP
3. Market Intel — benchmark vs market
4. Skill Analyzer — match%, gaps, apply?
5. Gap Strategist — action plan per gap
6. Drafter — 6-format output
7. Critic — 5-dim rubric, loops back
8. Report Writer — final assembly
    """)

# ── Input form ────────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("Job Description")
    jd_url = st.text_input("Job URL", placeholder="https://boards.greenhouse.io/...")
    jd_text = st.text_area("Or paste JD text", height=180)
with col2:
    st.subheader("Resume")
    resume_file = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])
    resume_text = st.text_area("Or paste resume text", height=180)

company_override = st.text_input("Company name (optional)", placeholder="Anthropic")
thread_id = st.text_input("Resume interrupted run (thread_id)", placeholder="Leave blank for new run")

if st.button("🚀 Run Agent", type="primary", use_container_width=True):
    if not jd_url and not jd_text:
        st.error("Provide a JD URL or paste the text")
        st.stop()
    if not resume_file and not resume_text:
        st.error("Provide your resume")
        st.stop()

    if resume_file and not resume_text:
        file_bytes = resume_file.read()
        if resume_file.name.endswith(".pdf"):
            reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
            resume_text = "\n".join(p.extract_text() or "" for p in reader.pages)
        else:
            resume_text = file_bytes.decode("utf-8", errors="ignore")

    headers = {"Authorization": f"Bearer {st.session_state.get('token', 'demo-token')}"}
    payload = {
        "jd_url": jd_url or None,
        "jd_text": jd_text or None,
        "resume_text": resume_text,
        "company_name": company_override or None,
        "thread_id": thread_id or None,
    }

    # Progress display
    status = st.empty()
    bar = st.progress(0)
    PROGRESS = {
        "jd_parser": 10, "deep_researcher": 30, "market_intel": 45,
        "skill_analyzer": 55, "gap_strategist": 65, "drafter": 75,
        "critic": 85, "report_writer": 95, "done": 100,
    }

    final_report = None
    match_score = apply_now = quality_score = draft_iters = 0
    quick_wins = dealbreakers = []

    try:
        with httpx.Client(timeout=300) as client:
            with client.stream("POST", f"{API_BASE}/v1/analyze/stream",
                               json=payload, headers=headers) as resp:
                for line in resp.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    try:
                        ev = json.loads(line[5:].strip())
                    except Exception:
                        continue

                    node = ev.get("node", "")
                    label = ev.get("label", node)
                    bar.progress(PROGRESS.get(node, 50))
                    status.info(f"{label}...")

                    if "match_score" in ev:
                        match_score = ev["match_score"]
                        apply_now = ev.get("apply_now", True)
                    if "quick_wins" in ev:
                        quick_wins = ev["quick_wins"]
                        dealbreakers = ev["dealbreakers"]
                    if "quality_score" in ev:
                        quality_score = ev["quality_score"]
                    if "final_report" in ev:
                        final_report = ev["final_report"]
                    if node == "drafter":
                        draft_iters += 1

        bar.progress(100)
        status.success("✅ Done!")

    except Exception as e:
        st.error(f"Error: {e}")
        st.info("Make sure the API is running: `python run.py api`")
        st.stop()

    # Results
    st.divider()
    rec = "✅ APPLY NOW" if apply_now else "⏳ PREP FIRST"
    st.subheader(f"Recommendation: {rec}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Skill Match", f"{int(match_score * 100)}%")
    c2.metric("Draft Quality", f"{int(quality_score * 100)}%")
    c3.metric("Drafts", draft_iters)

    if quick_wins:
        st.info(f"**Quick wins** (mention as learning): {', '.join(quick_wins)}")
    if dealbreakers:
        st.warning(f"**Dealbreakers** (be aware): {', '.join(dealbreakers)}")

    if final_report:
        st.download_button("⬇️ Download Report", data=final_report,
                           file_name=f"application_{company_override or 'report'}.md",
                           mime="text/markdown")

        tabs = st.tabs(["📄 Full Report", "📝 Cover Letter", "💼 ATS Bullets",
                        "💬 LinkedIn DM", "📮 Follow-up", "📊 Gap Plan"])
        with tabs[0]:
            st.markdown(final_report)
        with tabs[1]:
            cl_start = final_report.find("## 5. Cover Letter")
            cl_end = final_report.find("## 6.")
            if cl_start > 0:
                st.markdown(final_report[cl_start:cl_end if cl_end > 0 else cl_start+2000])
        # Other tabs would parse respective sections

        # Outcome feedback
        st.divider()
        st.subheader("📬 Did you get a response? (helps the system learn)")
        col_y, col_n = st.columns(2)
        if col_y.button("✅ Yes, got response"):
            httpx.post(f"{API_BASE}/v1/outcome", json={
                "company_name": company_override or "unknown",
                "quality_score": quality_score,
                "got_response": True,
            })
            st.success("Recorded! This improves future runs.")
        if col_n.button("❌ No response"):
            httpx.post(f"{API_BASE}/v1/outcome", json={
                "company_name": company_override or "unknown",
                "quality_score": quality_score,
                "got_response": False,
            })
            st.info("Recorded.")
