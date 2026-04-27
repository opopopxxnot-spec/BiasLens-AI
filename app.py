import streamlit as st
import google.generativeai as genai
import os
import json
import re
from dotenv import load_dotenv

# ── Load environment ──────────────────────────────────────────────────────────
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BiasLens AI – Inclusive Language Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark gradient background */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    min-height: 100vh;
}

/* Hero header */
.hero-header {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
}
.hero-header h1 {
    font-size: 3rem;
    font-weight: 700;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.4rem;
}
.hero-header p {
    font-size: 1.1rem;
    color: #94a3b8;
    margin-top: 0;
}

/* Glass cards */
.glass-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px;
    padding: 1.6rem;
    margin-bottom: 1.2rem;
    backdrop-filter: blur(12px);
}

/* Bias badge */
.bias-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-right: 6px;
    margin-bottom: 4px;
}
.badge-gender   { background: rgba(167,139,250,0.25); color: #a78bfa; border: 1px solid #a78bfa55; }
.badge-age      { background: rgba(251,191,36,0.25);  color: #fbbf24; border: 1px solid #fbbf2455; }
.badge-racial   { background: rgba(248,113,113,0.25); color: #f87171; border: 1px solid #f8717155; }
.badge-other    { background: rgba(96,165,250,0.25);  color: #60a5fa; border: 1px solid #60a5fa55; }

/* Metric boxes */
.metric-box {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}
.metric-box .metric-num {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(90deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.metric-box .metric-label {
    font-size: 0.8rem;
    color: #94a3b8;
    margin-top: 2px;
}

/* Rewrite block */
.rewrite-block {
    background: rgba(52,211,153,0.07);
    border: 1px solid rgba(52,211,153,0.3);
    border-radius: 12px;
    padding: 1.4rem;
    color: #e2e8f0;
    line-height: 1.8;
    white-space: pre-wrap;
    font-size: 0.95rem;
}

/* Disclaimer */
.disclaimer {
    background: rgba(251,191,36,0.08);
    border-left: 4px solid #fbbf24;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem;
    font-size: 0.82rem;
    color: #cbd5e1;
    margin-top: 1rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(15,12,41,0.9) !important;
    border-right: 1px solid rgba(255,255,255,0.08);
}

/* Buttons */
.stButton > button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(167,139,250,0.35) !important;
}

/* Text area */
.stTextArea > div > textarea {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    font-size: 0.95rem !important;
}

/* Progress / spinner */
.stSpinner > div { border-top-color: #a78bfa !important; }

/* Divider */
hr { border-color: rgba(255,255,255,0.08) !important; }
</style>
""", unsafe_allow_html=True)

# ── Sample texts ──────────────────────────────────────────────────────────────
SAMPLE_JD = """We are looking for a young, energetic rockstar developer to join our boys' club culture.
The ideal candidate is a recent graduate who can work long hours and handle pressure like a man.
We need someone who is a native English speaker and fits our company culture.
Candidates should be physically able to perform all job duties and have a clean background."""

SAMPLE_PR = """John is an aggressive go-getter who dominates meetings. He is a real team player.
Sarah, while emotional at times, does good work when she focuses. 
The older team members like Bob are set in their ways and resist change.
We need people with the right cultural fit who understand how things work here."""

# ── Gemini setup ──────────────────────────────────────────────────────────────
def get_gemini_model():
    api_key = st.session_state.get("api_key", GEMINI_API_KEY)
    if not api_key or api_key == "your_api_key_here":
        return None
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-2.5-flash")

# ── Prompt builder ────────────────────────────────────────────────────────────
def build_prompt(text: str) -> str:
    return f"""
You are an expert HR consultant and linguistic bias analyst specializing in inclusive language.

Analyze the following text for unconscious bias. Return ONLY valid JSON (no markdown, no code fences).

Text to analyze:
\"\"\"
{text}
\"\"\"

Return this exact JSON structure:
{{
  "overall_bias_score": <integer 0-100, where 100 = extremely biased>,
  "bias_level": "<Low | Medium | High>",
  "summary": "<2-3 sentence overall summary of bias found>",
  "biased_phrases": [
    {{
      "phrase": "<exact phrase from text>",
      "bias_type": "<Gender | Age | Racial | Cultural | Ableism | Other>",
      "explanation": "<why this is biased>",
      "severity": "<Low | Medium | High>"
    }}
  ],
  "inclusive_rewrite": "<full rewritten version of the text with all bias removed, preserving meaning>",
  "key_improvements": ["<improvement 1>", "<improvement 2>", "<improvement 3>"]
}}
"""

# ── Parse Gemini response ─────────────────────────────────────────────────────
def parse_response(raw: str) -> dict:
    # Strip markdown code fences if present
    clean = re.sub(r"```[a-zA-Z]*", "", raw)
    clean = clean.replace("```", "").strip()
    return json.loads(clean)

# ── Analyze function ──────────────────────────────────────────────────────────
def analyze_text(text: str):
    model = get_gemini_model()
    if not model:
        return None, "❌ No valid API key found. Please enter your Gemini API key in the sidebar."
    try:
        response = model.generate_content(build_prompt(text))
        data = parse_response(response.text)
        return data, None
    except json.JSONDecodeError:
        return None, "⚠️ Could not parse the AI response. Please try again."
    except Exception as e:
        return None, f"❌ API Error: {str(e)}"

# ── Bias badge helper ─────────────────────────────────────────────────────────
BADGE_CLASS = {
    "Gender":   "badge-gender",
    "Age":      "badge-age",
    "Racial":   "badge-racial",
    "Cultural": "badge-racial",
    "Ableism":  "badge-age",
    "Other":    "badge-other",
}

def bias_badge(bias_type: str) -> str:
    cls = BADGE_CLASS.get(bias_type, "badge-other")
    return f'<span class="bias-badge {cls}">{bias_type}</span>'

# ── Score color ───────────────────────────────────────────────────────────────
def score_color(score: int) -> str:
    if score < 30: return "#34d399"
    if score < 60: return "#fbbf24"
    return "#f87171"

# ── Session state init ────────────────────────────────────────────────────────
if "input_text" not in st.session_state:
    st.session_state.input_text = ""
if "results" not in st.session_state:
    st.session_state.results = None
if "api_key" not in st.session_state:
    st.session_state.api_key = GEMINI_API_KEY

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 BiasLens AI")
    st.markdown("---")

    st.markdown("### 🔑 API Configuration")
    st.text_input(
        "Gemini API Key",
        type="password",
        placeholder="AIza...",
        help="Get your free key at https://aistudio.google.com",
        key="api_key",
    )

    st.markdown("---")
    st.markdown("### 📋 Load Sample Text")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("📄 Job Post", use_container_width=True):
            st.session_state.input_text = SAMPLE_JD
            st.session_state.results = None
            st.rerun()
    with col2:
        if st.button("📊 Perf. Review", use_container_width=True):
            st.session_state.input_text = SAMPLE_PR
            st.session_state.results = None
            st.rerun()

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
BiasLens AI uses **Gemini 1.5 Flash** to detect unconscious bias in:
- 📋 Job descriptions
- 📊 Performance reviews
- 📝 Corporate communications

Bias types detected:
- 🟣 Gender bias
- 🟡 Age bias  
- 🔴 Racial / cultural bias
- 🔵 Ableism & other
    """)

    st.markdown("---")
    st.markdown(
        "<div style='color:#64748b;font-size:0.75rem;text-align:center'>"
        "BiasLens AI v1.0 · Built for Hackathon 2026"
        "</div>",
        unsafe_allow_html=True,
    )

# ── Hero Header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <h1>🔍 BiasLens AI</h1>
    <p>Detect unconscious bias in job descriptions & performance reviews — then rewrite for inclusion.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Input Section ─────────────────────────────────────────────────────────────
st.markdown("### ✍️ Paste Your Text")

input_text = st.text_area(
    label="Text to analyze",
    height=220,
    placeholder="Paste a job description, performance review, or any workplace text here...",
    label_visibility="collapsed",
    key="input_text",
)

col_a, col_b, col_c = st.columns([2, 1, 1])
with col_a:
    analyze_clicked = st.button(
        "🔍 Analyze Bias",
        type="primary",
        use_container_width=True,
        disabled=not input_text.strip(),
    )
with col_b:
    if st.button("🗑️ Clear", use_container_width=True):
        st.session_state.input_text = ""
        st.session_state.results = None
        st.rerun()
with col_c:
    word_count = len(input_text.split()) if input_text.strip() else 0
    st.markdown(
        f"<div style='text-align:center;padding:0.5rem;color:#94a3b8;font-size:0.85rem'>"
        f"📝 {word_count} words</div>",
        unsafe_allow_html=True,
    )

# ── Run analysis ──────────────────────────────────────────────────────────────
if analyze_clicked:
    if not input_text.strip():
        st.warning("⚠️ Please enter some text before analyzing.")
    else:
        with st.spinner("🤖 Gemini is scanning for bias patterns..."):
            results, error = analyze_text(input_text)
        if error:
            st.error(error)
            st.session_state.results = None
        else:
            st.session_state.results = results
            st.success("✅ Analysis complete! Scroll down to see your Bias Detection Report.")

# ── Results ───────────────────────────────────────────────────────────────────
if st.session_state.results:
    data = st.session_state.results
    try:
        score = int(data.get("overall_bias_score", 0))
    except (ValueError, TypeError):
        score = 0
    bias_level = data.get("bias_level", "Unknown")
    summary = data.get("summary", "")
    phrases = data.get("biased_phrases", [])
    rewrite = data.get("inclusive_rewrite", "")
    improvements = data.get("key_improvements", [])

    st.markdown("---")
    st.markdown("## 📊 Bias Detection Report")

    # ── Metrics row
    m1, m2, m3, m4 = st.columns(4)
    scolor = score_color(score)
    with m1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-num" style="background:linear-gradient(90deg,{scolor},{scolor}aa);
                -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">
                {score}
            </div>
            <div class="metric-label">Bias Score / 100</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        lvl_color = {"Low":"#34d399","Medium":"#fbbf24","High":"#f87171"}.get(bias_level,"#60a5fa")
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-num" style="background:linear-gradient(90deg,{lvl_color},{lvl_color}aa);
                -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">
                {bias_level}
            </div>
            <div class="metric-label">Bias Level</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-num">{len(phrases)}</div>
            <div class="metric-label">Issues Found</div>
        </div>""", unsafe_allow_html=True)
    with m4:
        bias_types = list(set(p.get("bias_type","Other") for p in phrases))
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-num">{len(bias_types)}</div>
            <div class="metric-label">Bias Categories</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Score bar
    st.markdown(f"""
    <div class="glass-card">
        <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
            <span style="color:#94a3b8;font-size:0.9rem;">Overall Bias Score</span>
            <span style="color:{scolor};font-weight:700;">{score}/100</span>
        </div>
        <div style="background:rgba(255,255,255,0.08);border-radius:999px;height:10px;">
            <div style="background:linear-gradient(90deg,{scolor}99,{scolor});
                width:{score}%;height:10px;border-radius:999px;
                transition:width 0.5s ease;"></div>
        </div>
        <p style="color:#cbd5e1;margin-top:1rem;font-size:0.9rem;line-height:1.6;">{summary}</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Biased phrases
    st.markdown("### ⚠️ Detected Bias Issues")
    if phrases:
        for i, item in enumerate(phrases, 1):
            phrase    = item.get("phrase", "")
            btype     = item.get("bias_type", "Other")
            expl      = item.get("explanation", "")
            severity  = item.get("severity", "Medium")
            sev_color = {"Low":"#34d399","Medium":"#fbbf24","High":"#f87171"}.get(severity,"#60a5fa")

            with st.expander(f"#{i}  \"{phrase}\"  —  {btype} Bias", expanded=(i <= 3)):
                st.markdown(f"""
                <div style="display:flex;gap:8px;align-items:center;margin-bottom:10px;flex-wrap:wrap;">
                    {bias_badge(btype)}
                    <span class="bias-badge" style="background:rgba(255,255,255,0.07);
                        color:{sev_color};border:1px solid {sev_color}55;">
                        {severity} Severity
                    </span>
                </div>
                <p style="color:#cbd5e1;line-height:1.7;margin:0;">{expl}</p>
                """, unsafe_allow_html=True)
    else:
        st.success("🎉 No significant bias phrases detected!")

    # ── Inclusive rewrite
    st.markdown("---")
    st.markdown("### ✨ Inclusive Rewrite")
    st.markdown("""
    <div class="glass-card" style="border-color:rgba(52,211,153,0.3);">
        <p style="color:#94a3b8;font-size:0.82rem;margin-bottom:0.8rem;">
            🤖 AI-generated inclusive version — review before using
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        f'<div class="rewrite-block">{rewrite}</div>',
        unsafe_allow_html=True,
    )

    # Copy helper
    st.text_area(
        "📋 Copy-friendly version",
        value=rewrite,
        height=160,
        help="Select all and copy from here",
    )

    # ── Key improvements
    if improvements:
        st.markdown("---")
        st.markdown("### 💡 Key Improvements Made")
        for imp in improvements:
            st.markdown(f"- ✅ {imp}")

    # ── Disclaimer
    st.markdown("""
    <div class="disclaimer">
        ⚖️ <strong>Disclaimer:</strong> BiasLens AI is an assistive tool powered by generative AI.
        Results are indicative and may not capture all forms of bias or cultural nuance.
        This tool <strong>does not replace</strong> professional HR guidance, legal review, or
        qualified diversity & inclusion consulting. Always apply human judgment before making
        hiring or employment decisions.
    </div>
    """, unsafe_allow_html=True)

    # ── Reset
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔄 Analyze New Text", type="secondary"):
        st.session_state.results = None
        st.session_state.input_text = ""
        st.rerun()

# ── Empty state ───────────────────────────────────────────────────────────────
elif not analyze_clicked:
    st.markdown("""
    <div class="glass-card" style="text-align:center;padding:3rem 2rem;">
        <div style="font-size:3rem;margin-bottom:1rem;">🔍</div>
        <h3 style="color:#e2e8f0;margin-bottom:0.5rem;">Ready to Detect Bias</h3>
        <p style="color:#94a3b8;">
            Paste any job description or performance review above, then click <strong>Analyze Bias</strong>.<br>
            Or load a sample text from the sidebar to see BiasLens in action.
        </p>
    </div>
    """, unsafe_allow_html=True)
