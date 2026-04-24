import streamlit as st
import re
import gc
import json
from langchain_openai import ChatOpenAI

# ==========================================
# PAGE CONFIGURATION & LAYOUT
# ==========================================
st.set_page_config(page_title="RFP Bid Response Evaluator", page_icon="📝", layout="wide")

# Custom CSS for Premium Design aesthetics
st.markdown("""
<style>
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
        font-family: 'Inter', sans-serif;
    }
    .main .block-container {
        padding-top: 2rem;
    }
    h1, h2, h3 {
        color: #58a6ff;
    }
    .metric-card {
        background: #161b22;
        border-radius: 10px;
        padding: 20px;
        border: 1px solid #30363d;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        margin-bottom: 20px;
        transition: transform 0.2s ease-in-out;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #58a6ff;
    }
    .stButton>button {
        background: linear-gradient(135deg, #238636 0%, #2ea043 100%);
        color: white;
        border: none;
        border-radius: 6px;
        font-weight: 600;
        transition: background 0.3s;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #2ea043 0%, #3fb950 100%);
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# PYTHON-NATIVE LOGIC / HEURISTICS 
# ==========================================
def extract_themes_count(text, themes):
    """Pure Python keyword density checker for Win Themes."""
    counts = {}
    if not themes.strip():
        return counts
        
    theme_list = [t.strip().lower() for t in themes.split(",")]
    text_lower = text.lower()
    
    for theme in theme_list:
        matches = len(re.findall(r'\b' + re.escape(theme) + r'\b', text_lower))
        counts[theme] = matches
    return counts

def extract_shipley_metrics(text, client_name):
    """Pure Python lexical check for Shipley customer-focus (we vs client)."""
    text_lower = text.lower()
    inward_words = ["we", "us", "our", "ourselves"]
    
    inward_count = sum(len(re.findall(r'\b' + word + r'\b', text_lower)) for word in inward_words)
    client_name_count = len(re.findall(r'\b' + re.escape(client_name.strip().lower()) + r'\b', text_lower)) if client_name else 0
    
    return {
        "inward_pronoun_count": inward_count,
        "client_name_count": client_name_count,
        "ratio_client_to_inward": round(client_name_count / max(1, inward_count), 2)
    }

# ==========================================
# LLM AGENT INVOCATIONS
# ==========================================
def invoke_agent_with_fallback(llm, prompt_text, system_message="You are an expert RFP Evaluator."):
    """Invokes LLM and aggressively handles memory and error checking."""
    try:
        messages = [
            ("system", system_message),
            ("human", prompt_text)
        ]
        response = llm.invoke(messages)
        # Force garbage collection to free memory
        gc.collect()
        content = response.content.strip() if response.content else ""
        if not content:
            print("[DEBUG] Model returned an empty string for prompt:\n", prompt_text)
            return "⚠️ The model returned an empty response. This occasionally happens with smaller local models under strict constraints. Please try clicking 'Evaluate' again."
        return content
    except Exception as e:
        gc.collect()
        return f"Error evaluating agent: {e}"

def run_compliance_agent(llm, question, draft):
    prompt = f"""
Analyze if the draft response meets ALL functional requirements asked in the RFP question.
Be extremely brief. Output must end with a definite: PASS, FAIL, or PARTIAL.

RFP Question: '{question}'
Draft Response: '{draft}'

Provide a 2 sentence explanation, then end with: "STATUS: [PASS/FAIL/PARTIAL]"
"""
    return invoke_agent_with_fallback(llm, prompt)

def run_theme_agent(llm, draft, python_theme_stats):
    prompt = f"""
We have tracked the following occurrences of Win Themes in the text using basic matching:
{python_theme_stats}

Given this draft response, evaluate conceptually if the themes are actually effectively woven into the narrative rather than just name-dropped.
Draft Response: '{draft}'

Provide a very short 1 paragraph explanation, and give a "Theme Quality Score" percentage (e.g., 85%).
"""
    return invoke_agent_with_fallback(llm, prompt)

def run_shipley_agent(llm, draft, python_shipley_stats):
    prompt = f"""
Shipley metrics calculated:
- First-person inward words (we/us/our): {python_shipley_stats['inward_pronoun_count']}
- Client Name Mentions: {python_shipley_stats['client_name_count']}

Evaluate the draft for active voice, clarity, and structural Shipley customer-focus.
Draft text: '{draft}'

Provide a brief paragraph on Active voice and Clarity, followed by a final "Shipley Score" out of 10.
"""
    return invoke_agent_with_fallback(llm, prompt)

# ==========================================
# UI BUILD & ORCHESTRATION 
# ==========================================
def main():
    st.title("🚀 RFP Bid Response Evaluator")
    st.markdown("Phase 1: Zero-RAG, Local LLM-as-a-Judge using Hybrid Python + GenAI logic.")

    with st.sidebar:
        st.header("⚙️ Local Server Config")
        local_url = st.text_input("LM Studio Endpoint", "http://localhost:1234/v1")
        model_name = st.text_input("Model Name", "Nemotron-3-nano-4b")
        st.caption("Settings bounded for speed and low CPU utilization.")
        
        st.divider()
        st.header("📋 Evaluation Context")
        win_themes = st.text_area("Win Themes (Comma Separated)", "Cost Efficiency, Seamless Integration, 24/7 Support")
        client_name = st.text_input("Target Client Name", "Acme Corp")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("RFP Question / Requirement")
        rfp_q = st.text_area("Paste the exact requirement here...", height=250)
        
    with col2:
        st.subheader("Draft Bid Response")
        rfp_draft = st.text_area("Paste your human-written answer here...", height=250)

    if st.button("Evaluate Response with AI Pipeline (Sequential)"):
        if not rfp_q or not rfp_draft:
            st.error("Please provide both the RFP Question and Draft Response before evaluating.")
            return
            
        # Initialize LangChain LLM optimized for speed / max_tokens
        llm = ChatOpenAI(
            base_url=local_url,
            api_key="lm-studio",
            model=model_name,
            temperature=0.1,
            max_tokens=300, # Increased limit in case the model's generation was abruptly clipped
            timeout=180, # Generous timeout for local inference
            max_retries=1
        )
        
        st.markdown("<hr/>", unsafe_allow_html=True)
        st.subheader("📊 Evaluation Dashboard")
        
        # Step 1: Python Engine Execution
        with st.status("Executing Python Lexical Analysis Engine...", expanded=True) as status:
            theme_metrics = extract_themes_count(rfp_draft, win_themes)
            shipley_metrics = extract_shipley_metrics(rfp_draft, client_name)
            st.write(f"Themes found lexically: {theme_metrics}")
            st.write(f"Shipley PR metrics: inward={shipley_metrics['inward_pronoun_count']}, client={shipley_metrics['client_name_count']}")
            status.update(label="Python Analytical execution completed successfully in < 1s.", state="complete", expanded=False)

        # Step 2: Sequential LLM Execution
        col_res1, col_res2, col_res3 = st.columns(3)
        
        # AGENT 1
        with col_res1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.subheader("🏛️ Agent 1: Compliance")
            with st.spinner("Analyzing Functional Fit..."):
                c_out = run_compliance_agent(llm, rfp_q, rfp_draft)
                st.write(c_out)
            st.markdown('</div>', unsafe_allow_html=True)

        # AGENT 2
        with col_res2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.subheader("🎯 Agent 2: Theme Qualifier")
            with st.spinner("Analyzing Narrative Win Themes..."):
                t_out = run_theme_agent(llm, rfp_draft, theme_metrics)
                st.write(t_out)
            st.markdown('</div>', unsafe_allow_html=True)

        # AGENT 3
        with col_res3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.subheader("✍️ Agent 3: Shipley Grader")
            with st.spinner("Analyzing Tone and Focus..."):
                s_out = run_shipley_agent(llm, rfp_draft, shipley_metrics)
                st.write(s_out)
            st.markdown('</div>', unsafe_allow_html=True)
            
        st.success("Sequential Multi-Agent Pipeline Completed. Memory successfully flushed.")

if __name__ == "__main__":
    main()
