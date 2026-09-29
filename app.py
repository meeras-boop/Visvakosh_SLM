"""
Visvakosh Style Converter — Streamlit UI (CPU / llama-cpp-python)
Loads a GGUF-quantized Visvakosh LoRA from Hugging Face and rewrites
any Gujarati or English content into Gujarati Visvakosh style.
"""
import re
import streamlit as st
from llama_cpp import Llama

# ─── CONFIG ─────────────────────────────────────────────────────────────────
HF_REPO       = "your-username/visvakosh-gguf"     # ← change this
GGUF_FILENAME = "visvakosh-model-q4_k_m.gguf"      # ← change if different
N_CTX         = 2048
N_THREADS     = 4
MAX_TOKENS    = 800

SYSTEM_PROMPT = (
    "You are a Gujarati Visvakosh encyclopedia writer. "
    "Rewrite the given Gujarati or English text into Gujarati Visvakosh style. "
    "Use dense narrative paragraphs, active voice, rich vocabulary, "
    "occasional (English) glosses in parentheses, and a pedagogical tone. "
    "Output ONLY the Visvakosh-style Gujarati text."
)

# ─── MODEL LOADING (cached) ─────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(repo_id: str, filename: str):
    llm = Llama.from_pretrained(
        repo_id=repo_id,
        filename=filename,
        n_ctx=N_CTX,
        n_threads=N_THREADS,
        n_gpu_layers=0,           # CPU-only
        verbose=False,
    )
    return llm

# ─── CLEANUP ────────────────────────────────────────────────────────────────
def clean_output(text: str) -> str:
    text = re.sub(r"<\|im[_ ]*\w*\|>", "", text)
    text = re.sub(r"<\|im.*?\|>", "", text)
    text = re.sub(r"<s>|</s>|\[INST\]|\[/INST\]", "", text)
    text = re.sub(r"^#+\s+", "", text, flags=re.MULTILINE)
    return text.strip()

# ─── INFERENCE ──────────────────────────────────────────────────────────────
def rewrite(llm, user_text: str,
            temperature: float, top_p: float,
            repetition_penalty: float) -> str:
    output = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_text},
        ],
        temperature=temperature,
        top_p=top_p,
        repeat_penalty=repetition_penalty,
        max_tokens=MAX_TOKENS,
    )
    raw = output["choices"][0]["message"]["content"]
    return clean_output(raw)

# ─── UI ─────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Visvakosh Style Converter",
                   page_icon="📖", layout="wide")

st.title("📖 Visvakosh Style Converter")
st.caption(
    "Rewrite any Gujarati or English content into Gujarati Visvakosh style "
    "using your own fine-tuned SLM (CPU inference via llama.cpp)."
)

with st.sidebar:
    st.header("Generation Settings")
    temperature = st.slider("Temperature", 0.1, 1.2, 0.5, 0.05,
                            help="Lower = more deterministic, higher = more creative")
    top_p = st.slider("Top-p", 0.5, 1.0, 0.9, 0.05,
                      help="Nucleus sampling threshold")
    rep_pen = st.slider("Repetition penalty", 1.0, 1.5, 1.15, 0.05,
                        help="Higher discourages repeated phrases")
    st.divider()
    st.caption(f"Model: `{HF_REPO}`")
    st.caption(f"File: `{GGUF_FILENAME}`")
    st.caption(f"Max tokens: `{MAX_TOKENS}`")

with st.spinner("Loading Visvakosh model (first load may take 1–3 minutes)…"):
    try:
        llm = load_model(HF_REPO, GGUF_FILENAME)
        st.success("Model loaded.")
    except Exception as e:
        st.error(f"Model load failed: {e}")
        st.stop()

col_in, col_out = st.columns(2)

with col_in:
    st.subheader("Input")
    input_text = st.text_area(
        "Paste Gujarati or English text:",
        height=350,
        placeholder="Paste any text here…",
        key="input_area",
    )
    convert = st.button("Convert to Visvakosh style",
                        type="primary", use_container_width=True)
    if st.button("Clear", use_container_width=True):
        st.session_state.pop("result", None)
        st.rerun()

with col_out:
    st.subheader("Visvakosh Output")
    if "result" in st.session_state:
        st.text_area("Result:",
                     value=st.session_state["result"],
                     height=350,
                     key="output_area")
        st.download_button(
            "Download output as .txt",
            data=st.session_state["result"],
            file_name="visvakosh_output.txt",
            mime="text/plain",
            use_container_width=True,
        )
    else:
        st.info("Output will appear here.")

if convert:
    if not input_text.strip():
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Converting to Visvakosh style…"):
            try:
                result = rewrite(llm, input_text.strip(),
                                 temperature=temperature,
                                 top_p=top_p,
                                 repetition_penalty=rep_pen)
                st.session_state["result"] = result
                st.rerun()
            except Exception as e:
                st.error(f"Generation failed: {e}")
