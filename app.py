"""
Visvakosh Style Converter — Streamlit UI
Reassembles the LoRA adapter from parts, then rewrites any Gujarati or
English input into Gujarati Visvakosh style.
"""
import os
import re
import glob
import streamlit as st
import torch
from unsloth import FastLanguageModel

# ─── CONFIG ─────────────────────────────────────────────────────────────────
LORA_PATH    = "visvakosh_lora_v4"   # unzipped folder (contains adapter_part_*)
MAX_SEQ_LEN  = 1536
MAX_NEW_TOKENS = 800

SYSTEM_PROMPT = (
    "You are a Gujarati Visvakosh encyclopedia writer. "
    "Rewrite the given Gujarati or English text into Gujarati Visvakosh style. "
    "Use dense narrative paragraphs, active voice, rich vocabulary, "
    "occasional (English) glosses in parentheses, and a pedagogical tone. "
    "Output ONLY the Visvakosh-style Gujarati text."
)

# ─── REASSEMBLE THE LORA FILE FROM PARTS ────────────────────────────────────
def reassemble_lora_if_needed(folder: str) -> str:
    """
    If adapter_model.safetensors is missing, join adapter_part_aa, _ab, ...
    in lexicographic order. Returns the path to the reassembled file.
    """
    target = os.path.join(folder, "adapter_model.safetensors")
    if os.path.exists(target) and os.path.getsize(target) > 0:
        return target

    parts = sorted(glob.glob(os.path.join(folder, "adapter_part_*")))
    if not parts:
        raise FileNotFoundError(
            f"No adapter_model.safetensors or adapter_part_* in '{folder}'. "
            "Did you upload all parts?"
        )

    st.info(f"Reassembling {len(parts)} parts into adapter_model.safetensors …")
    with open(target, "wb") as out:
        for p in parts:
            with open(p, "rb") as inp:
                out.write(inp.read())
    size_mb = os.path.getsize(target) / (1024 * 1024)
    st.success(f"Reassembled {len(parts)} parts → {size_mb:.1f} MB")
    return target


# ─── MODEL LOADING ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(lora_path: str):
    if not os.path.isdir(lora_path):
        raise FileNotFoundError(
            f"Folder '{lora_path}' not found. "
            "Make sure visvakosh_lora_v4/ is next to app.py."
        )
    reassemble_lora_if_needed(lora_path)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=lora_path,
        max_seq_length=MAX_SEQ_LEN,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


# ─── CLEANUP ────────────────────────────────────────────────────────────────
def clean_output(text: str) -> str:
    text = re.sub(r"<\|im[_ ]*\w*\|>", "", text)
    text = re.sub(r"<\|im.*?\|>", "", text)
    text = re.sub(r"<s>|</s>|\[INST\]|\[/INST\]", "", text)
    text = re.sub(r"^#+\s+", "", text, flags=re.MULTILINE)
    return text.strip()


# ─── INFERENCE ──────────────────────────────────────────────────────────────
def rewrite(model, tokenizer, user_text: str,
            temperature: float, top_p: float,
            repetition_penalty: float) -> str:
    prompt = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SEQ_LEN - MAX_NEW_TOKENS,
    ).to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=4,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    return clean_output(generated)


# ─── UI ─────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Visvakosh Style Converter",
                   page_icon="📖", layout="wide")

st.title("📖 Visvakosh Style Converter")
st.caption(
    "Rewrite any Gujarati or English content into Gujarati Visvakosh style "
    "using your own fine-tuned SLM."
)

with st.sidebar:
    st.header("Generation Settings")
    temperature = st.slider("Temperature", 0.1, 1.2, 0.5, 0.05)
    top_p = st.slider("Top-p", 0.5, 1.0, 0.9, 0.05)
    rep_pen = st.slider("Repetition penalty", 1.0, 1.5, 1.15, 0.05)
    st.divider()
    st.caption(f"LoRA path: `{LORA_PATH}`")
    st.caption(f"Max new tokens: `{MAX_NEW_TOKENS}`")

with st.spinner("Loading Visvakosh SLM (first load may take 1–2 minutes)…"):
    try:
        model, tokenizer = load_model(LORA_PATH)
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
                     height=350, key="output_area")
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
            result = rewrite(model, tokenizer, input_text.strip(),
                             temperature=temperature, top_p=top_p,
                             repetition_penalty=rep_pen)
        st.session_state["result"] = result
        st.rerun()
