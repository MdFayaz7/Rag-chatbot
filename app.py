import os
import re

import streamlit as st
import pandas as pd
import numpy as np
import faiss
import requests


def openrouter_key():
    k = os.getenv("OPENROUTER_API_KEY", "").strip()
    if k:
        return k
    try:
        return str(st.secrets["OPENROUTER_API_KEY"]).strip()
    except Exception:
        return ""


# ------------------ CONFIG ------------------
st.set_page_config(page_title="TVS Spare Parts Assistant", layout="wide")

# ------------------ LOAD DATA ------------------
@st.cache_data
def load_data():
    df = pd.read_csv("tvs_parts.csv")
    df["text"] = df.apply(lambda x: f"""
    Vehicle: {x['Vehicle']}
    Part Name: {x['Part Name']}
    Part Number: {x['Part Number']}
    Category: {x['Category']}
    Compatible Models: {x['Compatible Models']}
    """, axis=1)
    return df

df = load_data()

# ------------------ EMBEDDINGS ------------------
def get_embedding(text):
    key = openrouter_key()
    if not key:
        raise RuntimeError(
            "Missing API key: set OPENROUTER_API_KEY or add it to .streamlit/secrets.toml"
        )
    response = requests.post(
        "https://openrouter.ai/api/v1/embeddings",
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": "text-embedding-3-small",
            "input": text
        }
    )
    return response.json()["data"][0]["embedding"]

# ------------------ BUILD VECTOR DB ------------------
@st.cache_resource
def build_index():
    embeddings = [get_embedding(t) for t in df["text"]]
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype("float32"))
    return index


def get_faiss_index():
    """Lazy-load FAISS so the UI renders immediately (building the index calls the API once per row)."""
    if "faiss_index" not in st.session_state:
        with st.spinner(
            "Building search index (first time only: embedding all parts via OpenRouter)…"
        ):
            st.session_state.faiss_index = build_index()
    return st.session_state.faiss_index


# ------------------ SEARCH ------------------
def search(query, k=3):
    index = get_faiss_index()
    query_vec = np.array([get_embedding(query)]).astype("float32")
    distances, indices = index.search(query_vec, k)
    return df.iloc[indices[0]]

# ------------------ LLM ------------------
def ask_llm(context, query):
    prompt = f"""
You are a TVS spare parts assistant.

Context:
{context}

User Question:
{query}

Give structured answer:
- Part Name
- Part Number
- Compatible Models
- Availability
"""

    key = openrouter_key()
    if not key:
        raise RuntimeError(
            "Missing API key: set OPENROUTER_API_KEY or add it to .streamlit/secrets.toml"
        )
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": "openai/gpt-3.5-turbo",
            "messages": [{"role": "user", "content": prompt}]
        }
    )

    return response.json()["choices"][0]["message"]["content"]

# ------------------ HELPERS ------------------
# TVS part numbers in CSV look like TVS-JUP-BS-001, TVS-APR160-HA-029
PART_NUMBER_RE = re.compile(
    r"\bTVS-[A-Z0-9]+(?:-[A-Z0-9]+)*\b", re.IGNORECASE
)


def extract_part_numbers(text: str) -> list:
    if not text or not text.strip():
        return []
    found = PART_NUMBER_RE.findall(text)
    return list(dict.fromkeys(found))


def lookup_by_part_numbers(part_numbers: list) -> pd.DataFrame:
    if not part_numbers:
        return df.iloc[0:0]
    upper = df["Part Number"].str.upper()
    mask = False
    for pn in part_numbers:
        mask = mask | (upper == pn.upper())
    return df.loc[mask]


def is_only_part_numbers_in_query(query: str, codes: list) -> bool:
    """True if the user typed nothing meaningful beyond the part code(s) (no extra question words)."""
    if not codes:
        return False
    tmp = query
    for c in sorted(codes, key=len, reverse=True):
        tmp = re.sub(re.escape(c), "", tmp, flags=re.IGNORECASE)
    leftover = re.sub(r"[\s,;.?!:'\"]+", "", tmp).strip()
    return len(leftover) == 0

# ------------------ UI ------------------

st.title("🔧 TVS Spare Parts Assistant")
st.markdown("Check part numbers, compatibility, and availability")

if not openrouter_key():
    st.warning(
        "Set your OpenRouter key: environment variable `OPENROUTER_API_KEY` "
        "or `.streamlit/secrets.toml` with `OPENROUTER_API_KEY = \"...\"`."
    )

# Sidebar
st.sidebar.header("Select Vehicle")
vehicle = st.sidebar.selectbox("Vehicle Model", df["Vehicle"].unique())

# Chat Input
query = st.text_input("Ask about spare parts:")

if query:
    with st.spinner("Processing..."):

        codes = extract_part_numbers(query)

        # ------------------ PART NUMBER LOOKUP (code may be inside a sentence) ------------------
        if codes:
            results = lookup_by_part_numbers(codes)

            if len(results) > 0:
                st.success("✅ Part Found")

                for _, row in results.iterrows():
                    st.write(f"**Part Name:** {row['Part Name']}")
                    st.write(f"**Part Number:** {row['Part Number']}")
                    st.write(f"**Compatible:** {row['Compatible Models']}")
                    st.write(f"**Category:** {row['Category']}")
                    st.write("---")

                # Answer natural-language questions (availability, compatibility, etc.)
                if not is_only_part_numbers_in_query(query, codes):
                    try:
                        context = "\n".join(results["text"].tolist())
                        answer = ask_llm(context, query)
                        st.markdown("### 🤖 Answer")
                        st.write(answer)
                    except Exception as e:
                        st.warning(f"Could not generate a full answer: {e}")
            else:
                st.error(f"❌ Part Not Found for: {', '.join(codes)}")
                # Fall back to semantic search in case of typo or phrasing
                try:
                    results = search(query)
                    context = "\n".join(results["text"].tolist())
                    answer = ask_llm(context, query)
                    st.markdown("### 🤖 Answer (from catalog search)")
                    st.write(answer)
                    with st.expander("🔍 Retrieved Data"):
                        st.dataframe(results)
                except Exception as e:
                    st.error(f"Search failed: {e}")

        # ------------------ RAG SEARCH (no part code in message) ------------------
        else:
            try:
                results = search(query)

                context = "\n".join(results["text"].tolist())

                answer = ask_llm(context, query)

                st.markdown("### 🤖 Answer")
                st.write(answer)

                # Show retrieved results (debug + transparency)
                with st.expander("🔍 Retrieved Data"):
                    st.dataframe(results)
            except Exception as e:
                st.error(f"Search failed: {e}")

# Footer
st.markdown("---")
st.markdown("Built with RAG + OpenRouter + FAISS")