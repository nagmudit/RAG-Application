# ── first Streamlit call ────────────────────────────────────────
import streamlit as st
st.set_page_config(page_title="RAG Application", layout="wide")

# ── std libs ────────────────────────────────────────────────────
import os, re, pathlib, shutil, requests
from typing import List, Tuple

# ── third‑party ─────────────────────────────────────────────────
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from mistralai import Mistral
from mistralai.models import SDKError

# optional PDF extractor
try:
    from unstructured.partition.pdf import partition_pdf
    HAS_UNSTRUCTURED = True
except ImportError:
    HAS_UNSTRUCTURED = False

# ───────────── CONFIG ────────────────────────────────────────── #
EMBED_MODEL   = "mistral-embed"
LLM_MODEL     = "open-mistral-nemo-2407"          # ← your model
INDEX_DIR     = pathlib.Path("faiss_db")
CHUNK_SIZE    = 800
CHUNK_OVERLAP = 80
TIMEOUT_SEC   = 15
# ─────────────────────────────────────────────────────────────── #

# ---------- EMBEDDING + VECTOR STORE ----------
@st.cache_resource(show_spinner="Loading Mistral embedder…")
def get_embedder():
    key = os.getenv("MISTRAL_API_KEY")
    if not key:
        st.error("Set MISTRAL_API_KEY in the environment.")
        st.stop()
    return MistralAIEmbeddings(model=EMBED_MODEL, mistral_api_key=key)

def load_index_if_any():
    if (INDEX_DIR / "index.faiss").exists():
        return FAISS.load_local(
            str(INDEX_DIR),
            get_embedder(),
            allow_dangerous_deserialization=True,
        )
    return None

if "vstore" not in st.session_state:
    st.session_state.vstore = load_index_if_any()

# ---------- TEXT & INGESTION ---------- #
def chunk_text(text: str) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    return [Document(page_content=t) for t in splitter.split_text(text)]

def scrape_url(url: str) -> str:
    r = requests.get(url, timeout=TIMEOUT_SEC, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()
    return re.sub(r"\s+", " ", soup.get_text(" ", strip=True))

def read_file(file) -> str:
    name = file.name.lower()
    if name.endswith(".pdf") and HAS_UNSTRUCTURED:
        elements = partition_pdf(file=file)
        return "\n".join(e.text for e in elements if hasattr(e, "text"))
    return file.read().decode("utf-8", errors="ignore")

def ingest_docs(docs: List[Document]):
    if st.session_state.vstore is None:
        st.session_state.vstore = FAISS.from_documents(docs, get_embedder())
    else:
        st.session_state.vstore.add_documents(docs)
    st.session_state.vstore.save_local(str(INDEX_DIR))

def clear_kb():
    if INDEX_DIR.exists():
        shutil.rmtree(INDEX_DIR)
    st.session_state.vstore = None

def clear_chat():
    st.session_state.chat = []

# ---------- LLM CALL WITH RETRY ----------
@retry(
    reraise=True,
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=2, max=15),
    retry=retry_if_exception_type(SDKError),
)
def call_mistral(system: str, user: str) -> str:
    client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
    resp = client.chat.complete(
        model=LLM_MODEL,
        messages=[{"role": "system", "content": system},
                  {"role": "user",   "content": user}],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()

# ---------- RAG PIPELINE ----------
def rag_answer(query: str, k: int = 5) -> Tuple[str, List[Document]]:
    vs = st.session_state.vstore
    if vs is None or vs.index.ntotal == 0:
        return "Knowledge‑base is empty. Upload docs or ingest a URL first.", []
    matches = vs.similarity_search(query, k=k)
    context = "\n\n".join(m.page_content for m in matches)

    system_prompt = (
        "You are a helpful assistant. Answer ONLY from CONTEXT; cite as (S1),(S2)…; "
        "say 'I don't know' if answer not within context.\n\nCONTEXT:\n" + context
    )
    try:
        answer = call_mistral(system_prompt, query)
    except SDKError as e:
        if e.status_code == 429:
            return "⚠️ Rate‑limit hit. Please wait a few seconds and try again.", []
        raise
    return answer, matches

# ────────────────── UI ───────────────────────────────
st.title("🔎 RAG Application")

with st.sidebar:
    st.header("📚 Ingest knowledge")

    files = st.file_uploader(
        "Upload text / markdown / PDF files",
        accept_multiple_files=True,
        type=None,
    )
    url_in = st.text_input("…or paste a webpage URL")

    if st.button("➕ Add to KB"):
        try:
            docs: List[Document] = []
            if files:
                total_kb = 0
                for f in files:
                    txt = read_file(f)
                    docs.extend(chunk_text(txt))
                    total_kb += len(txt)
                ingest_docs(docs)
                st.success(f"Added {len(files)} file(s) ({total_kb//1024} KB) to KB.")
            elif url_in:
                txt = scrape_url(url_in)
                ingest_docs(chunk_text(txt))
                st.success(f"Ingested {len(txt)//1024} KB from {url_in}")
            else:
                st.info("Provide files or a URL first.")
        except Exception as e:
            st.error(f"Ingestion failed: {e}")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑 Clear KB", type="secondary"):
            clear_kb()
            st.success("Knowledge‑base cleared.")
    with col2:
        if st.button("🧹 Clear Chat", type="secondary"):
            clear_chat()
            st.success("Chat history cleared.")

# Chat history
if "chat" not in st.session_state:
    st.session_state.chat = []

for role, msg in st.session_state.chat:
    st.chat_message(role).write(msg)

# Chat input
query = st.chat_input("Ask something…")
if query:
    st.chat_message("user").write(query)
    ans, src = rag_answer(query)
    with st.chat_message("assistant"):
        st.write(ans)
        if src:
            with st.expander("Sources"):
                for i, s in enumerate(src, 1):
                    st.caption(f"(S{i}) {s.page_content[:200]}…")
    st.session_state.chat.extend([("user", query), ("assistant", ans)])
