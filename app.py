"""Streamlit client: upload a PDF, index it, and ask the scoped LangChain agent."""

import logging
import threading
import time
from io import BytesIO

import pymupdf
import streamlit as st
from PIL import Image

from scripts.agent import answer_with_agent
from scripts.ingest_pdf import ingest_pdf
from scripts.library import list_documents, local_asset, source_pdf


st.set_page_config(page_title="MaintiDoc · Assistant documentaire", page_icon="📘", layout="wide")
st.markdown("""
<style>
.block-container {max-width: 1120px; padding-top: 2.4rem;}
[data-testid="stSidebar"] {border-right: 1px solid #dce5e9;}
h1 {letter-spacing: -.04em;}
[data-testid="stChatMessage"] {border: 1px solid #e0e8ed; border-radius: 14px;}
</style>
""", unsafe_allow_html=True)

TOOL_LABELS = {
    "search_passages": "Passages du PDF",
    "search_descriptions": "Descriptions automatiques",
    "search_images": "Recherche visuelle CLIP",
}


@st.cache_resource
def workload_lock():
    # Avoid competing VLM/LLM jobs on a small local GPU.
    return threading.Lock()


def display_image(path):
    with Image.open(path) as image:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, "white")
        st.image(Image.alpha_composite(background, rgba).convert("RGB"), width="stretch")


def display_response(message, message_index):
    st.markdown(message["content"])
    sources = message.get("sources", [])
    if any(s["source_type"] == "visual_description" for s in sources):
        st.caption("Cette réponse utilise une description automatique d'image, qui peut contenir des erreurs.")
    if sources:
        with st.expander(f"Sources vérifiables · {len(sources)}"):
            for index, source in enumerate(sources):
                st.write(f"[{source['id']}] {source['document']} — page {source['page']}")
                if source["source_type"] == "visual_reference":
                    st.caption("Visuel retrouvé par similarité CLIP ; son contenu n'a pas été analysé pendant cette recherche.")
                else:
                    st.text(source["text"])
                if source.get("image_path"):
                    try:
                        display_image(local_asset(source["image_path"]))
                    except (ValueError, OSError) as error:
                        st.warning(f"Image indisponible : {error}")
                pdf_path = source_pdf(source["document"])
                if pdf_path and st.toggle(
                    "Afficher la page originale", key=f"page_{message_index}_{index}",
                ):
                    with pymupdf.open(pdf_path) as pdf:
                        page_number = int(source["page"]) - 1
                        if 0 <= page_number < len(pdf):
                            st.image(pdf[page_number].get_pixmap(dpi=110, alpha=False).tobytes("png"), width="stretch")
                st.divider()
    if message.get("trace"):
        with st.expander("Recherches effectuées"):
            for event in message["trace"]:
                st.write(f"**{TOOL_LABELS[event['tool']]}** · {event['results']} résultat(s)")
                st.caption(event["query"])


st.session_state.setdefault("messages", [])
with st.sidebar:
    st.title("📘 MaintiDoc")
    st.caption("Votre documentation, avec ses références.")
    st.subheader("Ajouter un document")
    upload = st.file_uploader("Manuel ou article PDF", type=["pdf"])
    st.caption("Texte extractible requis · 50 Mo maximum. Le premier traitement peut prendre plusieurs minutes.")
    if st.button("Indexer ce PDF", disabled=upload is None, type="primary", width="stretch"):
        try:
            with st.status("Indexation en cours…", expanded=True) as status:
                progress = st.empty()
                with workload_lock():
                    report = ingest_pdf(upload.getvalue(), upload.name, progress.write)
                status.update(label="Document prêt", state="complete", expanded=False)
            st.session_state["selected_document"] = report["document"]
            st.session_state["last_report"] = report
            st.session_state["messages"] = []
            st.rerun()
        except Exception as error:
            logging.exception("PDF ingestion failed")
            st.error("L'indexation a échoué. Après correction, relance le même PDF pour reprendre.")
            with st.expander("Détail de l'erreur"):
                st.code(str(error))

    try:
        documents = list_documents()
    except Exception:
        documents = []
        st.warning("Base indisponible. Démarre PostgreSQL dans Docker Desktop, puis actualise.")
    st.subheader("Document consulté")
    names = [d["document"] for d in documents]
    display_names = {d["document"]: d["display_name"] for d in documents}
    selected = st.session_state.get("selected_document")
    if names:
        if selected not in names:
            st.session_state["selected_document"] = names[0]
        selected = st.selectbox(
            "Les réponses concernent ce PDF uniquement", names,
            format_func=lambda name: display_names[name], key="selected_document",
        )
        record = next(d for d in documents if d["document"] == selected)
        st.caption(f"{record['passages']} passages · {record['descriptions']} descriptions indexées")
        pdf_path = source_pdf(selected)
        if pdf_path:
            st.download_button("Télécharger le PDF source", pdf_path.read_bytes(),
                               file_name=display_names[selected], mime="application/pdf", width="stretch")
    else:
        selected = None
        st.info("Ajoute et indexe un PDF pour commencer.")
    if st.button("Nouvelle conversation", width="stretch"):
        st.session_state["messages"] = []
        st.rerun()
    if st.button("Actualiser les documents", width="stretch"):
        st.rerun()
    st.divider()
    st.caption("Assistant documentaire. Aucun équipement n'est piloté. Les décisions de maintenance restent au technicien habilité.")

if st.session_state.get("conversation_document") != selected:
    st.session_state["messages"] = []
    st.session_state["conversation_document"] = selected

st.caption("ASSISTANT DOCUMENTAIRE · ÉTAPE 1")
st.title("Une question sur votre document ?")
st.write("Retrouvez les explications, tableaux et schémas, puis consultez les pages citées.")
if selected:
    st.caption(f"Document actif : {display_names[selected]}")
report = st.session_state.get("last_report", {})
if report.get("document") == selected:
    for warning in report.get("warnings", []):
        st.warning(warning)

if not st.session_state["messages"]:
    st.info("Exemples : « Explique ce mécanisme », « Quels libellés figurent dans le schéma ? », « Montre le tableau des résultats ».")

for index, message in enumerate(st.session_state["messages"]):
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            display_response(message, index)
        else:
            st.write(message["content"])

question = st.chat_input("Posez une question sur le PDF sélectionné…", disabled=selected is None)
if question:
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state["messages"]]
    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        try:
            started = time.monotonic()
            with st.spinner("Recherche dans le document…"), workload_lock():
                response = answer_with_agent(question, document=selected, history=history)
            message = {"role": "assistant", "content": response["answer"],
                       "sources": response["sources"], "trace": response["trace"]}
            st.session_state["messages"].append(message)
            display_response(message, len(st.session_state["messages"]) - 1)
            st.caption(f"Réponse préparée en {time.monotonic() - started:.1f} s")
        except Exception as error:
            logging.exception("Agent request failed")
            st.error("La réponse n'a pas pu être produite. Vérifie les services et réessaie.")
            with st.expander("Détail de l'erreur"):
                st.code(str(error))
