import uuid
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from chat.langraph_rag_backend import (
    chatbot,
    ingest_pdf,
    retrieve_all_threads,
    thread_document_metadata,
)

# =========================== Utilities ===========================

def generate_thread_id():
    return uuid.uuid4()


def extract_text(content):
    """Extract only text from AI/Human message content."""

    if isinstance(content, str):
        return content  

    if isinstance(content, list):
        texts = []

        for item in content:

            if isinstance(item, dict):
                if item.get("type") == "text":
                    texts.append(item.get("text", ""))

            elif isinstance(item, str):
                texts.append(item)

        return "\n".join(texts)

    return ""


def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)
    st.session_state["message_history"] = []


def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def load_conversation(thread_id):
    state = chatbot.get_state(
        config={"configurable": {"thread_id": thread_id}}
    )
    return state.values.get("messages", [])


# ======================= Session Initialization ===================

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

if "ingested_docs" not in st.session_state:
    st.session_state["ingested_docs"] = {}

add_thread(st.session_state["thread_id"])

thread_key = str(st.session_state["thread_id"])
thread_docs = st.session_state["ingested_docs"].setdefault(thread_key, {})
threads = st.session_state["chat_threads"][::-1]
selected_thread = None

# ============================ Sidebar ============================

st.sidebar.title("LangGraph PDF Chatbot")
st.sidebar.markdown(f"**Thread ID:** `{thread_key}`")

if st.sidebar.button("New Chat", use_container_width=True):
    reset_chat()
    st.rerun()

if thread_docs:
    latest_doc = list(thread_docs.values())[-1]
    st.sidebar.success(
        f"Using `{latest_doc.get('filename')}` "
        f"({latest_doc.get('chunks')} chunks from {latest_doc.get('documents')} pages)"
    )
else:
    st.sidebar.info("No PDF indexed yet.")

uploaded_pdf = st.sidebar.file_uploader(
    "Upload a PDF for this chat",
    type=["pdf"],
)

if uploaded_pdf:
    if uploaded_pdf.name in thread_docs:
        st.sidebar.info(f"`{uploaded_pdf.name}` already processed.")
    else:
        with st.sidebar.status("Indexing PDF...", expanded=True) as status:
            summary = ingest_pdf(
                uploaded_pdf.getvalue(),
                thread_id=thread_key,
                filename=uploaded_pdf.name,
            )
            thread_docs[uploaded_pdf.name] = summary
            status.update(
                label="✅ PDF indexed",
                state="complete",
                expanded=False,
            )

st.sidebar.subheader("Past Conversations")

if not threads:
    st.sidebar.write("No past conversations.")
else:
    for thread_id in threads:
        if st.sidebar.button(
            str(thread_id),
            key=f"thread-{thread_id}",
        ):
            selected_thread = thread_id

# ============================ Main UI ============================

st.title("Multi Utility Chatbot")

# Display chat history

for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ============================ Chat ===============================

user_input = st.chat_input("Ask anything...")

if user_input:

    st.session_state["message_history"].append(
        {
            "role": "user",
            "content": user_input,
        }
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    CONFIG = {
        "configurable": {"thread_id": thread_key},
        "metadata": {"thread_id": thread_key},
        "run_name": "chat_turn",
    }

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            result = chatbot.invoke(
                {
                    "messages": [
                        HumanMessage(content=user_input)
                    ]
                },
                config=CONFIG,
            )

        ai_message = ""

        # Get only the LAST non-empty AI message
        for msg in reversed(result["messages"]):

            if isinstance(msg, AIMessage):

                text = extract_text(msg.content)

                if text.strip():
                    ai_message = text
                    break

        st.markdown(ai_message)

    st.session_state["message_history"].append(
        {
            "role": "assistant",
            "content": ai_message,
        }
    )

    doc_meta = thread_document_metadata(thread_key)

    if doc_meta:
        st.caption(
            f"Document indexed: {doc_meta.get('filename')} "
            f"(Chunks: {doc_meta.get('chunks')}, "
            f"Pages: {doc_meta.get('documents')})"
        )

st.divider()

# ====================== Load Previous Chat ========================

if selected_thread:

    st.session_state["thread_id"] = selected_thread

    messages = load_conversation(selected_thread)

    temp_messages = []

    for msg in messages:

        # Human messages
        if isinstance(msg, HumanMessage):

            text = extract_text(msg.content)

            if text.strip():
                temp_messages.append(
                    {
                        "role": "user",
                        "content": text,
                    }
                )

        # AI messages
        elif isinstance(msg, AIMessage):

            text = extract_text(msg.content)

            # Ignore empty AI messages
            if text.strip():

                temp_messages.append(
                    {
                        "role": "assistant",
                        "content": text,
                    }
                )

        # Skip Tool Messages completely
        elif isinstance(msg, ToolMessage):
            continue

    st.session_state["message_history"] = temp_messages

    st.session_state["ingested_docs"].setdefault(
        str(selected_thread),
        {},
    )

    st.rerun()