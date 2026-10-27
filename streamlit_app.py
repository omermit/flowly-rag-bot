"""Flowly Help Bot: a chat UI on top of rag.py. Run: streamlit run streamlit_app.py"""
import streamlit as st
from rag import answer, get_api_key

st.set_page_config(page_title="Flowly Help Bot", page_icon="💬")
st.title("Flowly Help Bot")
st.caption("Ask about Flowly, a fictional company: vacation, plans and prices, support, security. "
           "Answers come only from the company docs, with sources.")

if not get_api_key():
    st.error("Missing GEMINI_API_KEY. Add it in the app's Secrets settings.")
    st.stop()

EXAMPLES = ["How much does the Pro plan cost?",
            "When do I need a doctor's note for sick leave?",
            "What is Flowly's parental leave policy?"]

if "messages" not in st.session_state:
    st.session_state.messages = []          # chat history survives reruns

# Show past messages
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Example buttons
clicked = None
cols = st.columns(len(EXAMPLES))
for col, ex in zip(cols, EXAMPLES):
    if col.button(ex, use_container_width=True):
        clicked = ex

question = st.chat_input("Ask a question...") or clicked
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching the docs..."):
            try:
                reply, chunks = answer(question)
            except Exception as e:
                reply, chunks = f"Error calling Gemini: {e}", []
        st.markdown(reply)
        if chunks:
            with st.expander("Sources (top 3 chunks)"):
                for c in chunks:
                    st.markdown(f"**{c['source']}**, score {c['score']:.2f}")
                    st.caption(c["text"][:300] + "...")
    st.session_state.messages.append({"role": "assistant", "content": reply})
