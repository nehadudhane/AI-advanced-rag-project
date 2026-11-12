import streamlit as st
from app import answer_question

st.title("Aetna Claims & Coverage Assistant")

query = st.text_input("Ask a question about claims, coverage, prior authorization, appeals, or benefits:")

ask_button = st.button("Ask")

if ask_button:
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        st.info("searching your documents....")
        answer = answer_question(query)
        st.subheader("Answer")
        st.write(answer)