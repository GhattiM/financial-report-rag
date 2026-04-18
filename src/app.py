import streamlit as st
import os
from query import setup_rag_chain, ask_finance

# Page configuration
st.set_page_config(
    page_title="Financial Analyst RAG",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Senior Financial Analyst RAG")
st.markdown("""
This application analyzes SEC 10-K filings for multiple companies. 
Ask questions about revenue, net income, or compare performance across companies.
""")

# Initialize the RAG chain
@st.cache_resource
def get_rag_chain():
    return setup_rag_chain()

try:
    with st.spinner("🧠 Loading AI Brain and Database... This may take a moment."):
        rag_chain = get_rag_chain()
    st.success("✅ System Ready!")
except Exception as e:
    st.error(f"❌ Error loading system: {e}")
    st.stop()

# Sidebar for information
with st.sidebar:
    st.header("About")
    st.info("""
    **Tech Stack:**
    - LangChain (LCEL)
    - Google Gemini 2.0 Flash (via OpenRouter)
    - ChromaDB (Local)
    - BGE Embeddings (HuggingFace)
    - Cohere Rerank v3.5
    
    **Features:**
    - Dynamic Multi-Query Retrieval
    - Automated Table Extraction
    - Long-Context Reordering
    """)
    
    if st.button("Clear Cache"):
        st.cache_resource.clear()
        st.rerun()

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Ask a financial question (e.g., 'Compare Apple and Nvidia revenue in 2023')"):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching filings and analyzing..."):
            try:
                result = ask_finance(rag_chain, prompt)
                answer = result["answer"]
                
                # Format sources
                sources = sorted(list(set([
                    f"- {doc.metadata.get('source', 'Unknown')} (Page {doc.metadata.get('page', '?')})" 
                    for doc in result['context']
                ])))
                
                full_response = answer + "\n\n**Sources:**\n" + "\n".join(sources)
                
                st.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            except Exception as e:
                error_msg = f"Sorry, I encountered an error: {e}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
