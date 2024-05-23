import streamlit as st
from streamlit_chat import message
import tempfile
import torch
import torch.nn as nn
from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import AutoModel, AutoTokenizer, AutoConfig, AutoModelForCausalLM
from langchain.llms import CTransformers
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import CTransformers
from langchain.docstore.document import Document
import PyPDF2

DB_FAISS_PATH = 'vectorstore/db_faiss'

# Loading the model
def load_llm():
    model_name = "TheBloke/nsql-llama-2-7B-GGUF"
    model_file = "nsql-llama-2-7b.Q6_K.gguf"
    llm = CTransformers(model=model_name, config_file=model_file, model_type="llama", gpu_layers=0)
    return llm

st.title("PDF reader using Llama2 🦙🦙")
st.markdown("<h3 style='text-align: center; color: grey;'>Built by <a href='https://github.com/TarunKatal'>Tarun Katal </a></h3>", unsafe_allow_html=True)

uploaded_file = st.sidebar.file_uploader("Upload your Data", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    # Optimize Page Parsing
    loader = PyPDF2.PdfReader(tmp_file_path)
    data = [Document(page_content=page.extract_text()) for page in loader.pages]

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})

    # Optimize FAISS Index
    db = FAISS.from_documents(data, embeddings)

    llm = load_llm()
    chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=db.as_retriever())

    @st.cache_data  # Cache expensive computations
    def conversational_chat(query):
        result = chain({"question": query, "chat_history": st.session_state['history']})
        st.session_state['history'].append((query, result["answer"]))
        return result["answer"]

    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello ! Ask me anything about " + uploaded_file.name + " 🤗"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hi!"]

    # Container for the chat history
    response_container = st.container()

    # Container for the user's text input
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Query:", placeholder="Talk to your pdf here (:", key='input')
            submit_button = st.form_submit_button(label='Send')

            if submit_button and user_input:
                output = conversational_chat(user_input)
                st.session_state['past'].append(user_input)
                st.session_state['generated'].append(output)

            if st.session_state['generated']:
                with response_container:
                    for i in range(len(st.session_state['generated'])):
                        message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="adventurer")
                        message(st.session_state["generated"][i], key=str(i), avatar_style="identicon")