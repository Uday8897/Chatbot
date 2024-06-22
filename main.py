import streamlit as st
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
import os
import traceback

def load_vectorstore():
    embeddings_model = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
    if os.path.exists("ipl2023embeddings"):
        st.write("Loading existing embeddings...")
        try:
            vectorstore = FAISS.load_local("ipl2023embeddings", embeddings_model, allow_dangerous_deserialization=True)
            return vectorstore
        except Exception as e:
            st.error(f"Error loading embeddings: {str(e)}")
            if st.checkbox("Show detailed error"):
                st.error(traceback.format_exc())
            return None
    else:
        st.error("Embeddings file 'ipl2023embeddings' not found. Please ensure it exists.")
        return None

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    
    if 'chat_history' in response:
        st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def get_conversation_chain(vectorstore):
    llm = HuggingFaceHub(repo_id="google/flan-t5-large", model_kwargs={"temperature": 0.5, "max_length": 512})
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    retriever = vectorstore.as_retriever()
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory
    )
    return conversation_chain

def main():
    load_dotenv()
    
    huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")
    if not huggingface_api_key:
        st.error("HuggingFace API key not found. Please set the HUGGINGFACE_API_KEY environment variable.")
        return

    st.set_page_config(page_title="Chat with IPL 2023", page_icon=":cricket_game:")
    st.write(css, unsafe_allow_html=True)

    st.header("Chat about IPL 2023 :cricket_game:")

    # Load vectorstore and initialize conversation at startup
    if "conversation" not in st.session_state:
        vectorstore = load_vectorstore()
        if vectorstore:
            st.session_state.conversation = get_conversation_chain(vectorstore)
            st.success("Ready to answer questions about IPL 2023!")
        else:
            st.session_state.conversation = None
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    user_question = st.text_input("Ask anything about IPL 2023")
    if user_question:
        if st.session_state.conversation is None:
            st.error("Error: Conversation not initialized. Embeddings may be missing or failed to load.")
        else:
            handle_userinput(user_question)

    # with st.sidebar:
    #     if debug_mode:
    #         st.write(f"Conversation initialized: {st.session_state.conversation is not None}")
    #         st.write(f"Embeddings file exists: {os.path.exists('ipl2023embeddings')}")
    #         if st.session_state.conversation:
    #             st.write(f"Conversation type: {type(st.session_state.conversation)}")

if __name__ == "__main__":
    main()